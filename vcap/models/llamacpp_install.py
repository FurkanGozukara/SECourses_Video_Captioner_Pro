"""Install a pinned CUDA build of llama.cpp for the GGUF backend.

The application pins b10621, the nightly referenced by llama.cpp v0.3.0.  It
is comfortably newer than the b8775 minimum needed for Qwen3-Omni audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Iterable
import zipfile

import requests

from vcap import APP_DIR
from vcap.core.logs import get_log, setup_utf8_stdio
from vcap.core.subprocess_runner import CancelledError, build_child_env


ProgressCallback = Callable[..., None]

LLAMACPP_BUILD = 10_621
LLAMACPP_TAG = f"b{LLAMACPP_BUILD}"
MIN_QWEN3_OMNI_BUILD = 8_775
RELEASE_API = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMACPP_TAG}"
RELEASE_ROOT = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TAG}"

# Pinned fallbacks make installation reproducible even if the GitHub API is
# unavailable. GitHub release metadata is still queried first and its digest is
# used when present.
_PINNED_WINDOWS_ASSETS: tuple[dict[str, Any], ...] = (
    {
        "name": f"llama-{LLAMACPP_TAG}-bin-win-cuda-13.3-x64.zip",
        "size": 146_446_450,
        "sha256": "23549ccc00b6a18d74348e95d4789f7e96c9efb11cf6e3f1b185baef34d7449f",
    },
    {
        "name": "cudart-llama-bin-win-cuda-13.3-x64.zip",
        "size": 390_970_417,
        "sha256": "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
    },
)


def _cancelled(token: object | None) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "is_set"):
        method = getattr(token, name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
    return bool(getattr(token, "cancelled", False))


def _emit(callback: ProgressCallback | None, message: str, **data: Any) -> None:
    get_log().log(message, scope="llama.cpp")
    if callback is None:
        return
    payload = {"message": message, **data}
    for args in ((message, payload), (payload,), (message,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _sha256(path: Path, cancel: object | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if _cancelled(cancel):
                raise CancelledError("download verification cancelled")
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(
    url: str,
    destination: str | os.PathLike[str],
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel: object | None = None,
    label: str | None = None,
) -> Path:
    """Download one file with HTTP Range resume and optional SHA-256 check."""

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    name = label or target.name
    expected_hash = (expected_sha256 or "").removeprefix("sha256:").casefold() or None

    if target.is_file():
        size_ok = expected_size is None or target.stat().st_size == int(expected_size)
        hash_ok = expected_hash is None or _sha256(target, cancel) == expected_hash
        if size_ok and hash_ok:
            _emit(progress_cb, f"Already downloaded: {name}", fraction=1.0)
            return target
        target.unlink(missing_ok=True)

    resumed = part.stat().st_size if part.is_file() else 0
    if expected_size is not None and resumed > int(expected_size):
        part.unlink(missing_ok=True)
        resumed = 0
    if expected_size is not None and resumed == int(expected_size):
        if expected_hash is not None:
            _emit(progress_cb, f"Verifying resumed SHA-256: {name}")
            actual_hash = _sha256(part, cancel)
            if actual_hash != expected_hash:
                part.unlink(missing_ok=True)
                resumed = 0
        if resumed:
            os.replace(part, target)
            _emit(
                progress_cb,
                f"Resumed download is complete: {name}",
                fraction=1.0,
                bytes_done=resumed,
                bytes_total=expected_size,
            )
            return target
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "SECourses-Video-Captioner-Pro/1.0",
    }
    if resumed:
        headers["Range"] = f"bytes={resumed}-"
    _emit(
        progress_cb,
        f"{'Resuming' if resumed else 'Downloading'} {name} at {resumed / 1e9:.2f} GB",
        bytes_done=resumed,
        bytes_total=expected_size,
    )

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download {name}: {exc}") from exc

    if resumed and response.status_code != 206:
        resumed = 0
        mode = "wb"
    else:
        mode = "ab" if resumed else "wb"
    content_length = response.headers.get("Content-Length")
    response_total = None
    try:
        response_total = resumed + int(content_length) if content_length is not None else None
    except ValueError:
        pass
    total = int(expected_size) if expected_size is not None else response_total
    done = resumed
    started = time.monotonic()
    last_report = started
    last_percent = -1
    try:
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if _cancelled(cancel):
                    raise CancelledError(f"download cancelled: {name}")
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                percent = int(done * 100 / total) if total else -1
                if now - last_report >= 2.0 or (percent >= 0 and percent >= last_percent + 2):
                    speed = max(0.0, done - resumed) / max(now - started, 1e-6)
                    progress = f"{done / 1e9:.2f} GB"
                    if total:
                        progress += f" / {total / 1e9:.2f} GB ({done * 100 / total:.1f}%)"
                    _emit(
                        progress_cb,
                        f"Downloading {name}: {progress}, {speed / 1e6:.1f} MB/s",
                        fraction=(done / total if total else None),
                        bytes_done=done,
                        bytes_total=total,
                        bytes_per_second=speed,
                    )
                    last_report = now
                    last_percent = percent
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        response.close()

    if expected_size is not None and done != int(expected_size):
        raise RuntimeError(
            f"Incomplete download for {name}: got {done} bytes, expected {int(expected_size)}"
        )
    if expected_hash is not None:
        _emit(progress_cb, f"Verifying SHA-256: {name}")
        actual_hash = _sha256(part, cancel)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: got {actual_hash}, expected {expected_hash}"
            )
    os.replace(part, target)
    _emit(progress_cb, f"Downloaded {name}", fraction=1.0, bytes_done=done, bytes_total=total)
    return target


def _release_assets(progress_cb: ProgressCallback | None) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            RELEASE_API,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "SECourses-Video-Captioner-Pro/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        assets = payload.get("assets", []) if isinstance(payload, dict) else []
        if isinstance(assets, list):
            return [asset for asset in assets if isinstance(asset, dict)]
    except (requests.RequestException, ValueError) as exc:
        _emit(progress_cb, f"GitHub release metadata unavailable; using pinned asset metadata: {exc}")
    return []


def _windows_assets(remote_assets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {str(asset.get("name", "")): asset for asset in remote_assets}
    selected: list[dict[str, Any]] = []
    for fallback in _PINNED_WINDOWS_ASSETS:
        remote = indexed.get(str(fallback["name"]), {})
        raw_digest = remote.get("digest")
        digest = (
            str(raw_digest).removeprefix("sha256:")
            if raw_digest
            else fallback["sha256"]
        )
        selected.append(
            {
                "name": fallback["name"],
                "size": int(remote.get("size") or fallback["size"]),
                "sha256": digest,
                "url": remote.get("browser_download_url") or f"{RELEASE_ROOT}/{fallback['name']}",
            }
        )
    return selected


def _linux_cuda_assets(remote_assets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for asset in remote_assets:
        name = str(asset.get("name", ""))
        lowered = name.casefold()
        if "bin-ubuntu" not in lowered or "cuda" not in lowered or "x64" not in lowered:
            continue
        raw_digest = asset.get("digest")
        matches.append(
            {
                "name": name,
                "size": int(asset.get("size") or 0) or None,
                "sha256": str(raw_digest).removeprefix("sha256:") if raw_digest else None,
                "url": asset.get("browser_download_url") or f"{RELEASE_ROOT}/{name}",
            }
        )
    return matches


def _safe_member_path(root: Path, member: str) -> Path:
    target = (root / member).resolve(strict=False)
    try:
        target.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise RuntimeError(f"Unsafe archive member: {member}") from exc
    return target


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix.casefold() == ".zip":
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                _safe_member_path(destination, member.filename)
            source.extractall(destination)
        return
    if archive.name.casefold().endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                _safe_member_path(destination, member.name)
            source.extractall(destination, filter="data")
        return
    raise RuntimeError(f"Unsupported llama.cpp archive format: {archive.name}")


def _find_binary(folder: Path, stem: str) -> Path | None:
    names = {stem.casefold(), f"{stem}.exe".casefold()}
    return next(
        (path for path in folder.rglob("*") if path.is_file() and path.name.casefold() in names),
        None,
    )


def _version_build(server: Path) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            [str(server), "--version"],
            cwd=str(server.parent),
            env=build_child_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not execute {server.name} --version: {exc}") from exc
    output = "\n".join(value.strip() for value in (completed.stdout, completed.stderr) if value.strip())
    candidates = [int(value) for value in re.findall(r"\bb(\d{4,})\b", output, flags=re.IGNORECASE)]
    candidates.extend(
        int(value)
        for value in re.findall(r"\b(?:version|build)\s*[:=]?\s*(\d{4,})\b", output, flags=re.IGNORECASE)
    )
    if not candidates:
        raise RuntimeError(f"Could not parse llama.cpp build number from:\n{output[-3000:]}")
    return max(candidates), output


def _validate_install(folder: Path) -> Path:
    server = _find_binary(folder, "llama-server")
    mtmd = _find_binary(folder, "llama-mtmd-cli")
    missing = [name for name, value in (("llama-server", server), ("llama-mtmd-cli", mtmd)) if value is None]
    if missing:
        raise RuntimeError(f"llama.cpp installation is missing {', '.join(missing)} in {folder}")
    assert server is not None
    build, output = _version_build(server)
    if build < MIN_QWEN3_OMNI_BUILD:
        raise RuntimeError(
            f"llama.cpp b{build} is too old for Qwen3-Omni audio; b{MIN_QWEN3_OMNI_BUILD}+ is required.\n{output}"
        )
    return server


def _external_install() -> Path | None:
    raw = os.environ.get("VCAP_LLAMACPP_SERVER", "").strip().strip('"').strip("'")
    if not raw:
        return None
    server = Path(raw).expanduser().resolve(strict=False)
    if not server.is_file():
        raise FileNotFoundError(f"VCAP_LLAMACPP_SERVER does not exist: {server}")
    _validate_install(server.parent)
    build, output = _version_build(server)
    if build < MIN_QWEN3_OMNI_BUILD:
        raise RuntimeError(
            f"External llama.cpp b{build} is too old for Qwen3-Omni audio; "
            f"b{MIN_QWEN3_OMNI_BUILD}+ is required.\n{output}"
        )
    return server


def describe_runtime() -> dict[str, Any]:
    """Describe the configured llama.cpp runtime without installing anything."""

    raw_external = os.environ.get("VCAP_LLAMACPP_SERVER", "").strip().strip('"').strip("'")
    external = Path(raw_external).expanduser().resolve(strict=False) if raw_external else None
    install_path = external.parent if external is not None else (APP_DIR / "llamacpp" / LLAMACPP_TAG).resolve(strict=False)
    server = external if external is not None else _find_binary(install_path, "llama-server") if install_path.is_dir() else None
    result: dict[str, Any] = {
        "pinned_tag": LLAMACPP_TAG,
        "pinned_build": LLAMACPP_BUILD,
        "minimum_build": MIN_QWEN3_OMNI_BUILD,
        "install_path": str(install_path),
        "server_path": str(server) if server is not None else None,
        "binary_exists": bool(server is not None and server.is_file()),
        "runs": False,
        "installed_build": None,
        "version_output": "",
        "cuda_build": False,
        "error": None,
    }
    cuda_files = False
    if install_path.is_dir():
        try:
            cuda_files = any(
                "cuda" in path.name.casefold()
                and path.suffix.casefold() in {".dll", ".so", ".dylib"}
                for path in install_path.rglob("*")
                if path.is_file()
            )
        except OSError:
            cuda_files = False
    if server is None or not server.is_file():
        result["cuda_build"] = cuda_files
        return result
    try:
        build, output = _version_build(server)
        result.update(
            runs=True,
            installed_build=build,
            version_output=output,
            cuda_build=cuda_files or "cuda" in output.casefold(),
        )
    except RuntimeError as exc:
        result.update(error=str(exc), cuda_build=cuda_files)
    return result


def ensure_llamacpp(
    progress_cb: ProgressCallback | None = None,
    cancel: object | None = None,
    *,
    force: bool = False,
) -> Path:
    """Return a verified ``llama-server`` path, installing b10621 if needed."""

    external = _external_install()
    if external is not None:
        _emit(progress_cb, f"Using external llama.cpp: {external}")
        return external
    if _cancelled(cancel):
        raise CancelledError("llama.cpp installation cancelled before launch")

    root = (APP_DIR / "llamacpp").resolve(strict=False)
    target = root / LLAMACPP_TAG
    root.mkdir(parents=True, exist_ok=True)
    if target.is_dir() and not force:
        try:
            server = _validate_install(target)
            _emit(progress_cb, f"llama.cpp {LLAMACPP_TAG} is ready: {server}", fraction=1.0)
            return server
        except RuntimeError as exc:
            _emit(progress_cb, f"Refreshing incomplete llama.cpp install: {exc}")

    remote_assets = _release_assets(progress_cb)
    if os.name == "nt":
        assets = _windows_assets(remote_assets)
    elif sys.platform.startswith("linux"):
        assets = _linux_cuda_assets(remote_assets)
        if not assets:
            raise RuntimeError(
                f"llama.cpp {LLAMACPP_TAG} publishes no Ubuntu x64 CUDA archive. "
                "Build llama-server and llama-mtmd-cli with GGML_CUDA=ON, set "
                "VCAP_LLAMACPP_SERVER to the resulting llama-server, then use -hf or "
                "the app's local --model/--mmproj files. See docs/GGUF_BACKEND.md."
            )
    else:
        raise RuntimeError(
            "Automatic llama.cpp installation currently supports Windows x64 CUDA. "
            "Set VCAP_LLAMACPP_SERVER to a b8775+ self-built CUDA llama-server."
        )

    download_dir = root / "downloads" / LLAMACPP_TAG
    archives: list[Path] = []
    for asset in assets:
        archives.append(
            download_resumable(
                str(asset["url"]),
                download_dir / str(asset["name"]),
                expected_size=asset.get("size"),
                expected_sha256=asset.get("sha256"),
                progress_cb=progress_cb,
                cancel=cancel,
            )
        )

    staging = root / f".{LLAMACPP_TAG}.extracting.{os.getpid()}"
    backup = root / f".{LLAMACPP_TAG}.previous.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for archive in archives:
            if _cancelled(cancel):
                raise CancelledError("llama.cpp installation cancelled before extraction")
            _emit(progress_cb, f"Extracting {archive.name}")
            _extract_archive(archive, staging)
        staged_server = _validate_install(staging)
        staged_build, version_text = _version_build(staged_server)
        metadata = {
            "tag": LLAMACPP_TAG,
            "build": staged_build,
            "minimum_build": MIN_QWEN3_OMNI_BUILD,
            "version_output": version_text,
            "assets": [
                {key: asset.get(key) for key in ("name", "size", "sha256", "url")}
                for asset in assets
            ],
        }
        (staging / "vcap_install.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)
        staging.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not target.exists() and backup.exists():
            backup.rename(target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    server = _validate_install(target)
    build, _ = _version_build(server)
    _emit(progress_cb, f"llama.cpp b{build} installed at {target}", fraction=1.0)
    return server


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the pinned llama.cpp CUDA runtime")
    parser.add_argument("--force", action="store_true", help="replace an existing pinned build")
    args = parser.parse_args(argv)
    setup_utf8_stdio()
    try:
        path = ensure_llamacpp(force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "LLAMACPP_BUILD",
    "LLAMACPP_TAG",
    "MIN_QWEN3_OMNI_BUILD",
    "download_resumable",
    "describe_runtime",
    "ensure_llamacpp",
]
