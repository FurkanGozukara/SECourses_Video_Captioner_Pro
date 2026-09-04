"""Bridge model readiness checks to the standalone downloader process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from vcap import APP_DIR
from vcap.core.logs import get_log
from vcap.core.subprocess_runner import build_child_env, kill_process_tree

from .registry import MODELS_DIR, get_variant, resolve_model_dir, variant_is_ready


ProgressCallback = Callable[..., None]
_DISK_USAGE_CACHE: dict[str, int] = {}
_DISK_USAGE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class DeleteReport:
    """Outcome of deleting one model variant's local artifacts."""

    variant_key: str
    folder: str
    files_removed: int
    bytes_freed: int
    errors: list[str]


def _variant_folder(variant_key: str, *, require_inside: bool) -> Path:
    folder = resolve_model_dir(str(variant_key)).resolve(strict=False)
    root = Path(MODELS_DIR).resolve(strict=False)
    inside = folder != root
    if inside:
        try:
            folder.relative_to(root)
        except ValueError:
            inside = False
    if require_inside and not inside:
        raise ValueError(f"Refusing to delete model files outside MODELS_DIR: {folder}")
    return folder


def invalidate_variant_disk_usage(variant_key: str | None = None) -> None:
    """Invalidate cached recursive size totals after a disk-changing action."""

    with _DISK_USAGE_CACHE_LOCK:
        if variant_key is None:
            _DISK_USAGE_CACHE.clear()
        else:
            _DISK_USAGE_CACHE.pop(str(variant_key), None)


def variant_disk_usage(variant_key: str) -> int:
    """Return bytes used by a variant folder, including partial downloads."""

    key = str(variant_key)
    with _DISK_USAGE_CACHE_LOCK:
        cached = _DISK_USAGE_CACHE.get(key)
    if cached is not None:
        return cached
    folder = _variant_folder(key, require_inside=False)
    try:
        if not folder.exists() or not folder.is_dir():
            total = 0
            with _DISK_USAGE_CACHE_LOCK:
                _DISK_USAGE_CACHE[key] = total
            return total
    except OSError:
        return 0
    total = 0
    pending = [folder]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                else:
                    total += max(0, int(entry.stat(follow_symlinks=False).st_size))
            except OSError:
                continue
    with _DISK_USAGE_CACHE_LOCK:
        _DISK_USAGE_CACHE[key] = total
    return total


def delete_variant_files(variant_key: str) -> DeleteReport:
    """Delete one variant folder without following links or leaving partial files."""

    key = str(variant_key)
    invalidate_variant_disk_usage(key)
    folder = _variant_folder(key, require_inside=True)
    errors: list[str] = []
    files_removed = 0
    bytes_freed = 0
    try:
        exists = folder.exists() or folder.is_symlink()
    except OSError as exc:
        return DeleteReport(key, str(folder), 0, 0, [f"{folder}: {exc}"])
    if not exists:
        return DeleteReport(key, str(folder), 0, 0, [])

    def remove_directory(directory: Path) -> None:
        nonlocal files_removed, bytes_freed
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            errors.append(f"{directory}: {exc}")
            return
        for entry in entries:
            path = Path(entry.path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
            if is_directory:
                remove_directory(path)
                try:
                    path.rmdir()
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
                continue
            try:
                size = max(0, int(entry.stat(follow_symlinks=False).st_size))
            except OSError:
                size = 0
            try:
                path.unlink()
            except OSError as exc:
                errors.append(f"{path}: {exc}")
            else:
                files_removed += 1
                bytes_freed += size

    if folder.is_symlink():
        try:
            size = max(0, int(folder.lstat().st_size))
            folder.unlink()
        except OSError as exc:
            errors.append(f"{folder}: {exc}")
        else:
            files_removed = 1
            bytes_freed = size
    else:
        remove_directory(folder)
        try:
            folder.rmdir()
        except OSError as exc:
            errors.append(f"{folder}: {exc}")
    invalidate_variant_disk_usage(key)
    return DeleteReport(key, str(folder), files_removed, bytes_freed, errors)


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


def _emit(callback: ProgressCallback | None, line: str, payload: dict[str, Any] | None = None) -> None:
    display = format_status_line(line, payload)
    get_log().log(display, scope="downloads")
    if callback is None:
        return
    for args in ((display, payload), (display,), (payload or {"message": display},)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _parse_status(line: str) -> dict[str, Any] | None:
    """Parse JSON status lines, the legacy text protocol, or plain percentages."""

    marker = "VCAP_STATUS"
    percent_match = re.search(r"(?<![\d.])(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%", line)
    fraction = (
        min(1.0, max(0.0, float(percent_match.group(1)) / 100.0))
        if percent_match
        else None
    )
    if marker not in line:
        return {"message": line, "fraction": fraction} if fraction is not None else None
    tail = line.split(marker, 1)[1].lstrip(" :=\t")
    if not tail:
        return {"message": line}
    try:
        value = json.loads(tail)
    except json.JSONDecodeError:
        parts = tail.split(maxsplit=2)
        if len(parts) >= 2:
            payload: dict[str, Any] = {
                "key": parts[0],
                "state": parts[1],
                "message": parts[2] if len(parts) > 2 else "",
            }
            if fraction is not None:
                payload["fraction"] = fraction
            return payload
        return {"message": tail, **({"fraction": fraction} if fraction is not None else {})}
    if not isinstance(value, dict):
        return {"value": value}
    payload = dict(value)
    raw_fraction = payload.get("fraction")
    if raw_fraction is not None:
        try:
            payload["fraction"] = min(1.0, max(0.0, float(raw_fraction)))
        except (TypeError, ValueError):
            payload["fraction"] = None
    elif fraction is not None:
        payload["fraction"] = fraction
    return payload


def format_status_line(
    line: str,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Render downloader protocol payloads as concise human-readable progress."""

    raw = str(line).strip()
    parsed = dict(payload) if isinstance(payload, Mapping) and payload else _parse_status(raw)
    if not parsed:
        return raw
    protocol = "VCAP_STATUS" in raw or bool(parsed.get("key"))
    if not protocol:
        return raw
    key = str(parsed.get("key") or "").strip()
    message = str(parsed.get("message") or parsed.get("state") or "Working").strip()
    if "VCAP_STATUS" in message:
        nested = _parse_status(message)
        if nested:
            message = str(nested.get("message") or nested.get("state") or "Working").strip()
    if key and not message.casefold().startswith(f"{key}:".casefold()):
        message = f"{key}: {message}"
    fraction = parsed.get("fraction")
    if fraction is not None and not re.search(r"\d(?:\.\d+)?\s*%", message):
        try:
            percent = min(100.0, max(0.0, float(fraction) * 100.0))
            percent_text = f"{percent:.0f}" if abs(percent - round(percent)) < 0.05 else f"{percent:.1f}"
            message += f" ({percent_text}%)"
        except (TypeError, ValueError):
            pass
    return message


def _find_downloader() -> Path | None:
    override = os.environ.get("SECOURSES_VCAP_DOWNLOADER", "").strip().strip('"').strip("'")
    candidates = [
        Path(override).expanduser() if override else None,
        APP_DIR.parent / "Models_Downloader.py",
        APP_DIR / "Models_Downloader.py",
    ]
    return next((path.resolve(strict=False) for path in candidates if path is not None and path.is_file()), None)


def _venv_python() -> Path:
    candidates = [
        APP_DIR / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        Path(sys.executable),
    ]
    return next((path.resolve(strict=False) for path in candidates if path.is_file()), Path(sys.executable))


def ensure_model(
    variant_key: str,
    progress_cb: ProgressCallback | None = None,
    cancel: object | None = None,
) -> tuple[bool, str]:
    """Ensure a local HF variant via ``Models_Downloader.py --ensure``."""

    variant = get_variant(variant_key)
    ready, detail = variant_is_ready(variant_key)
    if ready:
        invalidate_variant_disk_usage(variant_key)
        _emit(
            progress_cb,
            f"{variant_key}: {detail}",
            {"key": variant_key, "state": "ready", "fraction": 1.0, "message": detail},
        )
        return True, detail
    if variant.backend == "llamacpp":
        try:
            from .llamacpp_backend import ensure_gguf

            ensure_gguf(variant.key, progress_cb, cancel)
        except Exception as exc:
            message = f"GGUF download failed: {exc}"
            _emit(progress_cb, message)
            return False, message
        ready, detail = variant_is_ready(variant.key)
        invalidate_variant_disk_usage(variant.key)
        message = f"{variant.key}: {detail}"
        _emit(
            progress_cb,
            message,
            {"key": variant.key, "state": "ready" if ready else "error", "fraction": 1.0 if ready else None, "message": detail},
        )
        return ready, message
    downloader = _find_downloader()
    if downloader is None:
        message = (
            "Models_Downloader.py was not found. Set SECOURSES_VCAP_DOWNLOADER "
            "or place it beside the application folder."
        )
        _emit(progress_cb, message)
        return False, message
    if _cancelled(cancel):
        return False, "model download cancelled before launch"

    command = [str(_venv_python()), "-u", str(downloader), "--ensure", variant_key]
    _emit(progress_cb, f"Starting model downloader for {variant_key}")
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            cwd=str(APP_DIR),
            env=build_child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
            **popen_kwargs,
        )
    except OSError as exc:
        message = f"Could not start model downloader: {exc}"
        _emit(progress_cb, message)
        return False, message

    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for raw in iter(process.stdout.readline, ""):
                lines.put(raw)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, name="vcap-model-download-output", daemon=True)
    reader.start()
    cancelled = False
    output_finished = False
    while not output_finished:
        if _cancelled(cancel):
            cancelled = True
            kill_process_tree(process.pid)
            break
        try:
            raw = lines.get(timeout=0.1)
        except queue.Empty:
            if process.poll() is not None and not reader.is_alive():
                break
            continue
        if raw is None:
            output_finished = True
            continue
        line = raw.rstrip("\r\n")
        if line:
            _emit(progress_cb, line, _parse_status(line))
    reader.join(timeout=2.0)
    process.stdout.close()
    try:
        return_code = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid)
        return_code = -1
    if cancelled:
        invalidate_variant_disk_usage(variant_key)
        message = "model download cancelled"
        _emit(progress_cb, message)
        return False, message
    if return_code != 0:
        invalidate_variant_disk_usage(variant_key)
        message = f"model downloader exited with code {return_code}"
        _emit(progress_cb, message)
        return False, message

    ready, detail = variant_is_ready(variant_key)
    invalidate_variant_disk_usage(variant_key)
    message = f"{variant_key}: {detail}"
    _emit(progress_cb, message)
    return ready, message


__all__ = [
    "DeleteReport",
    "delete_variant_files",
    "ensure_model",
    "format_status_line",
    "invalidate_variant_disk_usage",
    "variant_disk_usage",
]
