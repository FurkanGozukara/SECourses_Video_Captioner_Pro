"""Select the current venv's CUDA libraries before native workers start.

PyTorch and our CTranslate2 build share CUDA 13. The CUDA 12 and CUDA 13
cuDNN wheels must never coexist: their shared file paths overwrite each other.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

_DLL_HANDLES: list[object] = []


def _distribution(name: str):
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def torch_cudnn_requirement() -> str | None:
    torch = _distribution("torch")
    for requirement in torch.requires or [] if torch else []:
        match = re.match(r"nvidia-cudnn-cu13\s*==\s*([\d.]+)", requirement)
        if match:
            return match.group(1)
    return None


def _within_env(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(sys.prefix).resolve())
        return True
    except (OSError, ValueError):
        return False


def runtime_library_dirs() -> list[str]:
    """Find runtime libraries in this environment, without importing Torch."""
    if sys.platform not in {"win32", "linux"}:
        return []
    torch = _distribution("torch")
    if torch is None:
        return []
    root = Path(torch.locate_file(""))
    candidates = [root / "torch/lib", root / "nvidia/cu13/bin", root / "nvidia/cu13/lib"]
    if sys.platform == "linux":
        candidates.append(root / "nvidia/cudnn/lib")
    return [str(path) for path in candidates if path.is_dir() and _within_env(path)]


def _contains_cuda_runtime(directory: str) -> bool:
    try:
        return any(re.match(r"(?:lib)?(?:cudnn|cublas|cudart)", path.name, re.I)
                   for path in Path(directory).iterdir())
    except (OSError, ValueError):
        return False


def cuda_runtime_env(environment: Mapping[str, str]) -> dict[str, str]:
    """Prefer this venv and exclude competing runtime folders; retain drivers."""
    env = dict(environment)
    directories = runtime_library_dirs()
    if not directories:
        return env
    variable = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
    separator = ";" if sys.platform == "win32" else ":"
    seen = {os.path.normcase(os.path.abspath(path)) for path in directories}
    kept = []
    for path in env.get(variable, "").split(separator):
        if not path:
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key in seen or _contains_cuda_runtime(path):
            continue
        seen.add(key)
        kept.append(path)
    env[variable] = separator.join([*directories, *kept])
    # LD_PRELOAD is evaluated before Python and can override every search path.
    if sys.platform == "linux" and env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = " ".join(value for value in re.split(r"[ :]+", env["LD_PRELOAD"])
                                         if value and not re.match(r"lib(?:cudnn|cublas|cudart)", Path(value).name))
    return env


def configure_cuda_runtime() -> list[str]:
    """Register venv DLLs in the current process and prepare future children."""
    os.environ.update(cuda_runtime_env(os.environ))
    directories = runtime_library_dirs()
    if sys.platform == "win32" and hasattr(os, "add_dll_directory") and not _DLL_HANDLES:
        for directory in directories:
            _DLL_HANDLES.append(os.add_dll_directory(directory))
    return directories


def bootstrap_cuda_runtime() -> None:
    """Refresh Linux's loader paths by re-executing once, before native imports."""
    environment = cuda_runtime_env(os.environ)
    if sys.platform == "linux" and any(environment.get(name) != os.environ.get(name)
                                      for name in ("LD_LIBRARY_PATH", "LD_PRELOAD")):
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
    configure_cuda_runtime()


def _backend_requirement():
    from packaging.requirements import Requirement
    requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
    if not requirements.is_file():
        return None
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if line.startswith("ctranslate2 @"):
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate():
                return requirement
    return None


def runtime_problems(*, verify_files: bool = False) -> list[str]:
    problems = []
    for name in ("nvidia-cudnn-cu12", "nvidia-cublas-cu12"):
        if _distribution(name) is not None:
            problems.append(f"{name} is installed alongside our CUDA 13 runtime")
    required = torch_cudnn_requirement() if sys.platform == "linux" else None
    cudnn = _distribution("nvidia-cudnn-cu13")
    if required and (cudnn is None or cudnn.version != required):
        problems.append(f"PyTorch requires nvidia-cudnn-cu13=={required}")
    backend = _backend_requirement()
    if backend:
        from packaging.utils import parse_wheel_filename
        from urllib.parse import unquote, urlsplit
        version = str(parse_wheel_filename(unquote(Path(urlsplit(backend.url).path).name))[1])
        installed = _distribution("ctranslate2")
        if installed is None or installed.version != version:
            problems.append(f"Whisper requires the CUDA 13 CTranslate2 build ({version})")
    if required and cudnn:
        expected = {Path(cudnn.locate_file(file)).resolve(): file for file in cudnn.files or []
                    if str(file).startswith("nvidia/cudnn/lib/") and ".so" in str(file)}
        library_dir = Path(cudnn.locate_file("nvidia/cudnn/lib"))
        for path in library_dir.glob("libcudnn*.so*"):
            if path.resolve() not in expected:
                problems.append(f"Untracked cuDNN sublibrary: {path.name}")
        if verify_files:
            for path, record in expected.items():
                if not path.is_file():
                    problems.append(f"Missing cuDNN library: {path.name}")
                elif record.hash and record.hash.mode == "sha256":
                    with path.open("rb") as handle:
                        digest = base64.urlsafe_b64encode(hashlib.file_digest(handle, "sha256").digest()).decode().rstrip("=")
                    if digest != record.hash.value:
                        problems.append(f"cuDNN library differs from its wheel: {path.name}")
    return problems


def _pip(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *arguments], check=True,
                   stdout=sys.stderr, stderr=sys.stderr,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def repair_runtime() -> None:
    """Remove overlapping CUDA 12 packages and restore Torch's exact cuDNN wheel."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this command with the application's venv Python")
    problems = runtime_problems(verify_files=True)
    required = torch_cudnn_requirement() if sys.platform == "linux" else None
    conflicting = [name for name in ("nvidia-cudnn-cu12", "nvidia-cublas-cu12") if _distribution(name)]
    for name in ["torch", "nvidia-cudnn-cu13", *conflicting]:
        distribution = _distribution(name)
        if distribution is not None and not _within_env(Path(distribution.locate_file(""))):
            raise RuntimeError(f"Refusing to modify {name} outside the active venv")
    if conflicting:
        _pip("uninstall", "-y", *conflicting)
    if required and any("cuDNN" in problem or "nvidia-cudnn" in problem for problem in problems):
        # Retain unexpected libraries left by an already-uninstalled newer
        # cuDNN wheel. They must not remain visible to dlopen after reinstall.
        cudnn = _distribution("nvidia-cudnn-cu13")
        if cudnn:
            expected = {Path(file).name for file in cudnn.files or []
                        if str(file).startswith("nvidia/cudnn/lib/")}
            directory = Path(cudnn.locate_file("nvidia/cudnn/lib"))
            extras = [path for path in directory.glob("libcudnn*.so*") if path.name not in expected]
            for path in extras:
                if not _within_env(path) or path.is_symlink():
                    raise RuntimeError(f"Refusing to move a cuDNN library outside the venv: {path}")
            if extras:
                import time
                quarantine = Path(sys.prefix) / "cuda_runtime_quarantine" / str(time.time_ns())
                quarantine.mkdir(parents=True)
                for path in extras:
                    print(f"Preserving foreign cuDNN library: {path} -> {quarantine / path.name}", file=sys.stderr)
                    path.rename(quarantine / path.name)
        _pip("install", "--no-deps", "--force-reinstall", f"nvidia-cudnn-cu13=={required}")
    backend = _backend_requirement()
    if backend and any("CTranslate2" in item for item in problems):
        _pip("install", "--no-deps", str(backend))
    remaining = runtime_problems(verify_files=True)
    if remaining:
        raise RuntimeError("; ".join(remaining))


def prepare_cuda_runtime() -> None:
    """Migrate older installations once, then initialize the native loader."""
    problems = runtime_problems(verify_files=True)
    if problems:
        print("[runtime] Repairing CUDA dependencies: " + "; ".join(problems), file=sys.stderr, flush=True)
        repair_runtime()
    bootstrap_cuda_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    if args.repair:
        repair_runtime()
    problems = runtime_problems(verify_files=True)
    print(json.dumps({"python": sys.executable, "runtime_directories": runtime_library_dirs(),
                      "problems": problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
