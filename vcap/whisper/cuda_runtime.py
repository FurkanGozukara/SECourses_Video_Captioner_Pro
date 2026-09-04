"""Find and preload pip-installed CUDA 12 libraries for CTranslate2.

The application also ships a CUDA 13 PyTorch environment. CTranslate2 does not
import torch and its Windows wheel currently needs CUDA 12 cuBLAS beside it, so
the Whisper worker configures these paths before importing ``faster_whisper``.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import platform
import site
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_DLL_DIRECTORY_HANDLES: list[object] = []
_PRELOADED_LIBRARY_HANDLES: list[object] = []
_CUDA_RUNTIME_CONFIGURED = False
_CONFIGURED_DIRECTORIES: list[str] = []

_NVIDIA_RUNTIME_MODULES = (
    "nvidia.cublas.bin",
    "nvidia.cublas.lib",
    "nvidia.cudnn.bin",
    "nvidia.cudnn.lib",
    "nvidia.cuda_runtime.bin",
    "nvidia.cuda_runtime.lib",
)

_WINDOWS_LIBRARY_PATTERNS = (
    "cudart64_*.dll",
    "cublasLt64_*.dll",
    "cublas64_*.dll",
    "cudnn*.dll",
    "nvblas64_*.dll",
)

_LINUX_LIBRARY_PATTERNS = (
    "libcudart.so*",
    "libcublasLt.so*",
    "libcublas.so*",
    "libcudnn*.so*",
)


def _dedupe_existing_dirs(paths: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not path:
            continue
        try:
            normalized = os.path.normpath(os.fspath(path))
            if not os.path.isdir(normalized):
                continue
        except (OSError, TypeError, ValueError):
            continue
        key = normalized.casefold() if os.name == "nt" else normalized
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _module_search_dirs(module: object) -> list[str]:
    candidates: list[str] = []
    try:
        module_file = getattr(module, "__file__", None)
        if module_file:
            module_path = Path(module_file).resolve(strict=False)
            candidates.append(str(module_path if module_path.is_dir() else module_path.parent))
        module_path_attr = getattr(module, "__path__", None)
        if module_path_attr:
            candidates.extend(str(Path(path).resolve(strict=False)) for path in module_path_attr)
        module_spec = getattr(module, "__spec__", None)
        search_locations = getattr(module_spec, "submodule_search_locations", None)
        if search_locations:
            candidates.extend(str(Path(path).resolve(strict=False)) for path in search_locations)
    except Exception as exc:
        logger.warning("Could not inspect a CUDA runtime package: %s", exc)
    return _dedupe_existing_dirs(candidates)


def _discover_dirs_from_nvidia_modules() -> list[str]:
    candidates: list[str] = []
    for module_name in _NVIDIA_RUNTIME_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidates.extend(_module_search_dirs(module))
    return _dedupe_existing_dirs(candidates)


def _candidate_site_package_roots() -> list[str]:
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception as exc:
        logger.warning("Could not inspect site-packages for CUDA libraries: %s", exc)
    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass
    candidates.extend(
        path
        for path in sys.path
        if "site-packages" in str(path) or "dist-packages" in str(path)
    )
    return _dedupe_existing_dirs(candidates)


def _is_within_current_env(path: str) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(Path(sys.prefix).resolve(strict=False))
        return True
    except Exception:
        return False


def _discover_dirs_from_site_packages(
    search_roots: Iterable[str] | None = None,
) -> list[str]:
    candidates: list[str] = []
    for root in search_roots or _candidate_site_package_roots():
        try:
            nvidia_root = Path(root) / "nvidia"
            if not nvidia_root.is_dir():
                continue
            for package_name in ("cublas", "cudnn", "cuda_runtime"):
                package_root = nvidia_root / package_name
                for subdir in ("bin", "lib"):
                    candidate = package_root / subdir
                    if candidate.is_dir():
                        candidates.append(str(candidate.resolve(strict=False)))
        except OSError as exc:
            logger.warning("Could not inspect CUDA package directory %s: %s", root, exc)
    return _dedupe_existing_dirs(candidates)


def _discover_dirs_from_current_env_site_packages() -> list[str]:
    roots = [root for root in _candidate_site_package_roots() if _is_within_current_env(root)]
    return _discover_dirs_from_site_packages(search_roots=roots)


def _discover_dirs_from_cuda_env() -> list[str]:
    candidates: list[Path] = []
    system = platform.system()
    try:
        environment = list(os.environ.items())
    except Exception:
        environment = []
    for env_name, env_value in environment:
        if not env_value:
            continue
        if env_name in {"CUDA_PATH", "CUDA_HOME", "CUDNN_HOME"} or env_name.startswith(
            "CUDA_PATH_V"
        ):
            base_path = Path(env_value)
            if system == "Windows":
                candidates.extend((base_path / "bin", base_path / "lib" / "x64"))
            elif system == "Linux":
                candidates.extend((base_path / "lib64", base_path / "lib"))
    return _dedupe_existing_dirs(candidates)


def discover_cuda_runtime_dirs() -> list[str]:
    """Return CUDA runtime package/toolkit library directories without raising."""

    try:
        if platform.system() not in {"Linux", "Windows"}:
            return []
        return _dedupe_existing_dirs(
            [
                *_discover_dirs_from_nvidia_modules(),
                *_discover_dirs_from_current_env_site_packages(),
                *_discover_dirs_from_cuda_env(),
                *_discover_dirs_from_site_packages(),
            ]
        )
    except Exception as exc:
        logger.warning("CUDA runtime discovery failed: %s", exc)
        return []


def _prepend_env_dirs(variable_name: str, directories: Iterable[str]) -> None:
    try:
        separator = ";" if platform.system() == "Windows" else ":"
        existing = os.environ.get(variable_name, "")
        merged = list(directories)
        if existing:
            merged.extend(part for part in existing.split(separator) if part)
        # Preserve existing entries that temporarily do not exist; only discovered
        # directories need filesystem validation.
        seen: set[str] = set()
        result: list[str] = []
        for entry in merged:
            normalized = os.path.normpath(str(entry))
            key = normalized.casefold() if os.name == "nt" else normalized
            if key not in seen:
                seen.add(key)
                result.append(normalized)
        os.environ[variable_name] = separator.join(result)
    except Exception as exc:
        logger.warning("Could not update %s for CUDA libraries: %s", variable_name, exc)


def _register_windows_dll_directories(directories: Iterable[str]) -> None:
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return
    for directory in directories:
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))
        except Exception as exc:
            logger.warning("Could not register CUDA DLL directory %s: %s", directory, exc)


def _iter_library_files(directories: Iterable[str]):
    patterns = (
        _WINDOWS_LIBRARY_PATTERNS
        if platform.system() == "Windows"
        else _LINUX_LIBRARY_PATTERNS
    )
    directory_list = list(directories)
    seen: set[str] = set()
    for pattern in patterns:
        for directory in directory_list:
            try:
                candidates = sorted(
                    Path(directory).glob(pattern),
                    key=lambda path: (len(path.name), path.name.casefold()),
                )
            except OSError as exc:
                logger.warning("Could not inspect CUDA library directory %s: %s", directory, exc)
                continue
            for candidate in candidates:
                try:
                    if not candidate.is_file():
                        continue
                    resolved = str(candidate.resolve(strict=False))
                except OSError:
                    continue
                key = resolved.casefold() if os.name == "nt" else resolved
                if key in seen:
                    continue
                seen.add(key)
                yield resolved


def _preload_libraries(directories: Iterable[str]) -> None:
    loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for library_path in _iter_library_files(directories):
        try:
            if platform.system() == "Windows":
                _PRELOADED_LIBRARY_HANDLES.append(loader(library_path))
            else:
                _PRELOADED_LIBRARY_HANDLES.append(loader(library_path, mode=mode))
        except Exception as exc:
            logger.warning("Could not preload CUDA library %s: %s", library_path, exc)


def enable_cuda_runtime_autodiscovery() -> list[str]:
    """Configure CUDA runtime search paths idempotently and never raise."""

    global _CUDA_RUNTIME_CONFIGURED, _CONFIGURED_DIRECTORIES
    try:
        if _CUDA_RUNTIME_CONFIGURED:
            return list(_CONFIGURED_DIRECTORIES)
        directories = discover_cuda_runtime_dirs()
        if not directories:
            logger.warning("No CUDA runtime library directories were discovered")
            return []
        if platform.system() == "Windows":
            _prepend_env_dirs("PATH", directories)
            _register_windows_dll_directories(directories)
        elif platform.system() == "Linux":
            _prepend_env_dirs("LD_LIBRARY_PATH", directories)
        _preload_libraries(directories)
        _CONFIGURED_DIRECTORIES = list(directories)
        _CUDA_RUNTIME_CONFIGURED = True
        logger.info("CUDA runtime library paths configured: %s", ", ".join(directories))
        return list(directories)
    except Exception as exc:
        logger.warning("CUDA runtime auto-configuration failed: %s", exc)
        return []


__all__ = ["discover_cuda_runtime_dirs", "enable_cuda_runtime_autodiscovery"]
