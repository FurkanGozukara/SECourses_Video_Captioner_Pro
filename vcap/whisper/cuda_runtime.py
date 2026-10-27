"""Use the same venv CUDA 13 libraries as the captioning engine."""
from __future__ import annotations

import logging

from vcap.core.cuda_runtime import configure_cuda_runtime, runtime_library_dirs

logger = logging.getLogger(__name__)
_CUDA_RUNTIME_CONFIGURED = False
_CONFIGURED_DIRECTORIES: list[str] = []


def discover_cuda_runtime_dirs() -> list[str]:
    """Return only the current venv's PyTorch/CUDA 13 runtime directories."""
    return runtime_library_dirs()


def enable_cuda_runtime_autodiscovery() -> list[str]:
    """Configure the shared runtime once; CPU transcription remains available."""
    global _CUDA_RUNTIME_CONFIGURED, _CONFIGURED_DIRECTORIES
    if _CUDA_RUNTIME_CONFIGURED:
        return list(_CONFIGURED_DIRECTORIES)
    try:
        _CONFIGURED_DIRECTORIES = configure_cuda_runtime()
        _CUDA_RUNTIME_CONFIGURED = True
        logger.info("Whisper CUDA 13 runtime directories: %s", ", ".join(_CONFIGURED_DIRECTORIES))
        return list(_CONFIGURED_DIRECTORIES)
    except Exception as exc:
        logger.warning("CUDA runtime configuration failed: %s", exc)
        return []


__all__ = ["discover_cuda_runtime_dirs", "enable_cuda_runtime_autodiscovery"]
