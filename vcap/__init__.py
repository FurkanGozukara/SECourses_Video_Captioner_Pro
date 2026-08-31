"""Application identity and filesystem locations.

Importing :mod:`vcap` only computes constants. Runtime directories are created
explicitly through :func:`ensure_app_dirs`.
"""

from __future__ import annotations

import os
from pathlib import Path

VERSION = "1.0.0"
APP_NAME = "SECourses Video Captioner Pro"
APP_DIR = Path(__file__).resolve().parent.parent


def _directory_from_env(name: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        return fallback.resolve()
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    expanded = os.path.expandvars(os.path.expanduser(value))
    return Path(expanded).resolve(strict=False)


MODELS_DIR = _directory_from_env("VCAP_MODELS_DIR", APP_DIR / "models")
OUTPUTS_DIR = _directory_from_env("VCAP_OUTPUTS_DIR", APP_DIR / "outputs")
TEMP_DIR = _directory_from_env("VCAP_TEMP_DIR", APP_DIR / "temp")
LOGS_DIR = _directory_from_env("VCAP_LOGS_DIR", APP_DIR / "logs")
PRESETS_DIR = (APP_DIR / "presets").resolve()
PRESETS_DEFAULT_DIR = (APP_DIR / "presets_default").resolve()


def ensure_app_dirs() -> None:
    """Create all application-owned runtime and preset directories."""

    for directory in (
        MODELS_DIR,
        OUTPUTS_DIR,
        TEMP_DIR,
        LOGS_DIR,
        PRESETS_DIR,
        PRESETS_DEFAULT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "APP_DIR",
    "APP_NAME",
    "LOGS_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "PRESETS_DEFAULT_DIR",
    "PRESETS_DIR",
    "TEMP_DIR",
    "VERSION",
    "ensure_app_dirs",
]
