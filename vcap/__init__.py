"""Application identity and filesystem locations.

Importing :mod:`vcap` only computes constants. Runtime directories are created
explicitly through :func:`ensure_app_dirs`.
"""

from __future__ import annotations

import os
from pathlib import Path

from .core.app_settings import APP_SETTINGS_PATH, load_app_settings
from .core.paths import normalize_path

VERSION = "1.4.0"
APP_NAME = "SECourses Video Captioner Pro"
APP_DIR = Path(__file__).resolve().parent.parent


_APP_SETTINGS = load_app_settings(APP_SETTINGS_PATH)


def _directory_from_env(name: str, setting_key: str, fallback: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raw = _APP_SETTINGS.get(setting_key)
    return normalize_path(raw) if raw else normalize_path(fallback)


MODELS_DIR = _directory_from_env("VCAP_MODELS_DIR", "models_dir", APP_DIR / "models")
OUTPUTS_DIR = _directory_from_env("VCAP_OUTPUTS_DIR", "outputs_dir", APP_DIR / "outputs")
TEMP_DIR = _directory_from_env("VCAP_TEMP_DIR", "temp_dir", APP_DIR / "temp")
LOGS_DIR = _directory_from_env("VCAP_LOGS_DIR", "logs_dir", APP_DIR / "logs")
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
    "APP_SETTINGS_PATH",
    "LOGS_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "PRESETS_DEFAULT_DIR",
    "PRESETS_DIR",
    "TEMP_DIR",
    "VERSION",
    "ensure_app_dirs",
]
