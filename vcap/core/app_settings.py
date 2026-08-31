"""Persistent application-wide paths and lightweight preferences."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .paths import normalize_path


APP_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "app_settings.json"
_PATH_KEYS = ("outputs_dir", "temp_dir", "models_dir")
_BOOL_KEYS = (
    "save_processed_files",
    "scan_subfolders",
    "desktop_notification_on_finish",
    "play_sound_on_finish",
    "open_output_folder_on_single_finish",
)


def _settings_path(path: str | os.PathLike[str] | None) -> Path:
    return normalize_path(path or APP_SETTINGS_PATH)


def load_app_settings(
    path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load known global settings, tolerating missing or malformed JSON."""

    target = _settings_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}

    result: dict[str, Any] = {}
    for key in _PATH_KEYS:
        raw = payload.get(key)
        if isinstance(raw, (str, os.PathLike)) and str(raw).strip():
            try:
                result[key] = str(normalize_path(raw))
            except (OSError, TypeError, ValueError):
                continue
    for key in _BOOL_KEYS:
        value = payload.get(key)
        if isinstance(value, bool):
            result[key] = value
    return result


def save_app_settings(
    settings: Mapping[str, Any],
    path: str | os.PathLike[str] | None = None,
) -> Path:
    """Atomically persist known global settings as readable UTF-8 JSON."""

    target = _settings_path(path)
    payload: dict[str, Any] = {}
    for key in _PATH_KEYS:
        raw = settings.get(key)
        if raw is None or not str(raw).strip():
            raise ValueError(f"{key} cannot be empty")
        payload[key] = str(normalize_path(raw))
    for key in _BOOL_KEYS:
        payload[key] = bool(settings.get(key, False))

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return target


__all__ = ["APP_SETTINGS_PATH", "load_app_settings", "save_app_settings"]
