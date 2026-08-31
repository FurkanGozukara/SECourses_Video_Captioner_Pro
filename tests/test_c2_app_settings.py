from __future__ import annotations

from pathlib import Path

import vcap
import pytest
from vcap.core.app_settings import load_app_settings, save_app_settings
from vcap.core.paths import normalize_path


def test_app_settings_unicode_round_trip_and_corrupt_fallback(tmp_path: Path) -> None:
    settings_path = tmp_path / "config" / "app_settings.json"
    values = {
        "outputs_dir": tmp_path / "çıktılar 日本語",
        "temp_dir": tmp_path / "geçici dosyalar",
        "models_dir": tmp_path / "mödels",
        "save_processed_files": True,
        "scan_subfolders": True,
        "desktop_notification_on_finish": True,
        "play_sound_on_finish": True,
        "open_output_folder_on_single_finish": True,
    }

    assert save_app_settings(values, settings_path) == normalize_path(settings_path)
    loaded = load_app_settings(settings_path)
    assert loaded == {
        "outputs_dir": str(normalize_path(values["outputs_dir"])),
        "temp_dir": str(normalize_path(values["temp_dir"])),
        "models_dir": str(normalize_path(values["models_dir"])),
        "save_processed_files": True,
        "scan_subfolders": True,
        "desktop_notification_on_finish": True,
        "play_sound_on_finish": True,
        "open_output_folder_on_single_finish": True,
    }
    assert not list(settings_path.parent.glob("*.tmp"))

    settings_path.write_text("{not valid json", encoding="utf-8")
    assert load_app_settings(settings_path) == {}


def test_directory_resolution_prefers_environment_then_app_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = tmp_path / "configured outputs"
    from_environment = tmp_path / "environment outputs"
    monkeypatch.setitem(vcap._APP_SETTINGS, "outputs_dir", str(configured))
    monkeypatch.delenv("VCAP_C2_OUTPUTS", raising=False)
    assert vcap._directory_from_env("VCAP_C2_OUTPUTS", "outputs_dir", tmp_path / "default") == normalize_path(configured)

    monkeypatch.setenv("VCAP_C2_OUTPUTS", str(from_environment))
    assert vcap._directory_from_env("VCAP_C2_OUTPUTS", "outputs_dir", tmp_path / "default") == normalize_path(from_environment)
