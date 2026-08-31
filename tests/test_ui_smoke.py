from __future__ import annotations

from pathlib import Path

from vcap import PRESETS_DIR
from vcap.ui.app import build_app
from vcap.ui.tabs import settings_tab


def test_build_app_registry_and_presets_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings_tab, "APP_SETTINGS_PATH", tmp_path / "app_settings.json")
    marker = Path(PRESETS_DIR) / ".last_used_preset.txt"
    marker_existed = marker.exists()
    marker_text = marker.read_text(encoding="utf-8") if marker_existed else ""
    demo = build_app()
    try:
        registry = demo.vcap_context.settings_registry
        assert len(registry.keys()) > 60
        assert all(entry.description for entry in registry.entries())
        defaults = registry.defaults()
        assert defaults["batch_limit_items"] == 0
        assert defaults["desktop_notification_on_finish"] is False
        assert defaults["play_sound_on_finish"] is False
        assert defaults["open_output_folder_on_single_finish"] is False
        assert demo.vcap_context.preset_store.list_presets()
        first = demo.vcap_context.preset_store.list_presets()[0]
        assert first.name == "Default - Qwen3-Omni Instruct video"
        if not marker_existed:
            assert demo.vcap_context.preset_store.startup_preset_name() == first.name
        loaded = demo.vcap_context.preset_store.load(first.name)
        values = registry.dict_to_values(loaded)
        roundtrip = registry.values_to_dict(values)
        assert all(roundtrip[key] == value for key, value in loaded.items())
    finally:
        demo.vcap_context.pipeline_client.shutdown()
        if marker_existed:
            marker.write_text(marker_text, encoding="utf-8")
        else:
            marker.unlink(missing_ok=True)
