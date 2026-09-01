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
        assert "gpu_layers" not in defaults
        assert defaults["block_swap_auto"] is True
        assert defaults["blocks_to_swap"] == 0
        assert defaults["vram_reserve_gb"] == 2.0
        assert defaults["swap_slots"] == 2
        assert defaults["pin_cpu"] is True
        entries = {entry.key: entry for entry in registry.entries()}
        assert entries["vram_reserve_gb"].section == "model"
        assert entries["swap_slots"].choices == (2, 3)
        assert entries["blocks_to_swap"].maximum == 48
        # Presets saved before the block-swap controls existed still apply.
        legacy, legacy_warnings = registry.coerce(
            {"model_key": "qwen3_omni_instruct_int8", "gpu_layers": "36"}
        )
        assert not legacy_warnings
        assert (legacy["block_swap_auto"], legacy["blocks_to_swap"]) == (False, 12)
        assert demo.vcap_context.preset_store.list_presets()
        first = demo.vcap_context.preset_store.list_presets()[0]
        assert first.name == "Default - Qwen3-Omni Instruct video"
        if not marker_existed:
            assert demo.vcap_context.preset_store.startup_preset_name() == first.name
        loaded = demo.vcap_context.preset_store.load(first.name)
        values = registry.dict_to_values(loaded)
        roundtrip = registry.values_to_dict(values)
        assert all(roundtrip[key] == value for key, value in loaded.items())
        # Every shipped preset must apply through the full registry without a
        # single adjustment, including the Chat tab's preset-owned controls.
        chat_keys = {entry.key for entry in registry.entries() if entry.section == "chat"}
        assert chat_keys and all(entries[key].in_preset for key in chat_keys)
        for entry in demo.vcap_context.preset_store.list_presets():
            if not entry.is_default:
                continue
            shipped = demo.vcap_context.preset_store.load(entry.name)
            assert chat_keys <= set(shipped), entry.name
            _, warnings = registry.coerce(shipped)
            assert not warnings, (entry.name, warnings)
    finally:
        demo.vcap_context.pipeline_client.shutdown()
        if marker_existed:
            marker.write_text(marker_text, encoding="utf-8")
        else:
            marker.unlink(missing_ok=True)
