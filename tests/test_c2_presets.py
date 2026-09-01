from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from vcap import PRESETS_DEFAULT_DIR
from vcap.core.logs import get_log
from vcap.core.presets import PresetStore
from vcap.core.registry import SettingsRegistry
from vcap.models.registry import variant_to_family
from vcap.pipeline.client import PipelineClient
from vcap.ui.app import UiContext
from vcap.ui.components import preset_bar, wire_preset_bar
from vcap.ui.tabs.chat_tab import model_chat_support


_CHAT_KEYS = {
    "chat_system_prompt",
    "chat_temperature",
    "chat_top_p",
    "chat_top_k",
    "chat_max_new_tokens",
    "chat_enable_thinking",
}


def _shipped_presets() -> list[tuple[Path, dict[str, object]]]:
    paths = sorted(Path(PRESETS_DEFAULT_DIR).glob("*.json"))
    assert paths
    return [
        (path, json.loads(path.read_text(encoding="utf-8"))["settings"])
        for path in paths
    ]


def test_shipped_presets_make_trigger_injection_explicit() -> None:
    for path, settings in _shipped_presets():
        is_lora = "character lora" in path.stem.casefold() or "motion lora" in path.stem.casefold()
        assert settings["trigger_mode"] == ("prefix" if is_lora else "none"), path.name
        assert settings["save_reasoning"] is True, path.name
        if "thinking" in str(settings.get("model_key", "")).casefold():
            assert settings["enable_thinking"] is True, path.name


def test_shipped_presets_cover_chat_settings() -> None:
    chat_presets = []
    for path, settings in _shipped_presets():
        assert _CHAT_KEYS <= set(settings), path.name
        assert isinstance(settings["chat_system_prompt"], str), path.name
        assert 0.0 <= float(settings["chat_temperature"]) <= 2.0, path.name
        assert 0.0 <= float(settings["chat_top_p"]) <= 1.0, path.name
        assert 0 <= int(settings["chat_top_k"]) <= 200, path.name
        # The Chat tab slider tops out at 8192; larger stored values are rejected by Gradio.
        assert 1 <= int(settings["chat_max_new_tokens"]) <= 8192, path.name
        thinking = "thinking" in str(settings["model_key"]).casefold()
        assert settings["chat_enable_thinking"] is thinking, path.name
        # The context window is a universal-preset setting like the other
        # model-specific generation controls; the caption reserve must fit in it.
        assert 1024 <= int(settings["context_tokens"]) <= 32768, path.name
        assert int(settings["max_new_tokens"]) < int(settings["context_tokens"]), path.name
        if path.stem.casefold().startswith("chat"):
            chat_presets.append((path, settings))
    # One Transformers and one GGUF preset per chat-capable family.
    assert len(chat_presets) == 4
    families = set()
    for path, settings in chat_presets:
        mode, note = model_chat_support(str(settings["model_key"]))
        assert mode == "multi", path.name
        assert "multi-turn" in note
        assert str(settings["chat_system_prompt"]).strip(), path.name
        assert settings["keep_model_loaded"] is True, path.name
        assert float(settings["idle_unload_minutes"]) >= 30, path.name
        assert int(settings["chat_max_new_tokens"]) < int(settings["context_tokens"]), path.name
        if "gguf" in str(settings["model_key"]):
            # llama-server reserves the KV cache for the whole window at load, so the
            # fast presets trade some window for resident layers.
            assert int(settings["context_tokens"]) <= 24576, path.name
        families.add(variant_to_family(str(settings["model_key"])))
    assert families == {"qwen3_omni_instruct", "qwen3_omni_thinking"}
    assert {"gguf" in str(settings["model_key"]) for _, settings in chat_presets} == {True, False}


def test_chat_support_matrix_covers_every_variant() -> None:
    from vcap.models.registry import MODEL_SPECS

    expected = {
        "qwen3_omni_instruct": "multi",
        "qwen3_omni_thinking": "multi",
        "timechat": "single",
        "avocado": "single",
        "qwen3_omni_captioner": "unsupported",
    }
    assert set(expected) == set(MODEL_SPECS)
    for family, spec in MODEL_SPECS.items():
        for variant in spec.variants:
            mode, note = model_chat_support(variant.key)
            assert mode == expected[family], variant.key
            assert spec.label in note and variant.label in note, variant.key


def _preset_bar_blocks(tmp_path: Path) -> tuple[UiContext, gr.Blocks, object, gr.Number]:
    default_dir = tmp_path / "defaults"
    default_dir.mkdir()
    (default_dir / "Balanced.json").write_text(
        json.dumps(
            {
                "_meta": {"format": "secourses_vcap_preset", "version": 1},
                "settings": {"fps": 3, "chat_temperature": 0.9},
            }
        ),
        encoding="utf-8",
    )
    ctx = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(tmp_path / "user", default_dir, default_preset_name="Balanced"),
        pipeline_client=PipelineClient(subprocess_mode=True),
        app_log=get_log(),
    )
    with gr.Blocks() as demo:
        handles = preset_bar(ctx)
        fps = gr.Number(value=2.0, label="fps")
        ctx.reg("fps", fps, 2.0, section="test", description="Frames per second.", kind="float")
        chat_temperature = gr.Number(value=0.2, label="chat temperature")
        ctx.reg(
            "chat_temperature",
            chat_temperature,
            0.2,
            section="chat",
            description="Chat sampling temperature.",
            kind="float",
        )
        # Stand-in for the Caption tab's tier-plan follow-up: preset-owned inputs
        # must come from the applied settings, the rest from live components.
        ctx.states["caption_auto_vram_binding"] = {
            "fn": lambda fps_value, temperature: (fps_value * 10, temperature),
            "inputs": [fps, chat_temperature],
            "input_keys": ["fps", None],
            "outputs": [fps, chat_temperature],
        }
        wire_preset_bar(ctx, demo)
    return ctx, demo, handles, fps


def _dependencies_for(config: dict, component_id: int, event: str) -> list[dict]:
    return [
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id == component_id and target_event == event
            for target_id, target_event in dependency.get("targets", [])
        )
    ]


def test_universal_preset_dropdown_applies_selection_immediately(tmp_path: Path) -> None:
    ctx, demo, handles, fps = _preset_bar_blocks(tmp_path)
    try:
        config = demo.get_config_file()
        select = _dependencies_for(config, handles.dropdown._id, "change")
        load = _dependencies_for(config, handles.load._id, "click")
        assert len(select) == 1 and len(load) == 1
        assert select[0]["inputs"][0] == handles.dropdown._id and len(select[0]["inputs"]) == 2
        # Selecting a preset applies exactly what Load applies: every registered
        # control, the status line, and the applied-preset guard state.
        assert select[0]["outputs"] == load[0]["outputs"]
        assert set(select[0]["outputs"]) >= {fps._id, handles.status._id}
        # Gradio 6 dropdowns fire ``input`` on every blur (twice per mouse pick),
        # so auto-apply must not be bound to it.
        assert not _dependencies_for(config, handles.dropdown._id, "input")

        select_fn = demo.fns[select[0]["id"]].fn
        values = select_fn("Balanced", {"name": "", "settings": {}})
        assert values[0] == 3.0
        assert values[1] == 0.9
        assert "Loaded Balanced" in values[-2]
        assert values[-1]["name"] == "Balanced"
        assert values[-1]["settings"]["fps"] == 3.0
        assert ctx.preset_store.get_last_used() == "Balanced"
        # Re-selecting the applied preset, including the programmatic re-selection
        # after save/delete/startup, is a no-op that tells follow-ups to skip.
        repeated = select_fn("Balanced", values[-1])
        assert not isinstance(repeated[0], float)
        assert repeated[-1] == {"name": "Balanced", "settings": {}}

        follow_ups = [d for d in config["dependencies"] if d.get("trigger_after") == select[0]["id"]]
        assert len(follow_ups) == 1
        follow_fn = demo.fns[follow_ups[0]["id"]].fn
        assert follow_fn(values[-1], 1.0, 0.5) == (30.0, 0.5)
        skipped = follow_fn(repeated[-1], 1.0, 0.5)
        assert len(skipped) == 2 and not isinstance(skipped[0], float)

        save = _dependencies_for(config, handles.save._id, "click")
        saved = demo.fns[save[0]["id"]].fn("Mine", 4.0, 0.5)
        assert saved[0]["value"] == "Mine" and saved[-1] == {"name": "Mine", "settings": {}}
        delete = _dependencies_for(config, handles.delete._id, "click")
        deleted = demo.fns[delete[0]["id"]].fn("Mine")
        # Deleting leaves nothing selected so the next pick is a real change.
        assert deleted[0]["value"] is None and deleted[-1] == {"name": "", "settings": {}}
    finally:
        ctx.pipeline_client.shutdown()


def test_preset_value_adapters_ship_bounds_with_values(tmp_path: Path) -> None:
    from vcap.ui.tabs import caption_tab

    client = PipelineClient(subprocess_mode=True)
    ctx = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(Path("presets"), Path("presets_default")),
        pipeline_client=client,
        app_log=get_log(),
    )
    try:
        with gr.Blocks():
            caption_tab.build(ctx)
        adapter = ctx.states["preset_value_adapters"]["max_frames"]
        # Values are clamped to the family cap; the backend bound stays global so
        # no handler racing this update can see a value above the bound.
        clamped = adapter({"model_key": "timechat_int8", "max_frames": 240})
        assert clamped["value"] == 160 and clamped["maximum"] == 768
        assert "160 frames" in clamped["info"]
        kept = adapter({"model_key": "qwen3_omni_instruct_int8", "max_frames": 240})
        assert kept["value"] == 240 and kept["maximum"] == 768
        assert adapter({"model_key": "qwen3_omni_captioner_int4", "max_frames": 0})["value"] == 0
        context = ctx.states["preset_value_adapters"]["context_tokens"]
        assert context({"model_key": "timechat_int8", "context_tokens": 99_999})["value"] == 32768
        binding = ctx.states["caption_auto_vram_binding"]
        assert binding["input_keys"] == ["model_key", "vram_preset", None, "show_all_variants"]
        assert len(binding["input_keys"]) == len(binding["inputs"])
    finally:
        client.shutdown()
