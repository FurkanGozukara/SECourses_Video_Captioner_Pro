from __future__ import annotations

import json
from pathlib import Path
import time

import gradio as gr
import pytest

from vcap.core.logs import get_log
from vcap.core.presets import PresetStore
from vcap.core.registry import SettingsRegistry
from vcap.pipeline.client import PipelineClient
from vcap.ui.app import UiContext
from vcap.ui.components import _folder_scan
from vcap.ui.tabs import caption_tab


def test_tier_filter_keeps_selected_oversize_variant_with_warning() -> None:
    selected = "qwen3_omni_instruct_bf16"
    filtered = caption_tab.variant_choices_for_tier(selected, 24)
    by_key = {key: label for label, key in filtered}
    assert selected in by_key
    assert "exceeds tier" in by_key[selected]
    assert "qwen3_omni_instruct_int4" in by_key
    assert "qwen3_omni_instruct_int8" not in by_key

    unfiltered = caption_tab.variant_choices_for_tier(selected, 24, show_all=True)
    assert len(unfiltered) > len(filtered)
    assert all("exceeds tier" not in label for label, _ in unfiltered)


def test_folder_scan_counts_mirrored_output_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "captions"
    (source / "nested").mkdir(parents=True)
    (output / "nested").mkdir(parents=True)
    (source / "nested" / "clip.mp4").write_bytes(b"")
    (source / "nested" / "clip.txt").write_text("input sidecar", encoding="utf-8")
    (output / "nested" / "clip.txt").write_text("output sidecar", encoding="utf-8")

    selected, summary = _folder_scan(str(source), True, str(output), False)
    assert selected == [str(source / "nested" / "clip.mp4")]
    assert "1 already captioned in output folder" in summary
    assert "will be skipped" in summary


def test_compile_probe_refreshes_stale_cache_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "compile_probe.json"
    monkeypatch.setattr(caption_tab, "_COMPILE_PROBE_CACHE", cache)
    monkeypatch.setattr(caption_tab, "_COMPILE_PROBE_HTML", None)
    monkeypatch.setattr(caption_tab, "_COMPILE_PROBE_RUNNING", False)

    def slow_probe():
        time.sleep(0.15)
        return "<span>ready</span>", {"inductor_ready": "full"}

    monkeypatch.setattr(caption_tab, "_run_compile_probe_in_child", slow_probe)
    started = time.monotonic()
    assert "Probing C++ toolchain" in caption_tab._probe_compile_in_child()
    assert time.monotonic() - started < 0.1

    deadline = time.monotonic() + 2
    while caption_tab._COMPILE_PROBE_RUNNING and time.monotonic() < deadline:
        time.sleep(0.02)
    assert caption_tab._probe_compile_in_child() == "<span>ready</span>"
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["key"]["python"]
    assert payload["key"]["torch"]


def test_caption_tab_action_button_hues_are_distinct() -> None:
    client = PipelineClient(subprocess_mode=True)
    ctx = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(Path("presets"), Path("presets_default")),
        pipeline_client=client,
        app_log=get_log(),
    )
    try:
        with gr.Blocks() as demo:
            caption_tab.build(ctx)
        entries = {entry.key: entry for entry in ctx.settings_registry.entries()}
        assert entries["gpu_index"].in_preset is False
        assert entries["gpu_indices"].kind == "list"
        assert entries["gpu_indices"].in_preset is False
        assert entries["gpu_indices"].in_metadata is True
        assert entries["trigger_mode"].default == "none"
        assert entries["save_reasoning"].default is True
        assert entries["context_carry_over"].default is False
        config = demo.get_config_file()
        cancel_components = [
            component
            for component in config["components"]
            if component.get("props", {}).get("elem_id") == "vc_caption_cancel"
        ]
        assert len(cancel_components) == 1
        assert cancel_components[0]["props"]["interactive"] is False
        tab_select_events = [
            dependency
            for dependency in config["dependencies"]
            if any(event == "select" for _, event in dependency.get("targets", []))
        ]
        assert len(tab_select_events) >= 3
        action_buttons = []
        for component in config["components"]:
            classes = component.get("props", {}).get("elem_classes") or []
            hues = [value for value in classes if value.startswith("vc-btn-")]
            if hues:
                assert len(hues) == 1
                action_buttons.append((component["props"].get("value"), hues[0]))
        assert len(action_buttons) >= 12
        hues = [hue for _, hue in action_buttons]
        assert len(hues) == len(set(hues)), action_buttons
    finally:
        client.shutdown()
