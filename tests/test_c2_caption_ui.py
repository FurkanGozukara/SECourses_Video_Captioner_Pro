from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import gradio as gr
import pytest

from vcap.core.logs import get_log
from vcap.core.presets import PresetStore
from vcap.core.registry import SettingsRegistry
from vcap.pipeline.client import PipelineClient
from vcap.pipeline.job import InputItem, JobResult, JobSpec, OutputSpec
from vcap.pipeline.runner import (
    _apply_batch_limit,
    _assign_batch_outputs,
    _resolve_inputs,
    _write_batch_summary,
)
from vcap.prompts.presets import TEMPLATE_VARIABLES, render_prompt
from vcap.ui.app import UiContext
from vcap.ui.components import _folder_scan, _resolved_after_preview_edit
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

    selected, summary = _folder_scan(
        str(source), True, str(output), False, include_kinds=["video"]
    )
    assert selected == [str(source / "nested" / "clip.mp4")]
    assert "1 already captioned in output folder" in summary
    assert "will be skipped" in summary
    _, limited_summary = _folder_scan(
        str(source), True, str(output), False, 2, ["video"]
    )
    assert "limiting to first 2" in limited_summary


def test_folder_preview_edit_never_replaces_scanned_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source" / "nested" / "clip.mp4"
    cached = tmp_path / "gradio" / "hash" / "clip.mp4"
    uploaded = tmp_path / "gradio" / "original" / "clip.mp4"
    source.parent.mkdir(parents=True)
    cached.parent.mkdir(parents=True)
    uploaded.parent.mkdir(parents=True)
    source.write_bytes(b"")
    cached.write_bytes(b"")
    uploaded.write_bytes(b"")
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))

    assert _resolved_after_preview_edit(str(cached), [str(source)], "folder") == [str(source)]
    assert _resolved_after_preview_edit(str(cached), [str(source)], "upload") == [str(source)]
    assert _resolved_after_preview_edit(str(cached), [str(uploaded)], "upload") == [str(cached)]


def test_nested_unicode_batch_output_is_mirrored_from_source_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "g\u00f6lge_\u0432\u0438\u0434\u0435\u043e"
    media = nested / "clip.mp4"
    output = tmp_path / "captions"
    nested.mkdir(parents=True)
    media.write_bytes(b"")
    spec = JobSpec(
        inputs=[InputItem(media)],
        output=OutputSpec(
            kind="batch",
            batch_output_dir=output,
            source_root=source,
            recursive=True,
        ),
    )

    resolved = _resolve_inputs(spec)
    _assign_batch_outputs(spec, resolved)

    assert resolved[0].path == media
    assert resolved[0].out_dir == output / nested.name


def test_batch_limit_applies_after_existing_caption_skips() -> None:
    entries = [
        SimpleNamespace(status="skipped", message="already captioned"),
        SimpleNamespace(status="pending", message=""),
        SimpleNamespace(status="pending", message=""),
        SimpleNamespace(status="pending", message=""),
    ]
    spec = JobSpec(output=OutputSpec(kind="batch", limit_items=2))

    _apply_batch_limit(spec, entries)

    assert [entry.status for entry in entries] == [
        "skipped",
        "pending",
        "pending",
        "skipped",
    ]
    assert "limit of 2" in entries[-1].message


def test_batch_summary_records_dry_run_limit(tmp_path: Path) -> None:
    summary_path = _write_batch_summary(tmp_path, [], 1.25, limit_items=3)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["limit_items"] == 3


def test_prompt_resolution_keeps_valid_presets_and_replaces_invalid_ones() -> None:
    family, choices, preset = caption_tab._resolve_prompt_preset(
        "avocado_int8",
        "video_audio",
        "timechat_flatten_wan",
    )
    assert family == "avocado"
    assert preset is not None and preset.id == "avocado_av_aligned"
    assert preset.id in {key for _, key in choices}

    _, _, shared = caption_tab._resolve_prompt_preset(
        "qwen3_omni_instruct_int4",
        "video_audio",
        "wan22_t2v_dense",
    )
    assert shared is not None and shared.id == "wan22_t2v_dense"
    assert (
        caption_tab._effective_prompt_modality(
            "qwen3_omni_instruct_int4",
            "video_audio",
            False,
        )
        == "video"
    )
    assert (
        caption_tab._effective_prompt_modality(
            "timechat_int4",
            "video",
            True,
        )
        == "video_audio"
    )


def test_model_change_without_inputs_uses_family_default_prompt() -> None:
    cases = {
        "timechat_int4": ("video_audio", "timechat_flatten_wan"),
        "avocado_int8": ("video_audio", "avocado_av_aligned"),
        "qwen3_omni_instruct_int4": ("video", "wan22_t2v_dense"),
        "qwen3_omni_thinking_int8": ("video", "wan22_t2v_dense"),
        "qwen3_omni_captioner_int4": ("audio", "qwen3_captioner_promptfree"),
    }
    variables = {name: data["default"] for name, data in TEMPLATE_VARIABLES.items()}

    for variant_key, (expected_modality, expected_preset) in cases.items():
        modality = caption_tab._effective_prompt_modality(
            variant_key,
            "video_audio",
            False,
            has_inputs=False,
        )
        _, choices, preset = caption_tab._resolve_prompt_preset(
            variant_key,
            modality,
            "incompatible_previous_family_preset",
        )

        assert modality == expected_modality
        assert choices
        assert preset is not None and preset.id == expected_preset
        rendered_user = render_prompt(preset, variables)[1]
        if variant_key.startswith("qwen3_omni_captioner"):
            assert rendered_user == ""
        else:
            assert rendered_user.strip()


def test_job_done_messages_cover_success_failure_and_cancel() -> None:
    success = JobResult([], {"done": 5, "failed": 0}, "run", "metadata", 1.0)
    failed = JobResult([], {"done": 0, "failed": 1}, "run", "metadata", 1.0)
    cancelled = JobResult([], {"done": 2, "cancelled": 1}, "run", "metadata", 1.0)

    assert caption_tab._job_done_message(success) == "Caption job finished: 5 done, 0 failed"
    assert caption_tab._job_done_message(failed) == "Job failed: 0 done, 1 failed"
    assert caption_tab._job_done_message(cancelled) == "Job cancelled: 2 done, 0 failed"
    payload = json.loads(
        caption_tab._job_done_payload(
            "done",
            {
                "desktop_notification_on_finish": True,
                "play_sound_on_finish": False,
            },
        )
    )
    assert payload == {"message": "done", "desktop": True, "sound": False}
    one_file = [InputItem("clip.mp4")]
    assert caption_tab._should_auto_open_output(success, one_file, "single", True)
    assert not caption_tab._should_auto_open_output(failed, one_file, "single", True)
    assert not caption_tab._should_auto_open_output(success, one_file, "batch", True)


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
        assert entries["batch_limit_items"].default == 0
        assert entries["batch_limit_items"].kind == "int"
        assert entries["trigger_mode"].default == "none"
        assert entries["save_reasoning"].default is True
        assert entries["context_carry_over"].default is False
        config = demo.get_config_file()
        replacement = next(
            component
            for component in config["components"]
            if component.get("props", {}).get("label") == "Word replacements"
        )
        replacement_events = {
            event
            for dependency in config["dependencies"]
            for component_id, event in dependency.get("targets", [])
            if component_id == replacement["id"]
        }
        assert {"input", "change"} <= replacement_events
        cancel_components = [
            component
            for component in config["components"]
            if component.get("props", {}).get("elem_id") == "vc_caption_cancel"
        ]
        assert len(cancel_components) == 1
        assert cancel_components[0]["props"]["interactive"] is False
        by_elem_id = {
            component.get("props", {}).get("elem_id"): component
            for component in config["components"]
        }
        assert by_elem_id["vc_caption_cancel_confirmation"]["props"]["visible"] is False
        assert by_elem_id["vc_caption_cancel_yes"]["props"]["value"] == "✔ Yes, cancel"
        assert by_elem_id["vc_caption_cancel_keep"]["props"]["value"] == "✖ Keep running"
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


def test_model_switch_handlers_keep_values_inside_reported_bounds() -> None:
    """Every handler keeps numeric values inside the bounds it reports.

    Gradio validates incoming values against the *backend* bound of a Number or
    Slider. Frame and pixel bounds stay global to avoid sibling-event races;
    Maximum new tokens intentionally follows the selected family cap in F2.
    """

    from vcap.models.registry import MODEL_SPECS
    from vcap.models.vram_presets import VRAM_TIERS

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
        config = demo.get_config_file()
        bounds = {
            component["id"]: (component["props"].get("minimum"), component["props"].get("maximum"))
            for component in config["components"]
            if component.get("type") in {"number", "slider"}
        }
        max_tokens_id = next(
            entry.component._id
            for entry in ctx.settings_registry.entries()
            if entry.key == "max_new_tokens"
        )
        checked = 0

        def check(component_id: int, value: object, context: str) -> None:
            nonlocal checked
            if component_id not in bounds:
                return
            low, high = bounds[component_id]
            if isinstance(value, dict):
                if "minimum" in value:
                    assert value["minimum"] == low, (context, component_id, value)
                if "maximum" in value:
                    if component_id == max_tokens_id:
                        assert 1 <= value["maximum"] <= caption_tab._GLOBAL_MAX_NEW_TOKENS
                        high = value["maximum"]
                    else:
                        assert value["maximum"] == high, (context, component_id, value)
                value = value.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return
            if low is not None:
                assert value >= low, (context, component_id, value, low)
            if high is not None:
                assert value <= high, (context, component_id, value, high)
            checked += 1

        variants = [variant.key for spec in MODEL_SPECS.values() for variant in spec.variants]
        tiers = ["auto", *[str(tier) for tier in VRAM_TIERS]]
        for dependency in config["dependencies"]:
            fn = demo.fns[dependency["id"]].fn
            name = getattr(fn, "__name__", "")
            if name in {"model_defaults", "model_constraints"}:
                for variant in variants:
                    for component_id, value in zip(dependency["outputs"], fn(variant)):
                        check(component_id, value, f"{name}({variant})")
            elif name in {"apply_vram", "apply_auto_vram"}:
                for variant in variants:
                    for tier in tiers:
                        outputs = fn(variant, tier, 0, False)
                        for component_id, value in zip(dependency["outputs"], outputs):
                            check(component_id, value, f"{name}({variant}, {tier})")
        binding = ctx.states["caption_auto_vram_binding"]
        for variant in variants:
            for tier in tiers:
                outputs = binding["fn"](variant, tier, 0, False)
                for component, value in zip(binding["outputs"], outputs):
                    check(component._id, value, f"auto_vram_startup({variant}, {tier})")
        components = {entry.key: entry.component for entry in ctx.settings_registry.entries()}
        for key, adapter in ctx.states["preset_value_adapters"].items():
            for path in sorted(Path("presets_default").glob("*.json")):
                settings = json.loads(path.read_text(encoding="utf-8"))["settings"]
                for variant in variants:
                    check(components[key]._id, adapter({**settings, "model_key": variant}), f"adapter {key} {path.name} {variant}")
        assert checked > 500
    finally:
        client.shutdown()
