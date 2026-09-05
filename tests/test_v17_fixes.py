"""Regressions for the v1.7.0 real-user Chrome pass."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from vcap.core import gpu
from vcap.core.logs import get_log
from vcap.core.media import MediaInfo
from vcap.models.llamacpp_backend import LlamaCppRuntimeOptions, server_plan_for_vram
from vcap.models.registry import MODEL_SPECS
from vcap.pipeline import runner
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.ui import components
from vcap.ui.app import build_app
from vcap.ui.tabs import caption_tab, editor_tab, health_tab


@pytest.fixture(scope="module")
def app() -> Any:
    demo = build_app()
    try:
        yield demo
    finally:
        demo.vcap_context.pipeline_client.shutdown()


def _skipped(value: Any) -> bool:
    return value == gr.skip()


# Loading a preset must apply exactly what it stores; the tier plan only replaces a
# variant the detected GPU cannot run. Startup and Reset still apply the tier plan.
def test_preset_load_keeps_stored_media_and_token_values(app: Any) -> None:
    binding = app.vcap_context.states["caption_auto_vram_binding"]
    kept = binding["preset_fn"]("qwen3_omni_instruct_int4", "auto", 0, False)
    assert kept[0]["value"] == "qwen3_omni_instruct_int4"
    assert all(_skipped(value) for value in kept[1:-1])
    assert "applied as saved" in kept[-1]
    planned = binding["fn"]("qwen3_omni_instruct_int4", "auto", 0, False)
    assert not _skipped(planned[5]), "startup still applies the tier plan's token budget"


def test_preset_load_swaps_only_a_variant_the_gpu_cannot_run(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    small = gpu.GpuInfo(0, "Small", 8.0, 7.0, 1.0, (8, 6), "616", True)
    monkeypatch.setattr(gpu, "list_gpus", lambda: [small])
    binding = app.vcap_context.states["caption_auto_vram_binding"]
    swapped = binding["preset_fn"]("qwen3_omni_instruct_int8", "auto", 0, False)
    assert swapped[0]["value"] == "qwen3_omni_instruct_int4"
    assert not _skipped(swapped[5])


def test_preset_follow_ups_use_the_preset_variant_for_user_loads(app: Any) -> None:
    config = app.get_config_file()
    handles = app.vcap_context.preset_handles
    dropdown_change = next(
        dependency
        for dependency in config["dependencies"]
        if any(target_id == handles.dropdown._id and event == "change" for target_id, event in dependency.get("targets", []))
    )
    follow_ups = [dependency for dependency in config["dependencies"] if dependency.get("trigger_after") == dropdown_change["id"]]
    assert len(follow_ups) == 1
    follow_fn = app.fns[follow_ups[0]["id"]].fn
    state = {"name": "LTX", "settings": {"model_key": "qwen3_omni_instruct_int4", "vram_preset": "auto", "show_all_variants": False}}
    outputs = follow_fn(state, "qwen3_omni_instruct_int4", "auto", 0, False)
    assert all(_skipped(value) for value in outputs[1:-1])


# Gradio's Dropdown fires ``select`` only for mouse picks; the app now reacts to
# ``input`` and ignores the blur-only repeats that event also produces.
def _browser_verdicts(stamp_js: str, listener_js: list[str], timeline: str) -> list[str]:
    """Run the browser-side classifier in node: ``stamp(value)`` and ``pick(listener, value)`` at ``at(ms, ...)``."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to run the browser-side pick classifier")
    program = (
        "globalThis.window = globalThis; let now = 0; globalThis.performance = {now: () => now}; "
        f"const stamp = {stamp_js}; const listeners = [{', '.join(listener_js)}]; const out = []; "
        "const at = (ms, fn) => { now = ms; fn(); }; const pick = (i, value) => out.push(listeners[i](value, '')[1]); "
        f"{timeline} console.log(JSON.stringify(out));"
    )
    return json.loads(subprocess.run([node, "-e", program], capture_output=True, text=True, check=True).stdout)


def test_user_pick_reacts_to_picks_and_ignores_blurs() -> None:
    calls: list[str] = []
    with gr.Blocks() as demo:
        dropdown = gr.Dropdown(choices=["a", "b"], value="a")
        first, second = gr.Textbox(), gr.Textbox()
        marker = components.pick_marker(dropdown, "probe")
        components.user_pick(dropdown, marker, lambda value: calls.append(value) or value.upper(), inputs=[], outputs=first)
        components.user_pick(dropdown, marker, lambda value: value, inputs=[], outputs=second)
    config = demo.get_config_file()
    events = {event for dependency in config["dependencies"] for target_id, event in dependency.get("targets", []) if target_id == dropdown._id}
    assert events == {"change", "input"}
    stamp = next(dependency["js"] for dependency in config["dependencies"] if dependency.get("js") and any(event == "change" for _, event in dependency["targets"]))
    picks = [dependency for dependency in config["dependencies"] if any(event == "input" for _, event in dependency.get("targets", []))]
    assert len(picks) == 2 and all(pick["inputs"][1] == marker._id for pick in picks)
    guarded = demo.fns[picks[0]["id"]].fn
    assert _skipped(guarded("a", "blur"))
    assert guarded("b", "pick") == "B"
    assert calls == ["b"]
    # Gradio runs the two listeners one round trip apart (900 ms here), in either order
    # relative to the ``change`` stamp; both must see the pick, and only the pick.
    timeline = (
        "at(0, () => stamp('b')); at(5, () => pick(0, 'b')); at(900, () => pick(1, 'b'));"
        " at(5000, () => pick(0, 'b')); at(5900, () => pick(1, 'b'));"
        " at(9000, () => pick(0, 'a')); at(9001, () => stamp('a')); at(9900, () => pick(1, 'a'));"
        " at(20000, () => stamp('b')); at(50000, () => pick(0, 'b')); at(50900, () => pick(1, 'b'));"
        " at(60000, () => stamp('a')); at(60005, () => pick(0, 'a')); at(60900, () => pick(1, 'a'));"
    )
    assert _browser_verdicts(stamp, [pick["js"] for pick in picks], timeline) == [
        "pick", "pick",  # keyboard pick: change stamp, then both listeners
        "blur", "blur",  # focus and blur without a new pick
        "pick", "pick",  # the first listener ran before the change stamp
        "blur", "blur",  # programmatic change (preset load), blur much later
        "pick", "pick",  # picking again after a programmatic change
    ]


# Gradio's frontend re-dispatches a whole event when a deferred (``always_last``)
# listener finishes; two deferred listeners on one control then re-trigger each
# other forever and saturate the browser, so the app keeps at most one per event.
def test_at_most_one_deferred_listener_per_event(app: Any) -> None:
    config = app.get_config_file()
    deferred: dict[tuple[int, str], int] = {}
    for dependency in config["dependencies"]:
        if dependency.get("backend_fn") and dependency.get("trigger_mode") == "always_last":
            for target_id, event in dependency.get("targets", []):
                if target_id is not None:
                    deferred[(target_id, event)] = deferred.get((target_id, event), 0) + 1
    assert not {key: count for key, count in deferred.items() if count > 1}
    token_slider = app.vcap_context.caption_handles.controls["max_new_tokens"]._id
    listeners = [d for d in config["dependencies"] if (token_slider, "change") in [tuple(t) for t in d.get("targets", [])]]
    assert len(listeners) == 1 and len(listeners[0]["outputs"]) == 4, "budget, block-swap preview, and ceiling refresh together"


def test_caption_dropdown_reactions_are_bound_to_input_not_select(app: Any) -> None:
    controls = app.vcap_context.caption_handles.controls
    config = app.get_config_file()
    for key in ("model_key", "prompt_preset_id", "vram_preset", "resolution_preset"):
        component_id = controls[key]._id
        events = {event for dependency in config["dependencies"] for target_id, event in dependency.get("targets", []) if target_id == component_id}
        assert "select" not in events and "input" in events, key
    chat_id = app.vcap_context.chat_handles.prompt_preset._id
    chat_events = {event for dependency in config["dependencies"] for target_id, event in dependency.get("targets", []) if target_id == chat_id}
    assert "select" not in chat_events and "input" in chat_events


def test_manual_pixel_edits_track_the_pixel_preset() -> None:
    assert caption_tab._resolution_choice(262144) == "262144"
    assert caption_tab._resolution_choice("297920") == "297920"
    assert caption_tab._resolution_choice(300000) == "custom"
    assert caption_tab._resolution_choice(None) == "custom"


# TimeChat always needs an audio timeline, so the audio toggle can neither be turned
# off in the UI nor make a video unsupported in the pipeline.
def test_timechat_audio_toggle_is_locked_and_video_stays_supported(tmp_path: Path) -> None:
    assert caption_tab._use_audio_update(MODEL_SPECS["timechat"]) == {"value": True, "interactive": False}
    assert caption_tab._use_audio_update(MODEL_SPECS["qwen3_omni_instruct"], False) == {"interactive": True, "value": False}
    info = MediaInfo(tmp_path / "clip.mp4", "video", duration=5.0, has_video=True, has_audio=True)
    spec = JobSpec.from_settings({"model_key": "timechat_int4", "use_audio_in_video": False}, [InputItem("clip.mp4")], OutputSpec())
    assert runner._required_capability(info, spec.inputs[0], spec, MODEL_SPECS["timechat"]) == ("video_audio", "video")
    avocado = JobSpec.from_settings({"model_key": "avocado_int4", "use_audio_in_video": False}, [InputItem("clip.mp4")], OutputSpec())
    assert runner._required_capability(info, avocado.inputs[0], avocado, MODEL_SPECS["avocado"]) == ("video", "video")


def test_third_party_noise_stays_out_of_the_live_log() -> None:
    log = get_log()
    before = log.revision
    log.log("INFO:root:Successfully loaded: 'mslk.dll'", console=False)
    log.log("W0905 _pytree.py:630 register_constant() on Enum subclasses is deprecated", console=False)
    assert log.revision == before
    assert "mslk.dll" in log.current_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    log.log("Model ready on GPU 0", console=False)
    assert log.revision == before + 1


# Backend knobs that were environment variables are settings now.
def test_gguf_layer_placement_and_allocator_cap_flow_from_settings() -> None:
    spec = JobSpec.from_settings(
        {"model_key": "qwen3_omni_instruct_gguf_q4", "gguf_gpu_layers": 12, "gguf_n_cpu_moe": 8, "vram_hard_cap": False},
        [InputItem("clip.mp4")],
        OutputSpec(),
    )
    assert (spec.runtime.gguf_gpu_layers, spec.runtime.gguf_n_cpu_moe, spec.runtime.vram_hard_cap) == (12, 8, False)
    plan = LlamaCppRuntimeOptions.from_spec(spec).place_layers(server_plan_for_vram(32.0))
    assert plan.gpu_layers == 12 and plan.fit is False and plan.n_cpu_moe == 8
    untouched = LlamaCppRuntimeOptions().place_layers(server_plan_for_vram(32.0))
    assert untouched.gpu_layers is None and untouched.fit is True and untouched.n_cpu_moe is None


def test_new_runtime_controls_are_registered(app: Any) -> None:
    entries = {entry.key: entry for entry in app.vcap_context.registry.entries()}
    assert entries["gguf_gpu_layers"].default == 0 and entries["gguf_gpu_layers"].maximum == 999
    assert entries["gguf_n_cpu_moe"].default == 0
    assert entries["vram_hard_cap"].default is True
    updates = caption_tab.gguf_control_updates("qwen3_omni_instruct_gguf_q4")
    assert updates["vram_hard_cap"]["interactive"] is False
    assert caption_tab.gguf_control_updates("timechat_int4")["vram_hard_cap"]["interactive"] is True


def test_editor_txt_edit_updates_the_json_sidecar(tmp_path: Path) -> None:
    caption = tmp_path / "clip.txt"
    caption.write_text("old caption\n", encoding="utf-8")
    sidecar = tmp_path / "clip.json"
    sidecar.write_text(json.dumps({"text": "old caption", "start_s": 0.0}), encoding="utf-8")
    editor_tab._write_caption(caption, "new caption")
    assert caption.read_text(encoding="utf-8") == "new caption\n"
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document == {"text": "new caption", "start_s": 0.0}


# Only the selected tab polls; the editor autosave keeps a slow background tick.
def test_only_the_selected_tab_keeps_its_timers_ticking(app: Any) -> None:
    timers = app.vcap_context.states["tab_timers"]
    assert {"caption", "transcribe", "chat", "editor", "health"} <= {tab for tab, found in timers.items() if found}
    config = app.get_config_file()
    main_tabs = app.vcap_context.states["main_tabs"]
    every = [timer._id for found in timers.values() for timer in found]
    (gate,) = [d for d in config["dependencies"] if (main_tabs._id, "select") in [tuple(t) for t in d.get("targets", [])] and d["outputs"] == every]
    select = app.fns[gate["id"]].fn
    by_timer = dict(zip(every, select(gr.SelectData(None, {"index": 2, "value": "🎙️ Transcribe"}))))
    assert all(update.constructor_args["active"] is True for update in (by_timer[t._id] for t in timers["transcribe"]))
    assert all(update.constructor_args["active"] is False for update in (by_timer[t._id] for t in timers["caption"]))
    (autosave,) = timers["editor"]
    assert by_timer[autosave._id].constructor_args == {"value": 3.0, "active": True}
    editor_view = dict(zip(every, select(gr.SelectData(None, {"index": 99, "value": "✏️ Caption Editor"}))))
    assert editor_view[autosave._id].constructor_args == {"value": autosave.value, "active": True}
    assert all(_skipped(update) for update in select(gr.SelectData(None, {"index": 99, "value": "nope"})))


def test_model_disk_usage_is_reported_in_decimal_gb() -> None:
    check = health_tab.request_model_delete("timechat_int4", {}, usage_fn=lambda _key: 6_467_930_328)
    assert check["state"] == "confirm" and "6.47 GB" in check["question"]
