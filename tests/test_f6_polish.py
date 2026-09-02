from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gradio as gr

from vcap.ui import components
from vcap.ui.app import build_app
from vcap.ui.tabs import editor_tab
from vcap.ui.tabs.editor_tab import (
    new_editor_state,
    resolve_regeneration_prompt_choices,
)


def _audio_item() -> dict[str, Any]:
    return {"kind": "audio", "media_path": None}


def test_regeneration_prompt_resolution_keeps_a_compatible_value() -> None:
    choices, value, message = resolve_regeneration_prompt_choices(
        "qwen3_omni_instruct_int4",
        _audio_item(),
        "qwen3_audio_caption",
    )

    choice_values = {choice_value for _, choice_value in choices}
    assert value == "qwen3_audio_caption"
    assert value in choice_values
    assert message is None


def test_regeneration_prompt_resolution_replaces_a_foreign_value() -> None:
    choices, value, message = resolve_regeneration_prompt_choices(
        "qwen3_omni_instruct_int4",
        _audio_item(),
        "timechat_flatten_wan",
    )

    choice_values = {choice_value for _, choice_value in choices}
    assert value == "qwen3_audio_caption"
    assert value in choice_values
    assert message is None


def test_regeneration_prompt_resolution_handles_no_presets() -> None:
    choices, value, message = resolve_regeneration_prompt_choices(
        "timechat_int4",
        _audio_item(),
        "timechat_flatten_wan",
    )

    assert choices == []
    assert value is None
    assert message == (
        "No prompt preset supports audio input with TimeChat Captioner GRPO 7B; "
        "pick another model"
    )


def test_incompatible_regeneration_handlers_do_not_submit_a_job(
    monkeypatch,
    tmp_path,
) -> None:
    media_path = tmp_path / "track.mp3"
    media_path.write_bytes(b"not needed by the compatibility guard")
    state = new_editor_state(tmp_path)
    state["items"] = [
        {
            "kind": "audio",
            "media_path": str(media_path),
            "caption_path": str(tmp_path / "track.txt"),
            "caption": "old caption",
        }
    ]
    state["selected_index"] = 0
    expected = (
        "No prompt preset supports audio input with TimeChat Captioner GRPO 7B; "
        "pick another model"
    )

    def reject_probe(path: str) -> None:
        del path
        raise RuntimeError("probe not needed for this test")

    monkeypatch.setattr(editor_tab, "probe_media", reject_probe)
    demo = build_app()
    submitted: list[object] = []
    try:
        monkeypatch.setattr(
            demo.vcap_context.pipeline,
            "run_job",
            lambda *args, **kwargs: submitted.append((args, kwargs)),
        )
        binding = demo.vcap_context.states["editor_regeneration_binding"]

        selected_updates = list(
            binding["regenerate_handler"](
                state,
                "timechat_int4",
                None,
                "",
            )
        )
        all_updates = list(
            binding["regenerate_all_handler"](
                state,
                [0],
                "timechat_int4",
                None,
                "",
            )
        )

        assert selected_updates[0][1] == expected
        assert all_updates[0][1] == expected
        assert submitted == []
    finally:
        demo.vcap_context.pipeline_client.shutdown()


def test_log_panel_explains_hidden_tab_timer_pause(monkeypatch) -> None:
    class FakeLog:
        revision = 0

        @staticmethod
        def tail_snapshot(limit: int) -> tuple[list[str], int]:
            del limit
            return [], 0

        @staticmethod
        def snapshot_for_poll(
            cursor: int,
            *,
            recovery_limit: int,
        ) -> tuple[list[str], int, bool]:
            del cursor, recovery_limit
            return [], 0, False

    monkeypatch.setattr(components.gpu, "resource_snapshot", lambda index: index)
    monkeypatch.setattr(
        components.gpu,
        "render_resource_meter_html",
        lambda snapshot: str(snapshot),
    )

    with gr.Blocks() as demo:
        gpu_index = gr.Number(value=0)
        ctx = SimpleNamespace(
            app_log=FakeLog(),
            states={"gpu_index": gpu_index, "gpu_index_default": 0},
        )
        components.log_panel(ctx)

    note = next(
        component
        for component in demo.get_config_file()["components"]
        if component.get("props", {}).get("value")
        == "Updates pause while this browser tab is hidden and catch up when it is visible again."
    )
    assert "vc-help" in note["props"]["elem_classes"]
