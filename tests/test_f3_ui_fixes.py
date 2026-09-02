from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import gradio as gr

from vcap.core.logs import get_log
from vcap.core.presets import PresetStore
from vcap.core.registry import SettingsRegistry
from vcap.pipeline.chat import ChatResponse
from vcap.pipeline.client import PipelineClient
from vcap.ui import components
from vcap.ui.app import UiContext, build_app
from vcap.ui.components import input_mode_from_tab, media_input_block
from vcap.ui.tabs.caption_tab import validate_model_variant


def _dependencies_for(config: dict, component_id: int, event: str) -> list[dict]:
    return [
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id == component_id and target_event == event
            for target_id, target_event in dependency.get("targets", [])
        )
    ]


def _media_blocks(tmp_path: Path) -> tuple[UiContext, gr.Blocks, object]:
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    (defaults / "Default.json").write_text(
        json.dumps(
            {
                "_meta": {"format": "secourses_vcap_preset", "version": 1},
                "settings": {},
            }
        ),
        encoding="utf-8",
    )
    client = PipelineClient(subprocess_mode=True)
    ctx = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(tmp_path / "presets", defaults),
        pipeline_client=client,
        app_log=get_log(),
        outputs_dir=tmp_path / "outputs",
        temp_dir=tmp_path / "temp",
        presets_dir=tmp_path / "presets",
        presets_default_dir=defaults,
    )
    with gr.Blocks() as demo:
        handles = media_input_block(ctx)
    return ctx, demo, handles


def test_model_variant_validation_restores_the_previous_registered_choice() -> None:
    previous = "qwen3_omni_instruct_int4"
    update, state, status = validate_model_variant(previous + "bogus", previous)
    assert update["value"] == previous
    assert state == previous
    assert "Unknown model variant; kept" in status
    assert "Qwen3-Omni" in status and "Instruct" in status

    update, state, status = validate_model_variant("timechat_int8", previous)
    assert state == "timechat_int8"
    assert "value" not in update
    assert "value" not in status


def test_media_mode_is_owned_by_tabs_and_scan_does_not_return_it(tmp_path: Path) -> None:
    ctx, demo, handles = _media_blocks(tmp_path)
    try:
        config = demo.get_config_file()
        folder_change = _dependencies_for(config, handles.folder._id, "change")
        assert len(folder_change) == 1
        assert handles.mode_state._id not in folder_change[0]["outputs"]
        scan = demo.fns[folder_change[0]["id"]].fn
        scanned = scan("", False, "", False, 0, ["video", "audio", "image", "text"], "")
        assert len(scanned) == len(folder_change[0]["outputs"])

        tabs_id = next(
            item["id"]
            for item in config["components"]
            if item.get("props", {}).get("elem_id") == "vc-input-tabs"
        )
        tab_select = _dependencies_for(config, tabs_id, "select")
        mode_dependency = next(
            dependency
            for dependency in tab_select
            if dependency["outputs"] == [handles.mode_state._id]
        )
        handler = demo.fns[mode_dependency["id"]].fn
        assert handler(SimpleNamespace(value="Upload files", index=0)) == "upload"
        assert handler(SimpleNamespace(value="File path", index=1)) == "path"
        assert handler(SimpleNamespace(value="Folder batch", index=2)) == "folder"
        assert input_mode_from_tab(SimpleNamespace(value="folder", index=None)) == "folder"

        for component in (handles.files, handles.path):
            change_events = _dependencies_for(config, component._id, "change")
            assert change_events
            assert all(handles.mode_state._id not in event["outputs"] for event in change_events)
        preview_select_events = [
            event
            for event in config["dependencies"]
            if any(target_event == "select" for _, target_event in event.get("targets", []))
            and handles.info._id in event["outputs"]
        ]
        assert len(preview_select_events) >= 3
        assert all(
            handles.mode_state._id not in event["outputs"]
            for event in preview_select_events
        )
    finally:
        ctx.pipeline.shutdown()


def test_zip_upload_descends_wrapper_folder_and_can_enable_recursive_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx, demo, handles = _media_blocks(tmp_path)
    monkeypatch.setattr(
        components,
        "_preview_updates",
        lambda paths: (None, None, None, "preview", "gallery", "video", 0.0),
    )
    try:
        config = demo.get_config_file()
        upload = _dependencies_for(config, handles.zip_upload._id, "upload")
        assert len(upload) == 1
        handler = demo.fns[upload[0]["id"]].fn

        archive = tmp_path / "wrapped.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("top/a.mp4", b"a")
            bundle.writestr("top/nested/b.png", b"b")
        result = handler(
            str(archive),
            True,
            "",
            False,
            0,
            ["video", "audio", "image", "text"],
            "",
        )
        assert Path(result[0]).name == "top"
        assert result[1]["value"] is True
        assert {Path(path).name for path in result[9]} == {"a.mp4", "b.png"}
        assert "Extracted 2 files" in result[10]
        assert f"using {result[0]}" in result[10]

        nested_archive = tmp_path / "nested-only.zip"
        with zipfile.ZipFile(nested_archive, "w") as bundle:
            bundle.writestr("batch/videos/a.mp4", b"a")
            bundle.writestr("batch/images/b.png", b"b")
        nested = handler(
            str(nested_archive),
            False,
            "",
            False,
            0,
            ["video", "audio", "image", "text"],
            "",
        )
        assert Path(nested[0]).name == "batch"
        assert nested[1]["value"] is True
        assert {Path(path).name for path in nested[9]} == {"a.mp4", "b.png"}
        assert (
            "Scan subfolders enabled because the media sit in subfolders."
            in nested[10]
        )
    finally:
        ctx.pipeline.shutdown()


def test_short_chat_stream_finishes_with_authoritative_usage() -> None:
    demo = build_app()
    ctx = demo.vcap_context
    config = demo.get_config_file()
    component_props = {
        item["id"]: item.get("props", {}) for item in config["components"]
    }
    for key in ("model_key", "vram_preset", "attention_backend"):
        component = ctx.caption_handles.controls[key]
        assert component_props[component._id].get("allow_custom_value", False) is False
    dependency = next(
        item
        for item in config["dependencies"]
        if item.get("api_name") == "chat"
    )
    send_message = demo.fns[dependency["id"]].fn

    def fake_chat(_request, on_event, _token):
        on_event(
            {
                "ev": "delta",
                "delta": "Done.",
                "reasoning_delta": "",
                "text": "Done.",
                "reasoning": "",
            }
        )
        on_event(
            {
                "ev": "status",
                "message": "Generating: 1 streamed chunks | 0.00 tok/s",
                "data": {"new_tokens": 1, "tok_per_s": 0.0},
            }
        )
        return ChatResponse(
            model_key="qwen3_omni_instruct_int4",
            text="Done.",
            raw_text="Done.",
            reasoning="",
            prompt_tokens=182,
            new_tokens=12,
            finish_reason="eos",
            prefill_s=0.1,
            decode_s=0.05,
            tokens_per_s=182.2,
            total_s=0.08,
            peak_vram_gb=0.0,
            cancelled=False,
            context_tokens=182,
            context_limit=32768,
        )

    ctx.pipeline_client.chat = fake_chat
    try:
        values = ctx.registry.dict_to_values(ctx.registry.defaults())
        updates = list(send_message(*values, {}, [], "", "hello"))
        assert all(len(update) == len(dependency["outputs"]) for update in updates)
        assert "Complete: 12 tokens" in updates[-1][2]
        assert "182.20 tok/s" in updates[-1][2]
        assert "Tokens:** 12" in updates[-1][3]
        assert "182 / 32,768" in updates[-1][3]
    finally:
        ctx.pipeline.shutdown()
