from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.core.export import export_dataset
from vcap.core.outputs import list_recent_runs
from vcap.core.subprocess_runner import CancelToken
from vcap.models.offload import BudgetHint
from vcap.prompts.presets import default_preset_for
from vcap.ui.app import build_app
from vcap.ui.tabs import caption_tab, chat_tab, dataset_tab, editor_tab
from vcap.ui.theme import HOTKEYS_HEAD


def _write_segment_run(root: Path, count: int = 2) -> tuple[Path, Path]:
    run = root / "batch_0083_qwen3"
    segments = run / "video20s_segments"
    segments.mkdir(parents=True)
    source = root / "video20s.mp4"
    source.write_bytes(b"source video")
    records = []
    windows = [(0.0, 4.1), (9.5095, 11.8118)]
    for index, (start, end) in enumerate(windows[:count], start=1):
        caption = segments / f"clip_{index:04d}.txt"
        sidecar = caption.with_suffix(".json")
        caption.write_text(f"caption {index}\n", encoding="utf-8")
        sidecar.write_text(json.dumps({"text": f"caption {index}"}), encoding="utf-8")
        records.append(
            {
                "index": index,
                "start_s": start,
                "end_s": end,
                "media_path": str(run / ".work" / "clips" / f"clip_{index:04d}.mp4"),
                "outputs": {"txt": str(caption), "json": str(sidecar)},
            }
        )
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "settings": {
                    "split_mode": "copy",
                    "encode_codec": "libx264",
                    "encode_crf": 21,
                    "encode_preset": "fast",
                    "encode_audio_bitrate": "128k",
                },
                "items_results": [
                    {"path": str(source), "status": "done", "segments": records}
                ],
            }
        ),
        encoding="utf-8",
    )
    return run, source


def _video_probe() -> SimpleNamespace:
    return SimpleNamespace(has_video=True, has_audio=True, kind="video", duration=20.0)


def test_segment_scan_rows_and_regeneration_submit_one_clipped_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run, source = _write_segment_run(tmp_path)
    monkeypatch.setattr(editor_tab, "probe_media", lambda _path: _video_probe())
    monkeypatch.setattr(editor_tab, "preview_safe_media", lambda path, _cache: Path(path))
    demo = build_app()
    submitted: list[Any] = []

    def fake_run_job(spec: Any, _sink: Any) -> SimpleNamespace:
        submitted.append(spec)
        return SimpleNamespace(counts={"done": 1, "failed": 0})

    try:
        ctx = demo.vcap_context
        monkeypatch.setattr(ctx.pipeline, "run_job", fake_run_job)
        binding = ctx.states["editor_regeneration_binding"]
        scanned = binding["scan_handler"](
            str(run), False, 25, "qwen3_omni_instruct_int4", None, ""
        )
        state = scanned[0]
        items = state["items"]
        assert len(items) == 2
        second = items[1]
        assert Path(second["media_path"]) == source
        assert (second["segment_index"], second["start_s"], second["end_s"]) == (
            2,
            pytest.approx(9.5095),
            pytest.approx(11.8118),
        )
        assert scanned[1][0][0] == "▶ 1"
        assert scanned[8] == "caption 1"
        assert scanned[4]["visible"] is True
        assert "No item selected" not in scanned[9]
        assert scanned[1][1][1] == "video20s_segments/clip_0002.txt · 00:09.5–00:11.8"
        assert editor_tab.editor_regeneration_log(second) == (
            "Regenerating clip 2 (00:09.510–00:11.812) of video20s.mp4"
        )

        state["selected_index"] = 1
        prompt_id = default_preset_for("qwen3_omni_instruct", "video_audio").id
        runtime_values = [entry.default for entry in ctx.settings_registry.entries()]
        list(
            binding["regenerate_handler"](
                state,
                "qwen3_omni_instruct_int4",
                prompt_id,
                "",
                *runtime_values,
            )
        )
        assert len(submitted) == 1
        spec = submitted[0]
        assert len(spec.inputs) == 1 and Path(spec.inputs[0].path) == source
        assert spec.preprocess.trim_start_s == pytest.approx(9.5095)
        assert spec.preprocess.trim_end_s == pytest.approx(11.8118)
        assert spec.split.mode == "whole"
        assert set(spec.post.formats) >= {"txt", "json"}

        submitted.clear()
        list(
            binding["regenerate_all_handler"](
                state,
                [0, 1],
                "qwen3_omni_instruct_int4",
                prompt_id,
                "",
                *runtime_values,
            )
        )
        assert len(submitted) == 2
        assert all(len(item.inputs) == 1 and item.split.mode == "whole" for item in submitted)
        assert [item.preprocess.trim_start_s for item in submitted] == [0.0, pytest.approx(9.5095)]

        produced = run / "video20s_clips" / "clip_0002.mp4"
        produced.parent.mkdir()
        produced.write_bytes(b"clip")
        rescanned = editor_tab.scan_folder(run, recursive=False)
        produced_item = next(item for item in rescanned if Path(item["caption_path"]).stem == "clip_0002")
        produced_spec = editor_tab.build_editor_regeneration_spec(
            {"model_key": "qwen3_omni_instruct_int4"},
            produced_item,
            variant="qwen3_omni_instruct_int4",
            prompt_id=prompt_id,
            outputs_root=tmp_path,
        )
        assert Path(produced_spec.inputs[0].path) == produced
        assert produced_spec.preprocess.trim_start_s == 0.0
        assert produced_spec.preprocess.trim_end_s is None
    finally:
        demo.vcap_context.pipeline.shutdown()


def test_segment_export_cuts_or_copies_the_segment_and_reports_unknown_windows(tmp_path: Path) -> None:
    source = tmp_path / "video20s.mp4"
    source.write_bytes(b"whole source")
    caption = tmp_path / "video20s_segments" / "clip_0002.txt"
    caption.parent.mkdir()
    caption.write_text("scene caption", encoding="utf-8")
    calls: list[tuple[Any, ...]] = []

    def fake_cutter(source_path: Path, ranges: list[Any], out_dir: Path, **kwargs: Any) -> list[Any]:
        calls.append((source_path, ranges, kwargs))
        target = Path(out_dir) / "segment_0001.mp4"
        target.write_bytes(b"precise scene")
        return [SimpleNamespace(path=target)]

    item = {
        "media_path": str(source),
        "source_media_path": str(source),
        "caption_path": str(caption),
        "caption": "scene caption",
        "flag": "approved",
        "segment_index": 2,
        "start_s": 9.5095,
        "end_s": 11.8118,
        "split_mode": "copy",
        "encode_codec": "libx264",
        "encode_crf": 21,
        "encode_preset": "fast",
        "encode_audio_bitrate": "128k",
    }
    report = export_dataset([item], tmp_path / "export", cutter=fake_cutter)
    assert report.media_files[0].name == "video20s_clip_0002.mp4"
    assert report.caption_files[0].name == "video20s_clip_0002.txt"
    assert report.media_files[0].read_bytes() == b"precise scene"
    assert calls[0][1][0].start_s == pytest.approx(9.5095)
    assert calls[0][2]["crf"] == 21

    clip = tmp_path / "clip_0002.mp4"
    clip.write_bytes(b"saved clip")
    copied = export_dataset(
        [{**item, "segment_media_path": str(clip)}], tmp_path / "copied", cutter=lambda *_a, **_k: pytest.fail("cutter called")
    )
    assert copied.media_files[0].read_bytes() == b"saved clip"

    unknown = {key: value for key, value in item.items() if key not in {"start_s", "end_s", "segment_media_path"}}
    fallback = export_dataset([unknown], tmp_path / "fallback")
    assert fallback.segment_full_source_fallbacks == 1
    message = editor_tab.editor_export_handler(
        {"items": [unknown]}, tmp_path / "fallback_ui", True, ".txt"
    )
    assert "1 segment captions exported against the full source" in message


def test_combobox_guards_restore_known_prompt_and_caption_length() -> None:
    previous = default_preset_for("qwen3_omni_instruct", "video_audio").id
    update, state, description, context = caption_tab.validate_prompt_preset(
        "garbage preset zzz",
        previous,
        "qwen3_omni_instruct_int4",
        "video_audio",
        True,
    )
    assert update["value"] == previous and state == previous
    assert description.startswith("Unknown task preset; kept ")
    assert context == ["qwen3_omni_instruct", "video_audio"]

    length_update, length_state, length_message = caption_tab.validate_caption_length(
        "a novel", "detailed"
    )
    assert length_update["value"] == "detailed"
    assert length_state == "detailed"
    assert "Unknown caption length" in length_message

    chat_choices, chat_previous = chat_tab.chat_prompt_choices(
        "qwen3_omni_instruct_int4", []
    )
    assert chat_choices and chat_previous
    chat_update, chat_state, chat_message = chat_tab.validate_chat_prompt_preset(
        "garbage preset zzz", chat_previous, "qwen3_omni_instruct_int4", []
    )
    assert chat_update["value"] == chat_previous and chat_state == chat_previous
    assert str(chat_message).startswith("Unknown task preset; kept ")


@pytest.mark.parametrize("order", [(8, 24), (24, 8)], ids=["caption-then-chat", "chat-then-caption"])
def test_caption_and_chat_reuse_the_same_resident_model(
    monkeypatch: pytest.MonkeyPatch, order: tuple[int, int]
) -> None:
    from vcap.models import loader

    loads: list[Any] = []
    unloads: list[Any] = []

    def fake_load(variant_key: str, **kwargs: Any) -> Any:
        loaded = SimpleNamespace(
            model=SimpleNamespace(),
            variant=SimpleNamespace(key=variant_key),
            spec=SimpleNamespace(family="qwen3_omni_instruct"),
            load_report=SimpleNamespace(activation_estimate_bytes=1),
        )
        loads.append((variant_key, kwargs))
        return loaded

    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(loader, "unload_model", lambda loaded: unloads.append(loaded))
    cache = loader.ModelCache()
    first = cache.load(
        "qwen3_omni_instruct_int4",
        budget_hint=BudgetHint(max_frames=order[0]),
        last_token_logits=True,
    )
    second = cache.load(
        "qwen3_omni_instruct_int4",
        budget_hint=BudgetHint(max_frames=order[1]),
        last_token_logits=False,
    )
    assert second is first
    assert len(loads) == 1
    assert unloads == []
    assert first.model._vcap_last_token_logits is False


def test_model_cache_reload_log_names_the_differing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    messages: list[str] = []
    logger = SimpleNamespace(log=lambda message, scope=None: messages.append(message), warn=lambda *_a, **_k: None)

    def fake_load(variant_key: str, **_kwargs: Any) -> Any:
        return SimpleNamespace(model=SimpleNamespace(), variant=SimpleNamespace(key=variant_key))

    monkeypatch.setattr(loader, "get_log", lambda: logger)
    monkeypatch.setattr(loader, "load_model", fake_load)
    monkeypatch.setattr(loader, "unload_model", lambda _loaded: None)
    cache = loader.ModelCache()
    cache.load("qwen3_omni_instruct_int4", attention="auto")
    cache.load("qwen3_omni_instruct_int4", attention="sdpa")
    assert "Reloading: attention auto→sdpa" in messages


def test_nested_clip_fitness_plan_is_allowed_and_plan_folders_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "outputs"
    normal = source / "clip.mp4"
    generated = source / "clip_fitness" / "should_skip.mp4"
    generated.parent.mkdir(parents=True)
    normal.write_bytes(b"video")
    generated.write_bytes(b"video")
    plan = {"source_folder": str(source), "items": []}
    target = dataset_tab.write_clip_fitness_plan(
        plan, source / "clip_fitness", timestamp="20260904_120000"
    )
    assert target.is_file()

    probed: list[Path] = []
    monkeypatch.setattr(dataset_tab, "list_media_files", lambda *_a, **_k: [normal, generated])

    def fake_probe(path: Path) -> Any:
        probed.append(Path(path))
        return SimpleNamespace(duration=4.0, fps=24.0, width=1280, height=720)

    monkeypatch.setattr(dataset_tab, "probe_media", fake_probe)
    monkeypatch.setattr(
        dataset_tab,
        "evaluate_clip",
        lambda *_a, **_k: SimpleNamespace(
            frames_available=96, frames_needed=81, suggested_frames=81, bucket=(1280, 720), warnings=[]
        ),
    )
    monkeypatch.setattr(dataset_tab, "_bucket_geometry", lambda *_a, **_k: ("1280x720", {}))
    rows, _summary, _plan = dataset_tab.analyze_clip_fitness(source, "wan", "720p", "keep_ar")
    assert len(rows) == 1 and probed == [normal]


def test_ready_check_is_log_only_and_progress_starts_preparing() -> None:
    from vcap.pipeline.runner import _emit_model_download_progress

    calls: list[Any] = []
    emitter = SimpleNamespace(
        log=lambda message, scope=None: calls.append(("log", message, scope)),
        phase_progress=lambda *args, **kwargs: calls.append(("progress", args, kwargs)),
    )
    _emit_model_download_progress(
        emitter,
        "qwen3_omni_instruct_int4: ready",
        {"state": "ready", "fraction": 1.0},
    )
    assert calls == [("log", "qwen3_omni_instruct_int4: ready", "models")]
    assert 'starting_label = "Preparing…"' in Path(caption_tab.__file__).read_text(encoding="utf-8")


def test_regeneration_history_has_kind_model_item_and_recovery_path(tmp_path: Path) -> None:
    run = tmp_path / "batch_0084_qwen3"
    run.mkdir()
    metadata = run / "editor_regeneration_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "model_info": {"variant_key": "qwen3_omni_instruct_int4"},
                "items_results": [
                    {
                        "status": "done",
                        "path": "video20s.mp4",
                        "outputs": {"txt": str(run / "video20s_segments" / "clip_0003.txt")},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summaries = list_recent_runs(tmp_path)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.kind == "regenerate"
    assert summary.model_key == "qwen3_omni_instruct_int4"
    assert summary.items == 1 and summary.counts == {"done": 1}
    assert summary.preview == "video20s_segments/clip_0003.txt"
    rows = caption_tab.run_history_rows(summaries)
    records = caption_tab.run_history_records(summaries)
    assert rows[0][1:5] == ["regenerate", "qwen3_omni_instruct_int4", 1, "1 / 0"]
    assert records[0]["metadata_path"] == str(metadata.resolve(strict=False))


def test_unload_waits_for_release_event_report() -> None:
    class Client:
        def __init__(self) -> None:
            self.released = False

        def ping(self) -> dict[str, Any]:
            return {"busy": False, "loaded_variant": "qwen3_omni_instruct_int4"}

        def release_model(self, timeout_s: float) -> dict[str, Any]:
            assert timeout_s == 30.0
            self.released = True
            return {
                "ev": "unloaded",
                "released": "qwen3_omni_instruct_int4",
                "report": {"vram_before_gb": 17.0, "vram_after_gb": 0.5},
            }

        def unload(self) -> None:
            pytest.fail("fire-and-forget unload was used")

    client = Client()
    message = caption_tab.unload_model_report(client)
    assert client.released
    assert "VRAM 17.00 → 0.50 GB" in message


def test_variant_disk_usage_is_cached_until_invalidated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vcap.models import downloads

    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "weights.bin").write_bytes(b"1234")
    scans = 0
    real_scandir = downloads.os.scandir

    def counting_scandir(path: Any) -> Any:
        nonlocal scans
        scans += 1
        return real_scandir(path)

    monkeypatch.setattr(downloads, "_variant_folder", lambda *_a, **_k: folder)
    monkeypatch.setattr(downloads.os, "scandir", counting_scandir)
    downloads.invalidate_variant_disk_usage("test")
    assert downloads.variant_disk_usage("test") == 4
    assert downloads.variant_disk_usage("test") == 4
    assert scans == 1
    downloads.invalidate_variant_disk_usage("test")
    assert downloads.variant_disk_usage("test") == 4
    assert scans == 2


def _layout_parent(config: dict[str, Any], component_id: int) -> int | None:
    parent: int | None = None

    def visit(node: dict[str, Any], parent_id: int | None = None) -> None:
        nonlocal parent
        if int(node.get("id", -1)) == component_id:
            parent = parent_id
            return
        for child in node.get("children") or []:
            visit(child, int(node.get("id", -1)))

    visit(config["layout"])
    return parent


def test_task_b_ui_defaults_layout_hotkey_registry_cancel_and_preset_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = build_app()
    try:
        ctx = demo.vcap_context
        config = demo.get_config_file()
        components = config["components"]
        by_elem = {
            item.get("props", {}).get("elem_id"): item for item in components
        }
        by_label = {
            item.get("props", {}).get("label"): item for item in components
        }
        assert by_label["Whole word"]["props"]["value"] is True
        assert "Enter sends, Shift+Enter adds a line" in by_elem["vc_chat_message"]["props"]["info"]
        assert "#vc_chat_message textarea" in HOTKEYS_HEAD and "vc_chat_send" in HOTKEYS_HEAD
        zip_parent = _layout_parent(config, by_elem["vc_results_zip_download"]["id"])
        button_parent = _layout_parent(config, by_elem["vc_results_zip"]["id"])
        assert zip_parent is not None and zip_parent != button_parent

        entries = {entry.key: entry for entry in ctx.settings_registry.entries()}
        assert entries["output_formats"].kind == "list"
        assert entries["output_formats"].choices == ("txt", "json", "srt", "vtt", "jsonl")
        assert entries["system_prompt"].kind == "str" and entries["system_prompt"].default == ""
        assert entries["trim_end_s"].kind == "float"

        conversation = {
            "messages": [{"role": "user", "content": "hello"}],
            "model_key": "qwen3_omni_instruct_int4",
        }
        same = chat_tab.chat_model_change_updates("qwen3_omni_instruct_bf16", conversation)
        assert same[3].get("__type__") == "update"
        assert "conversation kept" in same[-1]
        changed = chat_tab.chat_model_change_updates("qwen3_omni_thinking_int4", conversation)
        assert changed[3] == [] and changed[4]["messages"] == []
        assert "conversation cleared" in changed[-1]

        token = CancelToken()
        ctx.activate_cancel(token)
        cancel_calls: list[bool] = []
        monkeypatch.setattr(ctx.pipeline_client, "cancel", lambda force=False: cancel_calls.append(force))
        request = ctx.states["caption_cancel_handlers"]["request"]()
        assert request[0]["interactive"] is True and token.is_armed()
        ctx.states["caption_cancel_handlers"]["confirm"]()
        assert token.is_cancelled() and cancel_calls == []
        ctx.clear_active_cancel(token)

        handlers = ctx.states["preset_bar_handlers"]
        count = handlers["component_count"]
        monkeypatch.setattr(ctx.preset_store, "delete", lambda _name: True)
        monkeypatch.setattr(ctx.preset_store, "load", lambda _name: ctx.settings_registry.defaults())
        deleted = handlers["confirm_delete"]({"name": "Temporary preset"})
        assert deleted[count + 1] == ""
        reset = handlers["reset"]()
        assert reset[count]["value"] is None
        assert "(defaults)" in reset[count + 1]
    finally:
        demo.vcap_context.pipeline.shutdown()
