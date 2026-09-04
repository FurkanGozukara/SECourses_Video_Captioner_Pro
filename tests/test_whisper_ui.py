from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.core.subprocess_runner import CancelToken
from vcap.ui.app import build_app
from vcap.ui.tabs import transcribe_tab


def _fake_ui_transcription(request: dict[str, Any], **kwargs: Any) -> Any:
    from vcap.whisper.client import TranscriptionOutcome
    from vcap.whisper.engine import TranscriptResult, TranscriptSegment, TranscriptWord

    item = request["items"][0]
    result = TranscriptResult(
        [
            TranscriptSegment(
                0,
                0.0,
                1.25,
                "A test transcript.",
                [TranscriptWord(0.0, 1.25, "A test transcript.", 0.98)],
                avg_logprob=-0.02,
            )
        ],
        "en",
        0.99,
        1.25,
        0.25,
        request["params"]["model"],
        request["params"]["compute_type"],
        "cpu",
    )
    out_dir = Path(item["out_dir"])
    files: list[str] = []
    for fmt in request["output"]["formats"]:
        path = out_dir / f"{item['stem']}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "1\n00:00:00,000 --> 00:00:01,250\nA test transcript.\n" if fmt == "srt" else result.text
        path.write_text(content, encoding="utf-8")
        files.append(str(path))
    sink = kwargs["sink"]
    sink.on_segment({"item_index": item["index"], "id": 0, "start": 0.0, "end": 1.25, "text": "A test transcript."})
    sink.on_progress({"item_index": item["index"], "fraction": 1.0, "message": "Transcribing 00:01 / 00:01", "segments": 1, "elapsed_s": 0.25, "eta_s": 0.0})
    payload = {
        "event": "item_done",
        "item_index": item["index"],
        "files": files,
        "result": result.to_dict(),
        "skipped": False,
    }
    sink.on_item_done(payload)
    return TranscriptionOutcome(True, [payload], {item["index"]: result}, 0.25, False, None)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    demo = build_app()
    demo.vcap_context.outputs_dir = tmp_path / "outputs"
    demo.vcap_context.models_dir = tmp_path / "models"
    demo.vcap_context.temp_dir = tmp_path / "temp"
    try:
        yield demo
    finally:
        demo.vcap_context.pipeline.shutdown()


def test_registry_contains_whisper_contract_defaults(app: Any) -> None:
    registry = app.settings_registry
    defaults = registry.defaults()
    expected = {
        "whisper_model": "large-v1",
        "whisper_language": "english",
        "whisper_translate": False,
        "whisper_compute_type": "float16",
        "whisper_device": "auto",
        "whisper_gpu_index": 0,
        "whisper_cpu_threads": 0,
        "whisper_beam_size": 5,
        "whisper_best_of": 5,
        "whisper_patience": 1.0,
        "whisper_temperature": 0.0,
        "whisper_length_penalty": 1.0,
        "whisper_repetition_penalty": 1.2,
        "whisper_no_repeat_ngram_size": 0,
        "whisper_compression_ratio_threshold": 2.4,
        "whisper_log_prob_threshold": -1.0,
        "whisper_no_speech_threshold": 0.6,
        "whisper_condition_on_previous_text": True,
        "whisper_prompt_reset_on_temperature": 0.5,
        "whisper_initial_prompt": "",
        "whisper_repeat_initial_prompt": False,
        "whisper_prefix": "",
        "whisper_hotwords": "",
        "whisper_suppress_blank": True,
        "whisper_suppress_tokens": "[-1]",
        "whisper_max_initial_timestamp": 1.0,
        "whisper_word_timestamps": True,
        "whisper_normalize_word_timestamps": True,
        "whisper_highlight_words": False,
        "whisper_prepend_punctuations": "\"'([{-",
        "whisper_append_punctuations": "\"'.,!?:)]}",
        "whisper_max_new_tokens": 0,
        "whisper_chunk_length": 30,
        "whisper_hallucination_silence_threshold": 0.0,
        "whisper_language_detection_threshold": 0.5,
        "whisper_language_detection_segments": 1,
        "whisper_use_batched_inference": False,
        "whisper_batch_size": 1,
        "whisper_vad_filter": False,
        "whisper_vad_threshold": 0.5,
        "whisper_vad_min_speech_ms": 250,
        "whisper_vad_max_speech_s": 9999.0,
        "whisper_vad_min_silence_ms": 2000,
        "whisper_vad_speech_pad_ms": 400,
        "whisper_formats": ["srt", "vtt", "txt", "lrc", "tsv", "json"],
        "whisper_add_timestamp": False,
        "whisper_batch_recursive": False,
        "whisper_batch_overwrite": False,
        "whisper_batch_save_next_to_source": False,
        "whisper_batch_output_dir": "",
        "transcript_enabled": False,
        "transcript_formats": ["srt", "txt"],
        "transcript_inject_prompt": True,
        "transcript_prompt_wrapper": "Exact speech transcript for this clip (use it verbatim for dialogue, do not invent speech):\n{{TRANSCRIPT}}",
        "transcript_file_suffix": "_transcript",
    }
    assert {key: defaults[key] for key in expected} == expected
    entries = {entry.key: entry for entry in registry.entries()}
    assert all(entries[key].description for key in expected)
    assert all(entries[key].kind is not None for key in expected)


def test_tab_order_endpoint_and_hotkey_proxies(app: Any) -> None:
    config = app.get_config_file()
    tabs = [
        item["props"].get("id")
        for item in config["components"]
        if item.get("type") == "tabitem"
    ]
    assert tabs.index("processing") < tabs.index("transcribe") < tabs.index("chat")
    element_ids = {item.get("props", {}).get("elem_id") for item in config["components"]}
    assert {"hk_transcribe_start", "hk_transcribe_cancel"} <= element_ids
    public_names = {dependency.get("api_name") for dependency in config["dependencies"]}
    assert "transcribe" in public_names


def test_input_resolution_and_batch_plan_mirror_and_skip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    first = nested / "first.wav"
    second = nested / "second.mp4"
    ignored = nested / "notes.txt"
    first.write_bytes(b"wav")
    second.write_bytes(b"mp4")
    ignored.write_text("ignore", encoding="utf-8")
    output = tmp_path / "batch"
    existing = output / "nested" / "first.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("already done", encoding="utf-8")
    settings = {
        "whisper_batch_input_folder": str(source),
        "whisper_batch_recursive": True,
        "whisper_batch_output_dir": str(output),
        "whisper_batch_overwrite": False,
        "whisper_batch_save_next_to_source": False,
        "whisper_batch_limit_items": 0,
        "whisper_formats": ["txt", "srt"],
    }

    resolved = transcribe_tab.resolve_transcribe_inputs_at_start(settings, "folder", [])
    plan = transcribe_tab.prepare_transcription_plan(resolved, settings, "folder", tmp_path / "runs")

    assert resolved == [str(first), str(second)]
    assert plan.run_dir.name == "batch_0001_whisper"
    assert [Path(item["path"]).name for item in plan.items] == ["second.mp4"]
    assert plan.items[0]["out_dir"] == str(output / "nested")
    assert plan.skipped[0]["status"] == "skipped"


def test_cancel_flow_uses_shared_token(app: Any) -> None:
    ctx = app.vcap_context
    token = CancelToken()
    ctx.states["transcribe_job_token"] = token
    ctx.activate_cancel(token)

    cancel_update, confirmation_update, status = transcribe_tab.request_transcription_cancel(ctx)

    assert token.is_armed()
    assert not token.is_cancelled()
    assert cancel_update["interactive"] is True
    assert confirmation_update["visible"] is True
    assert "confirmation" in status

    cancel_update, confirmation_update, status = transcribe_tab.confirm_transcription_cancel(ctx)
    assert token.is_cancelled()
    assert cancel_update["interactive"] is False
    assert confirmation_update["visible"] is False
    assert "Cancellation requested" in status


def test_transcribe_handler_with_fake_stream_writes_metadata_and_results(
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"not decoded by fake")
    monkeypatch.setattr(transcribe_tab, "_run_transcription_client", _fake_ui_transcription)
    ctx = app.vcap_context
    settings = app.settings_registry.defaults()
    settings.update(
        whisper_device="cpu",
        whisper_compute_type="int8",
        whisper_formats=["srt", "txt", "json"],
    )
    values = app.settings_registry.dict_to_values(settings)
    handler = ctx.states["transcribe_run_handler"]

    updates = list(handler(*values, [str(source)], "upload"))
    final = updates[-1]
    state = final[10]

    assert final[5] == "A test transcript."
    assert final[4][0][2] == "Done"
    assert Path(state["metadata_path"]).is_file()
    metadata = json.loads(Path(state["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["extra"]["kind"] == "whisper_transcription"
    assert metadata["extra"]["counts"] == {"done": 1, "skipped": 0, "failed": 0}
    assert len(state["produced_files"]) == 3
    assert "Complete: 1 transcribed" in final[1]
    from vcap.ui.tabs.editor_tab import scan_folder

    editor_items = scan_folder(state["run_dir"])
    assert any(Path(item["caption_path"]).suffix == ".txt" for item in editor_items)


def _long_ui_transcription(request: dict[str, Any], **kwargs: Any) -> Any:
    from vcap.whisper.client import TranscriptionOutcome
    from vcap.whisper.engine import TranscriptResult, TranscriptSegment, TranscriptWord

    item = request["items"][0]
    out_dir = Path(item["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = kwargs["sink"]
    sink.on_download(
        {
            "fraction": 0.5,
            "bytes": 50,
            "total": 100,
            "speed_bps": 10,
            "message": "large-v1 model files 50%",
        }
    )
    time.sleep(0.55)
    segments = []
    srt_cues: list[str] = []
    for index in range(3000):
        start = float(index)
        text = f"segment {index + 1:04d} " + ("long transcript text " * 5)
        segment = TranscriptSegment(
            index + 1,
            start,
            start + 0.9,
            text,
            [TranscriptWord(start, start + 0.9, text, 0.97)],
            avg_logprob=-0.03,
        )
        segments.append(segment)
        sink.on_segment(
            {
                "item_index": item["index"],
                "id": index + 1,
                "start": start,
                "end": start + 0.9,
                "text": text,
            }
        )
        sink.on_progress(
            {
                "item_index": item["index"],
                "fraction": (index + 1) / 3000,
                "message": f"Transcribing segment {index + 1}",
                "segments": index + 1,
                "elapsed_s": max(0.01, (index + 1) / 300.0),
                "eta_s": 1.0,
            }
        )
        srt_cues.append(
            f"{index + 1}\n00:00:00,000 --> 00:00:00,900\n{text}\n"
        )
        if (index + 1) % 600 == 0:
            time.sleep(0.55)
    result = TranscriptResult(
        segments,
        "en",
        0.995,
        3000.0,
        10.0,
        request["params"]["model"],
        request["params"]["compute_type"],
        "cpu",
    )
    srt_path = out_dir / f"{item['stem']}.srt"
    txt_path = out_dir / f"{item['stem']}.txt"
    srt_path.write_text("\n".join(srt_cues), encoding="utf-8")
    txt_path.write_text(result.text, encoding="utf-8")
    files = [str(srt_path), str(txt_path)]
    payload = {
        "event": "item_done",
        "item_index": item["index"],
        "files": files,
        "skipped": False,
    }
    sink.on_item_done(payload)
    return TranscriptionOutcome(True, [payload], {item["index"]: result}, 10.0, False, None)


def test_long_transcribe_stream_is_throttled_and_final_views_are_bounded(
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "long.wav"
    source.write_bytes(b"fake")
    monkeypatch.setattr(transcribe_tab, "_run_transcription_client", _long_ui_transcription)
    monkeypatch.setattr(
        transcribe_tab,
        "_model_choices",
        lambda _root: [("large-v1 - 3.09 GB ✓ downloaded", "large-v1")],
    )
    monkeypatch.setattr(
        transcribe_tab,
        "_model_info_markdown",
        lambda _alias, _root: "✓ downloaded",
    )
    settings = app.settings_registry.defaults()
    settings.update(
        whisper_device="cpu",
        whisper_compute_type="int8",
        whisper_formats=["srt", "txt"],
    )
    values = app.settings_registry.dict_to_values(settings)

    started = time.monotonic()
    updates = list(app.vcap_context.states["transcribe_run_handler"](*values, [str(source)], "upload"))
    elapsed = time.monotonic() - started

    assert len(updates) - 3 <= int(elapsed / transcribe_tab.LIVE_UPDATE_INTERVAL_S) + 1
    live_updates = updates[1:-2]
    assert any("Downloading Whisper model" in str(update[0]) and "50.0%" in str(update[0]) for update in live_updates)
    assert all(len(update[7]) <= transcribe_tab.LIVE_SEGMENT_LIMIT for update in live_updates)
    assert all(
        len(str(update[5]).encode("utf-8")) <= transcribe_tab.LIVE_TRANSCRIPT_LIMIT_BYTES
        for update in live_updates
    )

    status_first, final = updates[-2], updates[-1]
    assert status_first[5] == {"__type__": "update"}
    assert "Complete: 1 transcribed" in status_first[1]
    assert final[3] == "**Speed:** 300.0× realtime"
    assert len(final[5].encode("utf-8")) <= transcribe_tab.FINAL_TRANSCRIPT_LIMIT_BYTES
    assert "transcript capped at 200 KB" in final[5]
    assert final[6].count("-->") == transcribe_tab.SRT_PREVIEW_CUE_LIMIT
    assert "… full file:" in final[6]
    assert len(final[7]) == transcribe_tab.FINAL_SEGMENT_LIMIT + 1
    assert final[7][0][0] == 1 and final[7][499][0] == 500
    assert final[7][-1][3] == "showing 500 of 3000 (open the JSON/TSV for all)"
    assert set(final[9]) == {
        "language",
        "language_probability",
        "duration_s",
        "elapsed_s",
        "model",
        "compute_type",
        "device",
        "segment_count",
        "word_count",
        "files",
    }
    assert final[9]["segment_count"] == final[9]["word_count"] == 3000
    assert "segments" not in final[9]
    assert final[21]["choices"][0][0].endswith("✓ downloaded")
    assert final[22] == "✓ downloaded"


def test_terminal_status_is_yielded_before_result_views_are_built(
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "ordered.wav"
    source.write_bytes(b"fake")
    monkeypatch.setattr(transcribe_tab, "_run_transcription_client", _fake_ui_transcription)
    original = transcribe_tab._result_views
    view_calls: list[bool] = []

    def tracked_views(*args: Any, **kwargs: Any):
        view_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(transcribe_tab, "_result_views", tracked_views)
    settings = app.settings_registry.defaults()
    settings.update(whisper_device="cpu", whisper_compute_type="int8", whisper_formats=["txt"])
    values = app.settings_registry.dict_to_values(settings)
    run = app.vcap_context.states["transcribe_run_handler"](*values, [str(source)], "upload")

    assert "Starting transcription" in next(run)[0]
    status = next(run)
    assert "Complete: 1 transcribed" in status[1]
    assert status[5] == {"__type__": "update"}
    assert view_calls == []
    result = next(run)
    assert result[5] == "A test transcript."
    assert view_calls == [True]
    with pytest.raises(StopIteration):
        next(run)


def test_cancel_timer_skips_unchanged_state_and_cancelled_run_can_restart(
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vcap.whisper.client import TranscriptionOutcome

    source = tmp_path / "cancel.wav"
    source.write_bytes(b"fake")
    started = threading.Event()
    runs = 0

    def fake_client(request: dict[str, Any], **kwargs: Any) -> Any:
        nonlocal runs
        runs += 1
        if runs > 1:
            return _fake_ui_transcription(request, **kwargs)
        token = kwargs["cancel"]
        sink = kwargs["sink"]
        started.set()
        index = request["items"][0]["index"]
        while not token.is_cancelled():
            sink.on_segment(
                {"item_index": index, "id": 1, "start": 0.0, "end": 0.2, "text": "waiting"}
            )
            time.sleep(0.01)
        return TranscriptionOutcome(False, [], {}, 0.1, True, None)

    monkeypatch.setattr(transcribe_tab, "_run_transcription_client", fake_client)
    settings = app.settings_registry.defaults()
    settings.update(whisper_device="cpu", whisper_compute_type="int8", whisper_formats=["txt"])
    values = app.settings_registry.dict_to_values(settings)
    ctx = app.vcap_context
    handlers = ctx.states["transcribe_cancel_handlers"]
    ctx.states["transcribe_job_token"] = None
    assert handlers["refresh"]() == {"__type__": "update"}

    run = ctx.states["transcribe_run_handler"](*values, [str(source)], "upload")
    first = next(run)
    assert "Starting transcription" in first[0]
    assert started.wait(1.0)
    assert handlers["refresh"]()["interactive"] is True
    assert handlers["refresh"]() == {"__type__": "update"}
    _button, confirmation, _status = handlers["request"]()
    assert confirmation["visible"] is True
    assert handlers["refresh"]() == {"__type__": "update"}
    handlers["confirm"]()
    cancelled = list(run)[-1]
    assert "Cancelled:" in cancelled[1]
    assert cancelled[4][0][2] == "Cancelled"
    assert "Cancelled" in ctx.app_log.tail(20)
    assert ctx.states["transcribe_job_token"] is None
    assert ctx.get_active_cancel() is None

    restarted = list(ctx.states["transcribe_run_handler"](*values, [str(source)], "upload"))[-1]
    assert "Complete: 1 transcribed" in restarted[1]
    assert restarted[4][0][2] == "Done"


def _parents(config: dict[str, Any]) -> dict[int, int | None]:
    result: dict[int, int | None] = {}

    def visit(node: dict[str, Any], parent: int | None = None) -> None:
        node_id = int(node.get("id", -1))
        result[node_id] = parent
        for child in node.get("children") or []:
            visit(child, node_id)

    visit(config["layout"])
    return result


def test_transcribe_result_actions_zip_srt_and_gpu_layout(app: Any) -> None:
    handles = app.vcap_context.transcribe_handles
    config = app.get_config_file()
    parents = _parents(config)
    first_row = {
        parents[handles.start._id],
        parents[handles.cancel._id],
        parents[handles.open_output._id],
        parents[handles.open_transcript._id],
    }
    second_row = {
        parents[handles.open_editor._id],
        parents[handles.copy_transcript._id],
        parents[handles.results_zip._id],
        parents[handles.retry_failed._id],
    }
    assert len(first_row) == len(second_row) == 1
    assert first_row != second_row
    assert parents[handles.results_zip_file._id] != parents[handles.results_zip._id]

    components = {int(item["id"]): item for item in config["components"]}
    gpu_parent = components[parents[handles.controls["whisper_gpu_index"]._id]]
    if gpu_parent["type"] == "form":
        gpu_parent = components[parents[int(gpu_parent["id"])]]
    assert gpu_parent["type"] == "column"
    assert gpu_parent["props"]["scale"] == 2
    assert gpu_parent["props"]["min_width"] == 240
    gpu_choices = components[handles.controls["whisper_gpu_index"]._id]["props"]["choices"]
    assert all("NVIDIA GeForce" not in str(choice[0]) for choice in gpu_choices)
    assert components[handles.srt._id]["type"] == "textbox"
    assert components[handles.srt._id]["props"]["max_lines"] == 16
    assert components[handles.progress.tokens._id]["props"]["value"] == "**Speed:** —"

    cancel_timer_dependency = next(
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id == handles.cancel_timer._id and event == "tick"
            for target_id, event in dependency.get("targets", [])
        )
    )
    assert cancel_timer_dependency["outputs"] == [handles.cancel._id]
