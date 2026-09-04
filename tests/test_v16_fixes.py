from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import pytest

from vcap import LOGS_DIR
from vcap.core.logs import get_log
from vcap.core.media import MediaInfo
from vcap.core.progress import ProgressEvent, ProgressTracker
from vcap.core.subprocess_runner import CancelToken
from vcap.pipeline.client import PipelineClient
from vcap.pipeline.job import InputItem, ItemResult, JobResult, JobSpec, OutputSpec
from vcap.pipeline import runner
from vcap.pipeline.chat import ChatResponse
from vcap.ui import components
from vcap.ui.app import build_app
from vcap.ui.tabs import caption_tab, dataset_tab, editor_tab
from vcap.whisper import worker as whisper_worker


@pytest.fixture(scope="module")
def app() -> Any:
    demo = build_app()
    try:
        yield demo
    finally:
        demo.vcap_context.pipeline_client.shutdown()


def _settings(**updates: Any) -> dict[str, Any]:
    values = {
        "model_key": "qwen3_omni_instruct_int8",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe this input.",
        "system_prompt": "",
        "fps": 1.0,
        "max_frames": 8,
        "max_pixels": 131_072,
        "min_pixels": 4_096,
        "use_audio_in_video": False,
        "output_formats": ["txt"],
        "keep_model_loaded": False,
        "subprocess_mode": False,
        "segment_mode": "whole",
    }
    values.update(updates)
    return values


def _video_info(path: Path, duration: float, *, audio: bool = False) -> MediaInfo:
    return MediaInfo(
        path=path,
        kind="video" if audio else "video_no_audio",
        duration=duration,
        width=160,
        height=96,
        fps=10.0,
        has_video=True,
        has_audio=audio,
    )


# D8 / D20: prompt selection remains family-safe and stable across input contexts.
def test_d8_prompt_preset_survives_upload_folder_upload(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_tab,
        "probe_media",
        lambda path: _video_info(Path(path), 20.0, audio=True),
    )
    handler = app.vcap_context.states["caption_prompt_context_handler"]
    variables = ("", "English", "Auto", "English", "detailed", "", "", "")
    preset = "qwen3_video_dense"
    upload = handler(
        "qwen3_omni_instruct_int4", "video_audio", True, ["video.mp4"], preset,
        ["qwen3_omni_instruct", "video_audio"], *variables,
    )
    assert upload[0]["value"] == preset
    folder = handler(
        "qwen3_omni_instruct_int4", "unknown", True, [], preset, upload[4], *variables,
    )
    assert folder[0]["value"] == preset
    returned = handler(
        "qwen3_omni_instruct_int4", "video_audio", True, ["video.mp4"], preset,
        folder[4], *variables,
    )
    assert returned[0]["value"] == preset


def test_d20_family_change_never_displays_foreign_preset(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_tab,
        "probe_media",
        lambda path: _video_info(Path(path), 10.0, audio=True),
    )
    handler = app.vcap_context.states["caption_prompt_context_handler"]
    result = handler(
        "timechat_int4",
        "image",
        True,
        ["actual-video.mp4"],
        "qwen3_image_describe",
        ["qwen3_omni_instruct", "image"],
        "", "English", "Auto", "English", "detailed", "", "", "",
    )
    update = result[0]
    assert update["value"] == "timechat_flatten_wan"
    selected_labels = [label for label, value in update["choices"] if value == update["value"]]
    assert selected_labels and "Qwen3" not in selected_labels[0]
    assert "using" in result[1]


# D16: selecting Upload owns and refreshes its preview; stale scans are discarded.
def test_d16_upload_refreshes_empty_state_and_late_folder_scan_is_ignored(
    app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handlers = app.vcap_context.states["media_mode_handlers"]
    empty = handlers["upload"]([])
    assert "No inputs selected" in empty[3]
    assert empty[-1] == []

    started = threading.Event()
    release = threading.Event()

    def slow_scan(*args: Any, **kwargs: Any) -> tuple[list[str], str]:
        del args, kwargs
        started.set()
        assert release.wait(2.0)
        return ["old-folder.png"], "folder scan"

    monkeypatch.setattr(components, "_folder_scan", slow_scan)
    monkeypatch.setattr(
        components,
        "_preview_updates",
        lambda paths: (None, None, None, str(paths), "tiles", "video", 1.0),
    )
    handlers["select"](SimpleNamespace(value="folder", index=2))
    result: dict[str, tuple[Any, ...]] = {}

    def scan() -> None:
        result["value"] = handlers["folder"](
            "folder", False, "output", False, 0,
            ["video", "audio", "image", "text"], "", ".txt",
        )

    thread = threading.Thread(target=scan)
    thread.start()
    assert started.wait(1.0)
    uploaded = tmp_path / "uploaded-video.mp4"
    uploaded.write_bytes(b"video")
    upload = handlers["upload"]([str(uploaded)])
    release.set()
    thread.join(2.0)
    assert upload[-1] == [str(uploaded.resolve())]
    assert all(value == gr.skip() for value in result["value"])


# D6: batch metadata/indexes recover editor media and enumerate external captions.
def test_d6_mirrored_batch_caption_recovers_source_media(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    captions = outputs / "qa_batch_v16"
    run_dir = outputs / "batch_0106_qwen3"
    source = tmp_path / "kaynak_日本語.mp4"
    caption = captions / "kaynak_日本語.txt"
    source.write_bytes(b"media")
    caption.parent.mkdir(parents=True)
    caption.write_text("caption", encoding="utf-8")
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {"items_results": [{"path": str(source), "kind": "video", "outputs": {"txt": str(caption)}}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resolved, recorded = editor_tab.resolve_media_from_metadata(caption, captions)
    assert resolved == source.resolve()
    assert recorded == source.resolve()


def test_d6_batch_run_directory_lists_external_caption_and_source(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "batch_0107_qwen3"
    source = tmp_path / "source.mp4"
    caption = tmp_path / "mirrored" / "source.txt"
    run_dir.mkdir(parents=True)
    caption.parent.mkdir()
    source.write_bytes(b"media")
    caption.write_text("visible caption", encoding="utf-8")
    (run_dir / "captions_index.json").write_text(
        json.dumps(
            {
                "_meta": {"format": "secourses_vcap_captions_index", "version": 1},
                "captions": {str(caption): {"source_path": str(source), "kind": "video", "start_s": None, "end_s": None}},
            }
        ),
        encoding="utf-8",
    )
    items = editor_tab.scan_folder(run_dir)
    assert len(items) == 1
    assert Path(str(items[0]["media_path"])) == source.resolve()
    label = editor_tab.editor_item_label(items[0], run_dir)
    assert caption.name in label and source.name in label


# D10: per-item clip windows are honored even for batch-shaped regeneration jobs.
def test_d10_editor_regeneration_captions_only_recorded_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    durations: list[float] = []

    def fake_probe(raw: str | os.PathLike[str]) -> MediaInfo:
        path = Path(raw)
        return _video_info(path, 7.767 if path.name == "trimmed.mp4" else 78.0)

    def fake_trim(source_path: Path, target: Path, start: float, end: float, **kwargs: Any) -> Path:
        del source_path, kwargs
        assert start == pytest.approx(10.0)
        assert end == pytest.approx(17.767)
        target.write_bytes(b"trimmed")
        return target

    class Captioner:
        load_report = SimpleNamespace(peak_vram_gb=0.0)

        def caption(self, media: Any, **kwargs: Any) -> Any:
            del kwargs
            durations.append(float(fake_probe(media.path).duration or 0.0))
            from vcap.models.base import CaptionResult

            return CaptionResult(text="window caption", raw_text="window caption")

        def unload(self) -> None:
            pass

    from vcap.models import loader

    monkeypatch.setattr(runner, "probe_media", fake_probe)
    monkeypatch.setattr(runner, "trim_media", fake_trim)
    monkeypatch.setattr(loader, "load_model", lambda *args, **kwargs: Captioner())
    spec = JobSpec.from_settings(
        _settings(),
        [InputItem(path=str(source), trim_start_s=10.0, trim_end_s=17.767)],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=tmp_path / "captions",
            overwrite=True,
        ),
    )
    result = runner.run_job(spec, None)
    assert result.counts["done"] == 1
    assert durations == pytest.approx([7.767], abs=0.001)
    index = json.loads(
        (Path(result.run_dir) / "captions_index.json").read_text(encoding="utf-8")
    )
    record = next(iter(index["captions"].values()))
    assert Path(record["source_path"]) == source.resolve()
    assert record["start_s"] == pytest.approx(10.0)
    assert record["end_s"] == pytest.approx(17.767)


# D11: editor jobs share the persistent caption worker and resident fake model.
def test_d11_consecutive_jobs_reuse_shared_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")

    class Sink:
        def __init__(self) -> None:
            self.logs: list[str] = []

        def on_log(self, message: str, **kwargs: Any) -> None:
            del kwargs
            self.logs.append(str(message))

        def on_progress(self, event: Any) -> None:
            del event

        def on_item(self, event: Any) -> None:
            del event

    client = PipelineClient(subprocess_mode=True)
    sink = Sink()
    try:
        spec = JobSpec.from_settings(
            _settings(
                model_key="qwen3_omni_instruct_int4",
                keep_model_loaded=True,
                subprocess_mode=True,
            ),
            [InputItem(path="", kind="text", text_prompt_only=True, text="describe")],
            OutputSpec(outputs_root=tmp_path),
        )
        first = client.run_job(spec, sink)
        first_pid = client._worker.pid if client._worker is not None else None
        first_log_count = len(sink.logs)
        second = client.run_job(spec, sink)
        second_pid = client._worker.pid if client._worker is not None else None
        assert first.counts["done"] == second.counts["done"] == 1
        assert first_pid is not None and second_pid == first_pid
        assert any(
            "Reusing resident qwen3_omni_instruct_int4" in line
            for line in sink.logs[first_log_count:]
        )
    finally:
        client.shutdown()


# D1 / D21 and Esc-Esc: cancel notes cannot overwrite terminal status.
def test_d1_d21_cancel_note_is_separate_and_escape_rearms_after_expiry(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = app.vcap_context
    handles = ctx.caption_handles
    config = app.get_config_file()
    confirm_dependency = next(
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id == handles.cancel_yes._id and event == "click"
            for target_id, event in dependency.get("targets", [])
        )
    )
    assert handles.cancel_note._id in confirm_dependency["outputs"]
    assert handles.progress.status._id not in confirm_dependency["outputs"]

    cancellations: list[bool] = []
    monkeypatch.setattr(ctx.pipeline_client, "cancel", lambda force=False: cancellations.append(force))
    handlers = ctx.states["caption_cancel_handlers"]
    token = CancelToken()
    ctx.states["caption_job_token"] = token
    handlers["escape"]()
    assert token.is_armed() and not token.is_cancelled()
    assert 7.0 < token._armed_until - time.monotonic() <= 8.0
    handlers["escape"]()
    assert token.is_cancelled() and cancellations == [False]

    expired = CancelToken()
    ctx.states["caption_job_token"] = expired
    expired.arm_confirmation(0.01)
    time.sleep(0.02)
    expiry_updates = ctx.states["caption_cancel_timer_handler"]({}, "")
    assert expiry_updates[-1] == ""
    assert expiry_updates[1]["visible"] is False
    handlers["escape"]()
    assert expired.is_armed() and not expired.is_cancelled()
    ctx.states["caption_job_token"] = None

    started = threading.Event()
    release = threading.Event()

    def cancelled_job(spec: JobSpec, sink: Any, token: CancelToken) -> JobResult:
        del token
        started.set()
        assert release.wait(2.0)
        sink.on_item(
            ProgressEvent(
                message="Cancelled",
                item_index=0,
                total_items=1,
                status="cancelled",
                data={"processed": 1, "remaining": 0},
            )
        )
        return JobResult(
            items=[ItemResult(0, spec.inputs[0].path, "video", "cancelled")],
            counts={"done": 0, "skipped": 0, "failed": 0, "cancelled": 1},
            run_dir="",
            metadata_path="",
            elapsed=0.1,
        )

    monkeypatch.setattr(ctx.pipeline_client, "run_job", cancelled_job)
    run_caption = ctx.states["caption_run_handler"]
    values = ctx.registry.dict_to_values(ctx.registry.defaults())
    updates = run_caption(*values, ["fake.mp4"], "upload", "video")
    next(updates)
    assert started.wait(1.0)
    handlers["request"]()
    handlers["confirm"]()
    release.set()
    final = list(updates)[-1]
    assert "Cancelled: 1 cancelled" in final[1]
    assert "Cooperative cancellation requested" not in final[1]

    transcribe = ctx.transcribe_handles
    transcribe_confirm = next(
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id == transcribe.confirm_cancel._id and event == "click"
            for target_id, event in dependency.get("targets", [])
        )
    )
    assert transcribe.cancel_note._id in transcribe_confirm["outputs"]
    assert transcribe.progress.status._id not in transcribe_confirm["outputs"]
    transcribe_token = CancelToken()
    ctx.states["transcribe_job_token"] = transcribe_token
    transcribe_handlers = ctx.states["transcribe_cancel_handlers"]
    transcribe_handlers["request"]()
    assert 7.0 < transcribe_token._armed_until - time.monotonic() <= 8.0
    transcribe_handlers["confirm"]()
    assert transcribe_token.is_cancelled()
    ctx.states["transcribe_job_token"] = None


# D2 / D3: dense result surfaces remain legible and explain empty clip runs.
def test_d2_run_history_has_sane_widths_and_short_preview(app: Any) -> None:
    handles = app.vcap_context.caption_handles
    props = next(
        item["props"] for item in app.get_config_file()["components"] if item["id"] == handles.run_history._id
    )
    assert len(props["column_widths"]) == 7
    assert props["wrap"] is True
    row = caption_tab.run_history_rows([{"name": "run", "preview": "x" * 200}])[0]
    assert len(row[-1]) <= 90 and row[-1].endswith("...")


def test_d3_clips_tab_shows_empty_run_hint(app: Any) -> None:
    handles = app.vcap_context.caption_handles
    props = {
        item["id"]: item.get("props", {}) for item in app.get_config_file()["components"]
    }
    assert "No clips were saved for this run" in props[handles.clips_empty_hint._id]["value"]
    assert props[handles.clips._id]["visible"] is False


# D4 / D5: estimates ignore no-work items and sub-split values match precision.
def test_d4_eta_waits_for_a_positive_duration() -> None:
    tracker = ProgressTracker(4)
    tracker.start_item(0)
    tracker.finish_item("skipped", 0.0)
    assert tracker.eta_seconds is None
    tracker.start_item(1)
    tracker.finish_item("done", 0.0)
    assert tracker.eta_seconds is None
    tracker.start_item(2)
    tracker.finish_item("done", 2.0)
    assert tracker.eta_seconds == pytest.approx(2.0)


def test_d5_dataset_target_seconds_uses_three_decimal_step(app: Any) -> None:
    _, seconds = dataset_tab.trainer_clip_suggestion("wan", 16.0, 81)
    assert seconds == pytest.approx(81 / 16)
    target = next(
        item for item in app.get_config_file()["components"]
        if item.get("props", {}).get("label") == "Target seconds"
    )
    assert target["props"]["value"] == round(81 / 16, 3)
    assert target["props"]["step"] == pytest.approx(0.001)
    assert target["props"]["precision"] == 3


# D7 / D9: editor counts refresh on save and raw worker lines stay out of status.
def test_d7_autosave_refreshes_stats_and_queue_cells(app: Any, tmp_path: Path) -> None:
    caption_path = tmp_path / "clip.txt"
    caption_path.write_text("old", encoding="utf-8")
    state = editor_tab.new_editor_state(tmp_path)
    state.update(
        items=[
            {
                "media_path": None,
                "caption_path": str(caption_path),
                "caption": "a much longer edited caption",
                "chars": 28,
                "tokens": 7,
                "status": "no media",
                "caption_formats": [str(caption_path)],
            }
        ],
        selected_index=0,
        dirty=True,
        draft_caption="a much longer edited caption",
        last_edit=time.monotonic() - 2.0,
    )
    result = app.vcap_context.states["editor_autosave_handler"](
        state, True, "a much longer edited caption"
    )
    assert caption_path.read_text(encoding="utf-8").strip() == "a much longer edited caption"
    assert result[1][0][3] == len("a much longer edited caption")
    assert "28 chars" in result[3]


def test_d9_editor_regeneration_filters_raw_worker_status(app: Any) -> None:
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    sink = app.vcap_context.states["editor_regeneration_sink_class"](events)
    sink.on_log("INFO:root:Successfully loaded: 'mslk.dll'", scope="worker")
    sink.on_progress(ProgressEvent(message="INFO:root:Successfully loaded: 'mslk.dll'"))
    assert events.empty()
    sink.on_progress(ProgressEvent(message="Loading checkpoint 50%", data={"phase": "model_load"}))
    assert events.get_nowait() == ("progress", "Loading model...")


# D12 / D13 / D14: concise diagnostics in UI, durable detail in logs.
def test_d12_unreadable_preview_hides_raw_ffprobe_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = "[mov @ 000001] moov atom not found G:\\private\\corrupt video.mp4"
    captured: list[str] = []
    monkeypatch.setattr(
        components,
        "get_log",
        lambda: SimpleNamespace(warn=lambda message, scope=None: captured.append(str(message))),
    )
    message = components._media_info_markdown(
        MediaInfo(path=tmp_path / "corrupt video.mp4", kind="unknown", error=raw)
    )
    assert message == "<span class='vc-warn'>1 unreadable file: corrupt video.mp4 (skipped)</span>"
    assert raw in captured[0]


def test_d13_idle_release_and_worker_stop_are_logged() -> None:
    class Worker:
        alive = True
        pid = 76543
        returncode = 0

        def is_alive(self) -> bool:
            return self.alive

        def send(self, value: dict[str, Any]) -> None:
            if value.get("cmd") == "exit":
                self.alive = False

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.alive = False
            return 0

        def kill_tree(self, grace: float = 0.0) -> None:
            del grace
            self.alive = False

    client = PipelineClient(subprocess_mode=True)
    worker = Worker()
    try:
        with client._state_lock:
            client._worker = worker  # type: ignore[assignment]
            client._idle_minutes = 0.001
            client._idle_unloaded = False
            client._idle_exited = False
            client._resident_variant = "qwen3_omni_instruct_int4"
            client._last_activity = time.monotonic() - 1.0
        deadline = time.monotonic() + 2.0
        expected = "released qwen3_omni_instruct_int4 and stopped the worker"
        while expected not in get_log().tail(50) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert expected in get_log().tail(50)
    finally:
        client.shutdown()


def test_d14_app_log_persists_daily_and_writes_worker_crash_file(
    app: Any, tmp_path: Path
) -> None:
    logger = get_log()
    try:
        for day in range(1, 17):
            (tmp_path / f"app_2000-01-{day:02d}.log").write_text("old\n", encoding="utf-8")
        path = logger.configure_persistence(tmp_path)
        logger.log("durable v1.6 line", scope="test")
        assert "durable v1.6 line" in path.read_text(encoding="utf-8")
        crash = logger.write_worker_crash(4321, ["native stderr", "trace tail"])
        assert crash.name == "worker_4321.log"
        assert "native stderr" in crash.read_text(encoding="utf-8")
        client = PipelineClient(subprocess_mode=False)
        try:
            client._worker_output_tail.append("pipeline native stderr")
            client._persist_worker_crash(SimpleNamespace(pid=9876, returncode=1))
            captured = tmp_path / "worker_9876.log"
            assert "pipeline native stderr" in captured.read_text(encoding="utf-8")
        finally:
            client.shutdown()
        assert len(list(tmp_path.glob("app_????-??-??.log"))) <= 14
        component_text = "\n".join(
            str(component.get("props", {}).get("value", ""))
            for component in app.get_config_file()["components"]
        )
        assert "logs/app_YYYY-MM-DD.log" in component_text
        assert "logs/worker_<pid>.log" in component_text
    finally:
        logger.configure_persistence(LOGS_DIR)


# D15: every shipped preset is portable and limited to registry preset keys.
def test_d15_all_shipped_presets_are_portable_subsets(app: Any) -> None:
    allowed = {
        entry.key for entry in app.vcap_context.settings_registry.entries() if entry.in_preset
    }

    def machine_path(value: Any) -> bool:
        if isinstance(value, str):
            return bool(re.search(r"[A-Za-z]:[\\/]", value)) or "/home/" in value
        if isinstance(value, dict):
            return any(machine_path(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(machine_path(item) for item in value)
        return False

    for path in Path("presets_default").glob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        meta = document.get("_meta") or {}
        settings = document.get("settings") or {}
        assert meta.get("name") and meta.get("description"), path.name
        assert not (set(settings) - allowed), path.name
        assert not machine_path(document), path.name


# D17 / D18: native startup noise is log-only and model identity updates first.
def test_d17_llama_stderr_is_log_only_during_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sink:
        def __init__(self) -> None:
            self.logs: list[str] = []
            self.progress: list[str] = []

        def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
            del level, scope
            self.logs.append(message)

        def on_progress(self, event: ProgressEvent) -> None:
            self.progress.append(event.message)

        def on_item(self, event: ProgressEvent) -> None:
            del event

    class Loaded:
        load_report = SimpleNamespace(peak_vram_gb=0.0)

        def caption(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return "caption"

        def unload(self) -> None:
            pass

    from vcap.models import loader

    def load_model(variant: str, **kwargs: Any) -> Loaded:
        del variant
        callback = kwargs["progress_cb"]
        callback("llama-server: raw native warning")
        callback("Starting llama-server (GGUF)... 2 s", {"phase": "llama_start"})
        return Loaded()

    monkeypatch.setattr(loader, "load_model", load_model)
    spec = JobSpec.from_settings(
        _settings(model_key="qwen3_omni_instruct_gguf_q4"),
        [],
        OutputSpec(outputs_root=tmp_path),
    )
    sink = Sink()
    tracker = ProgressTracker(1)
    tracker.start_item(0, "item")
    session = runner._ModelSession(spec, runner._Emitter(sink, tracker), CancelToken())
    try:
        session.ensure()
    finally:
        session.unload()
    assert any("raw native warning" in line for line in sink.logs)
    assert not any("raw native warning" in line for line in sink.progress)
    assert any("Starting llama-server (GGUF)" in line for line in sink.progress)


def test_d18_model_identity_is_first_select_chain(app: Any) -> None:
    handles = app.vcap_context.caption_handles
    model_id = handles.controls["model_key"]._id
    config = app.get_config_file()
    components_by_id = {item["id"]: item for item in config["components"]}
    selects = [
        dependency
        for dependency in config["dependencies"]
        if any(
            target_id == model_id and event == "select"
            for target_id, event in dependency.get("targets", [])
        )
    ]
    assert selects and selects[0]["queue"] is False
    output_values = [
        str(components_by_id[component_id].get("props", {}).get("value", ""))
        for component_id in selects[0]["outputs"]
    ]
    assert any("Precision:" in value and "Backend:" in value for value in output_values)


# D22: Whisper probes first, skips silent video, sanitizes corrupt media, and
# the caption pipeline merges video-only captions without invoking Whisper.
class _NoDecodeEngine:
    transcribe_calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def load(self) -> None:
        pass

    def transcribe(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        type(self).transcribe_calls += 1
        raise AssertionError("decode must not run")

    def unload(self) -> None:
        pass


def test_d22_whisper_worker_skips_video_without_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"silent")
    events: list[dict[str, Any]] = []
    _NoDecodeEngine.transcribe_calls = 0
    monkeypatch.setattr(whisper_worker, "_load_engine_class", lambda: _NoDecodeEngine)
    monkeypatch.setattr(whisper_worker, "probe_media", lambda path: _video_info(Path(path), 2.0))
    monkeypatch.setattr(
        whisper_worker,
        "_emit",
        lambda event, **payload: events.append({"event": event, **payload}),
    )
    code = whisper_worker._run_request(
        {"action": "transcribe", "params": {}, "output": {"formats": ["txt"]}, "items": [{"index": 0, "path": str(source)}]},
        [],
        threading.Event(),
    )
    skipped = next(event for event in events if event["event"] == "item_done")
    assert code == 0 and skipped["skipped"] is True
    assert skipped["message"] == "No audio track; skipped"
    assert _NoDecodeEngine.transcribe_calls == 0


def test_d22_whisper_worker_sanitizes_unreadable_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "corrupt.mp4"
    source.write_bytes(b"corrupt")
    raw = f"[Errno 1094995529] Invalid data found when processing input: '{source}'"
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(whisper_worker, "_load_engine_class", lambda: _NoDecodeEngine)
    monkeypatch.setattr(
        whisper_worker,
        "probe_media",
        lambda path: MediaInfo(path=Path(path), kind="unknown", error=raw),
    )
    monkeypatch.setattr(
        whisper_worker,
        "_emit",
        lambda event, **payload: events.append({"event": event, **payload}),
    )
    code = whisper_worker._run_request(
        {"action": "transcribe", "params": {}, "output": {"formats": ["txt"]}, "items": [{"index": 0, "path": str(source)}]},
        [],
        threading.Event(),
    )
    failure = next(event for event in events if event["event"] == "item_error")
    assert code == 1
    assert failure["message"] == "unreadable media (ffmpeg: Invalid data found when processing input)"
    assert "traceback" not in failure and str(source) not in failure["message"]
    assert any(raw in str(event.get("message")) for event in events if event["event"] == "log")


def test_d22_caption_pipeline_skips_whisper_for_silent_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"silent")
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setattr(runner, "probe_media", lambda path: _video_info(Path(path), 2.0))
    monkeypatch.setattr(
        runner,
        "_run_item_transcript",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Whisper must be skipped")),
    )
    spec = JobSpec.from_settings(
        _settings(audio_caption_source="whisper", caption_write_merged=True),
        [InputItem(path=str(source))],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=tmp_path / "captions",
            overwrite=True,
        ),
    )
    result = runner.run_job(spec, None)
    item = result.items[0]
    assert item.status == "done"
    assert item.audio_caption_path is None
    assert item.merged_caption_path
    assert Path(item.merged_caption_path).read_text(encoding="utf-8").strip()


# D23: terminal chat output bypasses stale queued load/generation progress.
def test_d23_chat_completion_coalesces_stale_progress(app: Any) -> None:
    config = app.get_config_file()
    dependency = next(item for item in config["dependencies"] if item.get("api_name") == "chat")
    send_message = app.fns[dependency["id"]].fn
    ctx = app.vcap_context
    original = ctx.pipeline_client.chat

    def fake_chat(request: Any, on_event: Any, token: Any) -> ChatResponse:
        del request, token
        for value in range(60):
            on_event({"ev": "status", "message": f"Loading checkpoint {value}%", "data": {}})
        on_event({"ev": "delta", "text": "Complete answer", "reasoning": ""})
        on_event({"ev": "status", "message": "Tokens: generating", "data": {"new_tokens": 21}})
        on_event({"ev": "chat_result", "result": {}})
        return ChatResponse(
            model_key="qwen3_omni_instruct_int4",
            text="Complete answer",
            raw_text="Complete answer",
            reasoning="",
            prompt_tokens=10,
            new_tokens=21,
            finish_reason="eos",
            prefill_s=0.1,
            decode_s=1.0,
            tokens_per_s=12.31,
            total_s=1.1,
            peak_vram_gb=0.0,
            cancelled=False,
            context_tokens=10,
            context_limit=32768,
        )

    ctx.pipeline_client.chat = fake_chat
    try:
        values = ctx.registry.dict_to_values(ctx.registry.defaults())
        updates = list(send_message(*values, {}, [], "", "hello"))
    finally:
        ctx.pipeline_client.chat = original
    assert len(updates) <= 4
    assert "Complete: 21 tokens" in updates[-1][2]
    assert "Tokens:** 21" in updates[-1][3]
    assert "12.31 tok/s" in updates[-1][3]


# Resident-selection regression: a stale pre-run choice cannot unload the model
# actually used by the run; a deliberate change while busy still can.
def test_model_release_ignores_stale_pre_run_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    client = PipelineClient(subprocess_mode=False)
    releases: list[str | None] = []
    monkeypatch.setattr(
        client,
        "release_model",
        lambda unless_variant=None, **kwargs: releases.append(unless_variant) or {"released": "B"},
    )
    try:
        client.record_variant_selection("A")
        client.record_job_variant("B")
        client._release_after_selection_change("B")
        assert releases == []
        with client._state_lock:
            client._busy = True
        client.record_variant_selection("A")
        with client._state_lock:
            client._busy = False
        client._release_after_selection_change("B")
        assert releases == ["A"]
    finally:
        client.shutdown()
