from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from vcap.core.media import find_ffmpeg, probe_media
from vcap.core.progress import ProgressEvent
from vcap.core.subprocess_runner import CancelToken, WorkerProcess, build_child_env
from vcap.pipeline.client import PipelineClient
from vcap.pipeline.job import InputItem, JobResult, JobSpec, OutputSpec
from vcap.pipeline.runner import run_job


class RecordingSink:
    def __init__(self) -> None:
        self.logs: list[tuple[str, str, str | None]] = []
        self.progress: list[ProgressEvent] = []
        self.items: list[ProgressEvent] = []

    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        self.logs.append((message, level, scope))

    def on_progress(self, event: ProgressEvent) -> None:
        self.progress.append(event)

    def on_item(self, event: ProgressEvent) -> None:
        self.items.append(event)


class FakeCaptioner:
    def __init__(self, variant_key: str, calls: list[str]) -> None:
        self.variant = SimpleNamespace(key=variant_key)
        self.load_report = SimpleNamespace(peak_vram_gb=0.0)
        self.model = object()
        self.processor = None
        self.calls = calls

    def caption(self, media: Any, prompt: Any, gen: Any, pre: Any, cb: Any) -> Any:
        del prompt, gen, pre
        from vcap.models.base import CaptionResult, CaptionTiming, TokenUsage

        path = Path(str(media.path)) if media.path else None
        label = path.name if path else "text"
        self.calls.append(label)
        deadline = time.monotonic() + 0.015
        while time.monotonic() < deadline:
            if cb.cancel.is_cancelled():
                raise RuntimeError("cancelled by test fake")
            time.sleep(0.002)
        if "broken" in label:
            raise RuntimeError("synthetic per-item failure")
        text = f"caption for {path.stem if path else 'text'}"
        if media.end is not None:
            duration = max(0.0, float(media.end) - float(media.start or 0.0))
        elif path and path.is_file():
            duration = float(probe_media(path).duration or 0.0)
        else:
            duration = 0.0
        segments = [(0.0, min(0.4, duration), text)] if duration > 0 else []
        return CaptionResult(
            text=text,
            raw_text=text,
            segments=segments,
            usage=TokenUsage(3, 5),
            timing=CaptionTiming(0.0, 0.01, 500.0, 0.01),
        )


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from vcap.models import loader

    calls: list[str] = []
    cache = loader.ModelCache()
    monkeypatch.setattr(loader, "MODEL_CACHE", cache)

    def load_model(variant_key: str, **kwargs: Any) -> FakeCaptioner:
        del kwargs
        return FakeCaptioner(variant_key, calls)

    monkeypatch.setattr(loader, "load_model", load_model)
    return calls


def _settings(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_key": "qwen3_omni_instruct_int8",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe this input.",
        "system_prompt": None,
        "fps": 2.0,
        "max_frames": 24,
        "max_pixels": 131_072,
        "min_pixels": 4_096,
        "use_audio_in_video": False,
        "output_formats": ["txt", "json"],
        "keep_model_loaded": True,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("settings", "expected_layers"),
    [
        ({"gpu_layers": "AUTO"}, "auto"),
        ({"gpu_layers": "all"}, "all"),
        ({"gpu_layers": "12"}, 12),
        ({"layers_on_gpu": 8}, 8),
    ],
)
def test_job_offload_parsing_accepts_auto_all_and_counts(
    settings: dict[str, Any], expected_layers: int | str, tmp_path: Path
) -> None:
    spec = JobSpec.from_settings(
        _settings(
            **settings,
            vram_reserve_gb="3.5",
            swap_slots="3",
            pin_cpu="true",
        ),
        [],
        OutputSpec(outputs_root=tmp_path),
    )

    assert spec.model.offload.gpu_layers == expected_layers
    assert spec.model.offload.vram_reserve_gb == 3.5
    assert spec.model.offload.swap_slots == 3
    assert spec.model.offload.pin_cpu is True
    assert JobSpec.from_json(spec.to_json()).model.offload == spec.model.offload


def test_job_offload_defaults_match_automatic_block_swap(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        _settings(), [], OutputSpec(outputs_root=tmp_path)
    )
    offload = spec.model.offload
    assert offload.gpu_layers == "auto"
    assert offload.pin_cpu is True
    assert offload.vram_reserve_gb == 2.0
    assert offload.swap_slots == 2
    payload = spec.to_dict()
    payload["model"]["offload"].update(
        gpu_layers="16", vram_reserve_gb="4.0", swap_slots="3", pin_cpu="false"
    )
    parsed = JobSpec.from_dict(payload).model.offload
    assert (parsed.gpu_layers, parsed.vram_reserve_gb, parsed.swap_slots, parsed.pin_cpu) == (
        16,
        4.0,
        3,
        False,
    )


def _write_wav(path: Path, seconds: float = 0.2) -> None:
    sample_rate = 8_000
    frames = bytearray()
    for index in range(int(sample_rate * seconds)):
        value = int(2_000 * math.sin(index * 2 * math.pi * 440 / sample_rate))
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _two_scene_video(path: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is unavailable")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:r=10:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_job_json_and_batch_skip_overwrite_metadata(
    tmp_path: Path, fake_model: list[str]
) -> None:
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "a.txt").write_text("first", encoding="utf-8")
    (source / "b.txt").write_text("second", encoding="utf-8")
    batch_out = tmp_path / "captions"
    batch_out.mkdir()
    (batch_out / "a.txt").write_text("existing\n", encoding="utf-8")
    output = OutputSpec(
        kind="batch",
        outputs_root=tmp_path / "runs",
        batch_output_dir=batch_out,
    )
    settings = _settings(overwrite_existing=False, metadata_probe="preserve-me")
    batch_inputs = [InputItem(source / "a.txt"), InputItem(source / "b.txt")]
    spec = JobSpec.from_settings(settings, batch_inputs, output)
    assert JobSpec.from_json(spec.to_json()) == spec

    first = run_job(spec, RecordingSink(), CancelToken())
    assert first.counts["skipped"] == 1
    assert first.counts["done"] == 1
    assert len(fake_model) == 1
    assert (batch_out / "a.txt").read_text(encoding="utf-8") == "existing\n"
    metadata = json.loads(Path(first.metadata_path).read_text(encoding="utf-8"))
    assert metadata["settings"] == settings
    assert metadata["processing_time_seconds"] >= 0
    assert Path(first.run_dir, "summary.json").is_file()
    assert Path(first.run_dir, "run_log.txt").is_file()

    overwrite = JobSpec.from_settings(
        {**settings, "overwrite_existing": True},
        batch_inputs,
        output,
    )
    second = run_job(overwrite, RecordingSink(), CancelToken())
    assert second.counts["done"] == 2
    assert second.counts["skipped"] == 0
    assert len(fake_model) == 3
    assert "caption for a" in (batch_out / "a.txt").read_text(encoding="utf-8")


def test_per_item_error_continues(tmp_path: Path, fake_model: list[str]) -> None:
    paths = []
    for name in ("good_1.txt", "broken.txt", "good_2.txt"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths.append(InputItem(path))
    spec = JobSpec.from_settings(
        _settings(),
        paths,
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    result = run_job(spec, RecordingSink(), CancelToken())
    statuses = {Path(item.path).name: item.status for item in result.items}
    assert statuses == {"good_1.txt": "done", "broken.txt": "failed", "good_2.txt": "done"}
    assert result.counts["failed"] == 1
    assert fake_model == ["good_1.txt", "broken.txt", "good_2.txt"]
    failed = next(item for item in result.items if item.status == "failed")
    assert "synthetic per-item failure" in failed.message
    assert "RuntimeError" in failed.traceback_tail


def test_batch_limit_counts_only_items_without_existing_captions(
    tmp_path: Path,
    fake_model: list[str],
) -> None:
    source = tmp_path / "inputs"
    output_dir = tmp_path / "captions"
    source.mkdir()
    output_dir.mkdir()
    paths = []
    for name in ("clip1.txt", "clip2.txt", "clip3.txt", "clip4.txt"):
        path = source / name
        path.write_text(name, encoding="utf-8")
        paths.append(InputItem(path))
    (output_dir / "clip1.txt").write_text("existing", encoding="utf-8")
    spec = JobSpec.from_settings(
        _settings(overwrite_existing=False, batch_limit_items=2),
        paths,
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=output_dir,
            source_root=source,
            limit_items=2,
        ),
    )

    result = run_job(spec, RecordingSink(), CancelToken())

    assert result.counts["done"] == 2
    assert result.counts["skipped"] == 2
    assert fake_model == ["clip2.txt", "clip3.txt"]
    summary = json.loads(Path(result.run_dir, "summary.json").read_text(encoding="utf-8"))
    assert summary["limit_items"] == 2
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert metadata["batch_limit_items"] == 2


def test_batch_sidecar_mirrors_nested_unicode_source_path(
    tmp_path: Path,
    fake_model: list[str],
) -> None:
    source = tmp_path / "source"
    nested = source / "g\u00f6lge_\u0432\u0438\u0434\u0435\u043e"
    media = nested / "clip.txt"
    output_dir = tmp_path / "captions"
    nested.mkdir(parents=True)
    media.write_text("source", encoding="utf-8")
    spec = JobSpec.from_settings(
        _settings(),
        [InputItem(media)],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=output_dir,
            source_root=source,
        ),
    )

    result = run_job(spec, RecordingSink(), CancelToken())

    assert result.items[0].path == str(media)
    assert Path(result.items[0].outputs["txt"]) == output_dir / nested.name / "clip.txt"
    assert (output_dir / nested.name / "clip.txt").is_file()


def test_scene_split_captions_and_combined_srt_offsets(
    tmp_path: Path, fake_model: list[str]
) -> None:
    video = tmp_path / "two_scenes.mp4"
    _two_scene_video(video)
    settings = _settings(
        scene_detect_enabled=True,
        scene_threshold=10.0,
        scene_min_len_s=0.1,
        scene_max_len_s=0.0,
        merge_short_scenes=False,
        split_mode="precise",
        max_clip_duration_s=30.0,
        save_clips=True,
        output_formats=["txt", "json", "srt"],
    )
    spec = JobSpec.from_settings(
        settings,
        [InputItem(video)],
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    result = run_job(spec, RecordingSink(), CancelToken())
    assert result.counts["done"] == 1
    item = result.items[0]
    completed = [segment for segment in item.segments if segment["status"] == "done"]
    assert len(completed) == 2
    assert len(fake_model) == 2
    assert Path(completed[0]["outputs"]["txt"]).parent.name.endswith("_clips")
    combined = Path(item.outputs["txt"]).read_text(encoding="utf-8")
    assert combined.count("[") >= 2
    srt = Path(item.outputs["srt"]).read_text(encoding="utf-8")
    cue_lines = [line for line in srt.splitlines() if " --> " in line]
    assert len(cue_lines) == 2
    assert cue_lines[0].startswith("00:00:00,000")
    assert cue_lines[1].startswith("00:00:01,")


def test_unsupported_audio_names_compatible_models_without_loading(
    tmp_path: Path, fake_model: list[str]
) -> None:
    audio = tmp_path / "tone.wav"
    _write_wav(audio)
    spec = JobSpec.from_settings(
        _settings(model_key="timechat_int8", use_audio_in_video=True),
        [InputItem(audio)],
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    result = run_job(spec, RecordingSink(), CancelToken())
    assert result.items[0].status == "unsupported"
    assert "does not support audio-only input" in result.items[0].message
    assert "Qwen3-Omni" in result.items[0].message
    assert fake_model == []


def test_worker_protocol_end_to_end_with_fake_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    worker = WorkerProcess().start(
        [sys.executable, "-u", "-m", "vcap.pipeline.worker", "--gpu", "0"],
        cwd=Path(__file__).resolve().parents[1],
        env=build_child_env(0, {"VCAP_FAKE_CAPTIONER": "1"}),
        name="pipeline-test",
    )
    stream = worker.events()
    assert next(stream) == {"ev": "ready"}
    spec = JobSpec.from_settings(
        _settings(),
        [InputItem("direct worker text", text_prompt_only=True)],
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    worker.send({"cmd": "run_job", "job": spec.to_dict()})
    events: list[dict[str, Any]] = []
    job_result: JobResult | None = None
    for event in stream:
        events.append(event)
        if event.get("ev") == "result":
            job_result = JobResult.from_dict(event["job_result"])
            break
    assert job_result is not None and job_result.counts["done"] == 1
    assert any(event.get("ev") == "progress" for event in events)
    assert any(event.get("ev") == "item" for event in events)
    worker.send({"cmd": "ping"})
    pong = next(event for event in stream if event.get("ev") == "pong")
    assert pong["loaded_variant"] == "qwen3_omni_instruct_int8"
    assert pong["block_swap"] is None
    worker.send({"cmd": "unload"})
    worker.send({"cmd": "exit"})
    list(stream)
    assert worker.wait(timeout=5) == 0


def test_worker_protocol_deduplicates_timestamped_log_mirror() -> None:
    import io

    from vcap.pipeline.worker import _ProtocolWriter

    stream = io.StringIO()
    protocol = _ProtocolWriter(stream)
    protocol.emit(
        {
            "ev": "stdout",
            "source": "stdout",
            "text": "[03:09:49] [models] Model ready",
        }
    )
    protocol.emit(
        {"ev": "log", "level": "info", "scope": "models", "text": "Model ready"}
    )
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert events == [
        {
            "ev": "log",
            "level": "info",
            "text": "Model ready",
            "timestamp": "03:09:49",
            "scope": "models",
        }
    ]


def test_pipeline_client_graceful_cancel_returns_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setenv("VCAP_FAKE_CAPTION_SLEEP", "30")
    spec = JobSpec.from_settings(
        _settings(subprocess_mode=True, keep_model_loaded=True),
        [InputItem("cancel this text", text_prompt_only=True)],
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    client = PipelineClient(subprocess_mode=True)
    token = CancelToken()
    state: dict[str, Any] = {}

    def execute() -> None:
        try:
            state["result"] = client.run_job(spec, RecordingSink(), token)
        except BaseException as exc:  # A forced-worker exit is also acceptable cancellation.
            state["error"] = exc

    thread = threading.Thread(target=execute)
    started = time.monotonic()
    thread.start()
    time.sleep(0.4)
    token.cancel()
    thread.join(timeout=4.0)
    client.shutdown()
    assert not thread.is_alive()
    assert time.monotonic() - started < 5.0
    if "result" in state:
        assert state["result"].counts["cancelled"] == 1
    else:
        assert "cancel" in str(state["error"]).casefold()


def test_worker_cancel_stops_fast_batch_between_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    monkeypatch.setenv("VCAP_FAKE_CAPTION_SLEEP", "0.35")
    spec = JobSpec.from_settings(
        _settings(subprocess_mode=True, keep_model_loaded=True),
        [InputItem(f"fast item {index}", text_prompt_only=True) for index in range(4)],
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    client = PipelineClient(subprocess_mode=True)
    token = CancelToken()
    state: dict[str, Any] = {}

    def execute() -> None:
        try:
            state["result"] = client.run_job(spec, RecordingSink(), token)
        except BaseException as exc:
            state["error"] = exc

    thread = threading.Thread(target=execute)
    thread.start()
    time.sleep(0.15)
    token.cancel()
    thread.join(timeout=4.0)
    client.shutdown()
    assert not thread.is_alive()
    if "result" in state:
        counts = state["result"].counts
        assert counts["cancelled"] >= 1
        assert counts["done"] < 4
    else:
        assert "cancel" in str(state["error"]).casefold()


def test_mixed_batch_adapts_automatic_prompt_to_each_modality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vcap.models import loader
    from vcap.prompts.presets import get_preset, render_prompt

    image_path = tmp_path / "still.png"
    audio_path = tmp_path / "song.wav"
    Image.new("RGB", (32, 32), color="navy").save(image_path)
    _write_wav(audio_path)
    variables: dict[str, Any] = {}
    _, image_prompt = render_prompt(get_preset("qwen3_image_describe"), variables)
    captured: list[tuple[str, str | None, str | None]] = []

    class CapturingCaptioner(FakeCaptioner):
        def caption(self, media: Any, prompt: Any, gen: Any, pre: Any, cb: Any) -> Any:
            captured.append((Path(str(media.path)).suffix, prompt.preset_id, prompt.user_prompt))
            return super().caption(media, prompt, gen, pre, cb)

    monkeypatch.setattr(loader, "MODEL_CACHE", loader.ModelCache())
    monkeypatch.setattr(
        loader,
        "load_model",
        lambda variant_key, **_kwargs: CapturingCaptioner(variant_key, []),
    )
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(
                prompt_preset_id="qwen3_image_describe",
                user_prompt=image_prompt,
                prompt_variables=variables,
            ),
            [InputItem(image_path), InputItem(audio_path)],
            OutputSpec(
                kind="batch",
                outputs_root=tmp_path / "runs",
                batch_output_dir=tmp_path / "captions",
            ),
        ),
        sink,
        CancelToken(),
    )
    assert result.counts["done"] == 2
    assert [preset for _, preset, _ in captured] == [
        "qwen3_image_describe",
        "qwen3_audio_caption",
    ]
    assert "audio" in str(captured[1][2]).casefold()
    assert any(
        "using qwen3_audio_caption" in message
        for message, _, scope in sink.logs
        if scope == "prompts"
    )


def test_multi_gpu_batch_dispatch_and_round_robin_partitioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vcap.pipeline.runner as pipeline_runner

    inputs = [InputItem(f"prompt {index}", text_prompt_only=True) for index in range(5)]
    spec = JobSpec.from_settings(
        _settings(
            subprocess_mode=True,
            gpu_indices="0, 1 2;2 invalid",
            keep_model_loaded=False,
        ),
        inputs,
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=tmp_path / "captions",
        ),
    )
    assert spec.runtime.gpu_indices == (0, 1, 2)
    list_strings = JobSpec.from_settings(
        _settings(gpu_indices=["0", "2", "2", "bad"]),
        [],
        OutputSpec(kind="batch", outputs_root=tmp_path / "list-strings"),
    )
    list_ints = JobSpec.from_settings(
        _settings(gpu_indices=[0, 2, 2, -1]),
        [],
        OutputSpec(kind="batch", outputs_root=tmp_path / "list-ints"),
    )
    assert list_strings.runtime.gpu_indices == (0, 2)
    assert list_ints.runtime.gpu_indices == (0, 2)
    assert pipeline_runner._round_robin_partitions(list(range(5)), 3) == [[0, 3], [1, 4], [2]]

    called: list[JobSpec] = []

    def fake_multi_gpu(job: JobSpec, _sink: Any, _cancel: Any) -> JobResult:
        called.append(job)
        return JobResult([], {"total": 0}, str(tmp_path), str(tmp_path / "metadata.json"), 0.0)

    monkeypatch.setattr(pipeline_runner, "_run_multi_gpu", fake_multi_gpu)
    pipeline_runner.run_job(spec, RecordingSink(), CancelToken())
    assert called == [spec]

    single = JobSpec.from_settings(
        _settings(subprocess_mode=True, gpu_indices="0 1"),
        [InputItem("one", text_prompt_only=True)],
        OutputSpec(outputs_root=tmp_path / "single-runs"),
    )
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    pipeline_runner.run_job(single, RecordingSink(), CancelToken())
    assert called == [spec]


def test_batch_file_list_mirrors_source_root_and_skips_mirrored_output(tmp_path: Path) -> None:
    import vcap.pipeline.runner as pipeline_runner

    source = tmp_path / "batch_in"
    first = source / "sub1" / "same.mp4"
    second = source / "sub2" / "same.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    batch_out = tmp_path / "batch_out"
    spec = JobSpec.from_settings(
        _settings(),
        [InputItem(first), InputItem(second)],
        OutputSpec(
            kind="batch",
            outputs_root=tmp_path / "runs",
            batch_output_dir=batch_out,
            source_root=source,
        ),
    )
    assert JobSpec.from_json(spec.to_json()).output.source_root == str(source)
    resolved = pipeline_runner._resolve_inputs(spec)
    pipeline_runner._assign_batch_outputs(spec, resolved)
    assert [entry.out_dir for entry in resolved] == [batch_out / "sub1", batch_out / "sub2"]
    assert [entry.stem for entry in resolved] == ["same", "same"]

    (batch_out / "sub1").mkdir(parents=True)
    (batch_out / "sub1" / "same.txt").write_text("done", encoding="utf-8")
    pipeline_runner._apply_batch_skip(spec, resolved)
    assert [entry.status for entry in resolved] == ["skipped", "pending"]

    duplicate_spec = JobSpec.from_settings(
        _settings(),
        [InputItem(first), InputItem(first)],
        spec.output,
    )
    duplicates = pipeline_runner._resolve_inputs(duplicate_spec)
    pipeline_runner._assign_batch_outputs(duplicate_spec, duplicates)
    assert [entry.stem for entry in duplicates] == ["same", "same_0002"]


def test_batch_directory_expansion_excludes_caption_sidecars(tmp_path: Path) -> None:
    import vcap.pipeline.runner as pipeline_runner

    source = tmp_path / "batch"
    source.mkdir()
    Image.new("RGB", (16, 16), color="red").save(source / "still.png")
    for name in ("still.txt", "notes.md", "data.json", "rows.jsonl", "captions.srt", "captions.vtt"):
        (source / name).write_text("sidecar", encoding="utf-8")
    spec = JobSpec.from_settings(
        _settings(batch_recursive=True),
        [InputItem(source)],
        OutputSpec(kind="batch", outputs_root=tmp_path / "runs", source_root=source, recursive=True),
    )
    resolved = pipeline_runner._resolve_inputs(spec)
    assert [entry.path.name for entry in resolved if entry.path is not None] == ["notes.md", "still.png"]


def test_batch_trim_is_ignored_once(
    tmp_path: Path,
    fake_model: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vcap.pipeline.runner as pipeline_runner

    video = tmp_path / "clip.mp4"
    _two_scene_video(video)
    monkeypatch.setattr(
        pipeline_runner,
        "trim_media",
        lambda *_args, **_kwargs: pytest.fail("batch trim_media must not be called"),
    )
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(trim_start_s=0.25, trim_end_s=1.25),
            [InputItem(video)],
            OutputSpec(
                kind="batch",
                outputs_root=tmp_path / "runs",
                batch_output_dir=tmp_path / "captions",
                source_root=tmp_path,
            ),
        ),
        sink,
        CancelToken(),
    )
    assert result.counts["done"] == 1
    notices = [message for message, _, _ in sink.logs if message == "Trim range is ignored for folder batches"]
    assert notices == ["Trim range is ignored for folder batches"]
    assert fake_model


def test_split_clips_are_temporary_and_subtitle_cues_only_get_replacements(
    tmp_path: Path,
    fake_model: list[str],
) -> None:
    video = tmp_path / "split.mp4"
    _two_scene_video(video)
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(
                segment_mode="fixed",
                fixed_chunk_s=0.8,
                trainer_target="wan",
                save_clips=False,
                caption_prefix="PRE",
                    trigger_word="TRIG",
                    trigger_mode="prefix",
                caption_suffix="POST",
                replace_pairs=[["caption", "description"]],
                output_formats=["txt", "srt"],
            ),
            [InputItem(video)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        sink,
        CancelToken(),
    )
    item = result.items[0]
    assert len(item.segments) >= 2
    assert all(not Path(str(record["media_path"])).exists() for record in item.segments)
    assert not any(path.name.endswith("_clips") for path in Path(result.run_dir).rglob("*"))
    assert any("Produced clips are temporary" in message for message, _, _ in sink.logs)
    caption = Path(item.outputs["txt"]).read_text(encoding="utf-8")
    subtitle = Path(item.outputs["srt"]).read_text(encoding="utf-8")
    assert all(value in caption for value in ("PRE", "TRIG", "POST", "description"))
    assert "description" in subtitle
    assert all(value not in subtitle for value in ("PRE", "TRIG", "POST"))
    assert len(fake_model) >= 2


def test_progress_payload_and_live_item_elapsed_keys(
    tmp_path: Path,
    fake_model: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    sink = RecordingSink()
    run_job(
        JobSpec.from_settings(
            _settings(),
            [InputItem("progress text", text_prompt_only=True)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        sink,
        CancelToken(),
    )
    required = {
        "processed",
        "remaining",
        "total",
        "elapsed_s",
        "eta_s",
        "item_index",
        "item_elapsed_s",
    }
    assert sink.progress and required <= sink.progress[0].data.keys()
    running = [event for event in sink.items if event.status == "running"]
    assert running and required <= running[0].data.keys()
    assert all(float(event.data["item_elapsed_s"]) >= 0 for event in running)
    console = capsys.readouterr().out
    assert "[0/1]" in console and "| elapsed " in console and "| ETA " in console


def test_loader_progress_is_written_once_to_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.core.logs import get_log
    from vcap.models import loader

    marker = "Loading checkpoint 50% - test.weight"

    def load_model(variant_key: str, **kwargs: Any) -> FakeCaptioner:
        get_log().log(marker, scope="models")
        kwargs["progress_cb"](marker, 0.5)
        return FakeCaptioner(variant_key, [])

    monkeypatch.setattr(loader, "load_model", load_model)
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(),
            [InputItem("log text", text_prompt_only=True)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        sink,
        CancelToken(),
    )
    run_log = Path(result.run_dir, "run_log.txt").read_text(encoding="utf-8")
    assert run_log.count(marker) == 1
    assert any(event.data.get("phase") == "model_load" for event in sink.progress)


def test_runner_passes_block_swap_budget_and_records_shared_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vcap.pipeline.runner as pipeline_runner
    from vcap.models import loader
    from vcap.models.registry import MODEL_SPECS

    captured: dict[str, Any] = {}

    def load_model(
        variant_key: str,
        *,
        budget_hint: Any = None,
        **kwargs: Any,
    ) -> FakeCaptioner:
        captured.update(variant_key=variant_key, budget_hint=budget_hint, **kwargs)
        captioner = FakeCaptioner(variant_key, [])
        captioner.load_report = SimpleNamespace(
            peak_vram_gb=7.0,
            block_swap={
                "mode": "block_swap",
                "layer_count": 48,
                "resident_layers": 40,
                "swapped_layers": 8,
                "slots": 3,
            },
            activation_estimate_bytes=int(1.25 * 2**30),
            vram_cap_bytes=int(27.5 * 2**30),
        )
        return captioner

    samples = iter(({"shared_gb": 0.1}, {"shared_gb": 2.5}))
    monkeypatch.setattr(loader, "load_model", load_model)
    monkeypatch.setattr(
        pipeline_runner.gpu,
        "shared_gpu_memory_usage",
        lambda: next(samples),
        raising=False,
    )
    monkeypatch.setattr(
        pipeline_runner.gpu,
        "resource_snapshot",
        lambda _index: {"gpu_index": 0, "vram_used_gb": 7.0},
    )
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(
                gpu_layers="auto",
                vram_reserve_gb=3.5,
                swap_slots=3,
                pin_cpu=True,
                max_frames=30,
                max_pixels=262_144,
                fps=1.5,
                max_new_tokens=321,
            ),
            [
                InputItem("first", text_prompt_only=True),
                InputItem("second", text_prompt_only=True),
            ],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        sink,
        CancelToken(),
    )

    offload = captured["offload"]
    assert offload.gpu_layers == "auto"
    assert offload.vram_reserve_gb == 3.5
    assert offload.swap_slots == 3
    assert offload.pin_cpu is True
    hint = captured["budget_hint"]
    # Text-only inputs need no vision or audio budget: the media-derived hint
    # reports zero frames and the kinds the job really contains.
    assert (hint.max_frames, hint.max_pixels, hint.fps, hint.max_new_tokens) == (
        0,
        262_144,
        1.5,
        321,
    )
    assert hint.media_kinds == ("text",)
    assert hint.context_tokens == MODEL_SPECS["qwen3_omni_instruct"].limits.context_tokens
    # 0.1 GiB is ordinary CUDA-context overhead (no warning); 2.5 GiB with no pinned
    # block-swap buffers means 1.9 GiB of device allocations were paged.
    assert any(
        level == "warning"
        and "WDDM shared GPU memory beyond the pinned block-swap buffers: 1.90 GiB" in message
        for message, level, _scope in sink.logs
    )
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    gpu_info = metadata["gpu_info"]
    assert gpu_info["block_swap"]["resident_layers"] == 40
    assert gpu_info["vram_reserve_gb"] == 3.5
    assert gpu_info["activation_estimate_gb"] == 1.25
    assert gpu_info["vram_cap_gb"] == 27.5
    assert gpu_info["shared_gpu_memory_peak_gb"] == 2.5
    assert gpu_info["shared_gpu_memory_excess_peak_gb"] == pytest.approx(1.9)


def test_runner_omits_budget_hint_for_legacy_loader_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    captured: dict[str, Any] = {}

    def load_model(variant_key: str, **kwargs: Any) -> FakeCaptioner:
        captured.update(kwargs)
        return FakeCaptioner(variant_key, [])

    monkeypatch.setattr(loader, "load_model", load_model)
    result = run_job(
        JobSpec.from_settings(
            _settings(),
            [InputItem("legacy", text_prompt_only=True)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        RecordingSink(),
        CancelToken(),
    )
    assert result.counts["done"] == 1
    assert "budget_hint" not in captured


def test_context_carry_over_appends_only_previous_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader

    video = tmp_path / "context.mp4"
    _two_scene_video(video)
    prompts: list[str | None] = []

    class ContextCaptioner(FakeCaptioner):
        def caption(self, media: Any, prompt: Any, gen: Any, pre: Any, cb: Any) -> Any:
            prompts.append(prompt.user_prompt)
            result = super().caption(media, prompt, gen, pre, cb)
            number = len(prompts)
            return replace(result, text=f"final words from segment {number}", raw_text=f"raw {number}")

    monkeypatch.setattr(loader, "load_model", lambda key, **_kwargs: ContextCaptioner(key, []))
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(
                segment_mode="fixed",
                fixed_chunk_s=0.8,
                context_carry_over=True,
            ),
            [InputItem(video)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        sink,
        CancelToken(),
    )
    assert result.counts["done"] == 1 and len(prompts) >= 2
    assert prompts[0] == "Describe this input."
    assert 'Context from the previous segment (do not repeat it): "final words from segment 1"' in str(prompts[1])
    if len(prompts) >= 3:
        assert "segment 2" in str(prompts[2]) and "segment 1" not in str(prompts[2])
    assert any("Applied previous-segment context" in message for message, _, _ in sink.logs)

    import vcap.pipeline.runner as pipeline_runner
    from vcap.models.registry import MODEL_SPECS

    assert len(pipeline_runner._context_excerpt(" ".join(f"w{i}" for i in range(100))).split()) == 60
    assert not pipeline_runner._supports_context_carry_over(MODEL_SPECS["timechat"])
    assert not pipeline_runner._supports_context_carry_over(MODEL_SPECS["qwen3_omni_captioner"])


def test_auto_reject_analysis_error_continues_item(
    tmp_path: Path,
    fake_model: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vcap.pipeline.runner as pipeline_runner

    video = tmp_path / "quality_error.mp4"
    _two_scene_video(video)
    monkeypatch.setattr(
        pipeline_runner,
        "analyze_clip_quality",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("quality unavailable")),
    )
    sink = RecordingSink()
    result = run_job(
        JobSpec.from_settings(
            _settings(auto_reject=True),
            [InputItem(video)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        sink,
        CancelToken(),
    )
    assert result.counts["done"] == 1 and fake_model
    assert result.items[0].segments[0]["quality_error"].endswith("quality unavailable")
    assert any("Auto-reject analysis failed" in message for message, _, _ in sink.logs)


def test_audio_only_auto_reject_runs_without_visual_failures(
    tmp_path: Path,
    fake_model: list[str],
) -> None:
    audio = tmp_path / "audio_quality.wav"
    _write_wav(audio, seconds=0.25)
    result = run_job(
        JobSpec.from_settings(
            _settings(
                auto_reject=True,
                reject_min_duration_s=0.1,
                reject_max_black_ratio=0.0,
                reject_max_static_score=1000.0,
                reject_min_sharpness=1000.0,
                reject_max_silence_ratio=1.0,
            ),
            [InputItem(audio)],
            OutputSpec(outputs_root=tmp_path / "runs"),
        ),
        RecordingSink(),
        CancelToken(),
    )
    assert result.counts["done"] == 1 and fake_model
    quality = result.items[0].segments[0]["quality"]
    assert quality["has_audio"] is True and quality["has_video"] is False


def test_metadata_records_finish_reason_and_new_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcap.models import loader
    from vcap.models.base import CaptionResult, CaptionTiming

    sampling_seen: list[str] = []

    class FinishCaptioner(FakeCaptioner):
        def caption(self, media: Any, prompt: Any, gen: Any, pre: Any, cb: Any) -> Any:
            del media, prompt, gen, cb
            sampling_seen.append(pre.sampling_strategy)
            return CaptionResult(
                text="complete",
                raw_text="complete",
                usage=SimpleNamespace(prompt_tokens=2, new_tokens=3, finish_reason="length"),
                timing=CaptionTiming(0.0, 0.01, 300.0, 0.01),
            )

    monkeypatch.setattr(loader, "load_model", lambda key, **_kwargs: FinishCaptioner(key, []))
    source_root = tmp_path / "source"
    source_root.mkdir()
    result = run_job(
        JobSpec.from_settings(
            _settings(sampling_strategy="uniform", context_carry_over=True),
            [InputItem("metadata text", text_prompt_only=True)],
            OutputSpec(outputs_root=tmp_path / "runs", source_root=source_root),
        ),
        RecordingSink(),
        CancelToken(),
    )
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert metadata["finish_reason"] == "length"
    assert metadata["sampling_strategy"] == "uniform"
    assert metadata["context_carry_over"] is True
    assert Path(metadata["source_root"]) == source_root
    assert metadata["processing_time_seconds"] >= 0
    assert metadata["items_results"][0]["segments"][0]["finish_reason"] == "length"
    assert sampling_seen == ["uniform"]

    import vcap.pipeline.runner as pipeline_runner
    from vcap.models.registry import MODEL_SPECS

    degraded = pipeline_runner._degrade_pre(
        {
            "fps": 2.0,
            "max_frames": 24,
            "max_pixels": 131_072,
            "min_pixels": 4_096,
            "use_audio_in_video": False,
            "sampling_strategy": "uniform",
        },
        MODEL_SPECS["qwen3_omni_instruct"],
    )
    assert degraded is not None and degraded[0]["sampling_strategy"] == "uniform"


def test_pipeline_client_mode_switch_stops_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    client = PipelineClient(subprocess_mode=True)
    try:
        client.run_job(
            JobSpec.from_settings(
                _settings(subprocess_mode=True, keep_model_loaded=True),
                [InputItem("worker text", text_prompt_only=True)],
                OutputSpec(outputs_root=tmp_path / "runs"),
            ),
            RecordingSink(),
            CancelToken(),
        )
        worker = client._worker
        assert worker is not None and worker.is_alive()
        client.set_subprocess_mode(False)
        assert client.subprocess_mode is False and client._worker is None
        assert not worker.is_alive()
        client.set_subprocess_mode(True)
        assert client.subprocess_mode is True and client._worker is None
    finally:
        client.shutdown()


def test_pipeline_client_in_app_idle_timer_unloads_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vcap.pipeline.runner as pipeline_runner

    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    unloaded = threading.Event()
    monkeypatch.setattr(pipeline_runner, "unload_cached_model", unloaded.set)
    client = PipelineClient(subprocess_mode=False)
    try:
        client.run_job(
            JobSpec.from_settings(
                _settings(
                    subprocess_mode=False,
                    keep_model_loaded=True,
                    idle_unload_minutes=0.001,
                ),
                [InputItem("local idle text", text_prompt_only=True)],
                OutputSpec(outputs_root=tmp_path / "runs"),
            ),
            RecordingSink(),
            CancelToken(),
        )
        assert unloaded.wait(timeout=1.5)
    finally:
        client.shutdown()
