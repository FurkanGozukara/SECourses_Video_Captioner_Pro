from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
import wave
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
    spec = JobSpec.from_settings(settings, [InputItem(source)], output)
    assert JobSpec.from_json(spec.to_json()) == spec

    first = run_job(spec, RecordingSink(), CancelToken())
    assert first.counts["skipped"] == 1
    assert first.counts["done"] == 1
    assert len(fake_model) == 1
    assert (batch_out / "a.txt").read_text(encoding="utf-8") == "existing\n"
    metadata = json.loads(Path(first.metadata_path).read_text(encoding="utf-8"))
    assert metadata["settings"] == settings
    assert Path(first.run_dir, "run_log.txt").is_file()

    overwrite = JobSpec.from_settings(
        {**settings, "overwrite_existing": True},
        [InputItem(source)],
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


def test_multi_gpu_round_robin_fake_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VCAP_FAKE_CAPTIONER", "1")
    inputs = [InputItem(f"prompt {index}", text_prompt_only=True) for index in range(4)]
    spec = JobSpec.from_settings(
        _settings(
            subprocess_mode=True,
            gpu_indices=[0, 1],
            keep_model_loaded=False,
        ),
        inputs,
        OutputSpec(outputs_root=tmp_path / "runs"),
    )
    sink = RecordingSink()
    result = run_job(spec, sink, CancelToken())
    assert result.counts["done"] == 4
    assert [item.gpu_index for item in result.items] == [0, 1, 0, 1]
    assert len({item.outputs["txt"] for item in result.items}) == 4
    assert any(message.startswith("[GPU 0]") for message, _, _ in sink.logs)
    assert any(message.startswith("[GPU 1]") for message, _, _ in sink.logs)
