from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vcap.core.subprocess_runner import CancelToken
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec, TranscriptSpec
from vcap.pipeline.runner import _prompt_with_transcript, run_job


def _write_wav(path: Path, seconds: float = 0.4) -> None:
    rate = 8_000
    frames = bytearray()
    for index in range(int(rate * seconds)):
        value = int(1_200 * math.sin(index * 2 * math.pi * 440 / rate))
        frames.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def _settings(**overrides: Any) -> dict[str, Any]:
    values = {
        "model_key": "qwen3_omni_instruct_int8",
        "prompt_preset_id": "custom",
        "user_prompt": "Describe the scene.",
        "system_prompt": None,
        "max_frames": 0,
        "use_audio_in_video": True,
        "output_formats": ["txt", "json"],
        "keep_model_loaded": False,
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
    }
    values.update(overrides)
    return values


class _Captioner:
    def __init__(self, key: str, prompts: list[Any]) -> None:
        self.variant = SimpleNamespace(key=key)
        self.load_report = SimpleNamespace(peak_vram_gb=0.0)
        self.model = object()
        self.processor = None
        self.prompts = prompts

    def caption(self, media: Any, prompt: Any, gen: Any, pre: Any, cb: Any) -> Any:
        del media, gen, pre, cb
        from vcap.models.base import CaptionResult, CaptionTiming, TokenUsage

        self.prompts.append(prompt)
        return CaptionResult(
            text="caption result",
            raw_text="caption result",
            segments=[(0.0, 0.4, "caption result")],
            structured={"caption": "caption result"},
            usage=TokenUsage(2, 3),
            timing=CaptionTiming(0.0, 0.01, 300.0, 0.01),
        )


@pytest.fixture
def caption_prompts(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from vcap.models import loader

    prompts: list[Any] = []
    monkeypatch.setattr(loader, "MODEL_CACHE", loader.ModelCache())
    monkeypatch.setattr(
        loader,
        "load_model",
        lambda key, **_kwargs: _Captioner(key, prompts),
    )
    return prompts


def _fake_transcription(request: dict[str, Any], **kwargs: Any) -> Any:
    from vcap.whisper.client import TranscriptionOutcome
    from vcap.whisper.engine import TranscriptResult, TranscriptSegment, TranscriptWord

    item = request["items"][0]
    result = TranscriptResult(
        segments=[
            TranscriptSegment(
                0,
                0.0,
                0.2,
                "hello ",
                [TranscriptWord(0.0, 0.2, "hello ", 0.97)],
            ),
            TranscriptSegment(
                1,
                0.2,
                0.4,
                "world",
                [TranscriptWord(0.2, 0.4, "world", 0.96)],
            ),
        ],
        language="en",
        language_probability=0.99,
        duration_s=0.4,
        elapsed_s=0.1,
        model=request["params"]["model"],
        compute_type=request["params"]["compute_type"],
        device="cpu",
    )
    out_dir = Path(item["out_dir"])
    suffix = request["output"]["file_suffix"]
    files: list[str] = []
    for fmt in request["output"]["formats"]:
        path = out_dir / f"{item['stem']}{suffix}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.text if fmt == "txt" else "fake subtitle\n", encoding="utf-8")
        files.append(str(path))
    sink = kwargs.get("sink")
    if sink is not None:
        sink.on_item_done(
            {
                "item_index": item["index"],
                "files": files,
                "result": result.to_dict(),
                "skipped": False,
            }
        )
    return TranscriptionOutcome(
        ok=True,
        items=[{"event": "item_done", "item_index": item["index"], "files": files, "skipped": False}],
        results={item["index"]: result},
        elapsed_s=0.1,
        cancelled=False,
        error=None,
    )


def test_job_spec_builds_and_round_trips_transcript_spec(tmp_path: Path) -> None:
    spec = JobSpec.from_settings(
        _settings(
            transcript_enabled=True,
            transcript_formats=["vtt", "txt"],
            transcript_inject_prompt=False,
            transcript_prompt_wrapper="Speech:\n{{TRANSCRIPT}}",
            transcript_file_suffix="_speech",
            whisper_model="small",
        ),
        [],
        OutputSpec(outputs_root=tmp_path),
    )

    assert spec.transcript == TranscriptSpec(
        enabled=True,
        formats=("vtt", "txt"),
        inject_prompt=False,
        prompt_wrapper="Speech:\n{{TRANSCRIPT}}",
        file_suffix="_speech",
        whisper=spec.transcript.whisper,
    )
    assert spec.transcript.whisper["model"] == "small"
    assert JobSpec.from_json(spec.to_json()).transcript == spec.transcript


def test_prompt_transcript_is_filled_last_per_clip() -> None:
    from vcap.models.base import PromptSpec
    from vcap.whisper.engine import TranscriptResult, TranscriptSegment

    result = TranscriptResult(
        [
            TranscriptSegment(0, 0.0, 1.0, "first", []),
            TranscriptSegment(1, 1.0, 2.0, "second", []),
        ],
        "en",
        1.0,
        2.0,
        0.1,
        "tiny",
        "int8",
        "cpu",
    )
    prompt = PromptSpec(user_prompt="Use exactly: {{TRANSCRIPT}}")
    rendered, text = _prompt_with_transcript(prompt, result, 1.0, 2.0, TranscriptSpec(enabled=True))
    assert text == "second"
    assert rendered.user_prompt == "Use exactly: second"
    assert "{{TRANSCRIPT}}" not in rendered.user_prompt


def test_runner_writes_transcripts_and_injects_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caption_prompts: list[Any],
) -> None:
    source = tmp_path / "speech.wav"
    _write_wav(source)
    monkeypatch.setattr("vcap.whisper.client.run_transcription", _fake_transcription)
    spec = JobSpec.from_settings(
        _settings(
            transcript_enabled=True,
            transcript_formats=["srt", "txt"],
            transcript_inject_prompt=True,
        ),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )

    result = run_job(spec, None, CancelToken())

    assert result.counts["done"] == 1
    item = result.items[0]
    assert Path(item.outputs["transcript_srt"]).is_file()
    assert Path(item.outputs["transcript_txt"]).read_text(encoding="utf-8") == "hello world"
    assert item.transcript and item.transcript["language"] == "en"
    assert caption_prompts and "hello world" in str(caption_prompts[0].user_prompt)
    structured = json.loads(Path(item.outputs["json"]).read_text(encoding="utf-8"))
    assert structured["transcript"]["text"] == "hello world"


def test_runner_transcript_failure_continues_captioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caption_prompts: list[Any],
) -> None:
    source = tmp_path / "failure.wav"
    _write_wav(source)
    monkeypatch.setattr(
        "vcap.whisper.client.run_transcription",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic Whisper failure")),
    )
    spec = JobSpec.from_settings(
        _settings(transcript_enabled=True),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )

    result = run_job(spec, None, CancelToken())

    assert result.counts["done"] == 1
    assert result.items[0].transcript == {"error": "RuntimeError: synthetic Whisper failure"}
    assert caption_prompts


def test_runner_disabled_does_not_call_whisper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caption_prompts: list[Any],
) -> None:
    source = tmp_path / "disabled.wav"
    _write_wav(source)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "vcap.whisper.client.run_transcription",
        lambda request, **_kwargs: calls.append(request),
    )
    spec = JobSpec.from_settings(
        _settings(transcript_enabled=False),
        [InputItem(source)],
        OutputSpec(outputs_root=tmp_path / "outputs"),
    )

    result = run_job(spec, None, CancelToken())

    assert result.counts["done"] == 1
    assert calls == []
    assert result.items[0].transcript is None
