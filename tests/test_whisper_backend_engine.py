from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import vcap.whisper.engine as engine_module
from vcap.core.subprocess_runner import CancelledError
from vcap.whisper.engine import WhisperEngine
from vcap.whisper.params import WhisperParams


class _FakeModel:
    def __init__(self, segments=None) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.segments = segments or []
        self.kwargs = None

    def transcribe(self, **kwargs):
        self.kwargs = kwargs
        info = SimpleNamespace(duration=2.0, language="en", language_probability=0.9)
        return iter(self.segments), info


def _raw_segment(index: int, start: float, end: float, text: str):
    return SimpleNamespace(
        id=index,
        start=start,
        end=end,
        text=text,
        words=[SimpleNamespace(start=start, end=end, word=" " + text, probability=0.8)],
        avg_logprob=-0.2,
        no_speech_prob=0.1,
        compression_ratio=1.0,
        temperature=0.0,
    )


def _fake_faster_whisper(monkeypatch: pytest.MonkeyPatch, audio: np.ndarray) -> ModuleType:
    module = ModuleType("faster_whisper")
    module.decode_audio = lambda *_args, **_kwargs: audio
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


def test_standard_transcribe_uses_exact_reference_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _fake_faster_whisper(monkeypatch, np.zeros(32_000, dtype=np.float32))
    model = _FakeModel([_raw_segment(0, 0.0, 2.0, "hello")])
    params = WhisperParams(
        model="tiny",
        device="cpu",
        language="english",
        condition_on_previous_text=False,
        max_new_tokens=12,
    )
    engine = WhisperEngine(params)
    engine.model = model
    engine.device = "cpu"
    engine.compute_type = "int8"
    monkeypatch.setattr(engine, "load", lambda: None)
    media = tmp_path / "audio.wav"
    media.write_bytes(b"fake")

    result = engine.transcribe(media)

    assert result.text == "hello"
    assert isinstance(model.kwargs["temperature"], float)
    assert set(model.kwargs) == {
        "audio",
        "language",
        "task",
        "beam_size",
        "log_prob_threshold",
        "no_speech_threshold",
        "best_of",
        "patience",
        "temperature",
        "initial_prompt",
        "compression_ratio_threshold",
        "length_penalty",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "prefix",
        "suppress_blank",
        "suppress_tokens",
        "max_initial_timestamp",
        "word_timestamps",
        "prepend_punctuations",
        "append_punctuations",
        "max_new_tokens",
        "chunk_length",
        "hallucination_silence_threshold",
        "hotwords",
        "language_detection_threshold",
        "language_detection_segments",
        "condition_on_previous_text",
        "prompt_reset_on_temperature",
    }
    assert model.kwargs["language"] == "en"
    assert model.kwargs["max_new_tokens"] == 12


def test_batched_path_passes_windows_and_disables_native_vad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _fake_faster_whisper(monkeypatch, np.zeros(64_000, dtype=np.float32))
    captured: dict = {}

    class FakeBatch:
        def __init__(self, model):
            assert model is fake_model

        def transcribe(self, **kwargs):
            captured.update(kwargs)
            return iter([_raw_segment(0, 0.0, 4.0, "batch")]), SimpleNamespace(
                duration=4.0, language="en", language_probability=1.0
            )

    module.BatchedInferencePipeline = FakeBatch
    fake_model = _FakeModel()
    params = WhisperParams(
        model="tiny",
        device="cpu",
        use_batched_inference=True,
        batch_size=4,
        chunk_length=2,
    )
    engine = WhisperEngine(params)
    engine.model = fake_model
    engine.device = "cpu"
    engine.compute_type = "int8"
    monkeypatch.setattr(engine, "load", lambda: None)
    media = tmp_path / "audio.wav"
    media.write_bytes(b"fake")

    engine.transcribe(media)

    assert captured["clip_timestamps"] == [
        {"start": 0, "end": 32_000},
        {"start": 32_000, "end": 64_000},
    ]
    assert captured["batch_size"] == 4
    assert captured["vad_filter"] is False
    assert captured["without_timestamps"] is False


def test_repeat_initial_prompt_wraps_every_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_faster_whisper(monkeypatch, np.zeros(32_000, dtype=np.float32))
    captured: list[int] = []

    class Tokenizer:
        def encode(self, text):
            assert text == " glossary"
            return [10, 11]

    class PromptModel(_FakeModel):
        def get_prompt(
            self,
            _tokenizer,
            previous_tokens,
            without_timestamps=False,
            prefix=None,
            hotwords=None,
        ):
            del without_timestamps, prefix, hotwords
            captured.extend(previous_tokens)
            return list(previous_tokens)

        def transcribe(self, **kwargs):
            assert kwargs["initial_prompt"] is None
            self.get_prompt(Tokenizer(), [99])
            return super().transcribe(**kwargs)

    model = PromptModel([_raw_segment(0, 0.0, 2.0, "hello")])
    engine = WhisperEngine(
        WhisperParams(
            model="tiny",
            device="cpu",
            initial_prompt="glossary",
            repeat_initial_prompt_every_window=True,
        )
    )
    engine.model = model
    engine.device = "cpu"
    engine.compute_type = "int8"
    monkeypatch.setattr(engine, "load", lambda: None)
    media = tmp_path / "audio.wav"
    media.write_bytes(b"fake")

    engine.transcribe(media)

    assert captured == [10, 11, 99]


def test_long_form_disables_previous_text_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_faster_whisper(monkeypatch, np.zeros(60 * 16_000, dtype=np.float32))
    model = _FakeModel([_raw_segment(0, 0.0, 1.0, "long")])
    logs: list[tuple[str, str]] = []
    engine = WhisperEngine(
        WhisperParams(model="tiny", device="cpu", chunk_length=1),
        log=lambda message, level: logs.append((message, level)),
    )
    engine.model = model
    engine.device = "cpu"
    engine.compute_type = "int8"
    monkeypatch.setattr(engine, "load", lambda: None)
    media = tmp_path / "long.wav"
    media.write_bytes(b"fake")

    engine.transcribe(media)

    assert model.kwargs["condition_on_previous_text"] is False
    assert any("Auto-disabling condition_on_previous_text" in message for message, _ in logs)


def test_cancel_is_checked_between_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_faster_whisper(monkeypatch, np.zeros(32_000, dtype=np.float32))
    model = _FakeModel(
        [_raw_segment(0, 0.0, 1.0, "one"), _raw_segment(1, 1.0, 2.0, "two")]
    )
    cancelled = {"value": False}
    engine = WhisperEngine(
        WhisperParams(model="tiny", device="cpu"),
        cancel_check=lambda: cancelled["value"],
    )
    engine.model = model
    engine.device = "cpu"
    engine.compute_type = "int8"
    monkeypatch.setattr(engine, "load", lambda: None)
    media = tmp_path / "audio.wav"
    media.write_bytes(b"fake")

    with pytest.raises(CancelledError):
        engine.transcribe(media, on_segment=lambda _segment: cancelled.update(value=True))


def test_auto_cuda_load_failure_falls_back_to_cpu_int8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "whisper" / "tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.bin").write_bytes(b"model")
    calls: list[dict] = []

    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 1
    faster_whisper = ModuleType("faster_whisper")

    def fake_constructor(**kwargs):
        calls.append(kwargs)
        if kwargs["device"] == "cuda":
            raise RuntimeError("Could not load cublas64_12.dll")
        return _FakeModel()

    faster_whisper.WhisperModel = fake_constructor
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setattr(engine_module, "enable_cuda_runtime_autodiscovery", lambda: [])
    logs: list[tuple[str, str]] = []
    engine = WhisperEngine(
        WhisperParams(model="tiny", device="auto", compute_type="float16"),
        models_dir=tmp_path,
        log=lambda message, level: logs.append((message, level)),
    )

    engine.load()

    assert [call["device"] for call in calls] == ["cuda", "cpu"]
    assert calls[-1]["compute_type"] == "int8"
    assert engine.device == "cpu"
    assert any("cuBLAS" in message for message, _ in logs)
