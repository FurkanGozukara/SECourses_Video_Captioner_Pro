from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from vcap.core.media import read_audio_with_rate, read_model_audio


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def test_48khz_offsets_are_applied_before_model_resampling(tmp_path: Path) -> None:
    rate = 48_000
    timeline = np.arange(rate * 2, dtype=np.float64) / rate
    samples = (0.8 * np.sin(2.0 * np.pi * 7.0 * timeline)).astype(np.float32)
    source = tmp_path / "source_48k.wav"
    _write_wav(source, samples, rate)

    decoded, actual_rate = read_audio_with_rate(source)
    assert actual_rate == 48_000
    assert decoded.shape == (96_000,)

    model_audio = read_model_audio(source, 0.25, 0.75)
    assert model_audio.dtype == np.float32
    assert model_audio.shape == (8_000,)
    # At 0.25 seconds the 7 Hz tone is at its negative peak. Applying the
    # offset as though the source were already 16 kHz lands at a different phase.
    assert model_audio[0] < -0.75
