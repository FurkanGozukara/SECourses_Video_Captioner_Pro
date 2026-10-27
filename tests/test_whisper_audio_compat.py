from __future__ import annotations

import wave

import numpy as np
import pytest

from vcap.whisper.audio import decode_audio


def test_current_installed_decoder_reads_pcm_with_expected_samples(tmp_path) -> None:
    source = tmp_path / "samples.wav"
    samples = np.tile(np.array([-16384, 0, 8192, 16384], dtype="<i2"), 4000)
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    actual = decode_audio(source)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, samples.astype(np.float32) / 32768)


def test_other_decoder_type_errors_propagate(monkeypatch) -> None:
    import faster_whisper

    def broken(*_args, **_kwargs):
        raise TypeError("invalid audio input")

    monkeypatch.setattr(faster_whisper, "decode_audio", broken)
    with pytest.raises(TypeError, match="invalid audio input"):
        decode_audio("unread.wav")


def test_compatible_upstream_decoder_is_retained(monkeypatch) -> None:
    import faster_whisper

    expected = np.zeros(12, dtype=np.float32)
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda *_args, **_kwargs: expected)
    assert decode_audio("unread.wav") is expected
