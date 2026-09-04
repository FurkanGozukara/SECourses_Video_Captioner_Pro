from __future__ import annotations

import numpy as np

from faster_whisper.vad import VadOptions

from vcap.whisper.engine import TranscriptSegment, TranscriptWord
from vcap.whisper.vad import SileroVAD, restore_speech_timestamps


def test_collect_chunks_concatenates_detected_audio() -> None:
    audio = np.arange(10, dtype=np.float32)
    collected = SileroVAD.collect_chunks(
        audio, [{"start": 1, "end": 3}, {"start": 7, "end": 10}]
    )
    assert collected.tolist() == [1.0, 2.0, 7.0, 8.0, 9.0]


def test_silero_timestamp_detection_with_fake_onnx_model() -> None:
    vad = SileroVAD()
    probabilities = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    vad.model = lambda _audio: probabilities
    audio = np.zeros(len(probabilities) * 512, dtype=np.float32)
    chunks = vad.get_speech_timestamps(
        audio,
        VadOptions(
            threshold=0.5,
            min_speech_duration_ms=0,
            min_silence_duration_ms=32,
            speech_pad_ms=0,
        ),
    )
    assert chunks == [{"start": 512, "end": 1536}]


def test_restore_speech_timestamps_restores_words_and_segment() -> None:
    segment = TranscriptSegment(
        0,
        0.0,
        1.0,
        "word",
        [TranscriptWord(0.0, 1.0, " word", 1.0)],
    )
    restored = restore_speech_timestamps(
        [segment],
        [{"start": 16_000, "end": 32_000}],
        16_000,
    )
    assert restored[0].start == 1.0
    assert restored[0].end == 2.0
    assert restored[0].words[0].start == 1.0
