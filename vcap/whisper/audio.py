"""Audio decoding across released faster-whisper and current PyAV versions."""

from __future__ import annotations

import os


def decode_audio(source: str | os.PathLike[str], sampling_rate: int = 16_000):
    from faster_whisper import decode_audio as faster_decode

    try:
        return faster_decode(str(source), sampling_rate=sampling_rate)
    except TypeError as exc:
        # faster-whisper 1.2.1 passes this removed PyAV 19 argument. Keep
        # newer upstream decoders in use, and propagate unrelated failures.
        if "unexpected keyword argument 'metadata_errors'" not in str(exc):
            raise
    from vcap.core.media import read_audio

    return read_audio(source, sample_rate=sampling_rate)
