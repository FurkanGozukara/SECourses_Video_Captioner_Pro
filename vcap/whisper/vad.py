"""CPU Silero VAD filtering and timestamp restoration."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, TypeVar

from vcap.core.subprocess_runner import CancelledError

from .params import WhisperVadParams

T = TypeVar("T")


def _cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise CancelledError("Whisper VAD cancelled")


def _vad_options(parameters: WhisperVadParams | Mapping[str, Any] | Any):
    from faster_whisper.vad import VadOptions

    if isinstance(parameters, VadOptions):
        return parameters
    if isinstance(parameters, WhisperVadParams):
        max_speech = (
            float("inf")
            if parameters.max_speech_duration_s >= 9999
            else parameters.max_speech_duration_s
        )
        return VadOptions(
            threshold=parameters.threshold,
            min_speech_duration_ms=parameters.min_speech_duration_ms,
            max_speech_duration_s=max_speech,
            min_silence_duration_ms=parameters.min_silence_duration_ms,
            speech_pad_ms=parameters.speech_pad_ms,
        )
    if isinstance(parameters, Mapping):
        values = dict(parameters)
        values.pop("enabled", None)
        if float(values.get("max_speech_duration_s", 0) or 0) >= 9999:
            values["max_speech_duration_s"] = float("inf")
        return VadOptions(**values)
    return VadOptions()


class SileroVAD:
    """Reference Silero VAD algorithm backed by ONNX Runtime on CPU."""

    def __init__(
        self,
        *,
        progress: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.sampling_rate = 16_000
        self.window_size_samples = 512
        self.model: Any | None = None
        self.progress = progress
        self.cancel_check = cancel_check

    def run(
        self,
        audio: Any,
        vad_parameters: WhisperVadParams | Mapping[str, Any] | Any,
    ) -> tuple[Any, list[dict[str, int]]]:
        """Remove silence and return the concatenated audio plus source chunks."""

        import numpy as np

        _cancelled(self.cancel_check)
        if not isinstance(audio, np.ndarray):
            from faster_whisper import decode_audio

            audio = decode_audio(audio, sampling_rate=self.sampling_rate)
        normalized = self.normalize_audio(audio)
        chunks = self.get_speech_timestamps(normalized, _vad_options(vad_parameters))
        return self.collect_chunks(normalized, chunks), chunks

    def get_speech_timestamps(
        self,
        audio: Any,
        vad_options: Any | None = None,
        **kwargs: Any,
    ) -> list[dict[str, int]]:
        """Split mono 16 kHz audio into padded speech sample ranges."""

        import numpy as np

        from faster_whisper.vad import VadOptions

        _cancelled(self.cancel_check)
        if self.model is None:
            self.update_model()
        audio = self.normalize_audio(audio)
        if vad_options is None:
            vad_options = VadOptions(**kwargs)

        threshold = vad_options.threshold
        neg_threshold = vad_options.neg_threshold
        min_speech_duration_ms = vad_options.min_speech_duration_ms
        max_speech_duration_s = vad_options.max_speech_duration_s
        min_silence_duration_ms = vad_options.min_silence_duration_ms
        window_size_samples = self.window_size_samples
        speech_pad_ms = vad_options.speech_pad_ms
        min_speech_samples = self.sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = self.sampling_rate * speech_pad_ms / 1000
        max_speech_samples = (
            self.sampling_rate * max_speech_duration_s
            - window_size_samples
            - 2 * speech_pad_samples
        )
        min_silence_samples = self.sampling_rate * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = self.sampling_rate * 98 / 1000
        audio_length_samples = len(audio)

        padding = window_size_samples - audio.shape[0] % window_size_samples
        padded_audio = np.pad(audio, (0, padding))
        speech_probs = self._get_speech_probs(padded_audio)
        if self.progress is not None:
            self.progress(
                {
                    "stage": "vad",
                    "fraction": 0.5,
                    "message": "Silero VAD analyzed audio",
                }
            )

        triggered = False
        speeches: list[dict[str, int]] = []
        current_speech: dict[str, int] = {}
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)
        temp_end = 0
        prev_end = 0
        next_start = 0

        for index, speech_prob in enumerate(speech_probs):
            if index % 1024 == 0:
                _cancelled(self.cancel_check)
            current_sample = window_size_samples * index
            if speech_prob >= threshold and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = current_sample
            if speech_prob >= threshold and not triggered:
                triggered = True
                current_speech["start"] = current_sample
                continue
            if (
                triggered
                and current_sample - current_speech["start"] > max_speech_samples
            ):
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech["end"] = current_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue
            if speech_prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = current_sample
                if current_sample - temp_end > min_silence_samples_at_max_speech:
                    prev_end = temp_end
                if current_sample - temp_end < min_silence_samples:
                    continue
                current_speech["end"] = temp_end
                if current_speech["end"] - current_speech["start"] > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False

        if (
            current_speech
            and audio_length_samples - current_speech["start"] > min_speech_samples
        ):
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        for index, speech in enumerate(speeches):
            if index == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if index != len(speeches) - 1:
                silence_duration = speeches[index + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[index + 1]["start"] = int(
                        max(0, speeches[index + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] = int(
                        min(audio_length_samples, speech["end"] + speech_pad_samples)
                    )
                    speeches[index + 1]["start"] = int(
                        max(0, speeches[index + 1]["start"] - speech_pad_samples)
                    )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
        if self.progress is not None:
            self.progress(
                {
                    "stage": "vad",
                    "fraction": 1.0,
                    "message": f"Silero VAD found {len(speeches)} speech region(s)",
                }
            )
        return speeches

    def update_model(self) -> None:
        """Load faster-whisper's bundled ONNX model (CPU provider only)."""

        from faster_whisper.vad import get_vad_model

        self.model = get_vad_model()

    @staticmethod
    def collect_chunks(audio: Any, chunks: list[dict[str, int]]):
        import numpy as np

        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate([audio[chunk["start"] : chunk["end"]] for chunk in chunks])

    @staticmethod
    def normalize_audio(audio: Any):
        import numpy as np

        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.ndim > 1:
            channel_axis = 0 if normalized.shape[0] <= normalized.shape[-1] else -1
            normalized = normalized.mean(axis=channel_axis)
        return np.ascontiguousarray(normalized.reshape(-1), dtype=np.float32)

    def _get_speech_probs(self, padded_audio: Any):
        import numpy as np

        assert self.model is not None
        try:
            speech_probs = self.model(padded_audio)
        except AssertionError as exc:
            if "multiple of num_samples" in str(exc):
                raise
            speech_probs = self.model(padded_audio.reshape(1, -1))
        return np.asarray(speech_probs, dtype=np.float32).reshape(-1)

    def restore_speech_timestamps(
        self,
        segments: list[T],
        speech_chunks: list[dict[str, int]],
        sampling_rate: int | None = None,
    ) -> list[T]:
        return restore_speech_timestamps(
            segments,
            speech_chunks,
            sampling_rate or self.sampling_rate,
        )


def restore_speech_timestamps(
    segments: list[T],
    speech_chunks: list[dict[str, int]],
    sampling_rate: int = 16_000,
) -> list[T]:
    """Map VAD-compressed segment and word times back to source media."""

    if not speech_chunks:
        return segments
    from faster_whisper.transcribe import SpeechTimestampsMap

    timestamp_map = SpeechTimestampsMap(speech_chunks, sampling_rate)
    for segment in segments:
        words = list(getattr(segment, "words", None) or [])
        if words:
            for word in words:
                middle = (float(word.start) + float(word.end)) / 2
                chunk_index = timestamp_map.get_chunk_index(middle)
                word.start = timestamp_map.get_original_time(float(word.start), chunk_index)
                word.end = timestamp_map.get_original_time(float(word.end), chunk_index)
            segment.start = words[0].start
            segment.end = words[-1].end
            segment.words = words
        else:
            segment.start = timestamp_map.get_original_time(float(segment.start))
            segment.end = timestamp_map.get_original_time(float(segment.end), is_end=True)
    return segments


def apply_silero_vad(
    audio: Any,
    parameters: WhisperVadParams,
    *,
    progress: Callable[[dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
):
    """Convenience wrapper used by tests and alternate callers."""

    return SileroVAD(progress=progress, cancel_check=cancel_check).run(audio, parameters)


__all__ = [
    "SileroVAD",
    "apply_silero_vad",
    "restore_speech_timestamps",
]
