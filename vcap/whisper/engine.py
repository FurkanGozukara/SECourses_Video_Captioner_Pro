"""Faster-whisper model lifecycle and transcription engine."""

from __future__ import annotations

import copy
import gc
import inspect
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable, Iterator, Mapping

from vcap.core.logs import get_log
from vcap.core.paths import normalize_path
from vcap.core.subprocess_runner import CancelledError

from .cuda_runtime import enable_cuda_runtime_autodiscovery
from .models import download_model, is_model_ready, model_dir
from .params import WhisperParams, language_to_code, parse_suppress_tokens

LONG_FORM_CONDITIONING_WINDOW_THRESHOLD = 60
WHISPER_TOKEN_LIMIT_FALLBACK = 448
PROMPT_TOKEN_RESERVE = 16


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranscriptWord":
        return cls(
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            word=str(data.get("word") or ""),
            probability=float(data.get("probability") or 0.0),
        )


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranscriptSegment":
        return cls(
            id=int(data.get("id") or 0),
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            text=str(data.get("text") or ""),
            words=[
                TranscriptWord.from_dict(word)
                for word in data.get("words") or []
                if isinstance(word, Mapping)
            ],
            avg_logprob=_optional_float(data.get("avg_logprob")),
            no_speech_prob=_optional_float(data.get("no_speech_prob")),
            compression_ratio=_optional_float(data.get("compression_ratio")),
            temperature=_optional_float(data.get("temperature")),
        )


@dataclass
class TranscriptResult:
    segments: list[TranscriptSegment]
    language: str | None
    language_probability: float | None
    duration_s: float
    elapsed_s: float
    model: str
    compute_type: str
    device: str

    @property
    def text(self) -> str:
        return " ".join(
            segment.text.strip() for segment in self.segments if segment.text.strip()
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        """Return the stable worker and JSON-writer representation."""

        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_s": float(self.duration_s),
            "elapsed_s": float(self.elapsed_s),
            "model": self.model,
            "compute_type": self.compute_type,
            "device": self.device,
            "segments": [asdict(segment) for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranscriptResult":
        """Restore a result from a worker event or JSON transcript."""

        source = data if isinstance(data, Mapping) else {}
        return cls(
            segments=[
                TranscriptSegment.from_dict(segment)
                for segment in source.get("segments") or []
                if isinstance(segment, Mapping)
            ],
            language=(
                str(source["language"])
                if source.get("language") is not None
                else None
            ),
            language_probability=_optional_float(source.get("language_probability")),
            duration_s=float(source.get("duration_s") or source.get("duration") or 0.0),
            elapsed_s=float(source.get("elapsed_s") or 0.0),
            model=str(source.get("model") or ""),
            compute_type=str(source.get("compute_type") or ""),
            device=str(source.get("device") or ""),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _overlaps(start: float, end: float, range_start: float, range_end: float) -> bool:
    return end > range_start and start < range_end


def text_between(result: TranscriptResult, start_s: float, end_s: float) -> str:
    """Return words or fallback segment text overlapping a time range."""

    start = max(0.0, float(start_s))
    end = max(start, float(end_s))
    chunks: list[str] = []
    for segment in result.segments:
        if segment.words:
            selected = [
                word.word
                for word in segment.words
                if _overlaps(word.start, word.end, start, end)
            ]
            if selected:
                chunks.append("".join(selected).strip())
        elif _overlaps(segment.start, segment.end, start, end):
            chunks.append(segment.text.strip())
    return " ".join(" ".join(chunks).split())


def _has_nonempty_prompt(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return True


def resolve_prompt_safe_params(
    params: WhisperParams,
    log: Callable[[str, str], None] | None = None,
) -> WhisperParams:
    """Apply faster-whisper's prompt-aware decoder-token safety rules."""

    resolved = copy.copy(params)
    try:
        requested = int(params.max_new_tokens)
    except (TypeError, ValueError, OverflowError):
        requested = 0
    if requested <= 0:
        resolved.max_new_tokens = None  # type: ignore[assignment]
        return resolved
    has_prompt_source = (
        bool(params.condition_on_previous_text)
        or _has_nonempty_prompt(params.initial_prompt)
        or _has_nonempty_prompt(params.prefix)
        or _has_nonempty_prompt(params.hotwords)
    )
    if has_prompt_source:
        if log is not None:
            log(
                "Auto-clearing max_new_tokens because prompt context is enabled. "
                "faster-whisper will use Whisper's decoder max_length dynamically so "
                "prompts, prefixes, hotwords, and previous text cannot overflow the "
                "token budget.",
                "info",
            )
        resolved.max_new_tokens = None  # type: ignore[assignment]
        return resolved
    safe_max = max(1, WHISPER_TOKEN_LIMIT_FALLBACK - PROMPT_TOKEN_RESERVE)
    if requested > safe_max:
        if log is not None:
            log(
                f"Auto-clamping max_new_tokens from {requested} to {safe_max} "
                "to fit Whisper's decoder token budget.",
                "info",
            )
        resolved.max_new_tokens = safe_max
    return resolved


def _encode_initial_prompt(initial_prompt: Any, tokenizer: Any) -> list[int]:
    if initial_prompt is None:
        return []
    if isinstance(initial_prompt, str):
        normalized = initial_prompt.strip()
        return tokenizer.encode(" " + normalized) if normalized else []
    return list(initial_prompt)


@contextmanager
def _repeat_initial_prompt_context(model: Any, initial_prompt: Any):
    if initial_prompt is None:
        yield
        return
    prompt_cache: dict[int, list[int]] = {}
    original_get_prompt = model.get_prompt

    def wrapped_get_prompt(
        model_self: Any,
        tokenizer: Any,
        previous_tokens: Iterable[int] | None,
        without_timestamps: bool = False,
        prefix: str | None = None,
        hotwords: str | None = None,
    ):
        del model_self
        cache_key = id(tokenizer)
        if cache_key not in prompt_cache:
            prompt_cache[cache_key] = _encode_initial_prompt(initial_prompt, tokenizer)
        merged_tokens = prompt_cache[cache_key] + list(previous_tokens or [])
        return original_get_prompt(
            tokenizer,
            merged_tokens,
            without_timestamps=without_timestamps,
            prefix=prefix,
            hotwords=hotwords,
        )

    model.get_prompt = MethodType(wrapped_get_prompt, model)
    try:
        yield
    finally:
        try:
            delattr(model, "get_prompt")
        except (AttributeError, TypeError):
            model.get_prompt = original_get_prompt


def _call_with_supported_kwargs(function: Callable[..., Any], kwargs: dict[str, Any]):
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def _segment_from_faster_whisper(segment: Any, fallback_id: int) -> TranscriptSegment:
    words = [
        TranscriptWord(
            start=float(getattr(word, "start", 0.0) or 0.0),
            end=float(getattr(word, "end", 0.0) or 0.0),
            word=str(getattr(word, "word", "") or ""),
            probability=float(getattr(word, "probability", 0.0) or 0.0),
        )
        for word in (getattr(segment, "words", None) or [])
    ]
    segment_id = getattr(segment, "id", fallback_id)
    return TranscriptSegment(
        id=fallback_id if segment_id is None else int(segment_id),
        start=float(getattr(segment, "start", 0.0) or 0.0),
        end=float(getattr(segment, "end", 0.0) or 0.0),
        text=str(getattr(segment, "text", "") or ""),
        words=words,
        avg_logprob=_optional_float(getattr(segment, "avg_logprob", None)),
        no_speech_prob=_optional_float(getattr(segment, "no_speech_prob", None)),
        compression_ratio=_optional_float(getattr(segment, "compression_ratio", None)),
        temperature=_optional_float(getattr(segment, "temperature", None)),
    )


def _format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class WhisperEngine:
    """Own one lazily imported CTranslate2 model for one worker request."""

    def __init__(
        self,
        params: WhisperParams,
        *,
        models_dir: Path | None = None,
        log: Callable[[str, str], None] | None = None,
        progress: Callable[[dict], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.params = params
        self.models_dir = normalize_path(models_dir) if models_dir is not None else None
        self._log_callback = log
        self.progress = progress
        self.cancel_check = cancel_check
        self.model: Any | None = None
        self.device = ""
        self.compute_type = ""
        self.model_path: Path | None = None

    def _log(self, message: str, level: str = "info") -> None:
        if self._log_callback is not None:
            self._log_callback(message, level)
            return
        logger = get_log()
        if level == "warning":
            logger.warn(message, scope="whisper")
        elif level == "error":
            logger.error(message, scope="whisper")
        else:
            logger.log(message, level=level, scope="whisper")

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress(payload)

    def _check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise CancelledError("Whisper transcription cancelled")

    def ensure_model(self) -> Path:
        """Download the selected model when its visible folder is incomplete."""

        self._check_cancel()
        target = model_dir(self.params.model, self.models_dir)
        if not is_model_ready(self.params.model, self.models_dir):
            target = download_model(
                self.params.model,
                self.models_dir,
                progress_cb=lambda payload: self._emit({"stage": "download", **payload}),
                cancel_check=self.cancel_check,
            )
        self.model_path = target
        return target

    @staticmethod
    def _cuda_cause(exc: BaseException) -> str:
        text = f"{type(exc).__name__}: {exc}".casefold()
        if "cublas" in text:
            return "cuBLAS could not be loaded"
        if "cudnn" in text:
            return "cuDNN could not be loaded"
        if (
            "out of memory" in text
            or "memory allocation" in text
            or "cuda_error_out_of_memory" in text
        ):
            return "CUDA ran out of memory"
        return "CUDA initialization failed"

    @staticmethod
    def _runtime_libraries(directories: Iterable[str]) -> tuple[bool, bool]:
        names: list[str] = []
        for directory in directories:
            try:
                names.extend(
                    path.name.casefold()
                    for path in Path(directory).iterdir()
                    if path.is_file()
                )
            except OSError:
                continue
        try:
            import ctranslate2

            package = Path(ctranslate2.__file__).resolve(strict=False).parent
            names.extend(path.name.casefold() for path in package.iterdir() if path.is_file())
        except Exception:
            pass
        return (
            any("cublas" in name for name in names),
            any("cudnn" in name for name in names),
        )

    def load(self) -> None:
        """Load CTranslate2, falling back to CPU only for automatic device mode."""

        if self.model is not None:
            return
        self._check_cancel()
        target = self.ensure_model()
        directories = enable_cuda_runtime_autodiscovery()
        import ctranslate2
        from faster_whisper import WhisperModel

        try:
            cuda_devices = max(0, int(ctranslate2.get_cuda_device_count()))
        except Exception as exc:
            cuda_devices = 0
            self._log(f"Could not query CTranslate2 CUDA devices: {exc}", "warning")
        requested_device = self.params.device
        effective_device = (
            "cuda"
            if requested_device == "auto" and cuda_devices > 0
            else "cpu"
            if requested_device == "auto"
            else requested_device
        )
        effective_compute = self.params.compute_type
        if effective_device == "cpu" and effective_compute in {
            "float16",
            "bfloat16",
            "int8_float16",
            "int8_bfloat16",
        }:
            effective_compute = "int8"

        kwargs = {
            "model_size_or_path": str(target),
            "device": effective_device,
            "device_index": 0,
            "compute_type": effective_compute,
            "cpu_threads": self.params.cpu_threads,
            "local_files_only": True,
        }
        load_started = time.perf_counter()
        try:
            self.model = WhisperModel(**kwargs)
        except Exception as exc:
            if requested_device != "auto" or effective_device != "cuda":
                cause = (
                    self._cuda_cause(exc)
                    if effective_device == "cuda"
                    else "model loading failed"
                )
                self._log(f"Whisper {cause}: {exc}", "error")
                raise
            cause = self._cuda_cause(exc)
            self._log(
                f"Whisper {cause}; retrying on CPU with int8 compute. Original error: {exc}",
                "warning",
            )
            effective_device = "cpu"
            effective_compute = "int8"
            kwargs.update(device=effective_device, compute_type=effective_compute)
            self.model = WhisperModel(**kwargs)

        self.device = effective_device
        self.compute_type = effective_compute
        load_s = time.perf_counter() - load_started
        cublas, cudnn = self._runtime_libraries(directories)
        message = (
            f"Whisper runtime: device={self.device}, compute_type={self.compute_type}, "
            f"CUDA devices={cuda_devices}"
        )
        self._emit(
            {
                "stage": "runtime",
                "device": self.device,
                "compute_type": self.compute_type,
                "cuda_devices": cuda_devices,
                "cublas": cublas,
                "cudnn": cudnn,
                "message": message,
            }
        )
        self._emit(
            {
                "stage": "model_loaded",
                "model": self.params.model,
                "load_s": load_s,
                "path": str(target),
            }
        )

    @staticmethod
    def resolve_prompt_safe_params(
        params: WhisperParams,
        log: Callable[[str, str], None] | None = None,
    ) -> WhisperParams:
        return resolve_prompt_safe_params(params, log)

    @staticmethod
    def build_clip_timestamps(
        audio: Any,
        chunk_length: int | None,
        sampling_rate: int,
    ) -> list[dict[str, int]]:
        total_samples = int(audio.shape[-1]) if getattr(audio, "size", 0) else 0
        if total_samples <= 0:
            return []
        if chunk_length is None or chunk_length <= 0:
            return [{"start": 0, "end": total_samples}]
        chunk_samples = max(1, int(chunk_length * sampling_rate))
        return [
            {"start": start, "end": min(start + chunk_samples, total_samples)}
            for start in range(0, total_samples, chunk_samples)
        ]

    def _standard_kwargs(self, audio: Any, params: WhisperParams) -> dict[str, Any]:
        repeat_prompt = bool(
            params.repeat_initial_prompt_every_window
            and _has_nonempty_prompt(params.initial_prompt)
        )
        return {
            "audio": audio,
            "language": language_to_code(params.language),
            "task": "translate" if params.translate_to_english else "transcribe",
            "beam_size": params.beam_size,
            "log_prob_threshold": params.log_prob_threshold,
            "no_speech_threshold": params.no_speech_threshold,
            "best_of": params.best_of,
            "patience": params.patience,
            "temperature": params.temperature,
            "initial_prompt": None if repeat_prompt else params.initial_prompt,
            "compression_ratio_threshold": params.compression_ratio_threshold,
            "length_penalty": params.length_penalty,
            "repetition_penalty": params.repetition_penalty,
            "no_repeat_ngram_size": params.no_repeat_ngram_size,
            "prefix": params.prefix,
            "suppress_blank": params.suppress_blank,
            "suppress_tokens": parse_suppress_tokens(
                params.suppress_tokens,
                warning=lambda message: self._log(message, "warning"),
            ),
            "max_initial_timestamp": params.max_initial_timestamp,
            "word_timestamps": params.word_timestamps,
            "prepend_punctuations": params.prepend_punctuations,
            "append_punctuations": params.append_punctuations,
            "max_new_tokens": params.max_new_tokens,
            "chunk_length": params.chunk_length,
            "hallucination_silence_threshold": (
                None
                if params.hallucination_silence_threshold <= 0
                else params.hallucination_silence_threshold
            ),
            "hotwords": params.hotwords,
            "language_detection_threshold": params.language_detection_threshold,
            "language_detection_segments": params.language_detection_segments,
            "condition_on_previous_text": params.condition_on_previous_text,
            "prompt_reset_on_temperature": params.prompt_reset_on_temperature,
        }

    def _transcribe_iter(self, audio: Any, params: WhisperParams, sampling_rate: int):
        assert self.model is not None
        repeat_prompt = bool(
            params.repeat_initial_prompt_every_window
            and _has_nonempty_prompt(params.initial_prompt)
        )
        context_prompt = params.initial_prompt if repeat_prompt else None
        if not params.use_batched_inference:
            with _repeat_initial_prompt_context(self.model, context_prompt):
                return self.model.transcribe(**self._standard_kwargs(audio, params))

        import faster_whisper

        pipeline_class = getattr(faster_whisper, "BatchedInferencePipeline", None)
        if pipeline_class is None:
            self._log(
                "Installed faster-whisper does not support BatchedInferencePipeline; "
                "falling back to standard transcription.",
                "warning",
            )
            with _repeat_initial_prompt_context(self.model, context_prompt):
                return self.model.transcribe(**self._standard_kwargs(audio, params))
        pipeline = pipeline_class(model=self.model)
        kwargs = self._standard_kwargs(audio, params)
        kwargs.update(
            without_timestamps=False,
            clip_timestamps=self.build_clip_timestamps(
                audio, params.chunk_length, sampling_rate
            ),
            batch_size=max(1, int(params.batch_size)),
            vad_filter=False,
        )
        with _repeat_initial_prompt_context(self.model, context_prompt):
            return _call_with_supported_kwargs(pipeline.transcribe, kwargs)

    def transcribe(
        self,
        media_path: str | os.PathLike[str],
        *,
        on_segment: Callable[[TranscriptSegment], None] | None = None,
    ) -> TranscriptResult:
        """Decode and transcribe one media file while streaming completed segments."""

        self.load()
        assert self.model is not None
        self._check_cancel()
        import numpy as np
        from faster_whisper import decode_audio

        source = normalize_path(media_path, must_exist=True)
        sampling_rate = int(
            getattr(getattr(self.model, "feature_extractor", None), "sampling_rate", 16_000)
            or 16_000
        )
        started = time.perf_counter()
        audio = np.ascontiguousarray(
            decode_audio(str(source), sampling_rate=sampling_rate), dtype=np.float32
        ).reshape(-1)
        original_duration = float(audio.shape[-1]) / float(sampling_rate) if sampling_rate else 0.0
        params = resolve_prompt_safe_params(self.params, self._log)
        windows = max(1, math.ceil(original_duration / max(1, params.chunk_length)))
        if params.condition_on_previous_text and windows >= LONG_FORM_CONDITIONING_WINDOW_THRESHOLD:
            params = copy.copy(params)
            params.condition_on_previous_text = False
            self._log(
                "Auto-disabling condition_on_previous_text for long-form audio "
                f"({original_duration / 60.0:.1f} minutes across ~{windows} windows) "
                "to prevent repetition drift.",
                "warning",
            )

        speech_chunks: list[dict[str, int]] = []
        if params.vad.enabled:
            from .vad import SileroVAD

            vad = SileroVAD(progress=self._emit, cancel_check=self.cancel_check)
            audio, speech_chunks = vad.run(audio, params.vad)
            if not speech_chunks:
                elapsed = time.perf_counter() - started
                return TranscriptResult(
                    [], None, None, original_duration, elapsed,
                    params.model, self.compute_type, self.device,
                )

        raw_segments, info = self._transcribe_iter(audio, params, sampling_rate)
        result_segments: list[TranscriptSegment] = []
        for index, raw_segment in enumerate(raw_segments):
            self._check_cancel()
            segment = _segment_from_faster_whisper(raw_segment, index)
            if speech_chunks:
                from .vad import restore_speech_timestamps

                restore_speech_timestamps([segment], speech_chunks, sampling_rate)
            result_segments.append(segment)
            if on_segment is not None:
                on_segment(segment)
            elapsed = time.perf_counter() - started
            fraction = (
                min(1.0, max(0.0, segment.end / original_duration))
                if original_duration > 0
                else 0.0
            )
            eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
            text_tail = " ".join(segment.text.split())[-80:]
            status = (
                f"Transcribing {_format_clock(segment.end)} / "
                f"{_format_clock(original_duration)} | elapsed {_format_clock(elapsed)}"
            )
            if eta is not None:
                status += f" | ETA {_format_clock(eta)}"
            if text_tail:
                status += f" | {text_tail}"
            self._emit(
                {
                    "stage": "transcribe",
                    "fraction": fraction,
                    "message": status,
                    "segments": len(result_segments),
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                }
            )

        elapsed_s = time.perf_counter() - started
        info_duration = _optional_float(getattr(info, "duration", None)) or 0.0
        duration = original_duration or info_duration
        return TranscriptResult(
            segments=result_segments,
            language=(
                str(getattr(info, "language"))
                if getattr(info, "language", None) is not None
                else language_to_code(params.language)
            ),
            language_probability=_optional_float(
                getattr(info, "language_probability", None)
            ),
            duration_s=duration,
            elapsed_s=elapsed_s,
            model=params.model,
            compute_type=self.compute_type,
            device=self.device,
        )

    def unload(self) -> None:
        """Release the CTranslate2 model without importing torch."""

        model = self.model
        self.model = None
        if model is not None:
            del model
        gc.collect()


__all__ = [
    "TranscriptResult",
    "TranscriptSegment",
    "TranscriptWord",
    "WhisperEngine",
    "resolve_prompt_safe_params",
    "text_between",
]
