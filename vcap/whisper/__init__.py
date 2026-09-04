"""Lightweight public API for the faster-whisper transcription backend."""

from __future__ import annotations

from .client import (
    TranscriptionOutcome,
    TranscriptionSink,
    build_request,
    run_transcription,
)
from .cuda_runtime import enable_cuda_runtime_autodiscovery
from .engine import (
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    WhisperEngine,
    text_between,
)
from .models import (
    MODEL_FILE_PATTERNS,
    WHISPER_MODELS,
    WhisperModelInfo,
    delete_model,
    download_model,
    format_size,
    get_model,
    is_model_ready,
    local_size_bytes,
    model_choices,
    model_dir,
    model_label,
    whisper_models_root,
)
from .params import (
    COMPUTE_TYPE_CHOICES,
    DEVICE_CHOICES,
    LANGUAGE_AUTO,
    LANGUAGE_CHOICES,
    TranscriptOutputOptions,
    WhisperParams,
    WhisperVadParams,
    code_to_language,
    language_to_code,
)
from .writers import (
    FORMAT_EXTENSIONS,
    format_timestamp,
    normalize_result_for_segment_subtitles,
    render_transcript,
    write_transcript_files,
)

__all__ = [
    "COMPUTE_TYPE_CHOICES",
    "DEVICE_CHOICES",
    "FORMAT_EXTENSIONS",
    "LANGUAGE_AUTO",
    "LANGUAGE_CHOICES",
    "MODEL_FILE_PATTERNS",
    "TranscriptionOutcome",
    "TranscriptionSink",
    "TranscriptOutputOptions",
    "TranscriptResult",
    "TranscriptSegment",
    "TranscriptWord",
    "WHISPER_MODELS",
    "WhisperEngine",
    "WhisperModelInfo",
    "WhisperParams",
    "WhisperVadParams",
    "build_request",
    "code_to_language",
    "delete_model",
    "download_model",
    "enable_cuda_runtime_autodiscovery",
    "format_size",
    "format_timestamp",
    "get_model",
    "is_model_ready",
    "language_to_code",
    "local_size_bytes",
    "model_choices",
    "model_dir",
    "model_label",
    "normalize_result_for_segment_subtitles",
    "render_transcript",
    "run_transcription",
    "text_between",
    "whisper_models_root",
    "write_transcript_files",
]
