"""Typed, JSON-serializable job and result contracts for the caption pipeline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from vcap import OUTPUTS_DIR
from vcap.core.dataset_captions import (
    DEFAULT_AUDIO_CAPTION_TEMPLATE,
    DEFAULT_CAPTION_MERGE_TEMPLATE,
)


DEFAULT_SUMMARY_PROMPT = (
    'You are given timestamped captions of consecutive segments of one video. Write (1) one '
    'paragraph summarizing the whole video in {{LANGUAGE}}, then (2) a chapter list with one line '
    'per chapter formatted as "MM:SS-MM:SS Title - one sentence". Use only information present '
    "in the captions and keep the chronological order."
)
DEFAULT_CONTEXT_CARRY_PROMPT = "Context from the previous segment (do not repeat it): {{CONTEXT}}"

_ENCODE_CODECS = frozenset({"libx264", "h264_nvenc", "libx265", "hevc_nvenc"})
_ENCODE_PRESETS = frozenset(
    {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"}
)
_AUDIO_BITRATES = frozenset({"96k", "128k", "192k", "256k", "320k"})
_GGUF_FLASH_ATTN = frozenset({"auto", "on", "off"})
_MEDIA_KINDS = ("video", "audio", "image", "text")
_TRANSCRIPT_FORMATS = frozenset({"srt", "vtt", "txt", "lrc", "tsv", "json"})
_AUDIO_CAPTION_SOURCES = frozenset({"none", "whisper", "captioner", "both"})
_VIDEO_CAPTION_SOURCES = frozenset({"generate", "existing"})
_AUDIO_TRANSCRIPT_STYLES = frozenset({"plain", "lines", "timestamped"})
_AUDIO_EMPTY_POLICIES = frozenset({"skip", "placeholder"})
DEFAULT_TRANSCRIPT_PROMPT_WRAPPER = (
    "Exact speech transcript for this clip (use it verbatim for dialogue, do not invent speech):\n"
    "{{TRANSCRIPT}}"
)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, os.PathLike)):
        return os.fspath(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_json_payload(payload: str | bytes | os.PathLike[str]) -> dict[str, Any]:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8")
    else:
        text = os.fspath(payload)
        stripped = text.lstrip()
        if not stripped.startswith(("{", "[")):
            candidate = Path(text)
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Serialized job data must be a JSON object")
    return value


def _setting(settings: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in settings:
            return settings[key]
    return default


def _optional_system_prompt(value: Any) -> str | None:
    """Normalize empty and JSON-like null sentinels to no system message."""

    if value is None:
        return None
    text = str(value)
    normalized = text.strip().casefold()
    return None if not normalized or normalized in {"none", "null"} else text


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", ""}:
        return False
    return default


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _gpu_layers(value: Any) -> int | str:
    if value is None:
        return "auto"
    normalized = str(value).strip().casefold()
    if normalized in {"auto", "all"}:
        return normalized
    return max(0, _int(value, 0))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _context_tokens(value: Any) -> int | None:
    """Return a positive requested context window; None means the model cap."""

    if value is None or str(value).strip() == "":
        return None
    parsed = _int(value, 0)
    return parsed if parsed > 0 else None


def _gpu_indices(value: Any, fallback: int) -> tuple[int, ...]:
    if value is None or value == "":
        return (int(fallback),)
    if isinstance(value, str):
        raw: Sequence[Any] = [part for part in re.split(r"[\s,;]+", value.strip()) if part]
    elif isinstance(value, Sequence):
        raw = value
    else:
        raw = [value]
    result: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index >= 0 and index not in result:
            result.append(index)
    return tuple(result or [int(fallback)])


def _formats(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ("txt",)
    normalized = [str(item).strip().casefold().lstrip(".") for item in values]
    normalized = [item for item in normalized if item]
    if "txt" not in normalized:
        normalized.insert(0, "txt")
    return tuple(dict.fromkeys(normalized))


def _media_kinds(value: Any, default: Sequence[str] = _MEDIA_KINDS) -> tuple[str, ...]:
    if value is None:
        values: Sequence[Any] = default
    elif isinstance(value, str):
        values = [part for part in re.split(r"[\s,;]+", value) if part]
    elif isinstance(value, Sequence):
        values = value
    else:
        values = default
    normalized = [str(item).strip().casefold() for item in values]
    return tuple(dict.fromkeys(item for item in normalized if item in _MEDIA_KINDS))


def _choice(value: Any, choices: frozenset[str], default: str) -> str:
    selected = str(value or "").strip().casefold()
    return selected if selected in choices else default


def _replace_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        result: list[tuple[str, str]] = []
        for line in value.splitlines():
            for piece in line.split("|"):
                if ";" not in piece:
                    continue
                source, target = piece.split(";", 1)
                if source.strip():
                    result.append((source.strip(), target.strip()))
        return tuple(result)
    result = []
    for item in value:
        if isinstance(item, Sequence) and len(item) >= 2 and str(item[0]).strip():
            result.append((str(item[0]).strip(), str(item[1]).strip()))
    return tuple(result)


def _transcript_formats(value: Any) -> tuple[str, ...]:
    if value is None:
        value = ("srt", "txt")
    if isinstance(value, str):
        values = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        values = list(value)
    result: list[str] = []
    for item in values:
        selected = str(item).casefold().lstrip(".")
        if selected == "webvtt":
            selected = "vtt"
        if selected in _TRANSCRIPT_FORMATS and selected not in result:
            result.append(selected)
    return tuple(result)


def _default_whisper_params() -> dict[str, Any]:
    """Return W1's JSON-safe defaults without importing Whisper at module load."""

    from vcap.whisper.params import WhisperParams

    params = WhisperParams()
    try:
        return dict(params.to_dict())
    except NotImplementedError:
        # Keeps W2's typed job tests usable while W1 is landing concurrently.
        return _json_safe(asdict(params))


@dataclass(frozen=True)
class InputItem:
    """One path, text file, or direct text prompt supplied to a job."""

    path: str | os.PathLike[str] = ""
    kind: str = "auto"
    text_prompt_only: bool = False
    text: str | None = None
    trim_start_s: float | None = None
    trim_end_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", os.fspath(self.path))
        object.__setattr__(self, "kind", str(self.kind or "auto").casefold())
        start = _optional_float(self.trim_start_s)
        end = _optional_float(self.trim_end_s)
        if start is not None:
            start = max(0.0, start)
        if end is not None:
            end = max(0.0, end)
        if start is not None and end is not None and end <= start:
            raise ValueError("InputItem trim_end_s must be greater than trim_start_s")
        object.__setattr__(self, "trim_start_s", start)
        object.__setattr__(self, "trim_end_s", end)


@dataclass(frozen=True)
class OffloadSpec:
    """Serializable counterpart of :class:`vcap.models.offload.OffloadPlan`."""

    gpu_layers: int | str = "auto"
    offload_experts: bool = False
    max_memory: dict[str, str] | None = None
    pin_cpu: bool = True
    vram_reserve_gb: float = 2.0
    swap_slots: int = 2
    pinned_ram_budget_gb: float = 0.0
    plan_slack_mib: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_layers", _gpu_layers(self.gpu_layers))
        object.__setattr__(self, "offload_experts", _bool(self.offload_experts, False))
        object.__setattr__(self, "pin_cpu", _bool(self.pin_cpu, True))
        object.__setattr__(self, "vram_reserve_gb", max(0.0, _float(self.vram_reserve_gb, 2.0)))
        object.__setattr__(self, "swap_slots", max(1, min(4, _int(self.swap_slots, 2))))
        object.__setattr__(
            self,
            "pinned_ram_budget_gb",
            max(0.0, min(1024.0, _float(self.pinned_ram_budget_gb, 0.0))),
        )
        object.__setattr__(
            self,
            "plan_slack_mib",
            max(0, min(8192, _int(self.plan_slack_mib, 512))),
        )


@dataclass(frozen=True)
class ModelChoice:
    """Selected checkpoint and model-loading policy."""

    variant_key: str = "qwen3_omni_instruct_int4"
    attention: str = "auto"
    vram_preset: str = "auto"
    offload: OffloadSpec = field(default_factory=OffloadSpec)


@dataclass(frozen=True)
class PromptSpec:
    """Prompt preset plus direct prompt overrides and template variables."""

    preset_id: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_prompt", _optional_system_prompt(self.system_prompt))


@dataclass(frozen=True)
class GenParams:
    """Generation controls shared by every captioner wrapper."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 2048
    do_sample: bool = False
    use_cache: bool = True
    enable_thinking: bool = False
    seed: int = -1
    no_repeat_ngram_size: int = 0
    # Requested context window; None defers to the model's cap.
    context_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", max(-1, min(2_147_483_647, _int(self.seed, -1))))
        object.__setattr__(
            self,
            "no_repeat_ngram_size",
            max(0, min(20, _int(self.no_repeat_ngram_size, 0))),
        )


@dataclass(frozen=True)
class PreprocessSpec:
    """Trim, frame sampling, pixel, and normalization controls."""

    trim_start_s: float = 0.0
    trim_end_s: float | None = None
    fps: float = 2.0
    max_frames: int = 160
    max_pixels: int = 297_920
    min_pixels: int | None = None
    sampling_strategy: str = "fps"
    normalize_clip: bool = False
    use_audio_in_video: bool = True
    audio_sample_rate: int = 16_000
    total_pixel_cap: int = 0
    adaptive_threshold: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_frames", max(0, _int(self.max_frames, 160)))
        object.__setattr__(self, "max_pixels", max(1, _int(self.max_pixels, 297_920)))
        object.__setattr__(self, "audio_sample_rate", max(1, _int(self.audio_sample_rate, 16_000)))
        object.__setattr__(
            self,
            "total_pixel_cap",
            max(0, min(400_000_000, _int(self.total_pixel_cap, 0))),
        )
        object.__setattr__(
            self,
            "adaptive_threshold",
            max(0.1, min(50.0, _float(self.adaptive_threshold, 2.0))),
        )


@dataclass(frozen=True)
class SplitSpec:
    """Scene/fixed/trainer segmentation and optional clip rejection policy."""

    mode: Literal["whole", "scenes", "fixed", "trainer"] = "whole"
    cut_mode: Literal["copy", "precise"] = "copy"
    scene_threshold: float = 27.0
    scene_min_len_s: float = 2.0
    scene_max_len_s: float = 60.0
    merge_short_scenes: bool = True
    merge_below_s: float = 2.0
    fade_detection: bool = False
    fade_threshold: float = 12.0
    scene_detector: Literal["content", "adaptive", "threshold"] = "content"
    scene_downscale: int = 0
    fixed_chunk_s: float = 30.0
    model_max_duration_s: float | None = None
    trainer_target: Any = None
    overlap_s: float = 0.5
    encode_codec: str = "libx264"
    encode_crf: int = 18
    encode_preset: str = "veryfast"
    encode_audio_bitrate: str = "192k"
    quality_frames: int = 8
    auto_reject: bool = False
    reject_min_duration_s: float = 0.0
    reject_max_black_ratio: float = 0.98
    reject_max_static_score: float = -1.0
    reject_min_sharpness: float = 0.0
    reject_require_audio: bool = False
    reject_max_silence_ratio: float = 1.0
    reject_black_luma: int = 16
    reject_silence_rms: float = 0.001

    def __post_init__(self) -> None:
        object.__setattr__(self, "fade_threshold", max(1.0, min(100.0, _float(self.fade_threshold, 12.0))))
        object.__setattr__(self, "encode_codec", _choice(self.encode_codec, _ENCODE_CODECS, "libx264"))
        object.__setattr__(self, "encode_crf", max(0, min(51, _int(self.encode_crf, 18))))
        object.__setattr__(self, "encode_preset", _choice(self.encode_preset, _ENCODE_PRESETS, "veryfast"))
        object.__setattr__(
            self,
            "encode_audio_bitrate",
            _choice(self.encode_audio_bitrate, _AUDIO_BITRATES, "192k"),
        )
        object.__setattr__(self, "quality_frames", max(4, min(32, _int(self.quality_frames, 8))))
        object.__setattr__(
            self,
            "reject_black_luma",
            max(0, min(255, _int(self.reject_black_luma, 16))),
        )
        object.__setattr__(
            self,
            "reject_silence_rms",
            max(0.0, min(0.1, _float(self.reject_silence_rms, 0.001))),
        )


@dataclass(frozen=True)
class PostSpec:
    """Caption cleanup, injection, replacement, and output format controls."""

    prefix: str = ""
    suffix: str = ""
    trigger: str = ""
    trigger_mode: Literal["prefix", "suffix", "none"] = "none"
    replace_pairs: tuple[tuple[str, str], ...] = ()
    replace_regex: bool = False
    replace_case_insensitive: bool = True
    replace_whole_words: bool = True
    collapse_whitespace: bool = False
    formats: tuple[str, ...] = ("txt",)
    save_reasoning: bool = True
    max_caption_chars: int = 0
    dedupe_repeated_sentences: bool = True
    subtitle_min_cue_s: float = 0.5
    subtitle_max_line_chars: int = 0
    join_separator: str = " "

    def __post_init__(self) -> None:
        object.__setattr__(self, "formats", _formats(self.formats))
        object.__setattr__(self, "replace_pairs", _replace_pairs(self.replace_pairs))
        object.__setattr__(
            self,
            "max_caption_chars",
            max(0, min(100_000, _int(self.max_caption_chars, 0))),
        )
        object.__setattr__(
            self,
            "dedupe_repeated_sentences",
            _bool(self.dedupe_repeated_sentences, True),
        )
        object.__setattr__(
            self,
            "subtitle_min_cue_s",
            max(0.0, min(5.0, _float(self.subtitle_min_cue_s, 0.5))),
        )
        object.__setattr__(
            self,
            "subtitle_max_line_chars",
            max(0, min(200, _int(self.subtitle_max_line_chars, 0))),
        )
        separator = " " if self.join_separator is None else str(self.join_separator)
        object.__setattr__(self, "join_separator", separator[:16])


@dataclass(frozen=True)
class OutputSpec:
    """Single-run or stable batch-caption output layout."""

    kind: Literal["single", "batch"] = "single"
    outputs_root: str | os.PathLike[str] = field(default_factory=lambda: str(OUTPUTS_DIR))
    batch_output_dir: str | os.PathLike[str] | None = None
    source_root: str | os.PathLike[str] | None = None
    mirror_names: bool = True
    overwrite: bool = False
    save_processed_files: bool = False
    save_clips: bool = False
    recursive: bool = False
    limit_items: int = 0
    include_kinds: tuple[str, ...] = _MEDIA_KINDS
    name_filter: str = ""
    save_next_to_source: bool = False

    def __post_init__(self) -> None:
        selected = str(self.kind).casefold()
        if selected not in {"single", "batch"}:
            raise ValueError("OutputSpec.kind must be 'single' or 'batch'")
        object.__setattr__(self, "kind", selected)
        object.__setattr__(self, "outputs_root", os.fspath(self.outputs_root))
        if self.batch_output_dir is not None:
            object.__setattr__(self, "batch_output_dir", os.fspath(self.batch_output_dir))
        if self.source_root is not None:
            object.__setattr__(self, "source_root", os.fspath(self.source_root))
        object.__setattr__(self, "limit_items", max(0, int(self.limit_items)))
        object.__setattr__(self, "include_kinds", _media_kinds(self.include_kinds))
        object.__setattr__(self, "name_filter", str(self.name_filter or "").strip())
        object.__setattr__(
            self,
            "save_next_to_source",
            _bool(self.save_next_to_source, False),
        )


@dataclass(frozen=True)
class RuntimeSpec:
    """Execution mode, model lifetime, GPU selection, and compile policy."""

    subprocess_mode: bool = True
    keep_model_loaded: bool = True
    idle_unload_minutes: float = 10.0
    gpu_index: int = 0
    gpu_indices: tuple[int, ...] = ()
    compile: bool = False
    oom_retries: int = 2
    gguf_max_frames: int = 32
    gguf_jpeg_quality: int = 90
    gguf_threads: int = 0
    gguf_batch_size: int = 2048
    gguf_ubatch_size: int = 512
    gguf_flash_attn: str = "auto"
    gguf_cache_reuse: int = 0
    gguf_ignore_tier_context: bool = False
    gguf_extra_args: str = ""
    oom_degrade_factor: float = 0.75
    gguf_min_p: float = 0.05
    gguf_repeat_last_n: int = 64
    gguf_presence_penalty: float = 0.0
    gguf_frequency_penalty: float = 0.0
    gguf_fit_headroom_mib: int = 1536
    gguf_startup_timeout_s: int = 900
    gguf_stream_idle_timeout_s: int = 120

    def __post_init__(self) -> None:
        primary = max(0, int(self.gpu_index))
        indices = _gpu_indices(self.gpu_indices, primary)
        object.__setattr__(self, "gpu_index", primary)
        object.__setattr__(self, "gpu_indices", indices)
        object.__setattr__(self, "oom_retries", max(0, min(4, _int(self.oom_retries, 2))))
        object.__setattr__(self, "gguf_max_frames", max(1, min(128, _int(self.gguf_max_frames, 32))))
        object.__setattr__(
            self,
            "gguf_jpeg_quality",
            max(50, min(100, _int(self.gguf_jpeg_quality, 90))),
        )
        object.__setattr__(self, "gguf_threads", max(0, min(256, _int(self.gguf_threads, 0))))
        object.__setattr__(
            self,
            "gguf_batch_size",
            max(64, min(8192, _int(self.gguf_batch_size, 2048))),
        )
        object.__setattr__(
            self,
            "gguf_ubatch_size",
            max(32, min(4096, _int(self.gguf_ubatch_size, 512))),
        )
        object.__setattr__(
            self,
            "gguf_flash_attn",
            _choice(self.gguf_flash_attn, _GGUF_FLASH_ATTN, "auto"),
        )
        object.__setattr__(
            self,
            "gguf_cache_reuse",
            max(0, min(4096, _int(self.gguf_cache_reuse, 0))),
        )
        object.__setattr__(self, "gguf_extra_args", str(self.gguf_extra_args or ""))
        object.__setattr__(
            self,
            "oom_degrade_factor",
            max(0.5, min(0.95, _float(self.oom_degrade_factor, 0.75))),
        )
        object.__setattr__(
            self,
            "gguf_min_p",
            max(0.0, min(1.0, _float(self.gguf_min_p, 0.05))),
        )
        object.__setattr__(
            self,
            "gguf_repeat_last_n",
            max(0, min(4096, _int(self.gguf_repeat_last_n, 64))),
        )
        object.__setattr__(
            self,
            "gguf_presence_penalty",
            max(-2.0, min(2.0, _float(self.gguf_presence_penalty, 0.0))),
        )
        object.__setattr__(
            self,
            "gguf_frequency_penalty",
            max(-2.0, min(2.0, _float(self.gguf_frequency_penalty, 0.0))),
        )
        object.__setattr__(
            self,
            "gguf_fit_headroom_mib",
            max(0, min(8192, _int(self.gguf_fit_headroom_mib, 1536))),
        )
        object.__setattr__(
            self,
            "gguf_startup_timeout_s",
            max(60, min(3600, _int(self.gguf_startup_timeout_s, 900))),
        )
        object.__setattr__(
            self,
            "gguf_stream_idle_timeout_s",
            max(0, min(3600, _int(self.gguf_stream_idle_timeout_s, 120))),
        )


@dataclass(frozen=True)
class TranscriptSpec:
    """Optional Whisper stage run before an item's caption clips."""

    enabled: bool = False
    formats: tuple[str, ...] = ("srt", "txt")
    inject_prompt: bool = True
    prompt_wrapper: str = DEFAULT_TRANSCRIPT_PROMPT_WRAPPER
    file_suffix: str = "_transcript"
    whisper: dict[str, Any] = field(default_factory=_default_whisper_params)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _bool(self.enabled, False))
        object.__setattr__(self, "formats", _transcript_formats(self.formats))
        object.__setattr__(self, "inject_prompt", _bool(self.inject_prompt, True))
        object.__setattr__(
            self,
            "prompt_wrapper",
            str(self.prompt_wrapper or DEFAULT_TRANSCRIPT_PROMPT_WRAPPER),
        )
        object.__setattr__(self, "file_suffix", str(self.file_suffix or ""))
        object.__setattr__(self, "whisper", _json_safe(dict(self.whisper or {})))


@dataclass(frozen=True)
class AudioCaptionSpec:
    """Audio-caption source, rendering, and split-layout policy."""

    source: Literal["none", "whisper", "captioner", "both"] = "none"
    video_source: Literal["generate", "existing"] = "generate"
    model_key: str = "auto"
    transcript_style: Literal["plain", "lines", "timestamped"] = "plain"
    template: str = DEFAULT_AUDIO_CAPTION_TEMPLATE
    write_merged: bool = True
    merge_template: str = DEFAULT_CAPTION_MERGE_TEMPLATE
    empty_policy: Literal["skip", "placeholder"] = "skip"
    empty_text: str = "No speech."

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _choice(self.source, _AUDIO_CAPTION_SOURCES, "none"))
        object.__setattr__(
            self,
            "video_source",
            _choice(self.video_source, _VIDEO_CAPTION_SOURCES, "generate"),
        )
        object.__setattr__(self, "model_key", str(self.model_key or "auto"))
        object.__setattr__(
            self,
            "transcript_style",
            _choice(self.transcript_style, _AUDIO_TRANSCRIPT_STYLES, "plain"),
        )
        object.__setattr__(
            self,
            "template",
            DEFAULT_AUDIO_CAPTION_TEMPLATE if self.template is None else str(self.template),
        )
        object.__setattr__(self, "write_merged", _bool(self.write_merged, True))
        object.__setattr__(
            self,
            "merge_template",
            DEFAULT_CAPTION_MERGE_TEMPLATE
            if self.merge_template is None
            else str(self.merge_template),
        )
        object.__setattr__(
            self,
            "empty_policy",
            _choice(self.empty_policy, _AUDIO_EMPTY_POLICIES, "skip"),
        )
        object.__setattr__(
            self,
            "empty_text",
            "No speech." if self.empty_text is None else str(self.empty_text),
        )

    @property
    def enabled(self) -> bool:
        return self.source != "none"

    @property
    def needs_whisper(self) -> bool:
        return self.source in {"whisper", "both"}

    @property
    def needs_captioner(self) -> bool:
        return self.source in {"captioner", "both"}


@dataclass(frozen=True)
class JobSpec:
    """Complete immutable input to the unified single/batch pipeline."""

    inputs: list[InputItem] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    model: ModelChoice = field(default_factory=ModelChoice)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    generation: GenParams = field(default_factory=GenParams)
    preprocess: PreprocessSpec = field(default_factory=PreprocessSpec)
    split: SplitSpec = field(default_factory=SplitSpec)
    post: PostSpec = field(default_factory=PostSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    transcript: TranscriptSpec = field(default_factory=TranscriptSpec)
    audio_caption: AudioCaptionSpec = field(default_factory=AudioCaptionSpec)
    context_carry_over: bool = False
    context_carry_words: int = 60
    context_carry_prompt: str = DEFAULT_CONTEXT_CARRY_PROMPT
    summarize_segments: bool = False
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    summary_max_new_tokens: int = 1024
    settings: dict[str, Any] = field(default_factory=dict)
    internal: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_carry_words",
            max(10, min(400, _int(self.context_carry_words, 60))),
        )
        object.__setattr__(self, "summary_prompt", str(self.summary_prompt or DEFAULT_SUMMARY_PROMPT))
        object.__setattr__(
            self,
            "context_carry_prompt",
            str(self.context_carry_prompt or DEFAULT_CONTEXT_CARRY_PROMPT),
        )
        object.__setattr__(
            self,
            "summary_max_new_tokens",
            max(64, min(8192, _int(self.summary_max_new_tokens, 1024))),
        )
        if "system_prompt" in self.settings:
            settings = dict(self.settings)
            raw_system = settings["system_prompt"]
            if isinstance(raw_system, str) and raw_system.strip().casefold() in {
                "none",
                "null",
            }:
                settings["system_prompt"] = ""
                object.__setattr__(self, "settings", settings)

    @property
    def gen(self) -> GenParams:
        """Short compatibility alias matching the model interface."""

        return self.generation

    @property
    def pre(self) -> PreprocessSpec:
        """Short compatibility alias matching the model interface."""

        return self.preprocess

    @classmethod
    def from_settings(
        cls,
        settings: dict[str, Any],
        inputs: list[InputItem],
        output: OutputSpec,
    ) -> "JobSpec":
        """Build typed specs from the UI's flat settings dictionary."""

        source = dict(settings or {})
        variant_key = str(
            _setting(source, "variant_key", "model_key", "model_variant", default=ModelChoice().variant_key)
        )
        family_defaults: dict[str, Any] = {}
        default_preset: str | None = None
        supports_audio_only = False
        try:
            from vcap.models.registry import MODEL_SPECS, variant_to_family

            model_spec = MODEL_SPECS[variant_to_family(variant_key)]
            supports_audio_only = "audio" in model_spec.capabilities
            family_defaults = {item.name: item.default for item in model_spec.param_schema}
            family_defaults.update(
                fps=model_spec.limits.default_fps,
                max_frames=model_spec.limits.max_frames or 768,
                max_pixels=model_spec.limits.default_max_pixels,
                min_pixels=model_spec.limits.min_pixels,
            )
            default_preset = model_spec.default_prompt_preset
        except (KeyError, ValueError):
            pass

        preset_id_raw = _setting(source, "prompt_preset_id", "preset_id", default=default_preset)
        preset_id = str(preset_id_raw) if preset_id_raw not in {None, ""} else None
        preset_generation: dict[str, Any] = {}
        if preset_id:
            try:
                from vcap.prompts.presets import get_preset

                preset_generation = dict(get_preset(preset_id).generation_overrides)
            except KeyError:
                pass

        raw_offload = _setting(source, "offload", "offload_plan", default={})
        if is_dataclass(raw_offload):
            offload_data = asdict(raw_offload)
        elif isinstance(raw_offload, Mapping):
            offload_data = dict(raw_offload)
        else:
            offload_data = {}
        raw_gpu_layers = _setting(source, "gpu_layers", "layers_on_gpu", default=None)
        if raw_gpu_layers is None and ("block_swap_auto" in source or "blocks_to_swap" in source):
            # The UI stores the block-swap controls; translate them with the
            # selected family's layer count (the loader caps the count again).
            layer_count = 48
            try:
                from vcap.models.offload import block_swap_to_gpu_layers, family_layer_count
                from vcap.models.registry import variant_to_family

                layer_count = family_layer_count(variant_to_family(variant_key))
            except (KeyError, ValueError):
                from vcap.models.offload import block_swap_to_gpu_layers
            raw_gpu_layers = block_swap_to_gpu_layers(
                _bool(source.get("block_swap_auto"), True),
                _int(source.get("blocks_to_swap"), 0),
                layer_count,
            )
        gpu_layers = _gpu_layers(
            offload_data.get("gpu_layers", "auto") if raw_gpu_layers is None else raw_gpu_layers
        )
        raw_max_memory = _setting(source, "offload_max_memory", "max_memory", default=offload_data.get("max_memory"))
        max_memory = (
            {str(key): str(value) for key, value in raw_max_memory.items()}
            if isinstance(raw_max_memory, Mapping)
            else None
        )
        model = ModelChoice(
            variant_key=variant_key,
            attention=str(_setting(source, "attention", "attention_backend", default="auto")),
            vram_preset=str(_setting(source, "vram_preset", default="auto")),
            offload=OffloadSpec(
                gpu_layers=gpu_layers,
                offload_experts=_bool(
                    _setting(source, "offload_experts", default=offload_data.get("offload_experts")),
                    False,
                ),
                max_memory=max_memory,
                pin_cpu=_bool(
                    _setting(source, "pin_cpu", default=offload_data.get("pin_cpu", True)),
                    True,
                ),
                vram_reserve_gb=max(
                    0.0,
                    _float(
                        _setting(
                            source,
                            "vram_reserve_gb",
                            default=offload_data.get("vram_reserve_gb", 2.0),
                        ),
                        2.0,
                    ),
                ),
                swap_slots=max(
                    1,
                    min(
                        4,
                        _int(
                            _setting(
                                source,
                                "swap_slots",
                                default=offload_data.get("swap_slots", 2),
                            ),
                            2,
                        ),
                    ),
                ),
                pinned_ram_budget_gb=max(
                    0.0,
                    min(
                        1024.0,
                        _float(
                            _setting(
                                source,
                                "pinned_ram_budget_gb",
                                default=offload_data.get("pinned_ram_budget_gb", 0.0),
                            ),
                            0.0,
                        ),
                    ),
                ),
                plan_slack_mib=max(
                    0,
                    min(
                        8192,
                        _int(
                            _setting(
                                source,
                                "plan_slack_mib",
                                default=offload_data.get("plan_slack_mib", 512),
                            ),
                            512,
                        ),
                    ),
                ),
            ),
        )

        variable_values = dict(_setting(source, "prompt_variables", "variables", default={}) or {})
        aliases = {
            "TRIGGER": ("trigger_word", "trigger"),
            "LANGUAGE": ("language",),
            "SOURCE_LANGUAGE": ("source_language",),
            "TARGET_LANGUAGE": ("target_language",),
            "CAPTION_LENGTH": ("caption_length",),
            "AVOID": ("avoid_list", "avoid"),
            "SUBJECT_CLASS": ("subject_class",),
            "EXTRA_INSTRUCTIONS": ("extra_instructions",),
        }
        for variable, keys in aliases.items():
            value = _setting(source, *keys, default=None)
            if value is not None:
                variable_values[variable] = value
        # Caption UI rendering happens before Whisper runs. Keeping this token
        # literal lets the runner substitute the per-clip transcript last.
        variable_values.setdefault("TRANSCRIPT", "{{TRANSCRIPT}}")
        prompt = PromptSpec(
            preset_id=preset_id,
            system_prompt=_optional_system_prompt(
                _setting(source, "system_prompt", default=None)
            ),
            user_prompt=_setting(source, "user_prompt", default=None),
            variables=variable_values,
        )

        def generation_value(key: str, fallback: Any) -> Any:
            return _setting(source, key, default=preset_generation.get(key, family_defaults.get(key, fallback)))

        def family_generation_default(key: str, fallback: Any) -> Any:
            return family_defaults.get(key, fallback)

        generation = GenParams(
            temperature=_float(
                generation_value("temperature", 0.0),
                _float(family_generation_default("temperature", 0.0), 0.0),
            ),
            top_p=_float(
                generation_value("top_p", 1.0),
                _float(family_generation_default("top_p", 1.0), 1.0),
            ),
            top_k=_int(
                generation_value("top_k", 0),
                _int(family_generation_default("top_k", 0), 0),
            ),
            repetition_penalty=_float(
                generation_value("repetition_penalty", 1.0),
                _float(family_generation_default("repetition_penalty", 1.0), 1.0),
            ),
            max_new_tokens=max(
                1,
                _int(
                    generation_value("max_new_tokens", 2048),
                    _int(family_generation_default("max_new_tokens", 2048), 2048),
                ),
            ),
            do_sample=_bool(
                generation_value("do_sample", False),
                _bool(family_generation_default("do_sample", False), False),
            ),
            use_cache=_bool(
                generation_value("use_cache", True),
                _bool(family_generation_default("use_cache", True), True),
            ),
            enable_thinking=_bool(
                generation_value("enable_thinking", False),
                _bool(family_generation_default("enable_thinking", False), False),
            ),
            seed=max(
                -1,
                min(2_147_483_647, _int(_setting(source, "seed", default=-1), -1)),
            ),
            no_repeat_ngram_size=max(
                0,
                min(
                    20,
                    _int(_setting(source, "no_repeat_ngram_size", default=0), 0),
                ),
            ),
            context_tokens=_context_tokens(_setting(source, "context_tokens", default=None)),
        )
        # ``family_defaults["max_frames"]`` is the family cap; the UI keeps a
        # global bound so switching models never rejects a stale value, and the
        # cap is enforced here instead.
        default_frames = max(1, _int(family_defaults.get("max_frames", 160), 160))
        selected_frames = _int(_setting(source, "max_frames", default=default_frames), default_frames)
        selected_frames = max(0, selected_frames)
        if selected_frames == 0 and not supports_audio_only:
            selected_frames = 4
        selected_frames = min(selected_frames, default_frames)
        default_fps = max(0.01, _float(family_defaults.get("fps", 2.0), 2.0))
        default_pixels = max(1, _int(family_defaults.get("max_pixels", 297_920), 297_920))
        preprocess = PreprocessSpec(
            trim_start_s=max(0.0, _float(_setting(source, "trim_start_s", "trim_start", default=0.0), 0.0)),
            trim_end_s=_optional_float(_setting(source, "trim_end_s", "trim_end", default=None)),
            fps=max(0.01, _float(_setting(source, "fps", default=default_fps), default_fps)),
            max_frames=selected_frames,
            max_pixels=max(1, _int(_setting(source, "max_pixels", default=default_pixels), default_pixels)),
            min_pixels=(
                max(
                    1,
                    _int(
                        _setting(source, "min_pixels", default=family_defaults.get("min_pixels")),
                        max(1, _int(family_defaults.get("min_pixels", 1), 1)),
                    ),
                )
                if _setting(source, "min_pixels", default=family_defaults.get("min_pixels")) is not None
                else None
            ),
            sampling_strategy=str(_setting(source, "sampling_strategy", "frame_sampling", default="fps")),
            normalize_clip=_bool(_setting(source, "normalize_clip", default=False), False),
            use_audio_in_video=_bool(
                _setting(source, "use_audio_in_video", default=family_defaults.get("use_audio_in_video", True)),
                True,
            ),
            audio_sample_rate=max(1, _int(_setting(source, "audio_sample_rate", default=16_000), 16_000)),
            total_pixel_cap=max(
                0,
                min(
                    400_000_000,
                    _int(_setting(source, "total_pixel_cap", default=0), 0),
                ),
            ),
            adaptive_threshold=max(
                0.1,
                min(
                    50.0,
                    _float(_setting(source, "adaptive_threshold", default=2.0), 2.0),
                ),
            ),
        )

        raw_segment_mode = str(_setting(source, "segment_mode", "segmentation_mode", default="")).casefold()
        raw_split_mode = str(_setting(source, "split_mode", default="copy")).casefold()
        if raw_segment_mode not in {"whole", "scenes", "fixed", "trainer"}:
            if raw_split_mode in {"whole", "scenes", "fixed", "trainer"}:
                raw_segment_mode = raw_split_mode
            else:
                raw_segment_mode = "scenes" if _bool(_setting(source, "scene_detect_enabled", default=False)) else "whole"
        cut_mode = raw_split_mode if raw_split_mode in {"copy", "precise"} else str(
            _setting(source, "cut_mode", "split_cut_mode", default="copy")
        ).casefold()
        if cut_mode not in {"copy", "precise"}:
            cut_mode = "copy"
        scene_data = _setting(source, "scene_params", default={})
        scene_data = dict(scene_data) if isinstance(scene_data, Mapping) else {}
        split = SplitSpec(
            mode=raw_segment_mode,  # type: ignore[arg-type]
            cut_mode=cut_mode,  # type: ignore[arg-type]
            scene_threshold=_float(_setting(source, "scene_threshold", default=scene_data.get("threshold", 27.0)), 27.0),
            scene_min_len_s=max(0.0, _float(_setting(source, "scene_min_len_s", default=scene_data.get("min_scene_len_s", 2.0)), 2.0)),
            scene_max_len_s=max(0.0, _float(_setting(source, "scene_max_len_s", default=scene_data.get("max_scene_len_s", 60.0)), 60.0)),
            merge_short_scenes=_bool(_setting(source, "merge_short_scenes", default=scene_data.get("merge_short_scenes", True)), True),
            merge_below_s=max(0.0, _float(_setting(source, "merge_below_s", default=scene_data.get("merge_below_s", 2.0)), 2.0)),
            fade_detection=_bool(_setting(source, "fade_detection", default=scene_data.get("fade_detection", False))),
            fade_threshold=max(
                1.0,
                min(
                    100.0,
                    _float(
                        _setting(source, "fade_threshold", default=scene_data.get("fade_threshold", 12.0)),
                        12.0,
                    ),
                ),
            ),
            scene_detector=str(_setting(source, "scene_detector", default=scene_data.get("detector", "content"))),  # type: ignore[arg-type]
            scene_downscale=max(0, _int(_setting(source, "scene_downscale", default=scene_data.get("downscale", 0)), 0)),
            fixed_chunk_s=max(0.0, _float(_setting(source, "fixed_chunk_s", "chunk_s", default=30.0), 30.0)),
            model_max_duration_s=_optional_float(
                _setting(source, "model_max_duration_s", "max_clip_duration_s", default=None)
            ),
            trainer_target=_setting(source, "trainer_target", default=None),
            overlap_s=max(0.0, _float(_setting(source, "sub_split_overlap_s", "overlap_s", default=0.5), 0.5)),
            encode_codec=_choice(
                _setting(source, "encode_codec", default="libx264"),
                _ENCODE_CODECS,
                "libx264",
            ),
            encode_crf=max(0, min(51, _int(_setting(source, "encode_crf", default=18), 18))),
            encode_preset=_choice(
                _setting(source, "encode_preset", default="veryfast"),
                _ENCODE_PRESETS,
                "veryfast",
            ),
            encode_audio_bitrate=_choice(
                _setting(source, "encode_audio_bitrate", default="192k"),
                _AUDIO_BITRATES,
                "192k",
            ),
            quality_frames=max(
                4,
                min(32, _int(_setting(source, "quality_frames", default=8), 8)),
            ),
            auto_reject=_bool(_setting(source, "auto_reject", "auto_reject_enabled", default=False)),
            reject_min_duration_s=max(0.0, _float(_setting(source, "reject_min_duration_s", default=0.0), 0.0)),
            reject_max_black_ratio=_float(_setting(source, "reject_max_black_ratio", default=0.98), 0.98),
            reject_max_static_score=_float(_setting(source, "reject_max_static_score", default=-1.0), -1.0),
            reject_min_sharpness=max(0.0, _float(_setting(source, "reject_min_sharpness", default=0.0), 0.0)),
            reject_require_audio=_bool(_setting(source, "reject_require_audio", default=False)),
            reject_max_silence_ratio=_float(_setting(source, "reject_max_silence_ratio", default=1.0), 1.0),
            reject_black_luma=max(
                0,
                min(255, _int(_setting(source, "reject_black_luma", default=16), 16)),
            ),
            reject_silence_rms=max(
                0.0,
                min(
                    0.1,
                    _float(_setting(source, "reject_silence_rms", default=0.001), 0.001),
                ),
            ),
        )
        post = PostSpec(
            prefix=str(_setting(source, "caption_prefix", "prefix", default="") or ""),
            suffix=str(_setting(source, "caption_suffix", "suffix", default="") or ""),
            trigger=str(_setting(source, "trigger_word", "trigger", default="") or ""),
            trigger_mode=str(_setting(source, "trigger_mode", default="none")),  # type: ignore[arg-type]
            replace_pairs=_replace_pairs(_setting(source, "replace_pairs", "replace_words", default=())),
            replace_regex=_bool(_setting(source, "replace_regex", "regex_replace", default=False)),
            replace_case_insensitive=_bool(_setting(source, "replace_case_insensitive", default=True), True),
            replace_whole_words=_bool(_setting(source, "replace_whole_words", "replace_whole_words_only", default=True), True),
            collapse_whitespace=_bool(_setting(source, "collapse_whitespace", default=False)),
            formats=_formats(_setting(source, "output_formats", "formats", default=("txt",))),
            save_reasoning=_bool(_setting(source, "save_reasoning", default=True), True),
            max_caption_chars=max(
                0,
                min(
                    100_000,
                    _int(_setting(source, "max_caption_chars", default=0), 0),
                ),
            ),
            dedupe_repeated_sentences=_bool(
                _setting(source, "dedupe_repeated_sentences", default=True),
                True,
            ),
            subtitle_min_cue_s=max(
                0.0,
                min(
                    5.0,
                    _float(_setting(source, "subtitle_min_cue_s", default=0.5), 0.5),
                ),
            ),
            subtitle_max_line_chars=max(
                0,
                min(
                    200,
                    _int(_setting(source, "subtitle_max_line_chars", default=0), 0),
                ),
            ),
            join_separator=(
                " "
                if _setting(source, "caption_join_separator", "join_separator", default=" ") is None
                else str(_setting(source, "caption_join_separator", "join_separator", default=" "))
            )[:16],
        )
        resolved_output = replace(
            output,
            overwrite=_bool(_setting(source, "overwrite_existing", "overwrite", default=output.overwrite), output.overwrite),
            save_processed_files=_bool(
                _setting(source, "save_processed_files", default=output.save_processed_files),
                output.save_processed_files,
            ),
            save_clips=_bool(_setting(source, "save_clips", default=output.save_clips), output.save_clips),
            mirror_names=_bool(_setting(source, "mirror_names", default=output.mirror_names), output.mirror_names),
            recursive=_bool(_setting(source, "recursive", "batch_recursive", default=output.recursive), output.recursive),
            include_kinds=_media_kinds(
                _setting(
                    source,
                    "batch_include_kinds",
                    default=output.include_kinds,
                ),
                output.include_kinds,
            ),
            name_filter=str(
                _setting(source, "batch_name_filter", default=output.name_filter) or ""
            ).strip(),
            save_next_to_source=_bool(
                _setting(
                    source,
                    "batch_save_next_to_source",
                    default=output.save_next_to_source,
                ),
                output.save_next_to_source,
            ),
        )
        gpu_index = max(0, _int(_setting(source, "gpu_index", default=0), 0))
        runtime = RuntimeSpec(
            subprocess_mode=_bool(_setting(source, "subprocess_mode", default=True), True),
            keep_model_loaded=_bool(_setting(source, "keep_model_loaded", default=True), True),
            idle_unload_minutes=max(0.0, _float(_setting(source, "idle_unload_minutes", default=10.0), 10.0)),
            gpu_index=gpu_index,
            gpu_indices=_gpu_indices(_setting(source, "gpu_indices", "multi_gpu_indices", default=None), gpu_index),
            compile=_bool(_setting(source, "compile", "torch_compile", default=False)),
            oom_retries=max(0, min(4, _int(_setting(source, "oom_retries", default=2), 2))),
            gguf_max_frames=max(
                1,
                min(128, _int(_setting(source, "gguf_max_frames", default=32), 32)),
            ),
            gguf_jpeg_quality=max(
                50,
                min(100, _int(_setting(source, "gguf_jpeg_quality", default=90), 90)),
            ),
            gguf_threads=max(
                0,
                min(256, _int(_setting(source, "gguf_threads", default=0), 0)),
            ),
            gguf_batch_size=max(
                64,
                min(8192, _int(_setting(source, "gguf_batch_size", default=2048), 2048)),
            ),
            gguf_ubatch_size=max(
                32,
                min(4096, _int(_setting(source, "gguf_ubatch_size", default=512), 512)),
            ),
            gguf_flash_attn=_choice(
                _setting(source, "gguf_flash_attn", default="auto"),
                _GGUF_FLASH_ATTN,
                "auto",
            ),
            gguf_cache_reuse=max(
                0,
                min(4096, _int(_setting(source, "gguf_cache_reuse", default=0), 0)),
            ),
            gguf_ignore_tier_context=_bool(
                _setting(source, "gguf_ignore_tier_context", default=False),
                False,
            ),
            gguf_extra_args=str(_setting(source, "gguf_extra_args", default="") or ""),
            oom_degrade_factor=max(
                0.5,
                min(
                    0.95,
                    _float(_setting(source, "oom_degrade_factor", default=0.75), 0.75),
                ),
            ),
            gguf_min_p=max(
                0.0,
                min(1.0, _float(_setting(source, "gguf_min_p", default=0.05), 0.05)),
            ),
            gguf_repeat_last_n=max(
                0,
                min(
                    4096,
                    _int(_setting(source, "gguf_repeat_last_n", default=64), 64),
                ),
            ),
            gguf_presence_penalty=max(
                -2.0,
                min(
                    2.0,
                    _float(_setting(source, "gguf_presence_penalty", default=0.0), 0.0),
                ),
            ),
            gguf_frequency_penalty=max(
                -2.0,
                min(
                    2.0,
                    _float(_setting(source, "gguf_frequency_penalty", default=0.0), 0.0),
                ),
            ),
            gguf_fit_headroom_mib=max(
                0,
                min(
                    8192,
                    _int(_setting(source, "gguf_fit_headroom_mib", default=1536), 1536),
                ),
            ),
            gguf_startup_timeout_s=max(
                60,
                min(
                    3600,
                    _int(_setting(source, "gguf_startup_timeout_s", default=900), 900),
                ),
            ),
            gguf_stream_idle_timeout_s=max(
                0,
                min(
                    3600,
                    _int(_setting(source, "gguf_stream_idle_timeout_s", default=120), 120),
                ),
            ),
        )
        from vcap.whisper.params import WhisperParams

        try:
            whisper_settings = WhisperParams.from_settings(source).to_dict()
        except NotImplementedError:
            # W1 and W2 are developed in parallel; remove this compatibility
            # path naturally once W1's normalizer is available.
            whisper_settings = _default_whisper_params()
        transcript = TranscriptSpec(
            enabled=_bool(_setting(source, "transcript_enabled", default=False), False),
            formats=_transcript_formats(
                _setting(source, "transcript_formats", default=("srt", "txt"))
            ),
            inject_prompt=_bool(
                _setting(source, "transcript_inject_prompt", default=True),
                True,
            ),
            prompt_wrapper=str(
                _setting(
                    source,
                    "transcript_prompt_wrapper",
                    default=DEFAULT_TRANSCRIPT_PROMPT_WRAPPER,
                )
                or DEFAULT_TRANSCRIPT_PROMPT_WRAPPER
            ),
            file_suffix=str(
                _setting(source, "transcript_file_suffix", default="_transcript")
                or ""
            ),
            whisper=whisper_settings,
        )
        audio_caption = AudioCaptionSpec(
            source=str(_setting(source, "audio_caption_source", default="none")),  # type: ignore[arg-type]
            video_source=str(_setting(source, "video_caption_source", default="generate")),  # type: ignore[arg-type]
            model_key=str(_setting(source, "audio_caption_model_key", default="auto") or "auto"),
            transcript_style=str(
                _setting(source, "audio_caption_transcript_style", default="plain")
            ),  # type: ignore[arg-type]
            template=(
                DEFAULT_AUDIO_CAPTION_TEMPLATE
                if _setting(source, "audio_caption_template", default=None) is None
                else str(_setting(source, "audio_caption_template"))
            ),
            write_merged=_bool(
                _setting(source, "caption_write_merged", default=True),
                True,
            ),
            merge_template=(
                DEFAULT_CAPTION_MERGE_TEMPLATE
                if _setting(source, "caption_merge_template", default=None) is None
                else str(_setting(source, "caption_merge_template"))
            ),
            empty_policy=str(
                _setting(source, "audio_caption_empty_policy", default="skip")
            ),  # type: ignore[arg-type]
            empty_text=(
                "No speech."
                if _setting(source, "audio_caption_empty_text", default=None) is None
                else str(_setting(source, "audio_caption_empty_text"))
            ),
        )
        return cls(
            inputs=[item if isinstance(item, InputItem) else InputItem(**item) for item in inputs],
            output=resolved_output,
            model=model,
            prompt=prompt,
            generation=generation,
            preprocess=preprocess,
            split=split,
            post=post,
            runtime=runtime,
            transcript=transcript,
            audio_caption=audio_caption,
            context_carry_over=_bool(_setting(source, "context_carry_over", default=False), False),
            context_carry_words=max(
                10,
                min(400, _int(_setting(source, "context_carry_words", default=60), 60)),
            ),
            context_carry_prompt=str(
                _setting(source, "context_carry_prompt", default=DEFAULT_CONTEXT_CARRY_PROMPT)
                or DEFAULT_CONTEXT_CARRY_PROMPT
            ),
            summarize_segments=_bool(
                _setting(source, "summarize_segments", default=False),
                False,
            ),
            summary_prompt=str(
                _setting(source, "summary_prompt", default=DEFAULT_SUMMARY_PROMPT)
                or DEFAULT_SUMMARY_PROMPT
            ),
            summary_max_new_tokens=max(
                64,
                min(
                    8192,
                    _int(_setting(source, "summary_max_new_tokens", default=1024), 1024),
                ),
            ),
            settings=_json_safe(source),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete worker-protocol mapping."""

        return {"_schema_version": 1, **_json_safe(asdict(self))}

    def to_json(self, path: str | os.PathLike[str] | None = None, *, indent: int = 2) -> str:
        """Serialize to UTF-8 JSON and optionally write it to ``path``."""

        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobSpec":
        """Rehydrate a job sent through the JSON-lines worker protocol."""

        data = dict(value)
        data.pop("_schema_version", None)
        model_data = dict(data.get("model") or {})
        model_data["offload"] = OffloadSpec(**dict(model_data.get("offload") or {}))
        post_data = dict(data.get("post") or {})
        output_data = dict(data.get("output") or {})
        runtime_data = dict(data.get("runtime") or {})
        transcript_data = dict(data.get("transcript") or {})
        audio_caption_data = dict(data.get("audio_caption") or {})
        return cls(
            inputs=[item if isinstance(item, InputItem) else InputItem(**dict(item)) for item in data.get("inputs", [])],
            output=OutputSpec(**output_data),
            model=ModelChoice(**model_data),
            prompt=PromptSpec(**dict(data.get("prompt") or {})),
            generation=GenParams(**dict(data.get("generation") or data.get("gen") or {})),
            preprocess=PreprocessSpec(**dict(data.get("preprocess") or data.get("pre") or {})),
            split=SplitSpec(**dict(data.get("split") or {})),
            post=PostSpec(**post_data),
            runtime=RuntimeSpec(**runtime_data),
            transcript=TranscriptSpec(**transcript_data),
            audio_caption=AudioCaptionSpec(**audio_caption_data),
            context_carry_over=_bool(data.get("context_carry_over"), False),
            context_carry_words=max(
                10,
                min(400, _int(data.get("context_carry_words"), 60)),
            ),
            context_carry_prompt=str(
                data.get("context_carry_prompt") or DEFAULT_CONTEXT_CARRY_PROMPT
            ),
            summarize_segments=_bool(data.get("summarize_segments"), False),
            summary_prompt=str(data.get("summary_prompt") or DEFAULT_SUMMARY_PROMPT),
            summary_max_new_tokens=max(
                64,
                min(8192, _int(data.get("summary_max_new_tokens"), 1024)),
            ),
            settings=dict(data.get("settings") or {}),
            internal=dict(data.get("internal") or {}),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | os.PathLike[str]) -> "JobSpec":
        """Read a job from a JSON string, bytes, or UTF-8 file path."""

        return cls.from_dict(_load_json_payload(payload))


@dataclass
class ItemResult:
    """Terminal status and artifacts for one resolved input item."""

    index: int
    path: str
    kind: str
    status: str
    message: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0
    peak_vram_gb: float = 0.0
    traceback_tail: str = ""
    gpu_index: int | None = None
    summary: str = ""
    summary_usage: dict[str, Any] = field(default_factory=dict)
    summary_timing: dict[str, Any] = field(default_factory=dict)
    transcript: dict[str, Any] | None = None
    video_caption_path: str | None = None
    audio_caption_path: str | None = None
    merged_caption_path: str | None = None
    audio_caption_source: str = "none"
    sound_caption_model: str | None = None
    audio_windows: int = 0

    def __getitem__(self, key: str) -> Any:
        """Allow lightweight mapping-style access in UI table adapters."""

        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if (
            self.audio_caption_source == "none"
            and self.video_caption_path is None
            and self.audio_caption_path is None
            and self.merged_caption_path is None
            and self.sound_caption_model is None
            and self.audio_windows == 0
        ):
            for key in (
                "video_caption_path",
                "audio_caption_path",
                "merged_caption_path",
                "audio_caption_source",
                "sound_caption_model",
                "audio_windows",
            ):
                data.pop(key, None)
        return _json_safe(data)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ItemResult":
        return cls(**dict(value))


@dataclass
class JobResult:
    """Merged terminal result returned by in-process and worker execution."""

    items: list[ItemResult]
    counts: dict[str, int]
    run_dir: str
    metadata_path: str
    elapsed: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobResult":
        data = dict(value)
        data["items"] = [
            item if isinstance(item, ItemResult) else ItemResult.from_dict(item)
            for item in data.get("items", [])
        ]
        data["counts"] = {str(key): int(count) for key, count in dict(data.get("counts") or {}).items()}
        return cls(**data)

    @classmethod
    def from_json(cls, payload: str | bytes | os.PathLike[str]) -> "JobResult":
        return cls.from_dict(_load_json_payload(payload))


__all__ = [
    "AudioCaptionSpec",
    "DEFAULT_CONTEXT_CARRY_PROMPT",
    "DEFAULT_SUMMARY_PROMPT",
    "DEFAULT_TRANSCRIPT_PROMPT_WRAPPER",
    "GenParams",
    "InputItem",
    "ItemResult",
    "JobResult",
    "JobSpec",
    "ModelChoice",
    "OffloadSpec",
    "OutputSpec",
    "PostSpec",
    "PreprocessSpec",
    "PromptSpec",
    "RuntimeSpec",
    "SplitSpec",
    "TranscriptSpec",
]
