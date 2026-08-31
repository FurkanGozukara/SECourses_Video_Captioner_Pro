"""Static model catalogue and local-checkpoint readiness checks.

This module deliberately has no Torch dependency.  It is imported by the UI
and downloader parent processes, where importing CUDA extensions would create
an unwanted CUDA context.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Literal

from vcap import MODELS_DIR


Scheme = Literal["bf16", "int8_convrot", "int4_convrot_w4a8", "gguf"]
ParamKind = Literal["float", "int", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    """One model-specific generation or preprocessing control."""

    name: str
    kind: ParamKind
    min: float | int | None
    max: float | int | None
    step: float | int | None
    default: float | int | bool
    description: str
    advanced: bool = False


@dataclass(frozen=True)
class ModelLimits:
    """Context, duration, media, and batching limits for a model family."""

    max_duration_s: float | None
    default_fps: float
    default_max_pixels: int
    min_pixels: int
    max_frames: int | None
    context_tokens: int
    max_new_tokens_cap: int
    audio_tokens_per_s: float
    requires_audio_track: bool
    single_audio_only: bool
    supports_batch: bool
    size_multiple: int = 28

    def compute_max_duration(
        self,
        fps: float | None = None,
        max_pixels: int | None = None,
        *,
        context: int | None = None,
        reserve_tokens: int = 4096,
        include_audio: bool = True,
    ) -> float:
        """Return a conservative duration ceiling for the selected media budget.

        Fixed-duration specialists return their documented ceiling. Qwen3
        derives the ceiling from its temporal-patch token rate, optionally adds
        its audio-token rate, and then applies the frame-count cap.
        """

        if self.max_duration_s is not None:
            return float(self.max_duration_s)
        sample_fps = max(0.01, float(fps if fps is not None else self.default_fps))
        pixels = max(self.size_multiple**2, int(max_pixels or self.default_max_pixels))
        available = max(1, int(context or self.context_tokens) - max(0, int(reserve_tokens)))
        # Qwen3 merges 2 temporal frames and a 2x2 spatial patch group:
        # tokens/s = fps/2 * pixels/(16*16*4) = fps/2 * pixels/1024.
        audio_rate = self.audio_tokens_per_s if include_audio else 0.0
        tokens_per_second = max(1.0, sample_fps * pixels / 2048.0 + audio_rate)
        duration = available / tokens_per_second
        if self.max_frames:
            duration = min(duration, self.max_frames / sample_fps)
        return max(0.0, float(duration))


@dataclass(frozen=True)
class VariantSpec:
    """One precision/backend choice for a model family."""

    key: str
    label: str
    scheme: Scheme
    size_gb: float
    folder_name: str
    gguf_files: tuple[str, ...] | None = None
    gguf_repo: str | None = None
    gguf_file_sizes: tuple[int, ...] | None = None
    gguf_sha256: tuple[str, ...] | None = None
    backend: Literal["transformers", "llamacpp"] = "transformers"


@dataclass(frozen=True)
class ModelSpec:
    """Authoritative product-facing description of one model family."""

    family: str
    label: str
    architecture: Literal["qwen2_5_omni_thinker", "qwen3_omni_moe_thinker"]
    variants: tuple[VariantSpec, ...]
    capabilities: frozenset[str]
    limits: ModelLimits
    param_schema: tuple[ParamSpec, ...]
    default_prompt_preset: str


def _param(
    name: str,
    kind: ParamKind,
    minimum: float | int | None,
    maximum: float | int | None,
    step: float | int | None,
    default: float | int | bool,
    description: str,
    advanced: bool = False,
) -> ParamSpec:
    return ParamSpec(name, kind, minimum, maximum, step, default, description, advanced)


def _generation_schema(
    *,
    max_tokens: int,
    default_max_tokens: int,
    fps: float,
    max_frames: int,
    max_pixels: int,
    sampled: bool = False,
    thinking: bool = False,
    video_audio: bool = True,
) -> tuple[ParamSpec, ...]:
    values = [
        _param("temperature", "float", 0.0, 2.0, 0.05, 0.6 if sampled else 0.0,
               "Sampling temperature; zero forces deterministic decoding."),
        _param("top_p", "float", 0.0, 1.0, 0.01, 0.95 if sampled else 1.0,
               "Nucleus-sampling probability mass.", True),
        _param("top_k", "int", 0, 200, 1, 20 if sampled else 0,
               "Maximum candidate-token count; zero leaves it unrestricted.", True),
        _param("repetition_penalty", "float", 0.5, 2.0, 0.01, 1.0,
               "Penalty applied to tokens already generated.", True),
        _param("max_new_tokens", "int", 1, max_tokens, 1, default_max_tokens,
               "Maximum number of assistant tokens to generate."),
        _param("do_sample", "bool", None, None, None, sampled,
               "Use sampling rather than greedy token selection."),
        _param("fps", "float", 0.25, 8.0, 0.25, fps,
               "Actual video sampling rate passed to the processor."),
        _param("max_frames", "int", 2, max_frames, 2, max_frames,
               "Maximum decoded frame count before model preprocessing."),
        _param("max_pixels", "int", 4 * 28 * 28, 1280 * 32 * 32, 1024, max_pixels,
               "Maximum resized pixel area for each decoded frame."),
        _param("use_audio_in_video", "bool", None, None, None, video_audio,
               "Interleave a video's audio with its visual tokens."),
    ]
    if thinking:
        values.append(
            _param("enable_thinking", "bool", None, None, None, True,
                   "Allow the Thinking model to emit a reasoning section.")
        )
    return tuple(values)


def _hf_variants(prefix: str, *, bf16: float, int8: float, int4: float) -> tuple[VariantSpec, ...]:
    return (
        VariantSpec(f"{prefix}_bf16", "BF16", "bf16", bf16, f"{prefix}_bf16"),
        VariantSpec(f"{prefix}_int8", "INT8 ConvRot", "int8_convrot", int8, f"{prefix}_int8"),
        VariantSpec(
            f"{prefix}_int4",
            "INT4 ConvRot W4A8",
            "int4_convrot_w4a8",
            int4,
            f"{prefix}_int4",
        ),
    )


def _gguf_variants(
    prefix: str,
    model_stem: str,
    repo: str,
    *,
    captioner: bool = False,
) -> tuple[VariantSpec, ...]:
    if captioner:
        q4_file = f"{model_stem}.Q4_K_M.gguf"
        q8_file = f"{model_stem}.Q8_0.gguf"
        mmproj = f"{model_stem}.mmproj-Q8_0.gguf"
        q4_size, q8_size, mmproj_size = 18_557_053_888, 32_484_494_272, 1_325_020_384
        q4_sha = "1ea2e29a75bd7f5cff9f78915c07f9e8c2a70e0bf159604de06b2d2691d100d9"
        q8_sha = "d58b165f8f57cceb2c9bfee4eb44fc9bd117beb8ab1012fea56553e46d7789cd"
        mmproj_sha = "ef6ee1e11745ca7a88c71a401f7e261078c41980d15d1e8ffb34fe25ed56c95a"
    else:
        q4_file = f"{model_stem}-Q4_K_M.gguf"
        q8_file = f"{model_stem}-Q8_0.gguf"
        mmproj = f"mmproj-{model_stem}-Q8_0.gguf"
        if model_stem.endswith("Instruct"):
            q4_size, q8_size, mmproj_size = 18_557_053_952, 32_484_494_336, 1_325_020_128
            q4_sha = "d9e2876556e7873e02c0359f832432ee2d67ab7dd0cee3efe0f77fd7a1f4dd85"
            q8_sha = "8a50e5a7d29ae6a28fea9ca45e3bb0a142e76ec07e6787a7703cd498eb08ffaa"
            mmproj_sha = "1104376db833f1e89c84834144ac3863340c2cd1ddaeddb39cb0247fb5c20c8d"
        else:
            q4_size, q8_size, mmproj_size = 18_557_053_248, 32_484_493_632, 1_325_020_224
            q4_sha = "afdaeff6f23c740429aadb3fa180f9d53b78278fe0d331b594b0b71bd9bf4835"
            q8_sha = "a146426ed58329962e3f9b582123d1e06dad64cb2454728d7be883bc7a5658c7"
            mmproj_sha = "2bd5459571f8230a0c251d3d0dd36267753f0800ed145449a34f220a31f93898"
    return (
        VariantSpec(
            f"{prefix}_gguf_q4",
            "GGUF Q4_K_M + mmproj Q8_0",
            "gguf",
            (q4_size + mmproj_size) / 1_000_000_000,
            f"{prefix}_gguf_q4",
            gguf_files=(q4_file, mmproj),
            gguf_repo=repo,
            gguf_file_sizes=(q4_size, mmproj_size),
            gguf_sha256=(q4_sha, mmproj_sha),
            backend="llamacpp",
        ),
        VariantSpec(
            f"{prefix}_gguf_q8",
            "GGUF Q8_0 + mmproj Q8_0",
            "gguf",
            (q8_size + mmproj_size) / 1_000_000_000,
            f"{prefix}_gguf_q8",
            gguf_files=(q8_file, mmproj),
            gguf_repo=repo,
            gguf_file_sizes=(q8_size, mmproj_size),
            gguf_sha256=(q8_sha, mmproj_sha),
            backend="llamacpp",
        ),
    )


_QWEN3_ALL = frozenset({"video", "video_audio", "image", "audio", "text"})

_TIMECHAT_LIMITS = ModelLimits(60.0, 2.0, 297_920, 100_352, 160, 32_768, 9_216, 25.0, True, False, False, 28)
_AVOCADO_LIMITS = ModelLimits(100.0, 2.0, 401_408, 100_352, 256, 32_768, 2_048, 25.0, False, False, False, 28)
_QWEN3_LIMITS = ModelLimits(None, 2.0, 256 * 32 * 32, 4 * 32 * 32, 768, 32_768, 32_768, 13.0, False, False, True, 32)
_QWEN3_INSTRUCT_LIMITS = replace(_QWEN3_LIMITS, max_new_tokens_cap=8_192)
_CAPTIONER_LIMITS = replace(
    _QWEN3_LIMITS,
    max_duration_s=30.0,
    max_frames=None,
    max_new_tokens_cap=8_192,
    single_audio_only=True,
    supports_batch=False,
)


MODEL_SPECS: dict[str, ModelSpec] = {
    "timechat": ModelSpec(
        "timechat",
        "TimeChat Captioner GRPO 7B",
        "qwen2_5_omni_thinker",
        _hf_variants("timechat", bf16=17.9, int8=10.3, int4=6.5),
        frozenset({"video_audio"}),
        _TIMECHAT_LIMITS,
        _generation_schema(max_tokens=9_216, default_max_tokens=9_216, fps=2.0, max_frames=160,
                           max_pixels=297_920, video_audio=True),
        "timechat_6d_raw",
    ),
    "avocado": ModelSpec(
        "avocado",
        "AVoCaDO",
        "qwen2_5_omni_thinker",
        _hf_variants("avocado", bf16=17.9, int8=10.3, int4=6.5),
        frozenset({"video_audio", "video"}),
        _AVOCADO_LIMITS,
        _generation_schema(max_tokens=2_048, default_max_tokens=2_048, fps=2.0, max_frames=256,
                           max_pixels=401_408, video_audio=True),
        "avocado_av_aligned",
    ),
    "qwen3_omni_instruct": ModelSpec(
        "qwen3_omni_instruct",
        "Qwen3-Omni 30B-A3B Instruct",
        "qwen3_omni_moe_thinker",
        _hf_variants("qwen3_omni_instruct", bf16=63.4, int8=33.1, int4=19.9)
        + _gguf_variants(
            "qwen3_omni_instruct",
            "Qwen3-Omni-30B-A3B-Instruct",
            "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF",
        ),
        _QWEN3_ALL,
        _QWEN3_INSTRUCT_LIMITS,
        _generation_schema(max_tokens=8_192, default_max_tokens=4_096, fps=2.0, max_frames=768,
                           max_pixels=256 * 32 * 32, video_audio=True),
        "qwen3_video_dense",
    ),
    "qwen3_omni_thinking": ModelSpec(
        "qwen3_omni_thinking",
        "Qwen3-Omni 30B-A3B Thinking",
        "qwen3_omni_moe_thinker",
        _hf_variants("qwen3_omni_thinking", bf16=63.4, int8=33.1, int4=19.9)
        + _gguf_variants(
            "qwen3_omni_thinking",
            "Qwen3-Omni-30B-A3B-Thinking",
            "ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF",
        ),
        _QWEN3_ALL,
        _QWEN3_LIMITS,
        _generation_schema(max_tokens=32_768, default_max_tokens=16_384, fps=2.0, max_frames=768,
                           max_pixels=256 * 32 * 32, sampled=True, thinking=True, video_audio=True),
        "qwen3_thinking_dense",
    ),
    "qwen3_omni_captioner": ModelSpec(
        "qwen3_omni_captioner",
        "Qwen3-Omni 30B-A3B Captioner",
        "qwen3_omni_moe_thinker",
        _hf_variants("qwen3_omni_captioner", bf16=63.4, int8=33.1, int4=19.9)
        + _gguf_variants(
            "qwen3_omni_captioner",
            "Qwen3-Omni-30B-A3B-Captioner",
            "mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF",
            captioner=True,
        ),
        frozenset({"audio"}),
        _CAPTIONER_LIMITS,
        _generation_schema(max_tokens=8_192, default_max_tokens=2_048, fps=2.0, max_frames=768,
                           max_pixels=256 * 32 * 32, sampled=True, video_audio=False),
        "qwen3_captioner_promptfree",
    ),
}


def _variant_index() -> dict[str, tuple[ModelSpec, VariantSpec]]:
    return {
        variant.key: (spec, variant)
        for spec in MODEL_SPECS.values()
        for variant in spec.variants
    }


_GGUF_VARIANT_ALIASES = {
    "qwen3_omni_instruct_gguf_q4_k_m": "qwen3_omni_instruct_gguf_q4",
    "qwen3_omni_instruct_gguf_q8_0": "qwen3_omni_instruct_gguf_q8",
    "qwen3_omni_thinking_gguf_q4_k_m": "qwen3_omni_thinking_gguf_q4",
    "qwen3_omni_thinking_gguf_q8_0": "qwen3_omni_thinking_gguf_q8",
    "qwen3_omni_captioner_gguf_q4_k_m": "qwen3_omni_captioner_gguf_q4",
    "qwen3_omni_captioner_gguf_q8_0": "qwen3_omni_captioner_gguf_q8",
}


def _canonical_variant_key(variant_key: str) -> str:
    return _GGUF_VARIANT_ALIASES.get(variant_key, variant_key)


def get_variant(variant_key: str) -> VariantSpec:
    """Return a variant by key with a descriptive error for invalid keys."""

    try:
        return _variant_index()[_canonical_variant_key(variant_key)][1]
    except KeyError as exc:
        raise KeyError(f"Unknown model variant: {variant_key}") from exc


def variant_to_family(variant_key: str) -> str:
    """Return the owning family for a variant key."""

    try:
        return _variant_index()[_canonical_variant_key(variant_key)][0].family
    except KeyError as exc:
        raise KeyError(f"Unknown model variant: {variant_key}") from exc


def resolve_model_dir(variant_key: str) -> Path:
    """Resolve a model variant to its application-local folder."""

    return (MODELS_DIR / get_variant(variant_key).folder_name).resolve(strict=False)


def _model_info_size(folder: Path) -> float | None:
    info_path = folder / "vcap_model_info.json"
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        total = int(payload.get("total_bytes", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return total / 1_000_000_000 if total > 0 else None


def variant_size_gb(variant_key: str) -> float:
    """Return measured decimal GB when model metadata exists, else the estimate."""

    variant = get_variant(variant_key)
    return _model_info_size(resolve_model_dir(variant_key)) or variant.size_gb


def all_variant_choices() -> list[tuple[str, str]]:
    """Return stable ``(label with size, key)`` choices for UI controls."""

    choices: list[tuple[str, str]] = []
    for spec in MODEL_SPECS.values():
        for variant in spec.variants:
            choices.append((f"{spec.label} — {variant.label} ({variant_size_gb(variant.key):.1f} GB)", variant.key))
    return choices


def _required_hf_files(folder: Path) -> list[str]:
    missing = [name for name in ("model.safetensors", "config.json", "preprocessor_config.json") if not (folder / name).is_file()]
    if not (folder / "tokenizer.json").is_file() and not (folder / "tokenizer_config.json").is_file():
        missing.append("tokenizer.json/tokenizer_config.json")
    return missing


def variant_is_ready(variant_key: str) -> tuple[bool, str]:
    """Check that a complete local checkpoint exists without opening tensors."""

    variant = get_variant(variant_key)
    folder = resolve_model_dir(variant_key)
    if not folder.is_dir():
        return False, f"model folder is missing: {folder}"
    if variant.backend == "llamacpp":
        if not variant.gguf_files:
            return False, "GGUF registry entry has no files"
        missing = [name for name in variant.gguf_files if not (folder / name).is_file()]
        if missing:
            return False, "missing " + ", ".join(missing)
        expected_sizes = variant.gguf_file_sizes or ()
        for index, name in enumerate(variant.gguf_files):
            path = folder / name
            try:
                actual = path.stat().st_size
            except OSError as exc:
                return False, f"cannot inspect {name}: {exc}"
            expected = expected_sizes[index] if index < len(expected_sizes) else 0
            if actual <= 0:
                return False, f"{name} is empty"
            if expected and actual != expected:
                return False, f"{name} appears incomplete ({actual} of {expected} bytes)"
        return True, "ready"
    missing = _required_hf_files(folder)
    if missing:
        return False, "missing " + ", ".join(missing)
    checkpoint = folder / "model.safetensors"
    try:
        actual = checkpoint.stat().st_size
    except OSError as exc:
        return False, f"cannot inspect model.safetensors: {exc}"
    expected_info = _model_info_size(folder)
    expected_bytes = int((expected_info or variant.size_gb) * 1_000_000_000)
    if expected_bytes > 0 and actual < expected_bytes * 0.90:
        return False, f"model.safetensors appears incomplete ({actual / 1e9:.2f} of about {expected_bytes / 1e9:.2f} GB)"
    if actual <= 0 or not math.isfinite(float(actual)):
        return False, "model.safetensors is empty"
    return True, "ready"


__all__ = [
    "MODEL_SPECS",
    "ModelLimits",
    "ModelSpec",
    "ParamSpec",
    "VariantSpec",
    "all_variant_choices",
    "get_variant",
    "resolve_model_dir",
    "variant_is_ready",
    "variant_size_gb",
    "variant_to_family",
]
