"""VRAM-tier model, media, attention, and offload defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .offload import OffloadPlan
from .registry import MODEL_SPECS, variant_size_gb


VRAM_TIERS = [6, 8, 10, 12, 16, 24, 32, 48, 80]


@dataclass(frozen=True)
class VramPreset:
    """Complete settings overlay for one family and VRAM tier."""

    variant_scheme: str = field(metadata={"description": "Preferred local checkpoint precision scheme."})
    attention: str = field(metadata={"description": "Preferred attention backend."})
    fps: float = field(metadata={"description": "Video frames sampled per second."})
    max_frames: int = field(metadata={"description": "Maximum sampled video frames."})
    max_pixels: int = field(metadata={"description": "Maximum pixel area of each frame."})
    max_new_tokens: int = field(metadata={"description": "Maximum generated caption tokens."})
    offload: OffloadPlan = field(metadata={"description": "GPU-layer and CPU-offload placement."})
    notes: str = field(metadata={"description": "Human-readable performance and quality tradeoff."})


def auto_tier(gpu_total_gb: float) -> int:
    """Map physical VRAM to the greatest supported tier not exceeding it."""

    rounded = max(0, int(float(gpu_total_gb) + 0.5))
    eligible = [tier for tier in VRAM_TIERS if tier <= rounded]
    return eligible[-1] if eligible else VRAM_TIERS[0]


def _normalize_family(family: str) -> str:
    if family not in MODEL_SPECS:
        raise KeyError(f"Unknown model family: {family}")
    return family


def _normalize_tier(tier: int | float) -> int:
    value = int(tier)
    if value in VRAM_TIERS:
        return value
    return auto_tier(float(value))


def _seven_b_preset(family: str, tier: int) -> VramPreset:
    max_pixels = 297_920 if family == "timechat" else 401_408
    max_tokens = 9_216 if family == "timechat" else 2_048
    if tier <= 6:
        return VramPreset("int4_convrot_w4a8", "auto", 1.0, 32, 128 * 28 * 28,
                          min(max_tokens, 2_048), OffloadPlan(6, False, None, False),
                          "Heavy CPU offload; lowest-memory mode and substantially slower.")
    if tier <= 8:
        return VramPreset("int4_convrot_w4a8", "auto", 1.0, 48, 200_000,
                          min(max_tokens, 4_096), OffloadPlan(16, False, None, True),
                          "INT4 with partial decoder offload and conservative video input.")
    if tier <= 10:
        return VramPreset("int4_convrot_w4a8", "auto", 2.0, 80, max_pixels,
                          max_tokens, OffloadPlan("all", False, None, False),
                          "INT4 resident weights with the model's training resolution.")
    if tier <= 12:
        return VramPreset("int8_convrot", "auto", 2.0, 80, max_pixels,
                          max_tokens, OffloadPlan(22, False, None, True),
                          "INT8 with a small CPU-offloaded decoder tail.")
    if tier <= 16:
        return VramPreset("int8_convrot", "auto", 2.0, 128, max_pixels,
                          max_tokens, OffloadPlan("all", False, None, False),
                          "INT8 fully resident with room for vision and KV-cache peaks.")
    if tier <= 24:
        return VramPreset("bf16", "auto", 2.0, 128 if family == "avocado" else 120, max_pixels,
                          max_tokens, OffloadPlan("all", False, None, False),
                          "BF16 resident; preserves the reference checkpoint exactly.")
    return VramPreset("bf16", "auto", 2.0, 160 if family == "timechat" else 256, max_pixels,
                      max_tokens, OffloadPlan("all", False, None, False),
                      "BF16 resident with the full documented frame allowance.")


def _qwen3_preset(family: str, tier: int) -> VramPreset:
    if tier < 8:
        raise ValueError("Qwen3-Omni is not offered below the 8 GB tier")
    captioner = family == "qwen3_omni_captioner"
    thinking = family == "qwen3_omni_thinking"
    default_tokens = 2_048 if captioner else 16_384 if thinking else 4_096
    if tier <= 8:
        return VramPreset("int4_convrot_w4a8", "auto", 1.0, 32, 128 * 32 * 32,
                          min(default_tokens, 2_048), OffloadPlan(4, True, None, False),
                          "Experimental heavy offload; experts stay on CPU and generation is slow.")
    if tier <= 12:
        return VramPreset("int4_convrot_w4a8", "auto", 1.0, 48, 128 * 32 * 32,
                          min(default_tokens, 4_096), OffloadPlan(12, True, None, True),
                          "INT4 with expert offload and a conservative context budget.")
    if tier <= 16:
        return VramPreset("int4_convrot_w4a8", "auto", 1.0, 64, 192 * 32 * 32,
                          min(default_tokens, 8_192), OffloadPlan(32, False, None, True),
                          "INT4 with a light CPU-offloaded decoder tail.")
    if tier <= 24:
        return VramPreset("int4_convrot_w4a8", "auto", 2.0, 96, 256 * 32 * 32,
                          default_tokens, OffloadPlan("all", False, None, False),
                          "INT4 fully resident with the recommended 256-token frame budget.")
    if tier <= 32:
        return VramPreset("int8_convrot", "auto", 2.0, 96, 256 * 32 * 32,
                          default_tokens, OffloadPlan(42, False, None, True),
                          "INT8 with a small decoder tail offloaded to CPU.")
    if tier <= 48:
        return VramPreset("int8_convrot", "auto", 2.0, 128, 256 * 32 * 32,
                          default_tokens, OffloadPlan("all", False, None, False),
                          "INT8 fully resident.")
    return VramPreset("bf16", "auto", 2.0, 128, 256 * 32 * 32,
                      default_tokens, OffloadPlan("all", False, None, False),
                      "BF16 thinker fully resident.")


def preset_for(family: str, tier: int | float) -> VramPreset:
    """Return the recommended settings for a family and supported tier."""

    family = _normalize_family(family)
    resolved = _normalize_tier(tier)
    if family in {"timechat", "avocado"}:
        return _seven_b_preset(family, resolved)
    return _qwen3_preset(family, resolved)


def _scheme_threshold(family: str, scheme: str) -> int:
    if family in {"timechat", "avocado"}:
        return {"int4_convrot_w4a8": 6, "int8_convrot": 10, "bf16": 24}.get(scheme, 999)
    return {"int4_convrot_w4a8": 8, "int8_convrot": 32, "bf16": 80, "gguf": 24}.get(scheme, 999)


def allowed_variants(family: str, tier: int | float) -> list[str]:
    """List variants whose scheme is supported by the tier's placement policy."""

    family = _normalize_family(family)
    resolved = _normalize_tier(tier)
    result: list[str] = []
    for variant in MODEL_SPECS[family].variants:
        threshold = _scheme_threshold(family, variant.scheme)
        if variant.scheme == "gguf":
            threshold = 32 if variant.key.endswith("_gguf_q8") else 24
        # Refine only downward when a local conversion is materially smaller
        # than its conservative design estimate; runtime headroom still wins.
        measured = variant_size_gb(variant.key)
        if variant.scheme != "gguf" and measured + 1.4 <= resolved:
            threshold = min(threshold, resolved)
        if resolved >= threshold:
            result.append(variant.key)
    return result


def apply_preset(settings_dict: dict[str, Any], preset: VramPreset) -> dict[str, Any]:
    """Return a copy of settings with every preset field applied."""

    settings = dict(settings_dict)
    settings.update(
        {
            "variant_scheme": preset.variant_scheme,
            "attention": preset.attention,
            "fps": preset.fps,
            "max_frames": preset.max_frames,
            "max_pixels": preset.max_pixels,
            "max_new_tokens": preset.max_new_tokens,
            "offload": preset.offload,
            "notes": preset.notes,
        }
    )
    return settings


__all__ = [
    "VRAM_TIERS",
    "VramPreset",
    "allowed_variants",
    "apply_preset",
    "auto_tier",
    "preset_for",
]
