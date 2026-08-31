"""Model registry, loaders, wrappers, offload, and runtime helpers."""

from __future__ import annotations

from typing import Any

from .base import (
    BaseCaptioner,
    Callbacks,
    CaptionResult,
    CaptionTiming,
    GenParams,
    MediaInput,
    MediaPart,
    PreprocessParams,
    PromptSpec,
    TokenUsage,
)
from .registry import MODEL_SPECS


def captioner_for_loaded(loaded: Any) -> BaseCaptioner:
    """Create the correct family wrapper for a loaded model."""

    family = loaded.spec.family
    if family == "timechat":
        from .timechat import TimeChatCaptioner

        return TimeChatCaptioner(loaded)
    if family == "avocado":
        from .avocado import AvocadoCaptioner

        return AvocadoCaptioner(loaded)
    from .qwen3_omni import captioner_for_loaded as qwen_captioner_for_loaded

    return qwen_captioner_for_loaded(loaded)


__all__ = [
    "BaseCaptioner",
    "Callbacks",
    "CaptionResult",
    "CaptionTiming",
    "GenParams",
    "MODEL_SPECS",
    "MediaInput",
    "MediaPart",
    "PreprocessParams",
    "PromptSpec",
    "TokenUsage",
    "captioner_for_loaded",
]
