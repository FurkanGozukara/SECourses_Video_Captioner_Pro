"""Qwen3-Omni Instruct, Thinking, and audio Captioner wrappers."""

from __future__ import annotations

from typing import Any

from .base import BaseCaptioner
from .omni_common import OmniCaptionerBase, split_thinking


class Qwen3OmniCaptioner(OmniCaptionerBase):
    """Family-selected Qwen3-Omni thinker wrapper."""

    def __init__(self, family: str = "qwen3_omni_instruct", loaded: Any | None = None) -> None:
        if family not in {
            "qwen3_omni_instruct",
            "qwen3_omni_thinking",
            "qwen3_omni_captioner",
        }:
            raise ValueError(f"Unsupported Qwen3-Omni family: {family}")
        super().__init__(family, loaded)
        self.thinking_mode = family == "qwen3_omni_thinking"
        self.captioner_mode = family == "qwen3_omni_captioner"


class Qwen3OmniInstructCaptioner(Qwen3OmniCaptioner):
    """Convenience wrapper fixed to Qwen3-Omni Instruct."""

    def __init__(self, loaded: Any | None = None) -> None:
        super().__init__("qwen3_omni_instruct", loaded)


class Qwen3OmniThinkingCaptioner(Qwen3OmniCaptioner):
    """Convenience wrapper fixed to Qwen3-Omni Thinking."""

    def __init__(self, loaded: Any | None = None) -> None:
        super().__init__("qwen3_omni_thinking", loaded)


class Qwen3OmniAudioCaptioner(Qwen3OmniCaptioner):
    """Convenience wrapper fixed to the prompt-free audio Captioner."""

    def __init__(self, loaded: Any | None = None) -> None:
        super().__init__("qwen3_omni_captioner", loaded)


def captioner_for_loaded(loaded: Any) -> BaseCaptioner:
    """Construct the correct family wrapper for a :class:`LoadedModel`."""

    if getattr(loaded.variant, "backend", None) == "llamacpp" and isinstance(loaded.model, BaseCaptioner):
        return loaded.model
    family = loaded.spec.family
    if family == "qwen3_omni_instruct":
        return Qwen3OmniInstructCaptioner(loaded)
    if family == "qwen3_omni_thinking":
        return Qwen3OmniThinkingCaptioner(loaded)
    if family == "qwen3_omni_captioner":
        return Qwen3OmniAudioCaptioner(loaded)
    raise ValueError(f"Loaded model is not a Qwen3-Omni family: {family}")


__all__ = [
    "Qwen3OmniAudioCaptioner",
    "Qwen3OmniCaptioner",
    "Qwen3OmniInstructCaptioner",
    "Qwen3OmniThinkingCaptioner",
    "captioner_for_loaded",
    "split_thinking",
]
