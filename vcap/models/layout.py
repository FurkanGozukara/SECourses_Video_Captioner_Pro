"""Resolve Qwen thinker paths from the instantiated Transformers model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThinkerLayout:
    nested: bool = False

    def checkpoint_name(self, name: str) -> str:
        """Map the published checkpoint's legacy names to the runtime layout.

        Transformers 5.21 put the towers below ``model`` and the text decoder
        below ``model.language_model``. Our streamed checkpoints retain their
        original names; unlike ``from_pretrained``, streaming must map them here.
        Already converted checkpoint and device-map names remain unchanged.
        """
        if not self.nested:
            return name
        if name in {"audio_tower", "visual"} or name.startswith(("audio_tower.", "visual.")):
            return f"model.{name}"
        if name == "model":
            return "model.language_model"
        if name.startswith("model."):
            branch = name.split(".", 2)[1]
            if branch not in {"language_model", "audio_tower", "visual"}:
                return f"model.language_model.{name[len('model.') :]}"
        return name

    @property
    def tower_names(self) -> tuple[str, str]:
        return tuple(self.checkpoint_name(name) for name in ("audio_tower", "visual"))


def thinker_layout(model: Any) -> ThinkerLayout:
    backbone = getattr(model, "model", None)
    return ThinkerLayout(nested=getattr(backbone, "language_model", None) is not None)


def language_model(model: Any) -> Any:
    """Return the text decoder for either a thinker or its outer multimodal model."""
    thinker = getattr(model, "thinker", None)
    if thinker is not None:
        model = thinker
    backbone = getattr(model, "model", model)
    return getattr(backbone, "language_model", backbone)
