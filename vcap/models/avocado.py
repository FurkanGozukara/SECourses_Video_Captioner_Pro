"""AVoCaDO audiovisual captioner wrapper."""

from __future__ import annotations

from typing import Any

from .omni_common import OmniCaptionerBase


class AvocadoCaptioner(OmniCaptionerBase):
    """Qwen2.5-Omni thinker with official AVoCaDO prompt handling."""

    avocado_mode = True

    def __init__(self, loaded: Any | None = None) -> None:
        super().__init__("avocado", loaded)


AVoCaDOCaptioner = AvocadoCaptioner


__all__ = ["AVoCaDOCaptioner", "AvocadoCaptioner"]
