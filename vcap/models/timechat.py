"""TimeChat-Captioner GRPO wrapper."""

from __future__ import annotations

from typing import Any

from .omni_common import OmniCaptionerBase


class TimeChatCaptioner(OmniCaptionerBase):
    """Qwen2.5-Omni thinker specialized for timestamped audiovisual JSON."""

    force_video_audio = True

    def __init__(self, loaded: Any | None = None) -> None:
        super().__init__("timechat", loaded)


__all__ = ["TimeChatCaptioner"]
