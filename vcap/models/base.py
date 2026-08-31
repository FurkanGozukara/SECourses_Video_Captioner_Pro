"""Typed, backend-neutral captioner inputs, outputs, and lifecycle interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .offload import OffloadPlan
from .registry import MODEL_SPECS, ModelSpec


MediaKind = Literal["video", "video_audio", "audio", "image", "text"]


@dataclass(frozen=True)
class MediaPart:
    """One ordered item in a mixed-media user message."""

    type: MediaKind
    path: str | Path | None = None
    text: str | None = None
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class MediaInput:
    """A single path or an ordered mixed-media message."""

    path: str | Path | None = None
    kind: MediaKind | None = None
    text: str | None = None
    start: float | None = None
    end: float | None = None
    parts: tuple[MediaPart, ...] = ()

    @classmethod
    def mixed(cls, parts: Sequence[MediaPart]) -> "MediaInput":
        """Construct an ordered mixed-media input."""

        return cls(parts=tuple(parts))


@dataclass(frozen=True)
class PromptSpec:
    """Prompt preset selection plus optional direct overrides."""

    preset_id: str | None = None
    user_prompt: str | None = None
    system_prompt: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenParams:
    """Generation controls shared by all model wrappers."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 2048
    do_sample: bool = False
    use_cache: bool = True
    enable_thinking: bool = True


@dataclass(frozen=True)
class PreprocessParams:
    """Frame, pixel, segment, and embedded-audio preprocessing controls."""

    fps: float = 2.0
    max_frames: int = 160
    sampling_strategy: str = "fps"
    max_pixels: int = 297_920
    min_pixels: int | None = None
    use_audio_in_video: bool = True
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class Callbacks:
    """Cooperative cancellation and progress hooks for one caption call."""

    progress: Callable[..., None] | None = None
    cancel: object | None = None
    delta: Callable[..., None] | None = None


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One text message in a multimodal conversation history."""

    role: ChatRole
    content: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    new_tokens: int
    finish_reason: Literal["eos", "length", "cancelled"] = "length"

    def __getitem__(self, key: str) -> Any:
        """Allow metadata-style access while preserving the typed API."""

        return getattr(self, key)


@dataclass(frozen=True)
class CaptionTiming:
    prefill_s: float
    decode_s: float
    tokens_per_s: float
    total_s: float
    generated_tokens: int = 0

    @property
    def tok_per_s(self) -> float:
        """Backward-compatible alias used by the existing pipeline display."""

        return self.tokens_per_s

    def __getitem__(self, key: str) -> Any:
        """Allow metadata-style access while preserving attribute access."""

        return getattr(self, key)


@dataclass(frozen=True)
class CaptionResult:
    """Normalized generated caption and diagnostics."""

    text: str
    raw_text: str
    reasoning: str = ""
    structured: Any = None
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    timing: CaptionTiming = field(default_factory=lambda: CaptionTiming(0.0, 0.0, 0.0, 0.0))
    peak_vram_gb: float = 0.0
    cancelled: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatResult:
    """Normalized assistant response and multi-turn context diagnostics."""

    text: str
    raw_text: str
    reasoning: str = ""
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    timing: CaptionTiming = field(default_factory=lambda: CaptionTiming(0.0, 0.0, 0.0, 0.0))
    peak_vram_gb: float = 0.0
    cancelled: bool = False
    warnings: tuple[str, ...] = ()
    retained_history: tuple[ChatMessage, ...] = ()
    dropped_turns: int = 0
    context_tokens: int = 0


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text", item.get("content"))
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if value is None else str(value)


def normalize_chat_history(
    history: Sequence[ChatMessage | Mapping[str, Any]],
) -> list[ChatMessage]:
    """Validate and normalize a JSON/Gradio-style text message history."""

    normalized: list[ChatMessage] = []
    for index, value in enumerate(history):
        if isinstance(value, ChatMessage):
            message = value
        elif isinstance(value, Mapping):
            role = str(value.get("role") or "").strip().casefold()
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Chat message {index + 1} has an unsupported role: {role or '<empty>'}")
            message = ChatMessage(role, _message_text(value.get("content")))  # type: ignore[arg-type]
        else:
            raise TypeError(f"Chat message {index + 1} must be a mapping or ChatMessage")
        role = str(message.role).casefold()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Chat message {index + 1} has an unsupported role: {role}")
        normalized.append(ChatMessage(role, str(message.content)))  # type: ignore[arg-type]
    return normalized


def truncate_chat_history(
    history: Sequence[ChatMessage | Mapping[str, Any]],
    token_counter: Callable[[Sequence[ChatMessage]], int],
    context_tokens: int,
    *,
    threshold: float = 0.9,
) -> tuple[list[ChatMessage], int, int]:
    """Drop oldest complete turns while preserving the media and current turns.

    The first user/assistant pair is the media turn. The final user turn is the
    request being answered. System messages and both boundary turns are retained.
    """

    working = normalize_chat_history(history)
    budget = max(1, int(max(1, context_tokens) * min(1.0, max(0.1, float(threshold)))))
    tokens = max(0, int(token_counter(working)))
    dropped = 0
    while tokens > budget:
        user_positions = [index for index, item in enumerate(working) if item.role == "user"]
        if len(user_positions) < 3:
            break
        start = user_positions[1]
        end = user_positions[2]
        del working[start:end]
        dropped += 1
        tokens = max(0, int(token_counter(working)))
    return working, dropped, tokens


class BaseCaptioner:
    """Minimal lifecycle interface implemented by every model-family wrapper."""

    def __init__(self, family: str, loaded: Any | None = None) -> None:
        try:
            self.spec: ModelSpec = MODEL_SPECS[family]
        except KeyError as exc:
            raise KeyError(f"Unknown captioner family: {family}") from exc
        self.loaded = loaded

    def load(
        self,
        variant_key: str,
        device: str = "cuda:0",
        attention: str = "auto",
        offload_plan: OffloadPlan | None = None,
        compile_flag: bool = False,
        progress_cb: Callable[..., None] | None = None,
    ) -> Any:
        """Load this captioner's model and return the loader report."""

        from .loader import MODEL_CACHE

        self.loaded = MODEL_CACHE.load(
            variant_key,
            device=device,
            attention=attention,
            offload=offload_plan or OffloadPlan(),
            progress_cb=progress_cb,
            compile_model=compile_flag,
        )
        return self.loaded.load_report

    def unload(self) -> Any:
        """Unload this captioner's resident model and return freed-memory data."""

        if self.loaded is None:
            return None
        from .loader import MODEL_CACHE, unload_model

        report = MODEL_CACHE.unload() if MODEL_CACHE.loaded is self.loaded else unload_model(self.loaded)
        self.loaded = None
        return report

    def caption(
        self,
        media: MediaInput,
        prompt: PromptSpec | str | None = None,
        gen: GenParams | None = None,
        pre: PreprocessParams | None = None,
        cb: Callbacks | None = None,
    ) -> CaptionResult:
        """Generate one normalized caption."""

        raise NotImplementedError

    def chat(
        self,
        history: Sequence[ChatMessage | Mapping[str, Any]],
        media: MediaInput | None = None,
        *,
        system_prompt: str | None = None,
        gen: GenParams | None = None,
        pre: PreprocessParams | None = None,
        cb: Callbacks | None = None,
    ) -> ChatResult:
        """Generate one assistant turn from the complete conversation history."""

        del history, media, system_prompt, gen, pre, cb
        raise NotImplementedError(f"{self.spec.label} does not implement interactive chat")


__all__ = [
    "BaseCaptioner",
    "Callbacks",
    "CaptionResult",
    "CaptionTiming",
    "ChatMessage",
    "ChatResult",
    "ChatRole",
    "GenParams",
    "MediaInput",
    "MediaKind",
    "MediaPart",
    "PreprocessParams",
    "PromptSpec",
    "TokenUsage",
    "normalize_chat_history",
    "truncate_chat_history",
]
