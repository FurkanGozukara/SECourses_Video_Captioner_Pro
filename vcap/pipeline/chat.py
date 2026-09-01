"""Interactive chat request, model-cache execution, streaming, and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from vcap import OUTPUTS_DIR, VERSION
from vcap.core.media import probe_media
from vcap.core.outputs import OutputWriter, allocate_run_dir, model_short_name
from vcap.core.paths import normalize_path
from vcap.core.subprocess_runner import CancelToken
from vcap.models.base import (
    Callbacks,
    CaptionTiming,
    ChatMessage,
    ChatResult,
    GenParams,
    MediaInput,
    MediaPart,
    PreprocessParams,
    TokenUsage,
    normalize_chat_history,
)
from vcap.models.registry import MODEL_SPECS, variant_to_family


EventCallback = Callable[[dict[str, Any]], None]


def _json_safe(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


@dataclass(frozen=True)
class ChatRequest:
    """Serializable input to one assistant-turn generation."""

    settings: dict[str, Any]
    history: tuple[ChatMessage, ...]
    media: tuple[str, ...] = ()
    generation: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | "ChatRequest") -> "ChatRequest":
        if isinstance(value, cls):
            return value
        data = dict(value or {})
        settings = dict(data.get("settings") or {})
        history = tuple(normalize_chat_history(data.get("history") or ()))
        raw_media = data.get("media") or ()
        if isinstance(raw_media, (str, os.PathLike)):
            raw_media = [raw_media]
        media = tuple(str(item) for item in raw_media if str(item).strip())
        generation = dict(data.get("generation") or {})
        system_prompt = str(
            data.get("system_prompt", settings.get("chat_system_prompt", "")) or ""
        )
        return cls(_json_safe(settings), history, media, _json_safe(generation), system_prompt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": _json_safe(self.settings),
            "history": [asdict(item) for item in self.history],
            "media": list(self.media),
            "generation": _json_safe(self.generation),
            "system_prompt": self.system_prompt,
        }


@dataclass(frozen=True)
class ChatResponse:
    """JSON-safe assistant response returned by the pipeline client."""

    model_key: str
    text: str
    raw_text: str
    reasoning: str
    prompt_tokens: int
    new_tokens: int
    finish_reason: str
    prefill_s: float
    decode_s: float
    tokens_per_s: float
    total_s: float
    peak_vram_gb: float
    cancelled: bool
    warnings: tuple[str, ...] = ()
    dropped_turns: int = 0
    context_tokens: int = 0
    retained_history: tuple[ChatMessage, ...] = ()
    context_limit: int = 0

    @classmethod
    def from_result(cls, model_key: str, result: ChatResult) -> "ChatResponse":
        return cls(
            model_key=str(model_key),
            text=result.text,
            raw_text=result.raw_text,
            reasoning=result.reasoning,
            prompt_tokens=int(result.usage.prompt_tokens),
            new_tokens=int(result.usage.new_tokens),
            finish_reason=str(result.usage.finish_reason),
            prefill_s=float(result.timing.prefill_s),
            decode_s=float(result.timing.decode_s),
            tokens_per_s=float(result.timing.tokens_per_s),
            total_s=float(result.timing.total_s),
            peak_vram_gb=float(result.peak_vram_gb),
            cancelled=bool(result.cancelled),
            warnings=tuple(result.warnings),
            dropped_turns=int(result.dropped_turns),
            context_tokens=int(result.context_tokens),
            retained_history=tuple(result.retained_history),
            context_limit=int(getattr(result, "context_limit", 0) or 0),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChatResponse":
        data = dict(value)
        data["warnings"] = tuple(str(item) for item in data.get("warnings") or ())
        data["retained_history"] = tuple(
            normalize_chat_history(data.get("retained_history") or ())
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        data["retained_history"] = [asdict(item) for item in self.retained_history]
        return _json_safe(data)


class _SessionEmitter:
    def __init__(self, emit: EventCallback) -> None:
        self.emit = emit

    def log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        event: dict[str, Any] = {"ev": "log", "text": str(message), "level": str(level)}
        if scope:
            event["scope"] = str(scope)
        self.emit(event)

    def phase_progress(
        self,
        message: str,
        fraction: float | None = None,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(data or {})
        if fraction is not None:
            payload["fraction"] = float(fraction)
        self.emit({"ev": "status", "message": str(message), "data": _json_safe(payload)})


def _is_cancelled(token: object | None) -> bool:
    method = getattr(token, "is_cancelled", None)
    return bool(method()) if callable(method) else False


def _media_input(paths: Sequence[str]) -> MediaInput | None:
    parts: list[MediaPart] = []
    for raw in paths:
        path = normalize_path(raw, must_exist=True)
        if not path.is_file():
            raise ValueError(f"Chat attachment is not a file: {path}")
        info = probe_media(path)
        kind = {
            "video": "video_audio",
            "video_no_audio": "video",
            "audio": "audio",
            "image": "image",
        }.get(info.kind)
        if kind is None:
            raise ValueError(
                f"Unsupported chat attachment {path.name}: choose video, audio, or image media."
            )
        parts.append(MediaPart(kind, path))  # type: ignore[arg-type]
    return MediaInput.mixed(parts) if parts else None


def _callback_payload(args: tuple[Any, ...]) -> tuple[str, dict[str, Any]]:
    payload = next((dict(item) for item in args if isinstance(item, Mapping)), {})
    message = next((str(item) for item in args if isinstance(item, str)), "")
    if not message:
        message = str(payload.get("message") or "")
    return message, payload


def _partial_thinking(raw: str) -> tuple[str, str]:
    stripped = raw.lstrip()
    leading = len(raw) - len(stripped)
    if "</think>" in stripped:
        reasoning, answer = stripped.split("</think>", 1)
        return reasoning.removeprefix("<think>").strip(), answer
    if stripped.startswith("<think>"):
        return stripped[len("<think>") :].strip(), ""
    if stripped and "<think>".startswith(stripped):
        return "", ""
    return "", raw[leading:]


class _StreamAccumulator:
    def __init__(self, emit: EventCallback, thinking: bool) -> None:
        self.emit = emit
        self.thinking = thinking
        self.raw = ""
        self.direct_reasoning = ""
        self.answer = ""
        self.reasoning = ""

    def __call__(self, *args: Any) -> None:
        message, payload = _callback_payload(args)
        delta = str(payload.get("delta", message) or "")
        reasoning_delta = str(payload.get("reasoning_delta") or "")
        self.raw += delta
        self.direct_reasoning += reasoning_delta
        if self.direct_reasoning:
            reasoning, answer = self.direct_reasoning, self.raw
        elif self.thinking:
            reasoning, answer = _partial_thinking(self.raw)
        else:
            reasoning, answer = "", self.raw
        answer_delta = answer[len(self.answer) :] if answer.startswith(self.answer) else answer
        reason_delta = (
            reasoning[len(self.reasoning) :]
            if reasoning.startswith(self.reasoning)
            else reasoning
        )
        self.answer = answer
        self.reasoning = reasoning
        self.emit(
            {
                "ev": "delta",
                "delta": answer_delta,
                "reasoning_delta": reason_delta,
                "text": answer,
                "reasoning": reasoning,
            }
        )


def _halves(text: str) -> list[str]:
    middle = max(1, len(text) // 2)
    return [text[:middle], text[middle:]]


def _fake_chat(request: ChatRequest, emit: EventCallback, cancel: CancelToken) -> ChatResponse:
    model_key = str(request.settings.get("model_key") or "qwen3_omni_instruct_int4")
    last_user = next(item.content for item in reversed(request.history) if item.role == "user")
    prior = next(
        (item.content for item in reversed(request.history[:-1]) if item.role == "assistant"),
        "",
    )
    answer = f"Mock answer to: {last_user}"
    if prior:
        answer += f" Previous context: {prior}"
    # The Thinking family streams a reasoning block before its answer, exactly
    # like the real model, so the UI's live thought rendering can be exercised.
    thinking = variant_to_family(model_key) == "qwen3_omni_thinking" and bool(
        request.generation.get(
            "enable_thinking", request.settings.get("chat_enable_thinking", False)
        )
    )
    reasoning_pieces = _halves(f"Mock reasoning about: {last_user}") if thinking else []
    built = ""
    built_reasoning = ""
    started = time.perf_counter()
    try:
        delay_s = max(0.0, float(os.environ.get("VCAP_FAKE_CHAT_DELAY", "0") or 0.0))
    except ValueError:
        delay_s = 0.0

    def wait_turn() -> bool:
        deadline = time.monotonic() + delay_s
        while time.monotonic() < deadline and not _is_cancelled(cancel):
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return not _is_cancelled(cancel)

    for piece in reasoning_pieces:
        if not wait_turn():
            break
        built_reasoning += piece
        emit(
            {
                "ev": "delta",
                "delta": "",
                "reasoning_delta": piece,
                "text": "",
                "reasoning": built_reasoning,
            }
        )
    for piece in _halves(answer):
        if _is_cancelled(cancel) or not wait_turn():
            break
        built += piece
        emit(
            {
                "ev": "delta",
                "delta": piece,
                "reasoning_delta": "",
                "text": built,
                "reasoning": built_reasoning,
            }
        )
    elapsed = time.perf_counter() - started
    cancelled = _is_cancelled(cancel)
    tokens = max(1, math.ceil((len(built) + len(built_reasoning)) / 4))
    return ChatResponse(
        model_key=model_key,
        text=built,
        raw_text=f"<think>\n{built_reasoning}\n</think>\n\n{built}" if thinking else built,
        reasoning=built_reasoning,
        prompt_tokens=max(1, sum(len(item.content) for item in request.history) // 4),
        new_tokens=tokens,
        finish_reason="cancelled" if cancelled else "eos",
        prefill_s=0.0,
        decode_s=elapsed,
        tokens_per_s=tokens / max(elapsed, 1e-9),
        total_s=elapsed,
        peak_vram_gb=0.0,
        cancelled=cancelled,
        retained_history=request.history,
    )


def run_chat(
    request: ChatRequest | Mapping[str, Any],
    emit: EventCallback | None = None,
    cancel: CancelToken | None = None,
) -> ChatResponse:
    """Run one chat turn through the same worker and model cache as captioning."""

    selected = ChatRequest.from_dict(request)
    publish = emit or (lambda event: None)
    token = cancel or CancelToken()
    settings = dict(selected.settings)
    model_key = str(settings.get("model_key") or "qwen3_omni_instruct_int4")
    family = variant_to_family(model_key)
    if family == "qwen3_omni_captioner":
        raise ValueError(
            "Qwen3-Omni Captioner does not support chat. Pick Qwen3-Omni Instruct or Thinking."
        )
    if not selected.history or selected.history[-1].role != "user":
        raise ValueError("Chat history must end with the current user message")
    if family in {"timechat", "avocado"}:
        # Single-turn video Q&A: the current user turn and its one video only.
        current = next(item for item in reversed(selected.history) if item.role == "user")
        if len(selected.media) + len(current.media) != 1:
            raise ValueError(f"{MODEL_SPECS[family].label} chat requires exactly one video")
        selected = ChatRequest(
            selected.settings,
            tuple(item for item in selected.history if item.role == "system") + (current,),
            selected.media,
            selected.generation,
            selected.system_prompt,
        )
    publish({"ev": "status", "message": f"Preparing chat with {MODEL_SPECS[family].label}", "data": {}})
    if os.environ.get("VCAP_FAKE_CHAT", os.environ.get("VCAP_FAKE_CAPTIONER", "")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        response = _fake_chat(selected, publish, token)
        publish({"ev": "status", "message": "Mock chat finished", "data": {"finish_reason": response.finish_reason}})
        return response

    from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
    from vcap.pipeline.runner import _ModelSession

    output = OutputSpec(
        kind="single",
        outputs_root=str(settings.get("outputs_dir") or OUTPUTS_DIR),
    )
    job = JobSpec.from_settings(
        settings,
        [InputItem("", kind="text", text_prompt_only=True)],
        output,
    )
    session = _ModelSession(job, _SessionEmitter(publish), token)
    model = session.ensure()
    if not callable(getattr(model, "chat", None)):
        raise RuntimeError(f"{MODEL_SPECS[family].label} backend does not provide chat()")
    media = _media_input(selected.media)
    values = dict(selected.generation)
    temperature = float(values.get("temperature", settings.get("chat_temperature", 0.2)) or 0.0)
    try:
        requested_context = int(values.get("context_tokens", settings.get("context_tokens")) or 0)
    except (TypeError, ValueError):
        requested_context = 0
    generation = GenParams(
        temperature=temperature,
        top_p=float(values.get("top_p", settings.get("chat_top_p", 0.95)) or 1.0),
        top_k=int(values.get("top_k", settings.get("chat_top_k", 20)) or 0),
        repetition_penalty=float(
            values.get("repetition_penalty", settings.get("repetition_penalty", 1.0)) or 1.0
        ),
        max_new_tokens=max(
            1,
            int(values.get("max_new_tokens", settings.get("chat_max_new_tokens", 1024)) or 1024),
        ),
        do_sample=bool(values.get("do_sample", temperature > 0)),
        use_cache=bool(values.get("use_cache", True)),
        enable_thinking=bool(
            values.get("enable_thinking", settings.get("chat_enable_thinking", True))
        ),
        context_tokens=requested_context if requested_context > 0 else None,
    )
    pre = job.preprocess
    preprocessing = PreprocessParams(
        fps=pre.fps,
        max_frames=pre.max_frames,
        sampling_strategy=pre.sampling_strategy,
        max_pixels=pre.max_pixels,
        min_pixels=pre.min_pixels,
        use_audio_in_video=pre.use_audio_in_video,
        start=pre.trim_start_s or None,
        end=pre.trim_end_s,
    )
    accumulator = _StreamAccumulator(
        publish,
        family == "qwen3_omni_thinking" and generation.enable_thinking,
    )

    def progress(*args: Any) -> None:
        message, data = _callback_payload(args)
        publish({"ev": "status", "message": message or "Chat is running", "data": _json_safe(data)})

    try:
        result = model.chat(
            selected.history,
            media,
            system_prompt=selected.system_prompt,
            gen=generation,
            pre=preprocessing,
            cb=Callbacks(progress=progress, delta=accumulator, cancel=token),
        )
        response = ChatResponse.from_result(model_key, result)
        return response
    finally:
        if not job.runtime.keep_model_loaded:
            session.unload()


def conversation_markdown(
    messages: Sequence[Mapping[str, Any] | ChatMessage],
    metadata: Mapping[str, Any],
) -> str:
    """Render a saved conversation as readable UTF-8 Markdown."""

    lines = ["# Conversation", ""]
    model_key = str(metadata.get("model_key") or "unknown")
    lines.extend([f"- Model: `{model_key}`", f"- Saved: {metadata.get('saved_at', '')}", ""])
    system_prompt = str(metadata.get("system_prompt") or "").strip()
    if system_prompt:
        lines.extend(["## System", "", system_prompt, ""])
    for value in messages:
        if isinstance(value, ChatMessage):
            role, content, reasoning = value.role, value.content, ""
        else:
            role = str(value.get("role") or "message")
            content = str(value.get("content") or "")
            reasoning = str(value.get("reasoning") or "")
        if role == "system":
            continue
        lines.extend([f"## {role.title()}", ""])
        if reasoning:
            lines.extend(["<details>", "<summary>Reasoning</summary>", "", reasoning, "", "</details>", ""])
        lines.extend([content, ""])
    return "\n".join(lines).rstrip() + "\n"


def save_conversation(
    messages: Sequence[Mapping[str, Any] | ChatMessage],
    *,
    model_key: str,
    metadata: Mapping[str, Any] | None = None,
    outputs_root: str | os.PathLike[str] = OUTPUTS_DIR,
) -> Path:
    """Allocate a chat run directory and write JSON plus Markdown transcripts."""

    saved_at = datetime.now(timezone.utc).isoformat()
    run_dir = allocate_run_dir(
        normalize_path(outputs_root),
        f"chat_{model_short_name(model_key)}",
        kind="single",
    )
    message_data = [
        asdict(item) if isinstance(item, ChatMessage) else _json_safe(dict(item))
        for item in messages
    ]
    details = {
        "model_key": str(model_key),
        "saved_at": saved_at,
        "app_version": VERSION,
        **_json_safe(dict(metadata or {})),
    }
    document = {
        "_meta": {
            "format": "secourses_vcap_conversation",
            "version": 1,
            "created_at": saved_at,
        },
        "metadata": details,
        "messages": message_data,
    }
    writer = OutputWriter()
    writer.write_json(run_dir / "conversation.json", document, pretty=True)
    writer.write_text(run_dir / "conversation.md", conversation_markdown(message_data, details))
    return run_dir


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "conversation_markdown",
    "run_chat",
    "save_conversation",
]
