"""Shared PyAV/FFmpeg caption flow for Qwen-Omni thinker models."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import queue
import secrets
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

from vcap.core import console_progress
from vcap.core.logs import get_log
from vcap.core.media import (
    MediaInfo,
    probe_media,
    read_audio,
    read_video_frames,
    resample_audio,
)
from vcap.core.progress import TokenSpeedMeter
from vcap.prompts.postprocess import (
    POST_PROCESSORS,
    PostResult,
    plain,
    timechat_parse,
    trim_avocado_trailing_qa,
)
from vcap.prompts.presets import PromptPreset, get_preset, render_prompt

from .attention import is_flash_attention_failure, resolve as resolve_attention
from .base import (
    BaseCaptioner,
    Callbacks,
    CaptionResult,
    CaptionTiming,
    ChatMessage,
    ChatResult,
    GenParams,
    MediaInput,
    MediaPart,
    PreprocessParams,
    PromptSpec,
    TokenUsage,
    chat_media_parts,
    chat_media_placeholders,
    normalize_chat_history,
    truncate_chat_history,
)
from .registry import MODEL_SPECS


def _cancelled(token: object | None) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "is_set"):
        method = getattr(token, name, None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
    return bool(getattr(token, "cancelled", False))


def _callback(callback: Any, message: str, **data: Any) -> None:
    if callback is None:
        return
    payload = {"message": message, **data}
    for args in ((message, payload), (payload,), (message,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _delta_callback(callback: Any, delta: str = "", reasoning_delta: str = "") -> None:
    if callback is None or (not delta and not reasoning_delta):
        return
    payload = {"delta": delta, "reasoning_delta": reasoning_delta}
    for args in ((delta, payload), (payload,), (delta,)):
        try:
            callback(*args)
            return
        except TypeError:
            continue


def _record_generation_memory(
    loaded: Any,
    model: Any,
    torch: Any,
    device: Any,
    *,
    scope: str,
) -> None:
    """Persist activation peaks and emit swap counters without affecting generation."""

    try:
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            # Reserved bytes are what the card actually loses (allocator fragmentation
            # included), so the feedback ratio is based on them.
            peak_bytes = max(
                int(torch.cuda.max_memory_allocated(device)),
                int(torch.cuda.max_memory_reserved(device)),
            )
            resident_bytes = int(getattr(loaded.load_report, "resident_bytes", 0) or 0)
            planned = int(getattr(loaded.load_report, "activation_estimate_bytes", 0) or 0)
            observed = peak_bytes - resident_bytes
            if observed > 0:
                from .offload import record_observed_activation_bytes

                record_observed_activation_bytes(loaded.variant.key, observed, planned_bytes=planned)
    except Exception:
        pass

    try:
        manager = getattr(model, "_vcap_block_swap_manager", None)
        if manager is None:
            return
        stats = manager.stats()
        layer_loads = int(stats.get("layer_loads", 0) or 0)
        bytes_h2d = int(stats.get("bytes_h2d", 0) or 0)
        get_log().debug(
            f"Block swap: {layer_loads} layer loads, "
            f"{bytes_h2d / 2**30:.1f} GiB H2D this generation",
            scope=scope,
        )
        reset = getattr(manager, "reset_stats", None)
        if callable(reset):
            reset()
    except Exception:
        pass


def split_thinking(text: str) -> tuple[str, str]:
    """Separate Qwen3 Thinking's generated reasoning and final answer."""

    if "</think>" not in text:
        stripped = text.strip()
        if stripped.startswith("<think>"):
            # Generation stopped inside the reasoning block, so none of it is an answer.
            return stripped[len("<think>") :].strip(), ""
        return "", stripped
    reasoning, answer = text.split("</think>", 1)
    return reasoning.replace("<think>", "", 1).strip(), answer.strip()


def build_chat_conversation(
    history: Sequence[ChatMessage | Mapping[str, Any]],
    media_content: Sequence[Mapping[str, Any]] = (),
    system_prompt: str | None = None,
    *,
    turn_media: Any = None,
) -> list[dict[str, Any]]:
    """Render normalized history as a chat-template conversation.

    ``media_content`` is attached to the first user turn (legacy first-turn
    attachments). ``turn_media(index, message)`` returns the content entries
    for a message's own attachments, indexed within the non-system messages,
    so any later turn can carry media as well.
    """

    messages = normalize_chat_history(history)
    system_from_history = next((item.content for item in messages if item.role == "system"), "")
    selected_system = str(system_prompt) if system_prompt is not None else system_from_history
    text_messages = [item for item in messages if item.role != "system"]
    if not text_messages or text_messages[-1].role != "user":
        raise ValueError("Chat history must end with the current user message")
    first_user = next((index for index, item in enumerate(text_messages) if item.role == "user"), None)
    if first_user is None:
        raise ValueError("Chat history needs at least one user message")
    conversation: list[dict[str, Any]] = []
    if selected_system.strip():
        conversation.append(
            {"role": "system", "content": [{"type": "text", "text": selected_system.strip()}]}
        )
    for index, message in enumerate(text_messages):
        content: list[dict[str, Any]] = []
        if index == first_user:
            content.extend(dict(item) for item in media_content)
        if turn_media is not None and message.media:
            content.extend(dict(item) for item in turn_media(index, message))
        if message.content:
            content.append({"type": "text", "text": message.content})
        if not content:
            raise ValueError(f"The {message.role} message is empty")
        conversation.append({"role": message.role, "content": content})
    return conversation


def _default_gen(family: str) -> GenParams:
    values = {item.name: item.default for item in MODEL_SPECS[family].param_schema}
    return GenParams(
        temperature=float(values.get("temperature", 0.0)),
        top_p=float(values.get("top_p", 1.0)),
        top_k=int(values.get("top_k", 0)),
        repetition_penalty=float(values.get("repetition_penalty", 1.0)),
        max_new_tokens=int(values.get("max_new_tokens", 2048)),
        do_sample=bool(values.get("do_sample", False)),
        use_cache=bool(values.get("use_cache", True)),
        enable_thinking=bool(values.get("enable_thinking", False)),
        seed=-1,
        no_repeat_ngram_size=int(values.get("no_repeat_ngram_size", 0)),
    )


def _default_pre(family: str) -> PreprocessParams:
    limits = MODEL_SPECS[family].limits
    return PreprocessParams(
        fps=limits.default_fps,
        max_frames=int(limits.max_frames or 768),
        max_pixels=limits.default_max_pixels,
        min_pixels=limits.min_pixels,
        use_audio_in_video="video_audio" in MODEL_SPECS[family].capabilities,
    )


def _infer_part(media: MediaInput) -> MediaPart:
    if media.kind == "text" or (media.path is None and media.text is not None):
        return MediaPart("text", media.path, media.text, media.start, media.end)
    if media.path is None:
        raise ValueError("MediaInput needs a path, text, or mixed-media parts")
    info = probe_media(media.path)
    mapping = {
        "video": "video_audio",
        "video_no_audio": "video",
        "audio": "audio",
        "image": "image",
        "text": "text",
    }
    kind = media.kind or mapping.get(info.kind)
    if kind is None:
        raise ValueError(f"Unsupported or unreadable media {media.path}: {info.error or info.kind}")
    return MediaPart(kind, media.path, media.text, media.start, media.end)


def _parts(media: MediaInput) -> list[MediaPart]:
    return list(media.parts) if media.parts else [_infer_part(media)]


def _slice_audio(
    samples: np.ndarray,
    start: float | None,
    end: float | None,
    sample_rate: int = 16_000,
) -> np.ndarray:
    rate = max(1, int(sample_rate))
    first = min(len(samples), max(0, int(round(float(start or 0.0) * rate))))
    last = (
        len(samples)
        if end is None
        else min(len(samples), max(first, int(round(float(end) * rate))))
    )
    return np.ascontiguousarray(samples[first:last], dtype=np.float32)


def _read_model_audio(
    path: str | Path,
    start: float | None,
    end: float | None,
) -> np.ndarray:
    info = probe_media(path)
    sample_rate = max(1, int(getattr(info, "audio_sample_rate", None) or 16_000))
    try:
        samples = read_audio(path, sample_rate=sample_rate)
    except TypeError:
        # Some lightweight third-party/test decoders expose the older one-argument API.
        samples = read_audio(path)
    sliced = _slice_audio(samples, start, end, sample_rate)
    return resample_audio(sliced, sample_rate, 16_000)


def _sampling_seed(gen: GenParams, callbacks: Callbacks, scope: str) -> int | None:
    if not gen.do_sample:
        return None
    if gen.temperature <= 0:
        warning = "Sampling is enabled but temperature is 0; greedy decoding is used."
        get_log().warn(warning, scope=scope)
        _callback(callbacks.progress, warning, level="warning")
        return None
    return int(gen.seed) if int(gen.seed) >= 0 else secrets.randbits(32)


def _seed_torch(torch: Any, seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "manual_seed_all", None)):
        cuda.manual_seed_all(seed)


def _text_for_part(part: MediaPart) -> str:
    if part.text is not None:
        return str(part.text)
    if part.path is None:
        return ""
    try:
        return Path(part.path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(part.path).read_text(encoding="utf-8", errors="replace")


class _Prepared:
    def __init__(self) -> None:
        self.content: list[dict[str, Any]] = []
        self.audio: list[np.ndarray] = []
        self.images: list[Image.Image] = []
        self.videos: list[np.ndarray] = []
        self.text_parts: list[str] = []
        self.fps_values: list[float] = []
        self.video_infos: list[MediaInfo] = []
        self.warnings: list[str] = []
        self.use_audio_in_video = False


class OmniCaptionerBase(BaseCaptioner):
    """Full shared caption path for Qwen2.5-Omni and Qwen3-Omni thinkers."""

    size_multiple = 28
    force_video_audio = False
    avocado_mode = False
    thinking_mode = False
    captioner_mode = False

    def __init__(self, family: str, loaded: Any | None = None) -> None:
        super().__init__(family, loaded)
        self.size_multiple = self.spec.limits.size_multiple

    def _validate_loaded(self) -> None:
        if self.loaded is None or getattr(self.loaded, "model", None) is None:
            raise RuntimeError("The captioner model is not loaded")
        if self.loaded.spec.family != self.spec.family:
            raise RuntimeError(
                f"Loaded family {self.loaded.spec.family} does not match wrapper {self.spec.family}"
            )

    def _validate_capabilities(self, parts: Iterable[MediaPart]) -> None:
        part_list = list(parts)
        if self.captioner_mode:
            if len(part_list) != 1 or part_list[0].type != "audio":
                raise ValueError("Qwen3-Omni Captioner accepts exactly one audio input and no other media")
            return
        for part in part_list:
            kind = part.type
            if self.force_video_audio and kind in {"video", "video_audio"}:
                kind = "video_audio"
            if kind not in self.spec.capabilities:
                alternatives = [
                    spec.label for spec in MODEL_SPECS.values() if kind in spec.capabilities
                ]
                suffix = f" Models that support it: {', '.join(alternatives)}." if alternatives else ""
                raise ValueError(f"{self.spec.label} does not support {kind} input.{suffix}")
        if self.spec.family in {"timechat", "avocado"}:
            videos = [part for part in part_list if part.type in {"video", "video_audio"}]
            if len(videos) != 1 or len(part_list) != 1:
                raise ValueError(f"{self.spec.label} accepts one video per caption call")

    def _effective_bounds(self, part: MediaPart, pre: PreprocessParams) -> tuple[float | None, float | None]:
        return (
            part.start if part.start is not None else pre.start,
            part.end if part.end is not None else pre.end,
        )

    def _prepare_media(self, parts: list[MediaPart], pre: PreprocessParams) -> _Prepared:
        prepared = _Prepared()
        requested_video_audio = bool(pre.use_audio_in_video)
        video_records: list[tuple[MediaPart, MediaInfo]] = []
        for part in parts:
            if part.type in {"video", "video_audio"}:
                if part.path is None:
                    raise ValueError("Video parts require a path")
                info = probe_media(part.path)
                if not info.has_video:
                    raise ValueError(f"No video stream found in {part.path}")
                video_records.append((part, info))
        if self.force_video_audio:
            prepared.use_audio_in_video = bool(video_records)
        elif video_records:
            prepared.use_audio_in_video = requested_video_audio and all(info.has_audio for _, info in video_records)
        if requested_video_audio and video_records and not prepared.use_audio_in_video:
            warning = "A video has no audio track; falling back to visual-only video processing."
            prepared.warnings.append(warning)
            get_log().warn(warning, scope="caption")

        for part in parts:
            start, end = self._effective_bounds(part, pre)
            if part.type in {"video", "video_audio"}:
                assert part.path is not None
                info = next(info for record, info in video_records if record is part)
                duration = (end if end is not None else info.duration or 0.0) - float(start or 0.0)
                if self.spec.limits.max_duration_s is not None and duration > self.spec.limits.max_duration_s + 1e-6:
                    if self.captioner_mode:
                        warning = f"Audio exceeds the recommended {self.spec.limits.max_duration_s:g}s detail window."
                        prepared.warnings.append(warning)
                        get_log().warn(warning, scope="caption")
                    else:
                        raise ValueError(
                            f"{self.spec.label} supports clips up to {self.spec.limits.max_duration_s:g}s; split this input first"
                        )
                frame_max_pixels = int(pre.max_pixels)
                if pre.sampling_strategy in {"uniform", "keyframe"}:
                    estimated_frames = int(pre.max_frames)
                else:
                    estimated_frames = max(
                        4,
                        int(math.floor(max(0.0, duration) * pre.fps / 2.0)) * 2,
                    )
                estimated_frames = min(estimated_frames, pre.max_frames)
                if info.nb_frames:
                    estimated_frames = min(estimated_frames, info.nb_frames)
                total_pixel_cap = int(pre.total_pixel_cap or 20_070_400)
                family_frame_cap = 602_112 if self.size_multiple == 28 else frame_max_pixels
                adaptive_cap = max(
                    min(
                        family_frame_cap,
                        int(total_pixel_cap / max(1, estimated_frames) * 2),
                    ),
                    int((pre.min_pixels or self.spec.limits.min_pixels) * 1.05),
                )
                reduced_pixels = min(frame_max_pixels, adaptive_cap)
                if reduced_pixels < frame_max_pixels:
                    warning = (
                        f"Total pixel cap {total_pixel_cap:,} reduced per-frame pixels from "
                        f"{frame_max_pixels:,} to {reduced_pixels:,} for {part.path}."
                    )
                    prepared.warnings.append(warning)
                    get_log().warn(warning, scope="caption")
                frame_max_pixels = reduced_pixels
                frames = read_video_frames(
                    part.path,
                    start=start,
                    end=end,
                    target_fps=pre.fps,
                    max_frames=pre.max_frames,
                    min_frames=4,
                    max_pixels=frame_max_pixels,
                    min_pixels=pre.min_pixels or self.spec.limits.min_pixels,
                    size_multiple=self.size_multiple,
                    sampling=pre.sampling_strategy,
                    adaptive_threshold=float(pre.adaptive_threshold),
                    round_frames_to=2,
                )
                prepared.content.append({"type": "video"})
                prepared.videos.append(frames.frames)
                prepared.fps_values.append(float(frames.fps_effective or pre.fps))
                prepared.video_infos.append(info)
                if prepared.use_audio_in_video:
                    if info.has_audio:
                        samples = _read_model_audio(part.path, start, end)
                    else:
                        sample_count = max(1, int(round(max(duration, 0.01) * 16_000)))
                        samples = np.zeros(sample_count, dtype=np.float32)
                        warning = "TimeChat input had no audio track; a matching silent waveform was supplied."
                        prepared.warnings.append(warning)
                        get_log().warn(warning, scope="caption")
                    prepared.audio.append(samples)
            elif part.type == "audio":
                if part.path is None:
                    raise ValueError("Audio parts require a path")
                info = probe_media(part.path)
                duration = (end if end is not None else info.duration or 0.0) - float(start or 0.0)
                if self.captioner_mode and duration > 30.0:
                    warning = "Qwen3-Omni Captioner works best with audio no longer than 30 seconds."
                    prepared.warnings.append(warning)
                    get_log().warn(warning, scope="caption")
                prepared.content.append({"type": "audio"})
                prepared.audio.append(_read_model_audio(part.path, start, end))
            elif part.type == "image":
                if part.path is None:
                    raise ValueError("Image parts require a path")
                with Image.open(part.path) as image:
                    prepared.images.append(ImageOps.exif_transpose(image).convert("RGB").copy())
                prepared.content.append({"type": "image"})
            elif part.type == "text":
                text = _text_for_part(part).strip()
                if text:
                    prepared.text_parts.append(text)
        return prepared

    def _prompt(
        self,
        prompt: PromptSpec | str | None,
        prepared: _Prepared,
        callbacks: Callbacks | None = None,
    ) -> tuple[str | None, str, PromptPreset | None]:
        if isinstance(prompt, str):
            prompt = PromptSpec(user_prompt=prompt)
        prompt = prompt or PromptSpec()
        preset_id = prompt.preset_id
        if preset_id is None:
            media_types = [str(item.get("type")) for item in prepared.content]
            if self.spec.family.startswith("qwen3_") and media_types == ["audio"]:
                preset_id = "qwen3_captioner_promptfree" if self.captioner_mode else "qwen3_audio_caption"
            elif self.spec.family.startswith("qwen3_") and media_types and all(value == "image" for value in media_types):
                preset_id = "qwen3_image_describe"
            elif self.spec.family.startswith("qwen3_") and len(set(media_types)) > 1:
                preset_id = "qwen3_joint_describe"
            elif self.spec.family.startswith("qwen3_") and not media_types and prepared.text_parts:
                preset_id = None
            else:
                preset_id = self.spec.default_prompt_preset
        if self.avocado_mode and not prepared.use_audio_in_video and prompt.preset_id is None and prompt.user_prompt is None:
            preset_id = "avocado_visual_only"
        preset: PromptPreset | None = get_preset(preset_id) if preset_id else None
        system, user = render_prompt(preset, prompt.variables) if preset else (None, "")
        if prompt.system_prompt is not None:
            system = prompt.system_prompt
        if prompt.user_prompt is not None:
            user = prompt.user_prompt
        if self.captioner_mode:
            if (system or "").strip() or (user or "").strip() or prepared.text_parts:
                warning = "Qwen3-Omni Captioner is prompt-free; ignoring the provided prompt text"
                prepared.warnings.append(warning)
                get_log().warn(warning, scope="caption")
                if callbacks is not None:
                    _callback(callbacks.progress, warning, level="warning")
            return None, "", preset
        text_parts = [value for value in prepared.text_parts if value]
        if user:
            text_parts.append(user)
        return system, "\n\n".join(text_parts).strip(), preset

    def _conversation(
        self,
        prepared: _Prepared,
        system: str | None,
        user: str,
        gen: GenParams,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        conversation: list[dict[str, Any]] = []
        if system:
            conversation.append({"role": "system", "content": [{"type": "text", "text": system}]})
        content = list(prepared.content)
        if user:
            content.append({"type": "text", "text": user})
        if not content:
            raise ValueError("The user message is empty")
        conversation.append({"role": "user", "content": content})
        template_kwargs: dict[str, Any] = {}
        if self.thinking_mode:
            template_kwargs["enable_thinking"] = bool(gen.enable_thinking)
        return conversation, template_kwargs

    def _processor_inputs(
        self,
        prepared: _Prepared,
        text: str,
        pre: PreprocessParams,
    ) -> Any:
        processor = self.loaded.processor
        kwargs: dict[str, Any] = {
            "text": text,
            "audio": prepared.audio or None,
            "images": prepared.images or None,
            "videos": prepared.videos or None,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": prepared.use_audio_in_video,
        }
        if prepared.videos:
            fps = prepared.fps_values[0] if prepared.fps_values else pre.fps
            if any(abs(value - fps) > max(0.05, fps * 0.05) for value in prepared.fps_values[1:]):
                warning = "Mixed videos produced different effective frame rates; using the first video's scalar fps."
                prepared.warnings.append(warning)
                get_log().warn(warning, scope="caption")
            kwargs["fps"] = float(fps)
            kwargs["cap_pixels_per_frame"] = False
        if self.size_multiple == 32 and (prepared.images or prepared.videos):
            kwargs["size"] = {
                "shortest_edge": int(pre.min_pixels or self.spec.limits.min_pixels),
                "longest_edge": int(pre.max_pixels),
            }
        try:
            return processor(**kwargs)
        except TypeError as exc:
            if "cap_pixels_per_frame" not in str(exc):
                raise
            kwargs.pop("cap_pixels_per_frame", None)
            return processor(**kwargs)

    def _stopping(self, prompt_tokens: int, callbacks: Callbacks, started: float) -> tuple[Any, Any]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from vcap.core.progress import UiThrottle

        meter = TokenSpeedMeter()
        state = {"first": None, "tokens": 0, "speed": 0.0}
        key = f"caption-generate-{id(state)}"
        throttle = UiThrottle(0.1)
        stop_result = torch.zeros(1, dtype=torch.bool, device=self.loaded.device)

        class ConsoleProgressStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                del scores, kwargs
                count = max(0, int(input_ids.shape[-1]) - prompt_tokens)
                if count > 0 and state["first"] is None:
                    state["first"] = time.perf_counter()
                    meter.start()
                state["tokens"] = count
                state["speed"] = meter.update(max(0, count - 1))
                stop = _cancelled(callbacks.cancel)
                if throttle.should_emit(force=count <= 1 or stop):
                    message = f"Generating: {count} tokens | {state['speed']:.2f} tok/s"
                    console_progress.show_progress_line(message, key=key)
                    _callback(
                        callbacks.progress,
                        message,
                        new_tokens=count,
                        tok_per_s=state["speed"],
                    )
                if stop:
                    stop_result.fill_(True)
                return stop_result

        return StoppingCriteriaList([ConsoleProgressStoppingCriteria()]), (state, key, started)

    def _postprocess(self, raw: str, preset: PromptPreset | None) -> tuple[PostResult, str]:
        reasoning = ""
        if self.thinking_mode:
            reasoning, answer = split_thinking(raw)
            raw_for_post = answer
        else:
            raw_for_post = raw
        processor_name = preset.post_processor if preset and preset.post_processor else "plain"
        processor = POST_PROCESSORS.get(processor_name, plain)
        post = processor(raw_for_post, {})
        if self.spec.family == "timechat":
            native = timechat_parse(raw)
            post = PostResult(
                text=post.text,
                structured=post.structured if post.structured is not None else native.structured,
                segments=post.segments,
            )
        return post, reasoning

    def _context_limit(self, gen: GenParams) -> int:
        """Return the effective context window: the request capped by the model."""

        cap = int(self.spec.limits.context_tokens)
        requested = getattr(gen, "context_tokens", None)
        return min(cap, int(requested)) if requested else cap

    def _render_chat(
        self,
        history: Sequence[ChatMessage | Mapping[str, Any]],
        media_content: Sequence[Mapping[str, Any]],
        system_prompt: str | None,
        gen: GenParams,
        turn_media: Any = None,
    ) -> tuple[list[dict[str, Any]], str, int]:
        conversation = build_chat_conversation(
            history, media_content, system_prompt, turn_media=turn_media
        )
        template_kwargs: dict[str, Any] = {}
        if self.thinking_mode:
            template_kwargs["enable_thinking"] = bool(gen.enable_thinking)
        processor = self.loaded.processor
        rendered = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        try:
            encoded = tokenizer(rendered, add_special_tokens=False)
            token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
            if hasattr(token_ids, "shape"):
                tokens = int(token_ids.shape[-1])
            elif token_ids and isinstance(token_ids[0], (list, tuple)):
                tokens = len(token_ids[0])
            else:
                tokens = len(token_ids or [])
        except Exception:
            try:
                tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            except Exception:
                tokens = max(1, len(rendered) // 4)
        return conversation, str(rendered), max(0, int(tokens))

    def _stream_chat_generate(
        self,
        inputs: Any,
        prepared: _Prepared,
        generation: dict[str, Any],
        callbacks: Callbacks,
        input_length: int,
        seed: int | None,
    ) -> tuple[Any, dict[str, Any], float, float]:
        import torch
        from transformers import TextIteratorStreamer

        processor = self.loaded.processor
        tokenizer = getattr(processor, "tokenizer", processor)
        started = time.perf_counter()
        stopping, timing_state = self._stopping(input_length, callbacks, started)
        state, progress_key, _ = timing_state
        generation = {**generation, "stopping_criteria": stopping}
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            timeout=0.1,
        )
        shared: dict[str, Any] = {}

        def generate() -> None:
            try:
                _, runtime_context = resolve_attention(
                    self.loaded.attention,
                    self.spec.family,
                    self.loaded.dtype,
                )
                with torch.inference_mode(), runtime_context:
                    _seed_torch(torch, seed)
                    shared["output"] = self.loaded.model.generate(
                        **inputs,
                        use_audio_in_video=prepared.use_audio_in_video,
                        streamer=streamer,
                        **generation,
                    )
            except BaseException as exc:
                shared["error"] = exc
                end = getattr(streamer, "end", None)
                if callable(end):
                    try:
                        end()
                    except Exception:
                        pass

        thread = threading.Thread(target=generate, name="vcap-chat-generate", daemon=False)
        emitted = False
        thread.start()
        try:
            iterator = iter(streamer)
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue
                text = str(chunk)
                if text:
                    emitted = True
                    _delta_callback(callbacks.delta, text)
        finally:
            thread.join()
            console_progress.finalize_progress_line(key=progress_key)
        error = shared.get("error")
        if error is not None:
            try:
                setattr(error, "_vcap_chat_streamed", emitted)
            except Exception:
                pass
            raise error
        if "output" not in shared:
            raise RuntimeError("Chat generation ended without a model output")
        return shared["output"], state, started, time.perf_counter()

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
        """Stream one assistant turn while retaining Qwen3 multimodal history."""

        self._validate_loaded()
        if self.captioner_mode:
            raise ValueError(
                "Qwen3-Omni Captioner does not support chat; pick Qwen3-Omni Instruct or Thinking."
            )
        callbacks = cb or Callbacks()
        generation_params = gen or _default_gen(self.spec.family)
        preprocessing = pre or _default_pre(self.spec.family)
        max_tokens = min(
            int(generation_params.max_new_tokens),
            self.spec.limits.max_new_tokens_cap,
        )
        generation_params = replace(generation_params, max_new_tokens=max_tokens)
        normalized = normalize_chat_history(history)
        conversational = [item for item in normalized if item.role != "system"]
        if not conversational or conversational[-1].role != "user":
            raise ValueError("Chat history must end with the current user message")
        if self.spec.family in {"timechat", "avocado"}:
            if sum(item.role == "user" for item in conversational) != 1 or any(
                item.role == "assistant" for item in conversational
            ):
                raise ValueError(f"{self.spec.label} supports single-turn video Q&A only")
        # Legacy first-turn media plus each user turn's own attachments, probed once.
        legacy_parts = _parts(media) if media is not None else []
        part_by_path: dict[str, MediaPart] = {}
        for item in normalized:
            for raw, part in zip(item.media, chat_media_parts(item.media)):
                part_by_path[raw] = part
        all_parts = legacy_parts + [part_by_path[raw] for item in normalized for raw in item.media]
        if all_parts:
            self._validate_capabilities(all_parts)
        if self.spec.family in {"timechat", "avocado"} and (
            len(all_parts) != 1 or all_parts[0].type not in {"video", "video_audio"}
        ):
            raise ValueError(f"{self.spec.label} chat requires exactly one video")
        first_turn_content = chat_media_placeholders(legacy_parts)

        def turn_media(index: int, message: ChatMessage) -> list[dict[str, str]]:
            del index
            return chat_media_placeholders([part_by_path[raw] for raw in message.media])

        context_limit = self._context_limit(generation_params)

        def count_tokens(candidate: Sequence[ChatMessage]) -> int:
            return self._render_chat(
                candidate, first_turn_content, system_prompt, generation_params, turn_media
            )[2]

        retained, dropped_turns, rendered_tokens = truncate_chat_history(
            normalized,
            count_tokens,
            context_limit,
        )
        # Decode media only for the turns that survived truncation, in the order
        # the template will reference them (legacy media leads the first turn).
        retained_parts = list(legacy_parts) + [
            part_by_path[raw] for item in retained for raw in item.media
        ]
        prepared = self._prepare_media(retained_parts, preprocessing)
        if dropped_turns:
            warning = (
                f"Context limit: dropped {dropped_turns} oldest conversation "
                f"turn{'s' if dropped_turns != 1 else ''}; the first turn was kept."
            )
            prepared.warnings.append(warning)
            get_log().warn(warning, scope="chat")
            _callback(
                callbacks.progress,
                warning,
                level="warning",
                dropped_turns=dropped_turns,
                context_trimmed=True,
            )
        _, rendered, rendered_tokens = self._render_chat(
            retained,
            first_turn_content,
            system_prompt,
            generation_params,
            turn_media,
        )
        if rendered_tokens > int(context_limit * 0.9):
            warning = (
                f"Rendered chat prompt uses about {rendered_tokens} of "
                f"{context_limit} context tokens."
            )
            prepared.warnings.append(warning)
            _callback(callbacks.progress, warning, level="warning")
        _callback(callbacks.progress, "Preprocessing chat inputs")
        inputs = self._processor_inputs(prepared, rendered, preprocessing)

        import torch

        device = torch.device(self.loaded.device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        inputs = inputs.to(device)
        inputs = inputs.to(self.loaded.dtype)
        input_length = int(inputs["input_ids"].shape[-1])
        prompt_tokens = (
            int(inputs["attention_mask"].sum().item())
            if "attention_mask" in inputs
            else input_length
        )
        remaining = context_limit - prompt_tokens
        if remaining <= 0:
            raise ValueError(
                f"The media and retained conversation use {prompt_tokens} tokens, exceeding the "
                f"{context_limit}-token context window."
            )
        if generation_params.max_new_tokens > remaining:
            warning = f"Reduced this response to {remaining} tokens to stay within the context window."
            prepared.warnings.append(warning)
            generation_params = replace(generation_params, max_new_tokens=remaining)
            _callback(callbacks.progress, warning, level="warning")
        generation: dict[str, Any] = {
            "max_new_tokens": int(generation_params.max_new_tokens),
            "do_sample": bool(generation_params.do_sample and generation_params.temperature > 0),
            "repetition_penalty": float(generation_params.repetition_penalty),
            "use_cache": bool(generation_params.use_cache),
        }
        generation_config = getattr(self.loaded.model, "generation_config", None)
        eos_token_ids = getattr(generation_config, "eos_token_id", None)
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        resolved_pad_token_id = getattr(generation_config, "pad_token_id", None)
        if not eos_token_ids or not isinstance(resolved_pad_token_id, int) or resolved_pad_token_id < 0:
            from .loader import resolve_stop_token_ids

            resolved_eos, fallback_pad = resolve_stop_token_ids(self.loaded.processor)
            eos_token_ids = eos_token_ids or resolved_eos
            if not isinstance(resolved_pad_token_id, int) or resolved_pad_token_id < 0:
                resolved_pad_token_id = fallback_pad
        generation["eos_token_id"] = [int(value) for value in eos_token_ids]
        generation["pad_token_id"] = int(resolved_pad_token_id)
        if generation["do_sample"]:
            generation.update(
                temperature=float(generation_params.temperature),
                top_p=float(generation_params.top_p),
                top_k=int(generation_params.top_k),
            )
        actual_seed = _sampling_seed(generation_params, callbacks, "chat")
        try:
            output, state, started, ended = self._stream_chat_generate(
                inputs,
                prepared,
                generation,
                callbacks,
                input_length,
                actual_seed,
            )
        except Exception as exc:
            if (
                self.loaded.attention != "flash_attention_2"
                or not is_flash_attention_failure(exc)
                or bool(getattr(exc, "_vcap_chat_streamed", False))
            ):
                raise
            warning = f"FlashAttention 2 chat forward failed ({exc}); retrying with SDPA."
            prepared.warnings.append(warning)
            get_log().warn(warning, scope="attention")
            _callback(callbacks.progress, warning, level="warning")
            setter = getattr(self.loaded.model, "set_attn_implementation", None)
            if not callable(setter):
                raise RuntimeError("The loaded model cannot switch from FlashAttention 2 to SDPA") from exc
            setter("sdpa")
            self.loaded.attention = "sdpa"
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
            output, state, started, ended = self._stream_chat_generate(
                inputs,
                prepared,
                generation,
                callbacks,
                input_length,
                actual_seed,
            )
        _record_generation_memory(
            self.loaded,
            self.loaded.model,
            torch,
            device,
            scope="chat",
        )
        sequences = getattr(output, "sequences", output)
        new_ids = sequences[:, input_length:]
        raw = self.loaded.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        new_tokens = int(new_ids.shape[-1])
        terminal_token = int(new_ids[0, -1].item()) if new_tokens else None
        cancelled = _cancelled(callbacks.cancel)
        if cancelled:
            finish_reason = "cancelled"
        elif terminal_token in set(generation["eos_token_id"]):
            finish_reason = "eos"
        else:
            finish_reason = "length"
        reasoning, answer = split_thinking(raw) if self.thinking_mode else ("", raw)
        first = state["first"] or ended
        prefill = max(0.0, float(first - started))
        decode = max(0.0, float(ended - first))
        speed = (
            max(0, new_tokens - 1) / max(decode, 1e-9)
            if decode > 0
            else float(state["speed"] or 0.0)
        )
        peak = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if torch.cuda.is_available() and device.type == "cuda"
            else 0.0
        )
        elapsed = max(0.0, ended - started)
        detail = (
            "stopped by EOS"
            if finish_reason == "eos"
            else f"reached max_new_tokens {generation_params.max_new_tokens}"
            if finish_reason == "length"
            else "cancelled"
        )
        message = f"Chat finished: {new_tokens} new tokens in {elapsed:.1f}s ({detail})"
        get_log().log(message, scope="chat")
        _callback(
            callbacks.progress,
            message,
            new_tokens=new_tokens,
            tok_per_s=speed,
            cancelled=cancelled,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            context_limit=context_limit,
        )
        return ChatResult(
            text=answer,
            raw_text=raw,
            reasoning=reasoning,
            usage=TokenUsage(prompt_tokens, new_tokens, finish_reason, actual_seed),
            timing=CaptionTiming(prefill, decode, speed, elapsed, new_tokens),
            peak_vram_gb=float(peak),
            cancelled=cancelled,
            warnings=tuple(prepared.warnings),
            retained_history=tuple(retained),
            dropped_turns=dropped_turns,
            context_tokens=prompt_tokens,
            context_limit=context_limit,
        )

    def caption(
        self,
        media: MediaInput,
        prompt: PromptSpec | str | None = None,
        gen: GenParams | None = None,
        pre: PreprocessParams | None = None,
        cb: Callbacks | None = None,
    ) -> CaptionResult:
        """Decode media, process multimodal tensors, generate, and normalize output."""

        self._validate_loaded()
        pre = pre or _default_pre(self.spec.family)
        gen = gen or _default_gen(self.spec.family)
        cb = cb or Callbacks()
        parts = _parts(media)
        if (
            pre.max_frames == 0
            and self.spec.family.startswith("qwen3_omni_")
            and any(part.type in {"video", "video_audio"} for part in parts)
        ):
            converted: list[MediaPart] = []
            for part in parts:
                if part.type not in {"video", "video_audio"}:
                    converted.append(part)
                    continue
                if part.path is None or not probe_media(part.path).has_audio:
                    raise ValueError("Visual frames are disabled, but the video has no audio track")
                converted.append(replace(part, type="audio"))
            parts = converted
            message = "Visual frames disabled (Maximum frames = 0): captioning the audio track only."
            get_log().log(message, scope="preprocess")
            _callback(cb.progress, message)
        self._validate_capabilities(parts)
        max_tokens = min(int(gen.max_new_tokens), self.spec.limits.max_new_tokens_cap)
        if max_tokens != gen.max_new_tokens:
            gen = replace(gen, max_new_tokens=max_tokens)
        prepared = self._prepare_media(parts, pre)
        system, user, preset = self._prompt(prompt, prepared, cb)
        conversation, template_kwargs = self._conversation(prepared, system, user, gen)
        processor = self.loaded.processor
        rendered = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        _callback(cb.progress, "Preprocessing model inputs")
        inputs = self._processor_inputs(prepared, rendered, pre)

        import torch

        device = torch.device(self.loaded.device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        inputs = inputs.to(device)
        inputs = inputs.to(self.loaded.dtype)
        input_length = int(inputs["input_ids"].shape[-1])
        prompt_tokens = (
            int(inputs["attention_mask"].sum().item())
            if "attention_mask" in inputs
            else input_length
        )
        generation_started = time.perf_counter()
        stopping, timing_state = self._stopping(input_length, cb, generation_started)
        state, progress_key, _ = timing_state
        generation: dict[str, Any] = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": bool(gen.do_sample and gen.temperature > 0),
            "repetition_penalty": float(gen.repetition_penalty),
            "use_cache": bool(gen.use_cache),
            "stopping_criteria": stopping,
        }
        if int(gen.no_repeat_ngram_size) > 0:
            generation["no_repeat_ngram_size"] = int(gen.no_repeat_ngram_size)
        generation_config = getattr(self.loaded.model, "generation_config", None)
        eos_token_ids = getattr(generation_config, "eos_token_id", None)
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if not eos_token_ids:
            from .loader import resolve_stop_token_ids

            eos_token_ids, resolved_pad_token_id = resolve_stop_token_ids(processor)
        else:
            resolved_pad_token_id = getattr(generation_config, "pad_token_id", None)
            if not isinstance(resolved_pad_token_id, int) or resolved_pad_token_id < 0:
                from .loader import resolve_stop_token_ids

                _, resolved_pad_token_id = resolve_stop_token_ids(processor)
        generation["eos_token_id"] = [int(value) for value in eos_token_ids]
        generation["pad_token_id"] = int(resolved_pad_token_id)
        if generation["do_sample"]:
            generation.update(
                temperature=float(gen.temperature),
                top_p=float(gen.top_p),
                top_k=int(gen.top_k),
            )
        model = self.loaded.model
        _, runtime_context = resolve_attention(self.loaded.attention, self.spec.family, self.loaded.dtype)
        actual_seed = _sampling_seed(gen, cb, "caption")
        try:
            try:
                with torch.inference_mode(), runtime_context:
                    _seed_torch(torch, actual_seed)
                    output = model.generate(
                        **inputs,
                        use_audio_in_video=prepared.use_audio_in_video,
                        **generation,
                    )
            except Exception as exc:
                if self.loaded.attention != "flash_attention_2" or not is_flash_attention_failure(exc):
                    raise
                warning = f"FlashAttention 2 forward failed ({exc}); retrying this generation with SDPA."
                get_log().warn(warning, scope="attention")
                _callback(cb.progress, warning, level="warning")
                setter = getattr(model, "set_attn_implementation", None)
                if not callable(setter):
                    raise RuntimeError("The loaded model cannot switch from FlashAttention 2 to SDPA") from exc
                setter("sdpa")
                exc.__traceback__ = None
                self.loaded.attention = "sdpa"
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                stopping, timing_state = self._stopping(input_length, cb, generation_started)
                state, progress_key, _ = timing_state
                generation["stopping_criteria"] = stopping
                with torch.inference_mode():
                    _seed_torch(torch, actual_seed)
                    output = model.generate(
                        **inputs,
                        use_audio_in_video=prepared.use_audio_in_video,
                        **generation,
                    )
        finally:
            console_progress.finalize_progress_line(key=progress_key)
        _record_generation_memory(self.loaded, model, torch, device, scope="caption")
        ended = time.perf_counter()
        sequences = getattr(output, "sequences", output)
        new_ids = sequences[:, input_length:]
        raw = processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        new_tokens = int(new_ids.shape[-1])
        cancelled = _cancelled(cb.cancel)
        terminal_token = int(new_ids[0, -1].item()) if new_tokens else None
        if cancelled:
            finish_reason = "cancelled"
        elif terminal_token in set(generation["eos_token_id"]):
            finish_reason = "eos"
        else:
            finish_reason = "length"
        if self.avocado_mode and not cancelled:
            raw, cleanup_applied = trim_avocado_trailing_qa(
                raw,
                hit_token_cap=new_tokens >= int(gen.max_new_tokens),
            )
            if cleanup_applied:
                warning = (
                    "Trimmed an obvious trailing dataset-QA continuation from capped "
                    "AVoCaDO output."
                )
                prepared.warnings.append(warning)
                get_log().warn(warning, scope="caption")
        first = state["first"] or ended
        prefill = max(0.0, float(first - generation_started))
        decode = max(0.0, float(ended - first))
        decoded_after_first = max(0, new_tokens - 1)
        speed = decoded_after_first / max(decode, 1e-9) if decode > 0 else float(state["speed"] or 0.0)
        post, reasoning = self._postprocess(raw, preset)
        peak = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if torch.cuda.is_available() and device.type == "cuda"
            else 0.0
        )
        elapsed = ended - generation_started
        if finish_reason == "eos":
            finish_detail = "stopped by EOS"
        elif finish_reason == "length":
            finish_detail = f"reached max_new_tokens {int(gen.max_new_tokens)}"
        else:
            finish_detail = "cancelled"
        message = f"Generation finished: {new_tokens} new tokens in {elapsed:.1f}s ({finish_detail})"
        get_log().log(message, scope="caption")
        _callback(
            cb.progress,
            message,
            new_tokens=new_tokens,
            tok_per_s=speed,
            cancelled=cancelled,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            context_limit=self._context_limit(gen),
        )
        return CaptionResult(
            text=post.text,
            raw_text=raw,
            reasoning=reasoning,
            structured=post.structured,
            segments=list(post.segments),
            usage=TokenUsage(prompt_tokens, new_tokens, finish_reason, actual_seed),
            timing=CaptionTiming(prefill, decode, speed, elapsed, new_tokens),
            peak_vram_gb=float(peak),
            cancelled=cancelled,
            warnings=tuple(prepared.warnings),
        )


__all__ = ["OmniCaptionerBase", "build_chat_conversation", "split_thinking"]
