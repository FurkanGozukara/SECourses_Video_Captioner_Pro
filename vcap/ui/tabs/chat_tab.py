"""Interactive multimodal conversation tab using the Caption model worker."""

from __future__ import annotations

import html
import inspect
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gradio as gr

from vcap.core.media import probe_media
from vcap.core.paths import normalize_path
from vcap.core.subprocess_runner import CancelToken, CancelledError
from vcap.models.registry import MODEL_SPECS, get_variant, variant_to_family
from vcap.pipeline.chat import ChatRequest, ChatResponse, save_conversation
from vcap.prompts.presets import default_preset_for, get_preset, list_presets, render_prompt
from vcap.ui.components import action_button, context_usage_text

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


_INITIAL_STATE = {
    "messages": [],
    "media": [],
    "model_key": "",
    "system_prompt": "",
    "generation": {},
    "last_result": {},
}


@dataclass
class ChatTabHandles:
    chatbot: gr.Chatbot
    message: gr.Textbox
    files: gr.File
    path: gr.Textbox
    attachment_status: gr.Markdown
    model_note: gr.Markdown
    reasoning: gr.Textbox
    reasoning_accordion: gr.Accordion
    status: gr.Markdown
    tokens: gr.Markdown
    send: gr.Button
    stop: gr.Button
    clear: gr.Button
    copy_last: gr.Button
    save: gr.Button
    stop_timer: gr.Timer
    conversation_state: gr.State
    prompt_preset: gr.Dropdown
    prompt_helper: gr.Dropdown
    controls: dict[str, Any]


def model_chat_support(variant_key: str) -> tuple[str, str]:
    """Return the backend chat mode and its concise user-facing note."""

    family = variant_to_family(str(variant_key))
    name = html.escape(f"{MODEL_SPECS[family].label} · {get_variant(str(variant_key)).label}")
    if family in {"qwen3_omni_instruct", "qwen3_omni_thinking"}:
        return (
            "multi",
            f"<span class='vc-ok'><strong>{name}</strong> — multi-turn chat with video, audio, image, and text.</span>",
        )
    if family in {"timechat", "avocado"}:
        return (
            "single",
            f"<span class='vc-warn'><strong>{name}</strong> — video-only, single-turn Q&A. "
            "Each Send starts a fresh exchange with exactly one video.</span>",
        )
    return (
        "unsupported",
        f"<span class='vc-err'><strong>{name}</strong> has no chat mode. Pick Qwen3-Omni Instruct or Thinking "
        "in the Caption tab, or load a Chat preset.</span>",
    )


def _uploaded_paths(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else ([] if value is None else [value])
    result: list[str] = []
    for item in values:
        raw = getattr(item, "path", getattr(item, "name", item))
        if raw:
            result.append(str(raw))
    return result


def resolve_chat_attachments(files: Any, path_text: str) -> list[str]:
    """Resolve uploaded files and one-path-per-line text into unique media paths."""

    raw_paths = _uploaded_paths(files)
    raw_paths.extend(
        value.strip()
        for value in re.split(r"[\r\n]+", str(path_text or ""))
        if value.strip()
    )
    result: list[str] = []
    for raw in raw_paths:
        path = normalize_path(raw, must_exist=True)
        if not path.is_file():
            raise ValueError(f"Attachment is not a file: {path}")
        info = probe_media(path)
        if info.kind not in {"video", "video_no_audio", "audio", "image"}:
            raise ValueError(f"Unsupported attachment {path.name}: choose video, audio, or image media.")
        value = str(path)
        if value not in result:
            result.append(value)
    return result


def _attachment_line(files: Any, path_text: str) -> str:
    try:
        paths = resolve_chat_attachments(files, path_text)
    except Exception as exc:
        return f"<span class='vc-err'>{html.escape(str(exc))}</span>"
    if not paths:
        return "<span class='vc-help'>No media attached.</span>"
    labels = [f"{html.escape(Path(value).name)} ({probe_media(value).kind})" for value in paths]
    return f"<span class='vc-ok'>Attached: {' · '.join(labels)}</span>"


def _thought_message(reasoning: str, *, done: bool, seconds: Any = None) -> dict[str, Any]:
    """Chatbot "thought" entry: an expandable block holding streamed or saved reasoning.

    Gradio opens a pending thought (with a spinner) so the reasoning is visible
    while it streams, and collapses it once it is marked done.
    """

    metadata: dict[str, Any] = {
        "title": "🧠 Reasoning" if done else "🧠 Thinking…",
        "status": "done" if done else "pending",
    }
    try:
        duration = float(seconds) if seconds is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        metadata["duration"] = round(duration, 1)
    return {"role": "assistant", "content": reasoning, "metadata": metadata}


def _is_thought(item: Mapping[str, Any] | None) -> bool:
    metadata = (item or {}).get("metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("title"))


def _chatbot_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Render history for the Chatbot: attachments under each turn, reasoning as a thought."""

    display: list[dict[str, Any]] = []
    for item in messages:
        role = str(item.get("role") or "assistant")
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "")
        names = [Path(str(value)).name for value in (item.get("media") or []) if str(value)]
        if names:
            content = f"{content}\n\n📎 {' · '.join(names)}".strip()
        reasoning = str(item.get("reasoning") or "").strip() if role == "assistant" else ""
        if reasoning:
            display.append(_thought_message(reasoning, done=True, seconds=item.get("reasoning_s")))
        display.append({"role": role, "content": content})
    return display


def _tokens_line(tokens: Any = "—", speed: Any = "—", used: Any = None, limit: Any = None) -> str:
    """Tokens, speed, and context statistics shown under the composer."""

    return f"**Tokens:** {tokens} · **Speed:** {speed} · **Context:** {context_usage_text(used, limit)}"


def _last_answer(history: Sequence[Mapping[str, Any]] | None) -> str:
    for item in reversed(list(history or [])):
        if str(item.get("role") or "") == "assistant" and not _is_thought(item):
            return str(item.get("content") or "")
    return ""


def chat_prompt_modality(paths: Sequence[str] | None) -> str:
    """Return the preset modality represented by the first chat attachment."""

    if not paths:
        return "text"
    info = probe_media(paths[0])
    if info.kind in {"video", "video_no_audio"}:
        return "video_audio" if info.has_audio else "video"
    return str(info.kind)


def chat_prompt_choices(
    variant_key: str,
    paths: Sequence[str] | None,
    current: str | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Filter chat task presets by selected family and attached media kind."""

    family = variant_to_family(str(variant_key))
    modality = chat_prompt_modality(paths)
    presets = list_presets(family, modality)
    choices = [(preset.label, preset.id) for preset in presets]
    ids = {preset.id for preset in presets}
    selected = str(current or "")
    if selected not in ids:
        try:
            selected = default_preset_for(family, modality).id
        except Exception:
            selected = presets[0].id if presets else ""
    return choices, selected or None


def validate_chat_prompt_preset(
    value: str,
    previous_valid: str | None,
    variant_key: str,
    paths: Sequence[str] | None,
) -> tuple[Any, str, Any]:
    """Reject custom chat preset text and restore the prior compatible preset."""

    choices, default = chat_prompt_choices(variant_key, paths, previous_valid)
    aliases = {preset_id: preset_id for _, preset_id in choices}
    aliases.update({label: preset_id for label, preset_id in choices})
    selected = aliases.get(str(value or ""))
    if selected is not None:
        return (gr.update(value=selected) if selected != str(value or "") else gr.skip()), selected, gr.skip()
    valid_ids = {preset_id for _, preset_id in choices}
    fallback = str(previous_valid or "")
    if fallback not in valid_ids:
        fallback = str(default or "")
    if not fallback:
        return gr.update(choices=[], value=None), "", "Unknown task preset; no compatible preset is available"
    label = next((label for label, preset_id in choices if preset_id == fallback), fallback)
    return gr.update(choices=choices, value=fallback), fallback, f"Unknown task preset; kept {label}"


def _model_family_or_empty(variant_key: Any) -> str:
    try:
        return variant_to_family(str(variant_key or ""))
    except KeyError:
        return ""


def chat_model_change_updates(
    variant_key: str,
    current_state: Mapping[str, Any] | None,
) -> tuple[Any, ...]:
    """Update model-specific chat controls, clearing only across model families."""

    try:
        get_variant(str(variant_key))
    except KeyError:
        return tuple(gr.skip() for _ in range(8))
    mode, note = model_chat_support(variant_key)
    family = variant_to_family(str(variant_key))
    thinking = family == "qwen3_omni_thinking"
    cap = MODEL_SPECS[family].limits.max_new_tokens_cap
    state = dict(current_state or _INITIAL_STATE)
    previous_family = _model_family_or_empty(state.get("model_key"))
    family_changed = bool(previous_family and previous_family != family)
    has_conversation = bool(state.get("messages"))
    if family_changed and has_conversation:
        conversation_outputs: tuple[Any, Any, Any, Any] = (
            [],
            dict(_INITIAL_STATE),
            "",
            gr.update(visible=False),
        )
        status = "<span class='vc-warn'>Model family changed; conversation cleared.</span>"
    else:
        conversation_outputs = (gr.skip(), gr.skip(), gr.skip(), gr.skip())
        status = (
            note
            if mode == "unsupported"
            else (
                "<span class='vc-ok'>Model variant updated; conversation kept because the model family is unchanged.</span>"
                if has_conversation and previous_family == family
                else "<span class='vc-ok'>Ready.</span>"
            )
        )
    return (
        note,
        gr.update(interactive=thinking),
        gr.update(info=f"Hard limit for the next assistant response; {MODEL_SPECS[family].label} caps it at {cap} tokens."),
        *conversation_outputs,
        status,
    )


def build(ctx: "UiContext") -> ChatTabHandles:
    """Render chat controls and register preset-owned generation parameters."""

    controls: dict[str, Any] = {}
    initial_variant = str(getattr(ctx.caption_handles.controls["model_key"], "value", "qwen3_omni_instruct_int4"))
    _, initial_note = model_chat_support(initial_variant)
    initial_prompt_choices, initial_prompt_id = chat_prompt_choices(initial_variant, [])
    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=560):
            chatbot_kwargs: dict[str, Any] = {
                "value": [],
                "label": "Conversation",
                "height": 520,
                "buttons": ["copy", "copy_all"],
                "feedback_options": None,
                "placeholder": "Start a conversation with the selected model.",
            }
            if "type" in inspect.signature(gr.Chatbot).parameters:
                chatbot_kwargs["type"] = "messages"
            chatbot = gr.Chatbot(
                **chatbot_kwargs,
            )
            with gr.Accordion("Reasoning", open=False, visible=False) as reasoning_accordion:
                reasoning = gr.Textbox(
                    label="Reasoning",
                    lines=10,
                    max_lines=18,
                    interactive=False,
                    buttons=["copy"],
                    elem_classes=["vc-mono"],
                )
            message = gr.Textbox(
                label="Message",
                placeholder="Ask about the attached media…",
                info="Enter sends, Shift+Enter adds a line. Attachments are sent with the turn they accompany.",
                lines=3,
                max_lines=8,
                autofocus=False,
                elem_id="vc_chat_message",
            )
            with gr.Row():
                send = action_button("➤ Send", "cyan", variant="primary", scale=3, elem_id="vc_chat_send")
                stop = action_button(
                    "⏹ Stop",
                    "red",
                    variant="stop",
                    scale=2,
                    interactive=False,
                )
                clear = action_button("⌫ Clear history", "orange", scale=2)
                copy_last = action_button("⧉ Copy last answer", "blue", scale=2)
                save = action_button("💾 Save conversation", "green", scale=2)
            status = gr.Markdown("<span class='vc-ok'>Ready.</span>", elem_classes=["vc-status"])
            tokens = gr.Markdown(_tokens_line(), elem_classes=["vc-help"])
        with gr.Column(scale=4, min_width=420):
            with gr.Column():
                gr.Markdown("### Media")
                with gr.Group():
                    files = gr.File(
                        file_count="multiple",
                        file_types=["video", "audio", "image"],
                        type="filepath",
                        label="Attachments",
                        height=112,
                    )
                path = gr.Textbox(
                    label="Media path",
                    placeholder="Paste a local path; use one path per line",
                    info="Quoted, mixed-separator, Unicode, video, audio, and image paths are supported.",
                    lines=2,
                )
                attachment_status = gr.Markdown(
                    "<span class='vc-help'>No media attached.</span>",
                    elem_classes=["vc-help"],
                )
            model_note = gr.Markdown(initial_note, elem_classes=["vc-status"])
            prompt_preset = gr.Dropdown(
                choices=initial_prompt_choices,
                value=initial_prompt_id,
                allow_custom_value=True,
                label="Task / prompt preset",
                info="Filtered to the chat model family and the first attached media kind.",
                elem_id="vc_chat_prompt_preset",
            )
            valid_prompt_preset = gr.State(initial_prompt_id or "")
            system_prompt = gr.Textbox(
                value="",
                label="System prompt",
                info="Optional instruction applied before the conversation history.",
                lines=4,
                max_lines=10,
                elem_classes=["vc-mono"],
            )
            controls["chat_system_prompt"] = ctx.reg(
                "chat_system_prompt",
                system_prompt,
                "",
                section="chat",
                description="Optional system instruction for interactive chat.",
                kind="str",
                in_preset=True,
                in_metadata=False,
            )
            prompt_helper = gr.Dropdown(
                choices=[
                    ("Do not insert a Caption prompt", "none"),
                    ("Use current Caption user prompt", "caption_user"),
                    ("Use current Caption system + user prompts", "caption_both"),
                ],
                value="none",
                label="Caption prompt as first message",
                info="Copies the current Caption-tab prompt into the chat composer.",
            )
            with gr.Accordion("Generation", open=True):
                temperature = gr.Slider(
                    0.0,
                    2.0,
                    value=0.2,
                    step=0.01,
                    label="Temperature",
                    info="Sampling randomness; zero uses deterministic greedy decoding.",
                )
                controls["chat_temperature"] = ctx.reg(
                    "chat_temperature",
                    temperature,
                    0.2,
                    section="chat",
                    description="Interactive chat sampling temperature.",
                    kind="float",
                    minimum=0.0,
                    maximum=2.0,
                    in_preset=True,
                    in_metadata=False,
                )
                with gr.Row():
                    top_p = gr.Slider(
                        0.0,
                        1.0,
                        value=0.95,
                        step=0.01,
                        label="Top-p",
                        info="Nucleus probability mass used when chat sampling is active.",
                    )
                    top_k = gr.Slider(
                        0,
                        200,
                        value=20,
                        step=1,
                        precision=0,
                        label="Top-k",
                        info="Maximum token candidates; zero leaves the candidate set unrestricted.",
                    )
                controls["chat_top_p"] = ctx.reg(
                    "chat_top_p",
                    top_p,
                    0.95,
                    section="chat",
                    description="Interactive chat nucleus probability mass.",
                    kind="float",
                    minimum=0.0,
                    maximum=1.0,
                    in_preset=True,
                    in_metadata=False,
                )
                controls["chat_top_k"] = ctx.reg(
                    "chat_top_k",
                    top_k,
                    20,
                    section="chat",
                    description="Interactive chat top-k candidate limit.",
                    kind="int",
                    minimum=0,
                    maximum=200,
                    in_preset=True,
                    in_metadata=False,
                )
                max_new_tokens = gr.Slider(
                    1,
                    8192,
                    value=1024,
                    step=1,
                    precision=0,
                    label="Maximum new tokens",
                    info="Hard limit for the next assistant response, capped by the selected model context.",
                )
                controls["chat_max_new_tokens"] = ctx.reg(
                    "chat_max_new_tokens",
                    max_new_tokens,
                    1024,
                    section="chat",
                    description="Maximum tokens generated for one chat response.",
                    kind="int",
                    minimum=1,
                    maximum=32768,
                    in_preset=True,
                    in_metadata=False,
                )
                repetition_penalty = gr.Slider(
                    0.5,
                    2.0,
                    value=1.0,
                    step=0.01,
                    label="Repetition penalty",
                    info="Repetition penalty for chat replies (1.0 = off).",
                    elem_id="vc_chat_repetition_penalty",
                )
                controls["chat_repetition_penalty"] = ctx.reg(
                    "chat_repetition_penalty",
                    repetition_penalty,
                    1.0,
                    section="chat",
                    description="Repetition penalty for chat replies; 1.0 disables the penalty.",
                    kind="float",
                    minimum=0.5,
                    maximum=2.0,
                    in_preset=True,
                    in_metadata=False,
                )
                with gr.Row():
                    seed = gr.Number(
                        value=-1,
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        precision=0,
                        label="Seed",
                        info=(
                            "Seed for sampled decoding; -1 draws a fresh random seed every run. Greedy decoding "
                            "(Sample tokens off) is deterministic without it. The seed actually used is written to metadata."
                        ),
                        elem_id="vc_chat_seed",
                    )
                    enable_thinking = gr.Checkbox(
                        value=False,
                        label="Enable thinking",
                        info="Streams Qwen3-Omni Thinking reasoning live as a thought block above the answer and keeps a copy in the Reasoning panel.",
                        interactive=False,
                    )
                controls["chat_seed"] = ctx.reg(
                    "chat_seed", seed, -1, section="chat",
                    description=(
                        "Seed for sampled decoding; -1 draws a fresh random seed every run. Greedy decoding "
                        "(Sample tokens off) is deterministic without it. The seed actually used is written to metadata."
                    ),
                    kind="int", minimum=-1, maximum=2147483647,
                    in_preset=True, in_metadata=False,
                )
                controls["chat_enable_thinking"] = ctx.reg(
                    "chat_enable_thinking",
                    enable_thinking,
                    False,
                    section="chat",
                    description="Allow reasoning during Qwen3-Omni Thinking chat responses.",
                    kind="bool",
                    in_preset=True,
                    in_metadata=False,
                )
            gr.Markdown(
                "The model comes from the Caption tab's Model variant. The system prompt and generation "
                "values above are saved and applied with the universal preset bar; the shipped Chat presets "
                "select a chat-capable Qwen3-Omni model.",
                elem_classes=["vc-help"],
            )
    stop_timer = gr.Timer(1.0)
    conversation_state = gr.State(dict(_INITIAL_STATE))
    handles = ChatTabHandles(
        chatbot=chatbot,
        message=message,
        files=files,
        path=path,
        attachment_status=attachment_status,
        model_note=model_note,
        reasoning=reasoning,
        reasoning_accordion=reasoning_accordion,
        status=status,
        tokens=tokens,
        send=send,
        stop=stop,
        clear=clear,
        copy_last=copy_last,
        save=save,
        stop_timer=stop_timer,
        conversation_state=conversation_state,
        prompt_preset=prompt_preset,
        prompt_helper=prompt_helper,
        controls=controls,
    )
    ctx.chat_handles = handles

    for event in (files.change, path.change):
        event(
            _attachment_line,
            inputs=[files, path],
            outputs=attachment_status,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    caption_model = ctx.caption_handles.controls["model_key"]

    def update_prompt_presets(
        file_value: Any,
        path_text: str,
        variant_key: str,
        current: str,
    ) -> tuple[Any, str]:
        try:
            paths = resolve_chat_attachments(file_value, path_text)
        except Exception:
            paths = []
        try:
            choices, selected = chat_prompt_choices(str(variant_key), paths, current)
        except KeyError:
            return gr.skip(), str(current or "")
        return gr.update(choices=choices, value=selected), str(selected or "")

    for event in (files.change, path.change, caption_model.change):
        event(
            update_prompt_presets,
            inputs=[files, path, caption_model, prompt_preset],
            outputs=[prompt_preset, valid_prompt_preset],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def guard_prompt_preset(
        value: str,
        previous_valid: str,
        file_value: Any,
        path_text: str,
        variant_key: str,
    ) -> tuple[Any, str, Any]:
        try:
            paths = resolve_chat_attachments(file_value, path_text)
        except Exception:
            paths = []
        return validate_chat_prompt_preset(value, previous_valid, variant_key, paths)

    prompt_preset.change(
        guard_prompt_preset,
        inputs=[prompt_preset, valid_prompt_preset, files, path, caption_model],
        outputs=[prompt_preset, valid_prompt_preset, status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    caption_variable_keys = [
        "trigger_word",
        "language",
        "source_language",
        "target_language",
        "caption_length",
        "avoid_list",
        "subject_class",
        "extra_instructions",
    ]
    caption_variable_components = [
        ctx.caption_handles.controls[key] for key in caption_variable_keys
    ]

    def apply_chat_prompt_preset(preset_id: str, *values: Any) -> tuple[str, str]:
        try:
            preset = get_preset(str(preset_id))
            variables = dict(
                zip(
                    (
                        "TRIGGER",
                        "LANGUAGE",
                        "SOURCE_LANGUAGE",
                        "TARGET_LANGUAGE",
                        "CAPTION_LENGTH",
                        "AVOID",
                        "SUBJECT_CLASS",
                        "EXTRA_INSTRUCTIONS",
                    ),
                    values,
                )
            )
            rendered_system, rendered_user = render_prompt(preset, variables)
            return rendered_system or "", rendered_user or ""
        except Exception as exc:
            if not isinstance(exc, KeyError):
                gr.Warning(f"Could not render chat prompt preset: {exc}")
            return gr.skip(), gr.skip()

    prompt_preset.select(
        apply_chat_prompt_preset,
        inputs=[prompt_preset, *caption_variable_components],
        outputs=[system_prompt, message],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    caption_model.change(
        chat_model_change_updates,
        inputs=[caption_model, conversation_state],
        outputs=[
            model_note,
            enable_thinking,
            max_new_tokens,
            chatbot,
            conversation_state,
            reasoning,
            reasoning_accordion,
            status,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def insert_caption_prompt(
        selection: str,
        caption_user: str,
        caption_system: str,
    ) -> tuple[Any, Any]:
        if selection == "caption_user":
            return str(caption_user or ""), gr.skip()
        if selection == "caption_both":
            return str(caption_user or ""), str(caption_system or "")
        return gr.skip(), gr.skip()

    prompt_helper.change(
        insert_caption_prompt,
        inputs=[
            prompt_helper,
            ctx.caption_handles.controls["user_prompt"],
            ctx.caption_handles.controls["system_prompt"],
        ],
        outputs=[message, system_prompt],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    return handles


def wire(ctx: "UiContext") -> None:
    """Wire streaming send/stop/history/save events after the registry is complete."""

    handles = ctx.chat_handles
    if handles is None:
        raise RuntimeError("chat_tab.build() must run before wire()")
    registry = ctx.settings_registry
    components = registry.components()
    value_count = len(components)
    outputs = [
        handles.chatbot,
        handles.message,
        handles.status,
        handles.tokens,
        handles.reasoning,
        handles.reasoning_accordion,
        handles.conversation_state,
        handles.stop,
        handles.files,
        handles.path,
        handles.attachment_status,
    ]

    def response_display(
        messages: list[dict[str, Any]],
        text: str,
        reasoning: str = "",
        *,
        reasoning_done: bool = True,
        reasoning_s: float | None = None,
    ) -> list[dict[str, Any]]:
        """History plus the live turn: a thought block while reasoning streams, then the answer."""

        display = _chatbot_messages(messages)
        if reasoning:
            display.append(_thought_message(reasoning, done=reasoning_done, seconds=reasoning_s))
        if text:
            display.append({"role": "assistant", "content": text})
        return display

    def composer_reset(mode: str) -> tuple[Any, Any, Any]:
        """Clear sent attachments after a multi-turn exchange; single-turn Q&A keeps its video."""

        if mode != "multi":
            return gr.skip(), gr.skip(), gr.skip()
        return gr.update(value=None), "", "<span class='vc-help'>No media attached.</span>"

    def send_message(*args: Any):
        settings = registry.values_to_dict(args[:value_count])
        state = dict(args[value_count] or _INITIAL_STATE)
        files = args[value_count + 1]
        path_text = str(args[value_count + 2] or "")
        message = str(args[value_count + 3] or "").strip()
        model_key = str(settings.get("model_key") or "qwen3_omni_instruct_int4")
        mode, model_note = model_chat_support(model_key)
        keep_composer = (gr.skip(), gr.skip(), gr.skip())

        def rejected(status_html: str, tokens_html: Any = gr.skip()) -> tuple[Any, ...]:
            return (
                gr.skip(),
                gr.skip(),
                status_html,
                tokens_html,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.update(value="⏹ Stop", interactive=False),
                *keep_composer,
            )

        if mode == "unsupported":
            yield rejected(model_note, _tokens_line())
            return
        if not message:
            yield rejected("<span class='vc-warn'>Enter a message before sending.</span>")
            return
        try:
            selected_media = resolve_chat_attachments(files, path_text)
        except Exception as exc:
            yield rejected(f"<span class='vc-err'>{html.escape(str(exc))}</span>")
            return
        previous_messages = list(state.get("messages") or [])
        previous_model_key = str(state.get("model_key") or "")
        if (
            previous_model_key
            and previous_model_key != model_key
            and _model_family_or_empty(previous_model_key) != _model_family_or_empty(model_key)
        ):
            previous_messages = []
            state = dict(_INITIAL_STATE)
        if mode == "single":
            previous_messages = []
            if len(selected_media) != 1 or probe_media(selected_media[0]).kind not in {"video", "video_no_audio"}:
                yield rejected(
                    f"<span class='vc-warn'>{html.escape(MODEL_SPECS[variant_to_family(model_key)].label)} "
                    "chat requires exactly one video attachment.</span>"
                )
                return
        history = [
            {
                "role": str(item.get("role")),
                "content": str(item.get("content") or ""),
                "media": [str(value) for value in (item.get("media") or [])],
            }
            for item in previous_messages
            if str(item.get("role")) in {"user", "assistant"}
        ]
        # Attachments belong to the turn they are sent with, so any turn can add media.
        history.append({"role": "user", "content": message, "media": list(selected_media)})
        media = [value for item in history for value in (item.get("media") or [])]
        generation = {
            "temperature": float(settings.get("chat_temperature", 0.2)),
            "top_p": float(settings.get("chat_top_p", 0.95)),
            "top_k": int(settings.get("chat_top_k", 20)),
            "max_new_tokens": int(settings.get("chat_max_new_tokens", 1024)),
            "repetition_penalty": float(settings.get("chat_repetition_penalty", 1.0)),
            "enable_thinking": bool(settings.get("chat_enable_thinking", False)),
            "seed": int(settings.get("chat_seed", -1)),
        }
        request = ChatRequest.from_dict(
            {
                "settings": settings,
                "history": history,
                "media": [],
                "generation": generation,
                "system_prompt": str(settings.get("chat_system_prompt") or ""),
            }
        )
        set_mode = ctx.states.get("set_subprocess_mode")
        if callable(set_mode):
            set_mode(bool(settings.get("subprocess_mode", True)))
        token = CancelToken()
        ctx.activate_cancel(token)
        ctx.states["chat_job_token"] = token
        events: queue.Queue[dict[str, Any]] = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            try:
                terminal["result"] = ctx.pipeline_client.chat(request, events.put, token)
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put({"ev": "terminal"})

        thread = threading.Thread(target=work, name="vcap-chat-ui", daemon=True)
        thread.start()
        current_text = ""
        current_reasoning = ""
        current_status = "Starting chat."
        context_used: Any = None
        context_limit: Any = None
        token_line = _tokens_line("generating")
        reasoning_started: float | None = None
        reasoning_s: float | None = None

        def live_display(*, done: bool) -> list[dict[str, Any]]:
            seconds = reasoning_s
            if seconds is None and reasoning_started is not None and (done or current_text):
                seconds = time.monotonic() - reasoning_started
            return response_display(
                history,
                current_text,
                current_reasoning,
                reasoning_done=done or bool(current_text),
                reasoning_s=seconds,
            )

        def live_status(css_class: str = "vc-ok") -> str:
            phase = "Thinking · " if current_reasoning and not current_text else ""
            return f"<span class='{css_class}'>{html.escape(phase + current_status)}</span>"

        yield (
            response_display(history, ""),
            "",
            f"<span class='vc-ok'>{html.escape(current_status)}</span>",
            token_line,
            "",
            gr.update(visible=False),
            state,
            gr.update(value="⏹ Stop", interactive=True),
            *keep_composer,
        )
        last_emit = 0.0
        try:
            while True:
                batch = [events.get()]
                while True:
                    try:
                        batch.append(events.get_nowait())
                    except queue.Empty:
                        break
                if any(str(event.get("ev") or "") == "terminal" for event in batch):
                    break
                # chat_result is an ordering barrier. PipelineClient returns its
                # authoritative ChatResponse immediately after publishing it, so
                # do not send any older queued load/generation status afterward.
                if any(str(event.get("ev") or "") == "chat_result" for event in batch):
                    continue
                saw_delta = False
                for event in batch:
                    kind = str(event.get("ev") or "")
                    if kind == "delta":
                        saw_delta = True
                        current_text = str(event.get("text") or current_text)
                        current_reasoning = str(event.get("reasoning") or current_reasoning)
                        if current_reasoning and reasoning_started is None:
                            reasoning_started = time.monotonic()
                        if current_text and reasoning_started is not None and reasoning_s is None:
                            reasoning_s = time.monotonic() - reasoning_started
                    elif kind == "status":
                        current_status = str(event.get("message") or current_status)
                        data = dict(event.get("data") or {})
                        new_tokens = data.get("new_tokens")
                        speed = data.get("tok_per_s", data.get("tokens_per_second"))
                        token_text = str(new_tokens) if new_tokens is not None else "generating"
                        try:
                            speed_text = f"{float(speed):.2f} tok/s" if speed is not None else "—"
                        except (TypeError, ValueError):
                            speed_text = "—"
                        if data.get("prompt_tokens") is not None:
                            context_used = data.get("prompt_tokens")
                            context_limit = data.get("context_limit", context_limit)
                        token_line = _tokens_line(token_text, speed_text, context_used, context_limit)
                    elif kind == "log" and str(event.get("level") or "").casefold() in {"warning", "error"}:
                        current_status = str(event.get("text") or current_status)
                now = time.monotonic()
                # Coalesce UI updates: the browser cannot keep up with a yield per
                # token (each yield re-renders the conversation), so emit at most
                # ~6 times per second; the terminal event always flushes the final state.
                if now - last_emit < 0.15:
                    continue
                last_emit = now
                yield (
                    live_display(done=False),
                    "",
                    live_status(),
                    token_line,
                    current_reasoning,
                    gr.update(visible=bool(current_reasoning)),
                    state,
                    gr.update(value="⏹ Stop", interactive=True),
                    *keep_composer,
                )
            thread.join()
            if "error" in terminal:
                raise terminal["error"]
            result: ChatResponse = terminal["result"]
            current_text = result.text
            current_reasoning = result.reasoning
            if reasoning_s is None and reasoning_started is not None and current_reasoning:
                reasoning_s = time.monotonic() - reasoning_started
            saved_messages = list(history)
            if current_text:
                saved_messages.append(
                    {
                        "role": "assistant",
                        "content": current_text,
                        "reasoning": current_reasoning,
                        "reasoning_s": reasoning_s,
                        "result": result.to_dict(),
                    }
                )
            new_state = {
                "messages": saved_messages,
                "media": media,
                "model_key": model_key,
                "system_prompt": request.system_prompt,
                "generation": generation,
                "outputs_root": str(settings.get("outputs_dir") or ctx.outputs_dir),
                "last_result": result.to_dict(),
            }
            warning = " ".join(result.warnings)
            trim_note = (
                f" Context trimmed {result.dropped_turns} oldest turn(s)."
                if result.dropped_turns
                else ""
            )
            status_class = "vc-warn" if result.cancelled else "vc-ok"
            final_status = (
                f"{'Stopped' if result.cancelled else 'Complete'}: {result.new_tokens} tokens, "
                f"{result.tokens_per_s:.2f} tok/s, finish={result.finish_reason}.{trim_note} {warning}"
            ).strip()
            # Stream throttling is deliberately lossy; overwrite every live
            # statistic with the authoritative terminal response before the
            # generator closes, even when the whole reply fit inside one tick.
            current_status = final_status
            context_used = result.context_tokens or result.prompt_tokens
            context_limit = result.context_limit or settings.get("context_tokens")
            token_line = _tokens_line(
                result.new_tokens,
                f"{result.tokens_per_s:.2f} tok/s",
                context_used,
                context_limit,
            )
            # The turn (and its attachments) is now part of the conversation, so
            # the composer starts clean for the next one.
            yield (
                live_display(done=True),
                "",
                live_status(status_class),
                token_line,
                current_reasoning,
                gr.update(visible=bool(current_reasoning)),
                new_state,
                gr.update(value="⏹ Stop", interactive=False),
                *composer_reset(mode),
            )
        except (CancelledError, KeyboardInterrupt) as exc:
            yield (
                live_display(done=True),
                "",
                f"<span class='vc-warn'>{html.escape(str(exc))}</span>",
                token_line,
                current_reasoning,
                gr.update(visible=bool(current_reasoning)),
                state,
                gr.update(value="⏹ Stop", interactive=False),
                *keep_composer,
            )
        except BaseException as exc:
            ctx.app_log.exception(f"Chat failed: {exc}", scope="chat")
            yield (
                live_display(done=True),
                "",
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
                token_line,
                current_reasoning,
                gr.update(visible=bool(current_reasoning)),
                state,
                gr.update(value="⏹ Stop", interactive=False),
                *keep_composer,
            )
        finally:
            ctx.clear_active_cancel(token)
            if ctx.states.get("chat_job_token") is token:
                ctx.states["chat_job_token"] = None

    run_inputs = [
        *components,
        handles.conversation_state,
        handles.files,
        handles.path,
        handles.message,
    ]
    for trigger in (handles.send.click, handles.message.submit):
        trigger(
            send_message,
            inputs=run_inputs,
            outputs=outputs,
            concurrency_id="gpu_queue",
            concurrency_limit=1,
            show_progress="hidden",
            api_name="chat" if trigger == handles.send.click else False,
            api_description=(
                "Send one multimodal conversation turn with the Caption tab's selected model and runtime settings."
                if trigger == handles.send.click
                else None
            ),
            api_visibility="public" if trigger == handles.send.click else "private",
        )

    def stop_chat() -> tuple[Any, str]:
        token = ctx.states.get("chat_job_token")
        if token is None or token.is_cancelled():
            return gr.update(value="⏹ Stop", interactive=False), "<span class='vc-help'>No chat response is running.</span>"
        if not token.is_armed():
            token.arm_confirmation(window_s=6)
            return (
                gr.update(value="⚠ Click again to confirm stop", interactive=True),
                "<span class='vc-warn'>Click Stop again within 6 seconds to stop generation.</span>",
            )
        token.cancel()
        ctx.pipeline_client.cancel(force=False)
        return (
            gr.update(value="Stopping…", interactive=False),
            "<span class='vc-warn'>Cooperative stop requested; the loaded model will be retained.</span>",
        )

    handles.stop.click(
        stop_chat,
        outputs=[handles.stop, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def refresh_stop() -> Any:
        token = ctx.states.get("chat_job_token")
        if token is None or token.is_cancelled():
            return gr.update(value="⏹ Stop", interactive=False)
        if token.is_armed():
            return gr.skip()
        return gr.update(value="⏹ Stop", interactive=True)

    handles.stop_timer.tick(
        refresh_stop,
        outputs=handles.stop,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def clear_history() -> tuple[Any, ...]:
        token = ctx.states.get("chat_job_token")
        if token is not None and not token.is_cancelled():
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "<span class='vc-warn'>Stop the running response before clearing history.</span>",
            )
        return [], dict(_INITIAL_STATE), "", gr.update(visible=False), "<span class='vc-ok'>Conversation cleared.</span>"

    handles.clear.click(
        clear_history,
        outputs=[
            handles.chatbot,
            handles.conversation_state,
            handles.reasoning,
            handles.reasoning_accordion,
            handles.status,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    handles.copy_last.click(
        lambda history: (
            "<span class='vc-ok'>Copied the last answer.</span>"
            if _last_answer(history)
            else "<span class='vc-warn'>No assistant answer is available.</span>"
        ),
        inputs=handles.chatbot,
        outputs=handles.status,
        js=(
            "(history) => { const rows = Array.isArray(history) ? history : []; "
            "const item = [...rows].reverse().find(x => x && x.role === 'assistant' && !(x.metadata && x.metadata.title)); "
            "if (item && item.content) navigator.clipboard.writeText(String(item.content)); "
            "return [history]; }"
        ),
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def save_history(state: Mapping[str, Any]) -> str:
        messages = list((state or {}).get("messages") or [])
        if not messages:
            return "<span class='vc-warn'>No conversation is available to save.</span>"
        model_key = str((state or {}).get("model_key") or "qwen3_omni_instruct")
        metadata = {
            "system_prompt": str((state or {}).get("system_prompt") or ""),
            "media": list((state or {}).get("media") or []),
            "generation": dict((state or {}).get("generation") or {}),
            "last_result": dict((state or {}).get("last_result") or {}),
        }
        run_dir = save_conversation(
            messages,
            model_key=model_key,
            metadata=metadata,
            outputs_root=str((state or {}).get("outputs_root") or ctx.outputs_dir),
        )
        return f"<span class='vc-ok'>Saved conversation to {html.escape(str(run_dir))}</span>"

    handles.save.click(
        save_history,
        inputs=handles.conversation_state,
        outputs=handles.status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )


__all__ = [
    "ChatTabHandles",
    "build",
    "model_chat_support",
    "resolve_chat_attachments",
    "wire",
]
