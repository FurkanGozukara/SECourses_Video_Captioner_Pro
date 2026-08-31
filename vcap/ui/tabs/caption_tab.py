"""Primary caption workflow and all model/pipeline controls."""

from __future__ import annotations

import html
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import gradio as gr

from vcap.core.clip_fitness import TRAINER_TARGETS
from vcap.core.captions_post import to_srt
from vcap.core.media import probe_media
from vcap.core.paths import open_in_file_manager, reveal_in_file_manager
from vcap.core.preprocess import fits_context, token_budget_estimate
from vcap.core.progress import ProgressEvent, format_eta
from vcap.core.scene_split import (
    SceneDetectParams,
    cap_scene_lengths,
    detect_scenes,
    merge_short_scenes,
)
from vcap.core.subprocess_runner import CancelToken, CancelledError, build_child_env
from vcap.models.attention import ATTENTION_CHOICES, probe_available
from vcap.models.downloads import ensure_model
from vcap.models.registry import (
    MODEL_SPECS,
    all_variant_choices,
    get_variant,
    variant_is_ready,
    variant_to_family,
)
from vcap.models.torch_compile import clear_inductor_caches
from vcap.models.vram_presets import (
    VRAM_TIERS,
    allowed_variants,
    apply_preset,
    auto_tier,
    preset_for,
)
from vcap.pipeline.job import InputItem, JobResult, JobSpec, OutputSpec
from vcap.prompts.presets import (
    TEMPLATE_VARIABLES,
    default_preset_for,
    get_preset,
    list_presets,
    render_prompt,
)
from vcap.ui.components import (
    LogPanelHandles,
    MediaInputHandles,
    ProgressPanelHandles,
    action_button,
    log_panel,
    media_input_block,
    progress_panel,
    render_progress_html,
    replace_words_editor,
)

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


_INITIAL_VARIANT = "qwen3_omni_instruct_int8"
_INITIAL_MODALITY = "video_audio"


def _display(value: object) -> str:
    return (
        str(value)
        .replace("â€”", "—")
        .replace("â€“", "–")
        .replace("â†’", "→")
        .replace("â‰ˆ", "≈")
    )


def _variant_choices() -> list[tuple[str, str]]:
    return [(_display(label), key) for label, key in all_variant_choices()]


def _prompt_choices(family: str, modality: str) -> list[tuple[str, str]]:
    return [
        (f"{preset.group} · {_display(preset.label)}", preset.id)
        for preset in list_presets(family, modality)
    ]


def _prompt_variables(values: Iterable[Any]) -> dict[str, Any]:
    names = (
        "TRIGGER",
        "LANGUAGE",
        "SOURCE_LANGUAGE",
        "TARGET_LANGUAGE",
        "CAPTION_LENGTH",
        "AVOID",
        "SUBJECT_CLASS",
        "EXTRA_INSTRUCTIONS",
    )
    return dict(zip(names, values))


def _read_text(path: str | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: str | None) -> Any:
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _format_compile_report(data: dict[str, Any]) -> str:
    readiness = str(data.get("inductor_ready") or "unavailable")
    tooltip = (
        "Windows: install Desktop development with C++ in Visual Studio Build Tools. "
        "Linux: install gcc/g++ and ensure they are on PATH. Triton is also required for full Inductor."
    )
    if readiness == "full":
        if data.get("msvc_version"):
            match = re.search(r"MSVC[\\/]([0-9]+\.[0-9]+)", str(data.get("cl_path") or ""), re.I)
            version = match.group(1) if match else str(data["msvc_version"]).split("Version ")[-1].split()[0]
            compiler = f"MSVC {version} (VS Build Tools)"
        else:
            compiler = str(data.get("gcc_version") or "C++ compiler").splitlines()[0]
        message = f"✓ {html.escape(compiler)} — full Inductor available"
        css = "vc-ok"
    elif readiness == "triton_only":
        message = "⚠ No C++ build tools — Triton-only fallback will be used"
        css = "vc-warn"
    elif readiness == "cudagraphs_only":
        message = "⚠ Triton unavailable — CUDA graphs fallback only"
        css = "vc-warn"
    else:
        detail = "; ".join(str(item) for item in data.get("messages") or []) or "torch.compile is unavailable"
        message = f"✗ compile unavailable: {html.escape(detail)}"
        css = "vc-err"
    return f"<span class='{css}'>{message}</span> <span title='{html.escape(tooltip)}'>ⓘ</span>"


def _probe_compile_in_child() -> str:
    """Call the backend probe in a disposable process so this parent stays Torch-free."""

    code = (
        "import json; from dataclasses import asdict; "
        "from vcap.models.torch_compile import probe_compile_environment; "
        "print('VCAP_COMPILE_ENV '+json.dumps(asdict(probe_compile_environment())))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=build_child_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        marker = "VCAP_COMPILE_ENV "
        line = next(
            (item[len(marker) :] for item in reversed(completed.stdout.splitlines()) if item.startswith(marker)),
            None,
        )
        if line is None:
            detail = (completed.stderr or completed.stdout or "probe returned no data").strip()[-500:]
            return f"<span class='vc-err'>✗ compile probe failed: {html.escape(detail)}</span>"
        return _format_compile_report(json.loads(line))
    except Exception as exc:
        return f"<span class='vc-err'>✗ compile probe failed: {html.escape(str(exc))}</span>"


def _ready_line(variant_key: str) -> str:
    try:
        ready, detail = variant_is_ready(variant_key)
        variant = get_variant(variant_key)
        icon, css = ("✓", "vc-ok") if ready else ("✗", "vc-err")
        return (
            f"<span class='{css}'>{icon} {_display(variant.label)}: "
            f"{html.escape(_display(detail))}</span>"
        )
    except Exception as exc:
        return f"<span class='vc-err'>✗ {html.escape(str(exc))}</span>"


def _quant_line(variant_key: str) -> str:
    try:
        variant = get_variant(variant_key)
        backend = "llama.cpp" if variant.backend == "llamacpp" else "Transformers"
        return (
            f"**Precision:** `{html.escape(variant.scheme)}` · **Backend:** {backend} · "
            f"**Checkpoint:** {variant.size_gb:.1f} GB"
        )
    except Exception as exc:
        return f"<span class='vc-err'>{html.escape(str(exc))}</span>"


def _gpu_inventory() -> tuple[list[Any], int, float, str]:
    from vcap.core.gpu import list_gpus

    devices = list_gpus()
    if not devices:
        return [("CPU / GPU unavailable", 0)], 0, 0.0, "GPU telemetry unavailable"
    selected = next((item for item in devices if item.is_default), devices[0])
    choices = [
        (f"GPU {item.index} · {item.name} · {item.total_gb:.1f} GB", item.index)
        for item in devices
    ]
    summary = " · ".join(f"GPU {item.index}: {item.name} ({item.total_gb:.1f} GB)" for item in devices)
    return choices, selected.index, selected.total_gb, summary


@dataclass
class CaptionTabHandles:
    media: MediaInputHandles
    progress: ProgressPanelHandles
    logs: LogPanelHandles
    controls: dict[str, Any]
    start: gr.Button
    cancel: gr.Button
    hotkey_start: gr.Button
    hotkey_cancel: gr.Button
    open_output: gr.Button
    open_caption: gr.Button
    reveal_clip: gr.Button
    item_table: gr.Dataframe
    caption: gr.Textbox
    structured: gr.JSON
    srt: gr.Textbox
    reasoning: gr.Textbox
    reasoning_tab: gr.Tab
    clips: gr.Gallery
    last_outputs_state: gr.State


def build(ctx: "UiContext") -> CaptionTabHandles:
    """Render the complete Caption tab; app assembly wires global handlers later."""

    controls: dict[str, Any] = {}
    initial_family = variant_to_family(_INITIAL_VARIANT)
    initial_spec = MODEL_SPECS[initial_family]
    initial_prompt = default_preset_for(initial_family, _INITIAL_MODALITY)
    initial_vars = {name: data["default"] for name, data in TEMPLATE_VARIABLES.items()}
    initial_system, initial_user = render_prompt(initial_prompt, initial_vars)
    gpu_choices, gpu_default, gpu_total, _ = _gpu_inventory()
    ctx.states["gpu_index_default"] = gpu_default
    ctx.states["gpu_index"] = gr.State(gpu_default)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=500):
            media = media_input_block(ctx)

            with gr.Accordion("Trim range", open=False):
                with gr.Row(elem_classes=["vc-compact-row"]):
                    trim_start = gr.Number(
                        value=0.0,
                        minimum=0.0,
                        step=0.1,
                        precision=3,
                        label="Start (seconds)",
                        info="Applied after any trim made in the media player.",
                    )
                    controls["trim_start_s"] = ctx.reg(
                        "trim_start_s", trim_start, 0.0, section="preprocessing",
                        description="Start time for numeric trimming in seconds.", kind="float", minimum=0.0,
                    )
                    trim_end = gr.Number(
                        value=None,
                        minimum=0.0,
                        step=0.1,
                        precision=3,
                        label="End (seconds)",
                        info="Leave blank to process through the end of the input.",
                    )
                    controls["trim_end_s"] = ctx.reg(
                        "trim_end_s", trim_end, None, section="preprocessing",
                        description="Optional numeric trim end in seconds.", minimum=0.0,
                    )
                gr.Markdown(
                    "Use the player's built-in trim editor for visual trimming; its edited file becomes the first input automatically.",
                    elem_classes=["vc-help"],
                )

            with gr.Group(elem_classes=["vc-card", "vc-result-panel"]):
                gr.Markdown("### Result", elem_classes=["vc-section-title"])
                caption = gr.Textbox(
                    label="Caption",
                    lines=14,
                    max_lines=14,
                    buttons=["copy"],
                    show_label=True,
                    interactive=False,
                    autoscroll=True,
                )
                with gr.Tabs():
                    with gr.Tab("JSON"):
                        structured = gr.JSON(label="Structured result", buttons=["copy"], max_height=410)
                    with gr.Tab("SRT preview"):
                        srt = gr.Textbox(
                            label="Subtitles",
                            lines=12,
                            max_lines=16,
                            buttons=["copy"],
                            interactive=False,
                            elem_classes=["vc-mono"],
                        )
                    with gr.Tab("Reasoning", visible=False) as reasoning_tab:
                        reasoning = gr.Textbox(
                            label="Reasoning",
                            lines=12,
                            max_lines=18,
                            buttons=["copy"],
                            interactive=False,
                            elem_classes=["vc-mono"],
                        )
                    with gr.Tab("Clips"):
                        clips = gr.Gallery(
                            label="Produced clips",
                            columns=4,
                            rows=2,
                            height=330,
                            object_fit="cover",
                            allow_preview=True,
                            type="filepath",
                            buttons=["download", "fullscreen"],
                        )

            with gr.Row(elem_classes=["vc-action-row", "vc-compact-row"]):
                start = action_button("▶ Start Captioning", "emerald", variant="primary", size="lg", scale=3)
                cancel = action_button("⏹ Cancel", "red", variant="stop", size="lg", scale=2)
                open_output = action_button("📂 Open Output", "teal", size="md", scale=2)
                open_caption = action_button("📝 Open Last Caption", "violet", size="md", scale=2)
                reveal_clip = action_button("🎬 Reveal Clip", "amber", size="md", scale=2)
            hotkey_start = gr.Button("Start caption hotkey", elem_id="hk_caption_start", visible="hidden")
            hotkey_cancel = gr.Button("Cancel caption hotkey", elem_id="hk_caption_cancel", visible="hidden")

            progress = progress_panel(ctx)
            item_table = gr.Dataframe(
                headers=["#", "Input", "Status", "Message", "Elapsed"],
                value=[],
                type="array",
                datatype=["number", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                max_height=260,
                buttons=["copy", "fullscreen"],
                label="Items",
            )
            logs = log_panel(ctx)

        with gr.Column(scale=4, min_width=430):
            with gr.Accordion("1. Model", open=True):
                model_key = gr.Dropdown(
                    choices=_variant_choices(),
                    value=_INITIAL_VARIANT,
                    label="Model variant",
                    info="Model family, precision/backend variant, and estimated local checkpoint size.",
                )
                controls["model_key"] = ctx.reg(
                    "model_key", model_key, _INITIAL_VARIANT, section="model",
                    description="Selected model family and checkpoint variant.", choices=[key for _, key in _variant_choices()], kind="str",
                )
                quant_info = gr.Markdown(_quant_line(_INITIAL_VARIANT))
                with gr.Row(elem_classes=["vc-compact-row"]):
                    tier = auto_tier(gpu_total) if gpu_total else 32
                    vram_choices = [(f"Auto ({gpu_total:.0f} GB detected)" if gpu_total else "Auto", "auto")]
                    vram_choices.extend((f"{value} GB", str(value)) for value in VRAM_TIERS)
                    vram_preset = gr.Dropdown(
                        choices=vram_choices,
                        value="auto",
                        label="VRAM preset",
                        info="Applies a coordinated precision, frame, pixel, token, attention, and offload plan.",
                    )
                    controls["vram_preset"] = ctx.reg(
                        "vram_preset", vram_preset, "auto", section="model",
                        description="Detected or manually selected VRAM capacity tier.", choices=["auto", *map(str, VRAM_TIERS)], kind="str",
                    )
                    availability = probe_available()
                    attention_choices = [
                        (f"{name} {'✓' if availability.get(name) else '— unavailable'}", name)
                        for name in ATTENTION_CHOICES
                    ]
                    attention = gr.Dropdown(
                        choices=attention_choices,
                        value="auto",
                        label="Attention backend",
                        info="Unavailable optimized backends fall back safely to PyTorch SDPA.",
                    )
                    controls["attention_backend"] = ctx.reg(
                        "attention_backend", attention, "auto", section="model",
                        description="Requested attention implementation with safe runtime fallback.", choices=ATTENTION_CHOICES, kind="str",
                    )
                vram_note = gr.Markdown(
                    f"<span class='vc-help'>Auto tier: {tier} GB. Choose a tier to apply its complete plan.</span>"
                )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    gpu_picker = gr.Dropdown(
                        choices=gpu_choices,
                        value=gpu_default,
                        label="GPU",
                        info="The selected physical GPU is isolated for the caption worker.",
                    )
                    controls["gpu_index"] = ctx.reg(
                        "gpu_index", gpu_picker, gpu_default, section="runtime",
                        description="Physical NVIDIA GPU index used by the pipeline.", kind="int",
                    )
                    subprocess_mode = gr.Checkbox(
                        value=True,
                        label="Subprocess mode",
                        info="Recommended: isolates CUDA and allows process-tree cancellation.",
                    )
                    controls["subprocess_mode"] = ctx.reg(
                        "subprocess_mode", subprocess_mode, True, section="runtime",
                        description="Run model inference in an isolated worker process.", kind="bool",
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    keep_loaded = gr.Checkbox(
                        value=True,
                        label="Keep model loaded",
                        info="Reuse the worker and resident model between caption jobs.",
                    )
                    controls["keep_model_loaded"] = ctx.reg(
                        "keep_model_loaded", keep_loaded, True, section="runtime",
                        description="Keep model weights resident between runs.", kind="bool",
                    )
                    idle_minutes = gr.Number(
                        value=10,
                        minimum=0,
                        maximum=1440,
                        step=1,
                        precision=1,
                        label="Idle unload (minutes)",
                        info="Zero disables automatic idle unload.",
                    )
                    controls["idle_unload_minutes"] = ctx.reg(
                        "idle_unload_minutes", idle_minutes, 10, section="runtime",
                        description="Minutes before an idle persistent model is unloaded.", kind="float", minimum=0, maximum=1440,
                    )
                with gr.Accordion("Offload plan", open=False):
                    gpu_layers = gr.Dropdown(
                        choices=["all", *[str(value) for value in range(0, 49, 4)]],
                        value="all",
                        allow_custom_value=True,
                        label="Decoder layers on GPU",
                        info="Use all, or enter a leading decoder-layer count for CPU offload.",
                    )
                    controls["gpu_layers"] = ctx.reg(
                        "gpu_layers", gpu_layers, "all", section="model",
                        description="Leading decoder layers kept resident on GPU.",
                    )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        offload_experts = gr.Checkbox(
                            value=False,
                            label="Offload MoE experts",
                            info="Keep Qwen3 expert banks on CPU to reduce VRAM use.",
                        )
                        controls["offload_experts"] = ctx.reg(
                            "offload_experts", offload_experts, False, section="model",
                            description="Offload Qwen3 mixture-of-experts banks to CPU.", kind="bool",
                        )
                        pin_cpu = gr.Checkbox(
                            value=False,
                            label="Pin CPU tensors",
                            info="Use pinned host memory for faster transfers at the cost of RAM flexibility.",
                        )
                        controls["pin_cpu"] = ctx.reg(
                            "pin_cpu", pin_cpu, False, section="model",
                            description="Pin CPU-offloaded tensors in host memory.", kind="bool",
                        )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    compile_enabled = gr.Checkbox(
                        value=False,
                        label="torch.compile",
                        info=(
                            "CUDA graphs are the safe default. Full Inductor is opt-in because "
                            "DynamicCache specialization was slower in measured decode runs."
                        ),
                    )
                    controls["torch_compile"] = ctx.reg(
                        "torch_compile", compile_enabled, False, section="runtime",
                        description="Compile the language model forward pass with safe fallbacks.", kind="bool",
                    )
                    compile_mode = gr.Dropdown(
                        choices=[
                            ("CUDA graphs (recommended)", "cudagraphs"),
                            ("Full Inductor", "default"),
                            ("Max autotune, no CUDA graphs", "max-autotune-no-cudagraphs"),
                        ],
                        value="cudagraphs",
                        label="Compile mode",
                        info="Choose CUDA graphs for stable decode or explicitly opt into Inductor tuning.",
                    )
                    controls["torch_compile_mode"] = ctx.reg(
                        "torch_compile_mode", compile_mode, "cudagraphs", section="runtime",
                        description="Requested torch.compile tuning mode.",
                        choices=["default", "max-autotune-no-cudagraphs", "cudagraphs"], kind="str",
                    )
                compile_status = gr.Markdown(_probe_compile_in_child(), elem_classes=["vc-status"])
                with gr.Row(elem_classes=["vc-compact-row"]):
                    download = action_button("📥 Download / Verify model", "sky", size="md", scale=3)
                    refresh_ready = action_button("↻ Refresh", "cyan", size="md", scale=1)
                    clear_compile = action_button("⌫ Clear compile caches", "rose", size="md", scale=2)
                ready_status = gr.Markdown(_ready_line(_INITIAL_VARIANT), elem_classes=["vc-status"])

            with gr.Accordion("2. Task & Prompt", open=True):
                prompt_preset = gr.Dropdown(
                    choices=_prompt_choices(initial_family, _INITIAL_MODALITY),
                    value=initial_prompt.id,
                    allow_custom_value=True,
                    label="Task / prompt preset",
                    info="Filtered to the selected model family and first input modality.",
                )
                controls["prompt_preset_id"] = ctx.reg(
                    "prompt_preset_id", prompt_preset, initial_prompt.id, section="prompt",
                    description="Built-in task and prompt preset identifier.", kind="str",
                )
                prompt_description = gr.Markdown(_display(initial_prompt.description), elem_classes=["vc-help"])
                system_prompt = gr.Textbox(
                    value=initial_system or "",
                    label="System prompt",
                    info="Optional system instruction sent before the user request.",
                    lines=4,
                    max_lines=8,
                    elem_classes=["vc-mono"],
                )
                controls["system_prompt"] = ctx.reg(
                    "system_prompt", system_prompt, initial_system, section="prompt",
                    description="Rendered or custom system instruction.",
                )
                user_prompt = gr.Textbox(
                    value=initial_user,
                    label="User prompt",
                    info="The complete model request after template variables are rendered.",
                    lines=10,
                    max_lines=16,
                    elem_classes=["vc-mono"],
                )
                controls["user_prompt"] = ctx.reg(
                    "user_prompt", user_prompt, initial_user, section="prompt",
                    description="Rendered or custom user instruction.", kind="str",
                )
                with gr.Accordion("Template variables", open=False):
                    trigger_word = gr.Textbox(
                        value="ohwx",
                        label="Trigger word",
                        info="Concept token used in prompt templates and optional caption injection.",
                    )
                    controls["trigger_word"] = ctx.reg(
                        "trigger_word", trigger_word, "ohwx", section="prompt",
                        description="Concept trigger token used by templates and post-processing.", kind="str",
                    )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        language = gr.Textbox(value="English", label="Caption language", info="Requested caption language.")
                        controls["language"] = ctx.reg(
                            "language", language, "English", section="prompt",
                            description="Requested caption language.", kind="str",
                        )
                        source_language = gr.Textbox(value="English", label="Source language", info="Spoken language in the source audio.")
                        controls["source_language"] = ctx.reg(
                            "source_language", source_language, "English", section="prompt",
                            description="Language spoken in source audio.", kind="str",
                        )
                        target_language = gr.Textbox(value="English", label="Target language", info="Translation target language.")
                        controls["target_language"] = ctx.reg(
                            "target_language", target_language, "English", section="prompt",
                            description="Target language for translation tasks.", kind="str",
                        )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        caption_length = gr.Dropdown(
                            choices=["short", "medium", "detailed", "very detailed"],
                            value="detailed",
                            allow_custom_value=True,
                            label="Caption length",
                            info="Natural-language detail target inserted into compatible templates.",
                        )
                        controls["caption_length"] = ctx.reg(
                            "caption_length", caption_length, "detailed", section="prompt",
                            description="Requested caption length or detail level.", kind="str",
                        )
                        subject_class = gr.Textbox(value="person", label="Subject class", info="Generic identity class for LoRA captions.")
                        controls["subject_class"] = ctx.reg(
                            "subject_class", subject_class, "person", section="prompt",
                            description="Generic class noun used by character training prompts.", kind="str",
                        )
                    avoid_list = gr.Textbox(
                        value="",
                        label="Avoid list",
                        info="Concepts the generated caption should not mention.",
                        lines=2,
                    )
                    controls["avoid_list"] = ctx.reg(
                        "avoid_list", avoid_list, "", section="prompt",
                        description="Comma-separated concepts excluded by compatible prompt templates.", kind="str",
                    )
                    extra_instructions = gr.Textbox(
                        value="",
                        label="Extra instructions",
                        info="Optional task-specific directions appended by compatible templates.",
                        lines=3,
                    )
                    controls["extra_instructions"] = ctx.reg(
                        "extra_instructions", extra_instructions, "", section="prompt",
                        description="Additional task instructions inserted into prompt templates.", kind="str",
                    )
                reset_prompts = action_button("↺ Reset prompts to preset", "purple", size="md")

            schema = {item.name: item for item in initial_spec.param_schema}
            with gr.Accordion("3. Generation", open=True):
                temperature = gr.Slider(
                    0.0, 2.0, value=float(schema["temperature"].default), step=0.01,
                    label="Temperature", info=schema["temperature"].description, buttons=["reset"],
                )
                controls["temperature"] = ctx.reg(
                    "temperature", temperature, float(schema["temperature"].default), section="generation",
                    description=schema["temperature"].description, kind="float", minimum=0, maximum=2,
                )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    top_p = gr.Slider(0, 1, value=float(schema["top_p"].default), step=0.01, label="Top-p", info=schema["top_p"].description)
                    controls["top_p"] = ctx.reg(
                        "top_p", top_p, float(schema["top_p"].default), section="generation",
                        description=schema["top_p"].description, kind="float", minimum=0, maximum=1,
                    )
                    top_k = gr.Slider(0, 200, value=int(schema["top_k"].default), step=1, precision=0, label="Top-k", info=schema["top_k"].description)
                    controls["top_k"] = ctx.reg(
                        "top_k", top_k, int(schema["top_k"].default), section="generation",
                        description=schema["top_k"].description, kind="int", minimum=0, maximum=200,
                    )
                repetition = gr.Slider(
                    0.5, 2.0, value=float(schema["repetition_penalty"].default), step=0.01,
                    label="Repetition penalty", info=schema["repetition_penalty"].description,
                )
                controls["repetition_penalty"] = ctx.reg(
                    "repetition_penalty", repetition, float(schema["repetition_penalty"].default), section="generation",
                    description=schema["repetition_penalty"].description, kind="float", minimum=0.5, maximum=2,
                )
                max_new_tokens = gr.Slider(
                    1,
                    initial_spec.limits.max_new_tokens_cap,
                    value=int(schema["max_new_tokens"].default),
                    step=1,
                    precision=0,
                    label="Maximum new tokens",
                    info=schema["max_new_tokens"].description,
                )
                controls["max_new_tokens"] = ctx.reg(
                    "max_new_tokens", max_new_tokens, int(schema["max_new_tokens"].default), section="generation",
                    description=schema["max_new_tokens"].description, kind="int", minimum=1, maximum=32768,
                )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    do_sample = gr.Checkbox(
                        value=bool(schema["do_sample"].default),
                        label="Sample tokens",
                        info="Prompt presets set this automatically; disable for deterministic greedy decoding.",
                    )
                    controls["do_sample"] = ctx.reg(
                        "do_sample", do_sample, bool(schema["do_sample"].default), section="generation",
                        description=schema["do_sample"].description, kind="bool",
                    )
                    use_cache = gr.Checkbox(
                        value=True,
                        label="Use KV cache",
                        info="Speeds autoregressive decoding at the cost of additional VRAM.",
                    )
                    controls["use_cache"] = ctx.reg(
                        "use_cache", use_cache, True, section="generation",
                        description="Use the model key/value cache during generation.", kind="bool",
                    )
                    enable_thinking = gr.Checkbox(
                        value=False,
                        label="Enable thinking",
                        info="Available only for the Qwen3-Omni Thinking family.",
                        interactive=False,
                    )
                    controls["enable_thinking"] = ctx.reg(
                        "enable_thinking", enable_thinking, False, section="generation",
                        description="Allow the Thinking model to emit a reasoning section.", kind="bool",
                    )
                sample_note = gr.Markdown(
                    "<span class='vc-help'>Sampling is controlled explicitly and may also be overridden by a task preset.</span>"
                )

            with gr.Accordion("4. Preprocessing", open=False):
                with gr.Row(elem_classes=["vc-compact-row"]):
                    fps = gr.Number(
                        value=initial_spec.limits.default_fps,
                        minimum=0.25,
                        maximum=8.0,
                        step=0.25,
                        label="Sampling FPS",
                        info="Actual visual sampling rate passed to the processor.",
                    )
                    controls["fps"] = ctx.reg(
                        "fps", fps, initial_spec.limits.default_fps, section="preprocessing",
                        description="Video frames sampled per source second.", kind="float", minimum=0.25, maximum=8,
                    )
                    max_frames = gr.Number(
                        value=initial_spec.limits.max_frames or 768,
                        minimum=0,
                        maximum=768,
                        step=2,
                        precision=0,
                        label="Maximum frames",
                        info="Hard cap applied after sampling; zero is valid for audio-only presets.",
                    )
                    controls["max_frames"] = ctx.reg(
                        "max_frames", max_frames, initial_spec.limits.max_frames or 768, section="preprocessing",
                        description="Maximum decoded frames supplied to the processor; zero disables visual frames for audio workflows.", kind="int", minimum=0, maximum=768,
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    resolution_preset = gr.Dropdown(
                        choices=[
                            ("TimeChat default · 297,920", "297920"),
                            ("AVoCaDO default · 401,408", "401408"),
                            ("Qwen3 · 256·32·32", "262144"),
                            ("Low VRAM · 128·32·32", "131072"),
                            ("Custom", "custom"),
                        ],
                        value="262144",
                        label="Pixel preset",
                        info="Sets the maximum resized area per decoded frame.",
                    )
                    controls["resolution_preset"] = ctx.reg(
                        "resolution_preset", resolution_preset, "262144", section="preprocessing",
                        description="Convenient model-aware maximum-pixel preset.",
                        choices=["297920", "401408", "262144", "131072", "custom"], kind="str",
                    )
                    max_pixels = gr.Number(
                        value=initial_spec.limits.default_max_pixels,
                        minimum=4 * 28 * 28,
                        maximum=1280 * 32 * 32,
                        step=1024,
                        precision=0,
                        label="Maximum pixels",
                        info="Maximum resized frame area while preserving aspect ratio.",
                    )
                    controls["max_pixels"] = ctx.reg(
                        "max_pixels", max_pixels, initial_spec.limits.default_max_pixels, section="preprocessing",
                        description="Maximum resized pixel area for every sampled frame.", kind="int", minimum=3136, maximum=1310720,
                    )
                    min_pixels = gr.Number(
                        value=initial_spec.limits.min_pixels,
                        minimum=4 * 28 * 28,
                        maximum=1280 * 32 * 32,
                        step=1024,
                        precision=0,
                        label="Minimum pixels",
                        info="Lower bound used by the model-aware smart resize rule.",
                    )
                    controls["min_pixels"] = ctx.reg(
                        "min_pixels", min_pixels, initial_spec.limits.min_pixels, section="preprocessing",
                        description="Minimum resized pixel area for every sampled frame.", kind="int", minimum=3136, maximum=1310720,
                    )
                sampling = gr.Dropdown(
                    choices=[
                        ("Target FPS", "fps"),
                        ("Uniform across duration", "uniform"),
                        ("Keyframe biased", "keyframe"),
                        ("Adaptive motion", "adaptive"),
                    ],
                    value="fps",
                    label="Sampling strategy",
                    info="Controls how the frame budget is distributed over the clip.",
                )
                controls["sampling_strategy"] = ctx.reg(
                    "sampling_strategy", sampling, "fps", section="preprocessing",
                    description="Frame timestamp planning strategy.", choices=["fps", "uniform", "keyframe", "adaptive"], kind="str",
                )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    normalize_clip = gr.Checkbox(
                        value=False,
                        label="Normalize clip",
                        info="Create deterministic H.264, exact-FPS, model-divisible media before inference.",
                    )
                    controls["normalize_clip"] = ctx.reg(
                        "normalize_clip", normalize_clip, False, section="preprocessing",
                        description="Normalize video codec, cadence, geometry, and audio before captioning.", kind="bool",
                    )
                    use_audio = gr.Checkbox(
                        value=True,
                        label="Use video audio",
                        info="Interleave the video's audio features when supported by the selected model.",
                    )
                    controls["use_audio_in_video"] = ctx.reg(
                        "use_audio_in_video", use_audio, True, section="preprocessing",
                        description="Include and align a video's audio stream with visual tokens.", kind="bool",
                    )
                    audio_rate = gr.Dropdown(
                        choices=[("16 kHz", 16000), ("24 kHz", 24000), ("48 kHz", 48000)],
                        value=16000,
                        label="Audio sample rate",
                        info="Worker extraction rate; 16 kHz is the verified model path.",
                    )
                    controls["audio_sample_rate"] = ctx.reg(
                        "audio_sample_rate", audio_rate, 16000, section="preprocessing",
                        description="Mono audio extraction sample rate.", choices=[16000, 24000, 48000], kind="int",
                    )
                with gr.Accordion("Automatic rejection", open=False):
                    auto_reject = gr.Checkbox(
                        value=False,
                        label="Enable automatic rejection",
                        info="Analyze split clips and skip those failing any enabled quality rule.",
                    )
                    controls["auto_reject_enabled"] = ctx.reg(
                        "auto_reject_enabled", auto_reject, False, section="preprocessing",
                        description="Enable clip quality rejection before caption generation.", kind="bool",
                    )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        reject_min_duration = gr.Number(value=0, minimum=0, step=0.1, label="Minimum duration (s)", info="Reject clips shorter than this; zero disables.")
                        controls["reject_min_duration_s"] = ctx.reg(
                            "reject_min_duration_s", reject_min_duration, 0, section="preprocessing",
                            description="Reject clips shorter than this many seconds.", kind="float", minimum=0,
                        )
                        reject_black = gr.Slider(0, 1, value=0.98, step=0.01, label="Maximum black ratio", info="Reject clips with a larger near-black pixel fraction.")
                        controls["reject_max_black_ratio"] = ctx.reg(
                            "reject_max_black_ratio", reject_black, 0.98, section="preprocessing",
                            description="Maximum allowed sampled near-black pixel ratio.", kind="float", minimum=0, maximum=1,
                        )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        reject_static = gr.Number(value=-1, step=0.1, label="Minimum motion score", info="Reject at or below this frame-difference score; -1 disables.")
                        controls["reject_max_static_score"] = ctx.reg(
                            "reject_max_static_score", reject_static, -1, section="preprocessing",
                            description="Reject clips whose sampled motion score is at or below this value.", kind="float",
                        )
                        reject_sharp = gr.Number(value=0, minimum=0, step=1, label="Minimum sharpness", info="Reject clips below this Laplacian-variance score; zero disables.")
                        controls["reject_min_sharpness"] = ctx.reg(
                            "reject_min_sharpness", reject_sharp, 0, section="preprocessing",
                            description="Minimum acceptable sampled sharpness score.", kind="float", minimum=0,
                        )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        reject_audio = gr.Checkbox(value=False, label="Require audio", info="Reject video clips without an audio stream.")
                        controls["reject_require_audio"] = ctx.reg(
                            "reject_require_audio", reject_audio, False, section="preprocessing",
                            description="Reject clips lacking audio.", kind="bool",
                        )
                        reject_silence = gr.Slider(0, 1, value=1, step=0.01, label="Maximum silence ratio", info="Reject audio above this sampled silence fraction.")
                        controls["reject_max_silence_ratio"] = ctx.reg(
                            "reject_max_silence_ratio", reject_silence, 1, section="preprocessing",
                            description="Maximum allowed sampled silence ratio.", kind="float", minimum=0, maximum=1,
                        )
                token_budget = gr.Markdown("<span class='vc-help'>Upload media to estimate the live token budget.</span>")

            with gr.Accordion("5. Scene detection & splitting", open=False):
                with gr.Row(elem_classes=["vc-compact-row"]):
                    scene_enabled = gr.Checkbox(
                        value=True,
                        label="Enable scene detection",
                        info="When Scenes mode is selected, find content cuts before captioning.",
                    )
                    controls["scene_detect_enabled"] = ctx.reg(
                        "scene_detect_enabled", scene_enabled, True, section="splitting",
                        description="Enable PySceneDetect in scene segmentation mode.", kind="bool",
                    )
                    segment_mode = gr.Radio(
                        choices=[("Whole", "whole"), ("Scenes", "scenes"), ("Fixed", "fixed"), ("Trainer", "trainer")],
                        value="scenes",
                        label="Mode",
                        info="Choose whole input, detected scenes, fixed chunks, or trainer-sized clips.",
                    )
                    controls["segment_mode"] = ctx.reg(
                        "segment_mode", segment_mode, "scenes", section="splitting",
                        description="Segmentation planning mode.", choices=["whole", "scenes", "fixed", "trainer"], kind="str",
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    detector = gr.Dropdown(
                        choices=[("Content", "content"), ("Adaptive", "adaptive"), ("Threshold / fades", "threshold")],
                        value="content",
                        label="Detector",
                        info="PySceneDetect algorithm used to find boundaries.",
                    )
                    controls["scene_detector"] = ctx.reg(
                        "scene_detector", detector, "content", section="splitting",
                        description="PySceneDetect boundary detector algorithm.", choices=["content", "adaptive", "threshold"], kind="str",
                    )
                    threshold = gr.Slider(0, 100, value=27, step=0.1, label="Threshold", info="Lower detects more cuts; 27 is the content-detector default.")
                    controls["scene_threshold"] = ctx.reg(
                        "scene_threshold", threshold, 27, section="splitting",
                        description="Scene-change detection sensitivity threshold.", kind="float", minimum=0, maximum=100,
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    scene_min = gr.Number(value=2, minimum=0, step=0.1, label="Minimum scene (s)", info="Minimum detector scene duration.")
                    controls["scene_min_len_s"] = ctx.reg(
                        "scene_min_len_s", scene_min, 2, section="splitting",
                        description="Minimum detected scene length in seconds.", kind="float", minimum=0,
                    )
                    scene_max = gr.Number(value=60, minimum=0, step=1, label="Maximum scene (s)", info="Split longer scenes; zero leaves them uncapped before the model limit.")
                    controls["scene_max_len_s"] = ctx.reg(
                        "scene_max_len_s", scene_max, 60, section="splitting",
                        description="Maximum scene length before additional splitting.", kind="float", minimum=0,
                    )
                    merge_below = gr.Number(value=2, minimum=0, step=0.1, label="Merge below (s)", info="Merge shorter scenes into their nearest neighbor.")
                    controls["merge_below_s"] = ctx.reg(
                        "merge_below_s", merge_below, 2, section="splitting",
                        description="Duration threshold used when merging short scenes.", kind="float", minimum=0,
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    merge_short = gr.Checkbox(value=True, label="Merge short scenes", info="Combine detected scenes shorter than the merge threshold.")
                    controls["merge_short_scenes"] = ctx.reg(
                        "merge_short_scenes", merge_short, True, section="splitting",
                        description="Merge short scene ranges with an adjacent range.", kind="bool",
                    )
                    fade = gr.Checkbox(value=False, label="Detect fades", info="Add threshold-based fade detection to content/adaptive cuts.")
                    controls["fade_detection"] = ctx.reg(
                        "fade_detection", fade, False, section="splitting",
                        description="Detect fades in addition to hard content cuts.", kind="bool",
                    )
                    downscale = gr.Number(value=0, minimum=0, maximum=16, step=1, precision=0, label="Detection downscale", info="Zero lets PySceneDetect choose automatically.")
                    controls["scene_downscale"] = ctx.reg(
                        "scene_downscale", downscale, 0, section="splitting",
                        description="Explicit scene-analysis downscale factor; zero is automatic.", kind="int", minimum=0, maximum=16,
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    fixed_chunk = gr.Number(value=30, minimum=0.1, step=0.5, label="Fixed chunk (s)", info="Chunk duration for Fixed mode and custom trainer fallback.")
                    controls["fixed_chunk_s"] = ctx.reg(
                        "fixed_chunk_s", fixed_chunk, 30, section="splitting",
                        description="Fixed segmentation duration in seconds.", kind="float", minimum=0.1,
                    )
                    split_mode = gr.Radio(
                        choices=[("Fast stream copy", "copy"), ("Precise re-encode", "precise")],
                        value="copy",
                        label="Cut method",
                        info="Stream copy is fast; precise mode re-encodes exact boundaries.",
                    )
                    controls["split_mode"] = ctx.reg(
                        "split_mode", split_mode, "copy", section="splitting",
                        description="Physical clip cutting method.", choices=["copy", "precise"], kind="str",
                    )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    max_clip = gr.Number(value=120, minimum=0, step=1, label="Model limit override (s)", info="Zero uses the selected model's computed duration ceiling.")
                    controls["max_clip_duration_s"] = ctx.reg(
                        "max_clip_duration_s", max_clip, 120, section="splitting",
                        description="Maximum clip duration used for automatic model-limit splitting.", kind="float", minimum=0,
                    )
                    trainer_target = gr.Dropdown(
                        choices=[(config["label"], key) for key, config in TRAINER_TARGETS.items()],
                        value="wan",
                        label="Trainer target",
                        info="Uses the trainer's default frame count and FPS to choose clip length.",
                    )
                    controls["trainer_target"] = ctx.reg(
                        "trainer_target", trainer_target, "wan", section="splitting",
                        description="Target video trainer for trainer-sized sub-splits.", choices=list(TRAINER_TARGETS), kind="str",
                    )
                    overlap = gr.Dropdown(
                        choices=[("No overlap", 0.0), ("0.5 seconds", 0.5), ("1 second", 1.0)],
                        value=0.5,
                        label="Sub-split overlap",
                        info="Overlap between model-limit, fixed, or trainer sub-clips.",
                    )
                    controls["sub_split_overlap_s"] = ctx.reg(
                        "sub_split_overlap_s", overlap, 0.5, section="splitting",
                        description="Seconds of overlap between adjacent sub-clips.", choices=[0.0, 0.5, 1.0], kind="float",
                    )
                save_clips = gr.Checkbox(value=False, label="Save produced clips", info="Persist split clips beside their captions for dataset use.")
                controls["save_clips"] = ctx.reg(
                    "save_clips", save_clips, False, section="output",
                    description="Persist materialized split clips in the output dataset.", kind="bool",
                )
                initial_model_limit = initial_spec.limits.compute_max_duration(
                    fps=initial_spec.limits.default_fps,
                    max_pixels=initial_spec.limits.default_max_pixels,
                    reserve_tokens=int(schema["max_new_tokens"].default),
                    include_audio=True,
                )
                model_limit_info = gr.Markdown(
                    f"Model-limit auto-split ceiling: **{initial_model_limit:.1f} s** at the current FPS, pixel, audio, and output-token budget.",
                    elem_classes=["vc-help"],
                )
                detect_now = action_button("◫ Detect scenes now (preview)", "indigo", size="md")
                scene_status = gr.Markdown("<span class='vc-help'>Preview uses the first selected video.</span>")
                scene_table = gr.Dataframe(
                    headers=["#", "Start", "End", "Duration", "Warning"],
                    value=[],
                    type="array",
                    datatype=["number", "number", "number", "number", "str"],
                    interactive=False,
                    max_height=280,
                    label="Scene preview",
                    buttons=["copy", "fullscreen"],
                )

            with gr.Accordion("6. Post-processing", open=False):
                with gr.Row(elem_classes=["vc-compact-row"]):
                    caption_prefix = gr.Textbox(value="", label="Prefix", info="Text prepended to the final model caption.")
                    controls["caption_prefix"] = ctx.reg(
                        "caption_prefix", caption_prefix, "", section="postprocessing",
                        description="Text prepended to finalized captions.", kind="str",
                    )
                    caption_suffix = gr.Textbox(value="", label="Suffix", info="Text appended to the final model caption.")
                    controls["caption_suffix"] = ctx.reg(
                        "caption_suffix", caption_suffix, "", section="postprocessing",
                        description="Text appended to finalized captions.", kind="str",
                    )
                    trigger_mode = gr.Dropdown(
                        choices=[("Prefix", "prefix"), ("Suffix", "suffix"), ("Prompt only / none", "none")],
                        value="prefix",
                        label="Trigger injection",
                        info="Places the template trigger word before, after, or outside the saved caption.",
                    )
                    controls["trigger_mode"] = ctx.reg(
                        "trigger_mode", trigger_mode, "prefix", section="postprocessing",
                        description="Position where the trigger word is injected into final captions.", choices=["prefix", "suffix", "none"], kind="str",
                    )
                gr.Markdown("The **Trigger word** in Template variables is shared with caption injection.", elem_classes=["vc-help"])
                replacements = replace_words_editor()
                controls["replace_words"] = ctx.reg(
                    "replace_words", replacements.text, "", section="postprocessing",
                    description="One find;replace transformation per line.", kind="str",
                )
                controls["replace_case_insensitive"] = ctx.reg(
                    "replace_case_insensitive", replacements.case_insensitive, True, section="postprocessing",
                    description="Match replacement rules without case sensitivity.", kind="bool",
                )
                controls["replace_whole_words"] = ctx.reg(
                    "replace_whole_words", replacements.whole_words, True, section="postprocessing",
                    description="Restrict replacement matches to whole words.", kind="bool",
                )
                controls["replace_regex"] = ctx.reg(
                    "replace_regex", replacements.regex, False, section="postprocessing",
                    description="Interpret replacement find values as regular expressions.", kind="bool",
                )
                collapse = gr.Checkbox(value=False, label="Collapse whitespace", info="Normalize runs of spaces and line breaks in model text before injection.")
                controls["collapse_whitespace"] = ctx.reg(
                    "collapse_whitespace", collapse, False, section="postprocessing",
                    description="Collapse model-caption whitespace before adding prefix and suffix.", kind="bool",
                )
                formats = gr.CheckboxGroup(
                    choices=[("Plain text", "txt"), ("JSON", "json"), ("SRT", "srt"), ("WebVTT", "vtt"), ("Dataset JSONL", "jsonl")],
                    value=["txt", "json"],
                    label="Output formats",
                    info="Plain text is always written even when not explicitly selected.",
                    show_select_all=True,
                )
                controls["output_formats"] = ctx.reg(
                    "output_formats", formats, ["txt", "json"], section="output",
                    description="Caption and dataset file formats written for every item.",
                )
                save_reasoning = gr.Checkbox(
                    value=False,
                    label="Save reasoning",
                    info="Persist Thinking-model reasoning separately; it remains hidden from the caption.",
                )
                controls["save_reasoning"] = ctx.reg(
                    "save_reasoning", save_reasoning, False, section="output",
                    description="Write Thinking-model reasoning to a separate text file.", kind="bool",
                )

    last_outputs_state = gr.State({})
    ctx.states["last_outputs"] = last_outputs_state

    handles = CaptionTabHandles(
        media=media,
        progress=progress,
        logs=logs,
        controls=controls,
        start=start,
        cancel=cancel,
        hotkey_start=hotkey_start,
        hotkey_cancel=hotkey_cancel,
        open_output=open_output,
        open_caption=open_caption,
        reveal_clip=reveal_clip,
        item_table=item_table,
        caption=caption,
        structured=structured,
        srt=srt,
        reasoning=reasoning,
        reasoning_tab=reasoning_tab,
        clips=clips,
        last_outputs_state=last_outputs_state,
    )
    ctx.caption_handles = handles

    # Lightweight local interactions can be wired immediately.
    gpu_picker.change(
        lambda value: int(value or 0),
        inputs=gpu_picker,
        outputs=ctx.states["gpu_index"],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    refresh_ready.click(
        _ready_line,
        inputs=model_key,
        outputs=ready_status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    model_key.change(
        lambda value: (_quant_line(value), _ready_line(value)),
        inputs=model_key,
        outputs=[quant_info, ready_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    def set_resolution(value: str) -> Any:
        return gr.skip() if value == "custom" else int(value)

    resolution_preset.input(
        set_resolution,
        inputs=resolution_preset,
        outputs=max_pixels,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def sample_hint(temp: float, sampled: bool) -> str:
        if sampled:
            return "<span class='vc-ok'>Sampling enabled.</span> Temperature and top-p/top-k are active."
        if float(temp or 0) > 0:
            return "<span class='vc-warn'>Temperature is above zero, but sampling is disabled; decoding remains greedy.</span>"
        return "<span class='vc-help'>Deterministic greedy decoding.</span>"

    for event in (temperature.change, do_sample.change):
        event(
            sample_hint,
            inputs=[temperature, do_sample],
            outputs=sample_note,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def apply_vram(selected_variant: str, selected_tier: str, selected_gpu: int) -> tuple[Any, ...]:
        try:
            family = variant_to_family(selected_variant)
            total = next(
                (device.total_gb for device in __import__("vcap.core.gpu", fromlist=["list_gpus"]).list_gpus() if device.index == int(selected_gpu)),
                gpu_total or 32,
            )
            resolved_tier = auto_tier(total) if selected_tier == "auto" else int(selected_tier)
            preset = preset_for(family, resolved_tier)
            applied = apply_preset({}, preset)
            candidates = [
                variant.key
                for variant in MODEL_SPECS[family].variants
                if variant.scheme == applied["variant_scheme"]
                and variant.key in allowed_variants(family, resolved_tier)
            ]
            next_variant = candidates[0] if candidates else selected_variant
            offload = preset.offload
            note = f"<span class='vc-ok'>{resolved_tier} GB plan applied.</span> {html.escape(preset.notes)}"
            return (
                gr.update(value=next_variant),
                preset.attention,
                preset.fps,
                preset.max_frames,
                preset.max_pixels,
                preset.max_new_tokens,
                str(offload.gpu_layers),
                offload.offload_experts,
                offload.pin_cpu,
                note,
            )
        except Exception as exc:
            return (*[gr.skip() for _ in range(9)], f"<span class='vc-err'>{html.escape(str(exc))}</span>")

    vram_preset.input(
        apply_vram,
        inputs=[model_key, vram_preset, gpu_picker],
        outputs=[model_key, attention, fps, max_frames, max_pixels, max_new_tokens, gpu_layers, offload_experts, pin_cpu, vram_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def model_defaults(variant_key: str) -> tuple[Any, ...]:
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        values = {item.name: item for item in spec.param_schema}
        thinking = "enable_thinking" in values
        return (
            gr.update(value=values["temperature"].default, minimum=0, maximum=2, step=0.01),
            gr.update(value=values["top_p"].default, minimum=0, maximum=1, step=0.01),
            gr.update(value=values["top_k"].default, minimum=0, maximum=200, step=1),
            gr.update(value=values["repetition_penalty"].default, minimum=0.5, maximum=2, step=0.01),
            gr.update(value=values["max_new_tokens"].default, minimum=1, maximum=spec.limits.max_new_tokens_cap, step=1),
            gr.update(value=values["do_sample"].default),
            gr.update(value=bool(values.get("enable_thinking") and values["enable_thinking"].default), interactive=thinking),
            gr.update(value=values["fps"].default, minimum=values["fps"].min, maximum=values["fps"].max, step=values["fps"].step),
            gr.update(value=values["max_frames"].default, minimum=0, maximum=values["max_frames"].max, step=values["max_frames"].step),
            gr.update(value=values["max_pixels"].default, minimum=values["max_pixels"].min, maximum=values["max_pixels"].max, step=values["max_pixels"].step),
            gr.update(value=spec.limits.min_pixels, minimum=4 * spec.limits.size_multiple**2),
            gr.update(value=values["use_audio_in_video"].default, interactive="video_audio" in spec.capabilities),
        )

    def model_constraints(variant_key: str) -> tuple[Any, ...]:
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        values = {item.name: item for item in spec.param_schema}
        return (
            gr.update(minimum=0, maximum=2, step=0.01),
            gr.update(minimum=0, maximum=1, step=0.01),
            gr.update(minimum=0, maximum=200, step=1),
            gr.update(minimum=0.5, maximum=2, step=0.01),
            gr.update(minimum=1, maximum=spec.limits.max_new_tokens_cap, step=1),
            gr.skip(),
            gr.update(interactive="enable_thinking" in values),
            gr.update(minimum=values["fps"].min, maximum=values["fps"].max, step=values["fps"].step),
            gr.update(minimum=0, maximum=values["max_frames"].max, step=values["max_frames"].step),
            gr.update(minimum=values["max_pixels"].min, maximum=values["max_pixels"].max, step=values["max_pixels"].step),
            gr.update(minimum=4 * spec.limits.size_multiple**2, maximum=values["max_pixels"].max),
            gr.update(interactive="video_audio" in spec.capabilities),
        )

    model_key.change(
        model_constraints,
        inputs=model_key,
        outputs=[temperature, top_p, top_k, repetition, max_new_tokens, do_sample, enable_thinking, fps, max_frames, max_pixels, min_pixels, use_audio],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    model_key.input(
        model_defaults,
        inputs=model_key,
        outputs=[temperature, top_p, top_k, repetition, max_new_tokens, do_sample, enable_thinking, fps, max_frames, max_pixels, min_pixels, use_audio],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    variable_components = [
        trigger_word,
        language,
        source_language,
        target_language,
        caption_length,
        avoid_list,
        subject_class,
        extra_instructions,
    ]

    def render_selected(preset_id: str, *values: Any) -> tuple[str, str, str]:
        try:
            preset = get_preset(str(preset_id))
            system, user = render_prompt(preset, _prompt_variables(values))
            return _display(preset.description), system or "", user
        except Exception as exc:
            return f"<span class='vc-err'>{html.escape(str(exc))}</span>", "", ""

    def select_prompt(preset_id: str, variant_key: str, *values: Any) -> tuple[Any, ...]:
        description, system, user = render_selected(preset_id, *values)
        try:
            preset = get_preset(str(preset_id))
            schema_values = {item.name: item.default for item in MODEL_SPECS[variant_to_family(variant_key)].param_schema}
            merged = {**schema_values, **preset.generation_overrides}
            return (
                description,
                system,
                user,
                merged.get("temperature", gr.skip()),
                merged.get("top_p", gr.skip()),
                merged.get("top_k", gr.skip()),
                merged.get("repetition_penalty", gr.skip()),
                merged.get("max_new_tokens", gr.skip()),
                merged.get("do_sample", gr.skip()),
            )
        except Exception:
            return description, system, user, *[gr.skip() for _ in range(6)]

    prompt_preset.input(
        select_prompt,
        inputs=[prompt_preset, model_key, *variable_components],
        outputs=[prompt_description, system_prompt, user_prompt, temperature, top_p, top_k, repetition, max_new_tokens, do_sample],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    reset_prompts.click(
        render_selected,
        inputs=[prompt_preset, *variable_components],
        outputs=[prompt_description, system_prompt, user_prompt],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    prompt_preset.change(
        lambda preset_id: _display(get_preset(str(preset_id)).description) if preset_id else "",
        inputs=prompt_preset,
        outputs=prompt_description,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    for component in variable_components:
        component.input(
            render_selected,
            inputs=[prompt_preset, *variable_components],
            outputs=[prompt_description, system_prompt, user_prompt],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def filter_prompts(variant_key: str, modality: str, *values: Any) -> tuple[Any, str, str, str]:
        family = variant_to_family(variant_key)
        choices = _prompt_choices(family, modality)
        try:
            preset = default_preset_for(family, modality)
        except KeyError:
            presets = list_presets(family, modality)
            if not presets:
                return gr.update(choices=[], value=None), "<span class='vc-warn'>No compatible task preset for this model and modality.</span>", "", ""
            preset = presets[0]
        system, user = render_prompt(preset, _prompt_variables(values))
        return gr.update(choices=choices, value=preset.id), _display(preset.description), system or "", user

    model_key.input(
        filter_prompts,
        inputs=[model_key, media.modality_state, *variable_components],
        outputs=[prompt_preset, prompt_description, system_prompt, user_prompt],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    media.modality_state.change(
        filter_prompts,
        inputs=[model_key, media.modality_state, *variable_components],
        outputs=[prompt_preset, prompt_description, system_prompt, user_prompt],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def budget_line(variant_key: str, fps_value: float, frames_value: int, pixels_value: int, duration: float, output_tokens: int, include_audio: bool) -> str:
        if float(duration or 0) <= 0:
            return "<span class='vc-help'>Upload media to estimate the live token budget.</span>"
        try:
            family = variant_to_family(variant_key)
            spec = MODEL_SPECS[family]
            frame_count = min(max(1, int(frames_value or 1)), max(1, int(math.ceil(float(duration) * float(fps_value)))))
            factor = spec.limits.size_multiple
            width = max(factor, int(pixels_value or spec.limits.default_max_pixels) // factor)
            budget_family = "qwen3-omni" if family.startswith("qwen3") else family
            estimate = token_budget_estimate(
                budget_family,
                frame_count,
                factor,
                width,
                float(duration) if include_audio else 0.0,
            )
            total = int(estimate["total_input_tokens"])
            ok = fits_context(estimate, spec.limits.context_tokens, int(output_tokens or 0))
            css, word = ("vc-ok", "OK") if ok else ("vc-err", "OVER BUDGET")
            return (
                f"≈ {total:,} input tokens of {spec.limits.context_tokens:,} — "
                f"<span class='{css}'>{word}</span> · {frame_count} frames · reserves {int(output_tokens or 0):,} output tokens"
            )
        except Exception as exc:
            return f"<span class='vc-warn'>Token estimate unavailable: {html.escape(str(exc))}</span>"

    budget_inputs = [model_key, fps, max_frames, max_pixels, media.duration_state, max_new_tokens, use_audio]
    for event_component in budget_inputs:
        event_component.change(
            budget_line,
            inputs=budget_inputs,
            outputs=token_budget,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def limit_line(variant_key: str, fps_value: float, pixels_value: int, reserve: int, include_audio: bool) -> str:
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        limit = spec.limits.compute_max_duration(
            fps=float(fps_value or spec.limits.default_fps),
            max_pixels=int(pixels_value or spec.limits.default_max_pixels),
            reserve_tokens=int(reserve or 0),
            include_audio=bool(include_audio),
        )
        return f"Model-limit auto-split ceiling: **{limit:.1f} s** at the current FPS, pixel, audio, and output-token budget."

    limit_inputs = [model_key, fps, max_pixels, max_new_tokens, use_audio]
    for event_component in limit_inputs:
        event_component.change(
            limit_line,
            inputs=limit_inputs,
            outputs=model_limit_info,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def scene_preview(
        selected: list[str],
        detector_name: str,
        threshold_value: float,
        minimum: float,
        maximum: float,
        merge: bool,
        merge_limit: float,
        fades: bool,
        scale: int,
        model_limit: float,
    ):
        yield [], "<span class='vc-warn'>Detecting scenes…</span>"
        first = next((Path(value) for value in selected or [] if probe_media(value).has_video), None)
        if first is None:
            yield [], "<span class='vc-err'>Select at least one video first.</span>"
            return
        params = SceneDetectParams(
            threshold=float(threshold_value),
            min_scene_len_s=float(minimum),
            max_scene_len_s=float(maximum),
            merge_short_scenes=bool(merge),
            merge_below_s=float(merge_limit),
            fade_detection=bool(fades),
            detector=str(detector_name),
            downscale=int(scale or 0),
        )
        token = CancelToken()
        ctx.activate_cancel(token)
        try:
            scenes = detect_scenes(
                first,
                params,
                progress_cb=lambda fraction, message: ctx.app_log.log(f"{message} ({fraction * 100:.1f}%)", scope="scene-preview"),
                cancel=token,
            )
            if merge:
                scenes = merge_short_scenes(scenes, float(merge_limit))
            if float(maximum or 0) > 0:
                scenes = cap_scene_lengths(scenes, float(maximum))
            rows = []
            for index, scene in enumerate(scenes, start=1):
                warnings: list[str] = []
                if scene.duration_s < float(minimum):
                    warnings.append("below minimum")
                if float(model_limit or 0) > 0 and scene.duration_s > float(model_limit):
                    warnings.append("will auto-split")
                rows.append([index, round(scene.start_s, 3), round(scene.end_s, 3), round(scene.duration_s, 3), ", ".join(warnings)])
            status = (
                f"<span class='vc-ok'>Detected {len(rows)} scene(s) in {html.escape(first.name)}.</span>"
                if rows
                else "<span class='vc-warn'>No cuts detected; the pipeline will use the selected range as one clip.</span>"
            )
            yield rows, status
        except Exception as exc:
            yield [], f"<span class='vc-err'>{html.escape(str(exc))}</span>"
        finally:
            ctx.clear_active_cancel(token)

    detect_now.click(
        scene_preview,
        inputs=[media.resolved_state, detector, threshold, scene_min, scene_max, merge_short, merge_below, fade, downscale, max_clip],
        outputs=[scene_table, scene_status],
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    def clear_caches() -> str:
        report = clear_inductor_caches()
        removed = len(report.get("removed") or [])
        errors = report.get("errors") or []
        ctx.app_log.log(f"Cleared {removed} compile cache location(s).", scope="compile")
        for error in errors:
            ctx.app_log.warn(error, scope="compile")
        return _probe_compile_in_child()

    clear_compile.click(
        clear_caches,
        outputs=compile_status,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    def download_model(variant_key: str):
        token = CancelToken()
        ctx.activate_cancel(token)
        status_queue: queue.Queue[str] = queue.Queue()
        result: dict[str, Any] = {}

        def progress_callback(line: str, payload: Any = None) -> None:
            del payload
            message = str(line)
            status_queue.put(message)
            ctx.app_log.log(message, scope="download")

        def work() -> None:
            try:
                result["value"] = ensure_model(variant_key, progress_callback, token)
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=work, daemon=True, name="vcap-model-download-ui")
        thread.start()
        last = "Starting model verification…"
        try:
            while thread.is_alive() or not status_queue.empty():
                try:
                    while True:
                        last = status_queue.get_nowait()
                except queue.Empty:
                    pass
                yield f"<span class='vc-warn'>{html.escape(last)}</span>"
                time.sleep(0.18)
            thread.join()
            if "error" in result:
                raise result["error"]
            yield _ready_line(variant_key)
        except BaseException as exc:
            yield f"<span class='vc-err'>{html.escape(str(exc))}</span>"
        finally:
            ctx.clear_active_cancel(token)

    download.click(
        download_model,
        inputs=model_key,
        outputs=ready_status,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    return handles


class _UiSink:
    def __init__(self, ctx: "UiContext", events: "queue.Queue[tuple[str, Any]]", mirror_logs: bool) -> None:
        self.ctx = ctx
        self.events = events
        self.mirror_logs = mirror_logs

    def on_log(self, message: str, level: str = "info", scope: str | None = None) -> None:
        if self.mirror_logs:
            self.ctx.app_log.log(message, level=level, scope=scope)

    def on_progress(self, event: ProgressEvent) -> None:
        self.events.put(("progress", event))

    def on_item(self, event: ProgressEvent) -> None:
        self.events.put(("item", event))


def _result_payload(result: JobResult) -> tuple[str, Any, str, str, list[Any], dict[str, Any]]:
    completed = [item for item in result.items if item.status == "done"]
    if not completed:
        return "", None, "", "", [], {"run_dir": result.run_dir}
    item = completed[-1]
    caption_path = item.outputs.get("txt")
    json_path = item.outputs.get("json")
    srt_path = item.outputs.get("srt")
    reasoning_path = item.outputs.get("reasoning")
    caption = _read_text(caption_path).rstrip()
    structured = _read_json(json_path)
    if structured is None:
        records = [record.get("structured") for record in item.segments if record.get("structured") is not None]
        structured = records[0] if len(records) == 1 else records or None
    srt = _read_text(srt_path).rstrip()
    if not srt:
        cues = [
            (
                float(record.get("start_s", 0.0) or 0.0),
                float(record.get("end_s", 0.0) or 0.0),
                str(record.get("caption") or ""),
            )
            for record in item.segments
            if record.get("status") == "done" and record.get("caption")
        ]
        srt = to_srt(cues).rstrip()
    reasoning = _read_text(reasoning_path).rstrip()
    if not reasoning:
        reasoning = "\n\n".join(
            str(record.get("reasoning") or "")
            for record in item.segments
            if record.get("reasoning")
        ).strip()
    gallery: list[Any] = []
    clip_paths: list[str] = []
    for record in item.segments:
        raw = record.get("media_path")
        if raw and Path(raw).is_file():
            clip_paths.append(str(raw))
            if probe_media(raw).kind in {"video", "video_no_audio", "image"}:
                gallery.append(
                    (
                        str(raw),
                        f"Clip {record.get('index', len(gallery) + 1)} · {float(record.get('start_s', 0)):.2f}–{float(record.get('end_s', 0)):.2f}s",
                    )
                )
    state = {
        "run_dir": result.run_dir,
        "metadata_path": result.metadata_path,
        "caption_path": caption_path,
        "clip_path": clip_paths[-1] if clip_paths else None,
    }
    return caption, structured, srt, reasoning, gallery, state


def _result_summary(result: JobResult) -> tuple[str, str, str, str]:
    """Return terminal label, message, status class, and ETA text."""

    counts = result.counts
    done = int(counts.get("done", 0))
    skipped = int(counts.get("skipped", 0))
    failed = int(counts.get("failed", 0))
    cancelled = int(counts.get("cancelled", 0))
    if cancelled:
        return (
            "Cancelled",
            f"Cancelled: {cancelled} cancelled, {done} done, {skipped} skipped, "
            f"{failed} failed in {result.elapsed:.1f}s",
            "vc-warn",
            "cancelled",
        )
    return (
        "Complete",
        f"Complete: {done} done, {skipped} skipped, {failed} failed in {result.elapsed:.1f}s",
        "vc-ok",
        "done",
    )


def wire(ctx: "UiContext") -> None:
    """Wire registry-wide start/cancel/preset-safe events after every tab is built."""

    handles = ctx.caption_handles
    if handles is None:
        raise RuntimeError("caption_tab.build() must run before wire()")
    registry = ctx.settings_registry
    registry_components = registry.components()
    output_components = [
        handles.progress.bars,
        handles.progress.status,
        handles.progress.eta,
        handles.progress.tokens,
        handles.item_table,
        handles.caption,
        handles.structured,
        handles.srt,
        handles.reasoning,
        handles.reasoning_tab,
        handles.clips,
        handles.last_outputs_state,
        handles.cancel,
    ]

    def run_caption(*args: Any):
        value_count = len(registry_components)
        settings = registry.values_to_dict(args[:value_count])
        resolved = [str(value) for value in (args[value_count] or [])]
        input_mode = str(args[value_count + 1] or "upload")
        token = CancelToken()
        ctx.activate_cancel(token)
        items: list[InputItem] = [InputItem(path=value) for value in resolved]
        if not items:
            family = variant_to_family(str(settings.get("model_key", _INITIAL_VARIANT)))
            if "text" not in MODEL_SPECS[family].capabilities:
                message = "Select at least one input. Text-only queries require a Qwen3-Omni Instruct or Thinking model."
                yield (
                    render_progress_html(0, "Input required", message),
                    f"<span class='vc-err'>{message}</span>",
                    "**ETA:** —",
                    "**Speed:** —",
                    [],
                    *[gr.skip() for _ in range(7)],
                    gr.update(value="⏹ Cancel"),
                )
                ctx.clear_active_cancel(token)
                return
            items = [InputItem(path="", kind="text", text_prompt_only=True, text=str(settings.get("user_prompt") or ""))]

        segment_mode = str(settings.get("segment_mode") or "whole")
        if segment_mode == "scenes" and not bool(settings.get("scene_detect_enabled")):
            segment_mode = "whole"
        settings["segment_mode"] = segment_mode
        settings["compile_mode"] = settings.get("torch_compile_mode", "cudagraphs")
        settings["recursive"] = bool(settings.get("batch_recursive", settings.get("scan_subfolders", False)))
        output_kind = "batch" if input_mode == "folder" else "single"
        output = OutputSpec(
            kind=output_kind,
            outputs_root=str(settings.get("outputs_dir") or ctx.outputs_dir),
            batch_output_dir=(str(settings.get("batch_output_folder") or ctx.outputs_dir / "batch_captions") if output_kind == "batch" else None),
            mirror_names=True,
            overwrite=bool(settings.get("overwrite_existing", False)),
            save_processed_files=bool(settings.get("save_processed_files", False)),
            save_clips=bool(settings.get("save_clips", False)),
            recursive=bool(settings["recursive"]),
        )
        spec = JobSpec.from_settings(settings, items, output)
        ctx.pipeline_client.subprocess_mode = bool(settings.get("subprocess_mode", True))
        event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        sink = _UiSink(ctx, event_queue, mirror_logs=ctx.pipeline_client.subprocess_mode)
        terminal: dict[str, Any] = {}

        def work() -> None:
            try:
                terminal["result"] = ctx.pipeline_client.run_job(spec, sink, token)
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                event_queue.put(("terminal", None))

        thread = threading.Thread(target=work, daemon=True, name="vcap-caption-ui")
        thread.start()
        item_rows = [
            [index, Path(item.path).name if item.path else "Text prompt", "queued", "Waiting", "—"]
            for index, item in enumerate(items, start=1)
        ]
        last_emit = 0.0
        last_fraction = 0.0
        last_message = "Starting caption job"
        last_eta = "—"
        last_speed = "—"
        yield (
            render_progress_html(0, "Starting", f"Preparing {len(items)} item(s)."),
            "**Status:** Starting caption worker",
            "**ETA:** —",
            "**Speed:** —",
            item_rows,
            "",
            None,
            "",
            "",
            gr.update(visible=False),
            [],
            {},
            gr.update(value="⏹ Cancel"),
        )
        finished = False
        try:
            while not finished:
                kind, payload = event_queue.get()
                if kind == "terminal":
                    finished = True
                    continue
                now = time.monotonic()
                if kind == "progress":
                    event: ProgressEvent = payload
                    last_fraction = float(event.fraction or 0.0)
                    last_message = event.message
                    data = event.data or {}
                    eta_seconds = data.get("eta_seconds")
                    last_eta = format_eta(eta_seconds) if eta_seconds is not None else "—"
                    speed = data.get("tok_per_s") or data.get("tokens_per_second")
                    if speed is not None:
                        last_speed = f"{float(speed):.2f} tok/s"
                    index = int(event.item_index or 0)
                    if 0 <= index < len(item_rows):
                        item_rows[index][2] = "running"
                        item_rows[index][3] = event.message
                elif kind == "item":
                    event = payload
                    index = int(event.item_index or 0)
                    if 0 <= index < len(item_rows):
                        item_rows[index][2] = str(event.status or "done")
                        item_rows[index][3] = event.message
                        item_rows[index][4] = f"{float((event.data or {}).get('elapsed', 0)):.1f}s"
                    last_message = event.message
                    last_fraction = float(event.fraction or last_fraction)
                if now - last_emit < 0.12 and kind == "progress":
                    continue
                last_emit = now
                yield (
                    render_progress_html(last_fraction, last_message, f"{len(items)} item(s) in this job"),
                    f"**Status:** {html.escape(last_message)}",
                    f"**ETA:** {last_eta}",
                    f"**Speed:** {last_speed}",
                    item_rows,
                    *[gr.skip() for _ in range(8)],
                )
            thread.join()
            if "error" in terminal:
                raise terminal["error"]
            result: JobResult = terminal["result"]
            final_caption, structured, srt_text, reasoning_text, gallery, state = _result_payload(result)
            terminal_label, message, status_class, eta_text = _result_summary(result)
            yield (
                render_progress_html(1, terminal_label, message),
                f"<span class='{status_class}'>**Status:** {html.escape(message)}</span>",
                f"**ETA:** {eta_text}",
                f"**Speed:** {last_speed}",
                item_rows,
                final_caption,
                structured,
                srt_text,
                reasoning_text,
                gr.update(visible=bool(reasoning_text)),
                gallery,
                state,
                gr.update(value="⏹ Cancel"),
            )
        except (CancelledError, KeyboardInterrupt) as exc:
            ctx.app_log.warn(str(exc), scope="cancel")
            yield (
                render_progress_html(last_fraction, "Cancelled", str(exc)),
                f"<span class='vc-warn'>**Status:** {html.escape(str(exc))}</span>",
                "**ETA:** cancelled",
                f"**Speed:** {last_speed}",
                item_rows,
                *[gr.skip() for _ in range(7)],
                gr.update(value="⏹ Cancel"),
            )
        except BaseException as exc:
            ctx.app_log.exception(f"Caption job failed: {exc}", scope="ui")
            yield (
                render_progress_html(last_fraction, "Failed", str(exc)),
                f"<span class='vc-err'>**Status:** {html.escape(str(exc))}</span>",
                "**ETA:** failed",
                f"**Speed:** {last_speed}",
                item_rows,
                *[gr.skip() for _ in range(7)],
                gr.update(value="⏹ Cancel"),
            )
        finally:
            ctx.clear_active_cancel(token)

    run_inputs = [*registry_components, handles.media.resolved_state, handles.media.mode_state]
    handles.start.click(
        run_caption,
        inputs=run_inputs,
        outputs=output_components,
        api_name="caption",
        api_description="Caption the selected files or batch folder with the current registered settings.",
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="public",
    )
    handles.hotkey_start.click(
        run_caption,
        inputs=run_inputs,
        outputs=output_components,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    def cancel_job() -> tuple[Any, str]:
        token = ctx.get_active_cancel()
        if token is None or token.is_cancelled():
            return gr.update(value="⏹ Cancel"), "**Status:** No active job to cancel."
        token.cancel()
        ctx.pipeline_client.cancel(force=False)
        return gr.update(value="Cancelling…"), "<span class='vc-warn'>**Status:** Cooperative cancellation requested.</span>"

    for trigger in (handles.cancel.click, handles.hotkey_cancel.click):
        trigger(
            cancel_job,
            outputs=[handles.cancel, handles.progress.status],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def open_last(state: dict[str, Any], key: str, reveal: bool = False) -> str:
        value = (state or {}).get(key)
        if not value:
            return f"<span class='vc-warn'>No {html.escape(key.replace('_', ' '))} is available yet.</span>"
        ok, message = reveal_in_file_manager(value) if reveal else open_in_file_manager(value)
        css = "vc-ok" if ok else "vc-err"
        return f"<span class='{css}'>{html.escape(message)}</span>"

    handles.open_output.click(
        lambda state: open_last(state, "run_dir"),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.open_caption.click(
        lambda state: open_last(state, "caption_path"),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.reveal_clip.click(
        lambda state: open_last(state, "clip_path", True),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )


__all__ = ["CaptionTabHandles", "build", "wire"]
