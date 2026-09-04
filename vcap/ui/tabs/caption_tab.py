"""Primary caption workflow and all model/pipeline controls."""

from __future__ import annotations

import html
import importlib.metadata
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import gradio as gr

from vcap import TEMP_DIR
from vcap.core.clip_fitness import TRAINER_TARGETS
from vcap.core.captions_post import to_srt
from vcap.core.dataset_captions import (
    DEFAULT_AUDIO_CAPTION_TEMPLATE,
    DEFAULT_CAPTION_MERGE_TEMPLATE,
)
from vcap.core.media import probe_media, read_video_frames
from vcap.core.paths import normalize_path, open_in_file_manager, reveal_in_file_manager
from vcap.core.preprocess import (
    FrameSamplingParams,
    fits_context,
    plan_frame_sampling,
    token_budget_estimate,
)
from vcap.ui.components import context_usage_text
from vcap.core.progress import ProgressEvent, format_eta
from vcap.core.scene_split import (
    SceneDetectParams,
    cap_scene_lengths,
    detect_scenes,
    merge_short_scenes,
)
from vcap.core.subprocess_runner import CancelToken, CancelledError, build_child_env
from vcap.core.gpu import _default_gpu_index, resource_snapshot
from vcap.models import attention as attention_module
from vcap.models.attention import ATTENTION_CHOICES
from vcap.models.downloads import ensure_model
from vcap.models.offload import (
    BudgetHint,
    OffloadPlan,
    block_swap_to_gpu_layers,
    family_layer_count,
    migrate_legacy_gpu_layers,
    plan_model_folder,
)
from vcap.models.registry import (
    MODEL_SPECS,
    all_variant_choices,
    get_variant,
    resolve_model_dir,
    variant_size_gb,
    variant_is_ready,
    variant_to_family,
)
from vcap.models.torch_compile import (
    DEFAULT_COMPILE_MODE,
    clear_inductor_caches,
    compile_mode_choices,
    compile_mode_values,
)
from vcap.models.vram_presets import (
    VRAM_TIERS,
    allowed_variants,
    apply_preset,
    auto_tier,
    preset_for,
)

# Slider ceilings never shrink below a loaded value: Gradio rejects out-of-range
# inputs at event time, so the UI keeps the global cap and the pipeline clamps
# max_new_tokens to the selected family's real limit.
_GLOBAL_MAX_NEW_TOKENS = max(spec.limits.max_new_tokens_cap for spec in MODEL_SPECS.values())
_GLOBAL_MAX_CONTEXT = max(spec.limits.context_tokens for spec in MODEL_SPECS.values())


def _schema_bound(name: str, attribute: str, pick: Any) -> float:
    return pick(
        float(getattr(next(item for item in spec.param_schema if item.name == name), attribute))
        for spec in MODEL_SPECS.values()
    )


# Backend bounds for Number/Slider controls are fixed at construction to the
# widest value across families and are never narrowed by updates. Gradio
# validates every incoming value against the *backend* bound, and sibling
# handlers on one trigger race: one could narrow the bound before the browser
# applied the value another one still carries, which raised
# "Value 768 is greater than maximum value 160". Family limits live in help
# text, in value clamping, and in JobSpec instead.
_GLOBAL_MAX_FRAMES = int(_schema_bound("max_frames", "max", max))
_GLOBAL_MIN_PIXELS = int(min(4 * spec.limits.size_multiple**2 for spec in MODEL_SPECS.values()))
_GLOBAL_MAX_PIXELS = int(_schema_bound("max_pixels", "max", max))
_GLOBAL_FPS_MIN = _schema_bound("fps", "min", min)
_GLOBAL_FPS_MAX = _schema_bound("fps", "max", max)

DEFAULT_SUMMARY_PROMPT = (
    'You are given timestamped captions of consecutive segments of one video. Write (1) one '
    'paragraph summarizing the whole video in {{LANGUAGE}}, then (2) a chapter list with one line '
    'per chapter formatted as "MM:SS-MM:SS Title - one sentence". Use only information present in '
    "the captions and keep the chronological order."
)

_GGUF_UNUSED_SUFFIX = " (not used by GGUF)"
_CONTROL_INFO = {
    "attention_backend": "Unavailable optimized backends fall back safely to PyTorch SDPA.",
    "block_swap_auto": (
        "Fits the decoder to free VRAM minus the reserve at load time and shows the resulting "
        "swapped-layer count; uncheck to set the count yourself."
    ),
    "swap_slots": "Advanced: GPU staging slots used to prefetch swapped decoder layers.",
    "offload_experts": "Legacy Accelerate expert offload; disables block swap.",
    "pin_cpu": "Use pinned host memory for faster decoder-layer transfers.",
    "pinned_ram_budget_gb": (
        "Maximum system RAM used for pinned block-swap layers; 0 = automatic (total RAM minus 6 GB)."
    ),
    "plan_slack_mib": (
        "Fixed CUDA-allocator slack (MiB) added to the activation estimate when planning how many "
        "decoder layers stay resident. Raise it if runs still go out of memory; lower it to keep "
        "more layers on the GPU."
    ),
    "torch_compile": (
        "The first generation can spend 1-5 minutes compiling kernels; later runs reuse them. "
        "A compile runtime failure restores the loaded model and retries that segment eagerly."
    ),
    "torch_compile_mode": (
        "Both choices avoid explicit CUDA graph replay, which is incompatible with the DynamicCache "
        "used by these decoders. Max autotune has a longer first run."
    ),
    "use_cache": "Speeds autoregressive decoding at the cost of additional VRAM.",
    "no_repeat_ngram_size": (
        "Blocks any word-piece n-gram of this size from repeating inside one generation "
        "(Transformers backends only; 0 disables). Stops looping captions at the cost of some "
        "natural repetition; 3-6 is typical. GGUF ignores it (llama.cpp has no n-gram blocking)."
    ),
}


def _frames_info(spec: Any) -> str:
    cap = int(next(item for item in spec.param_schema if item.name == "max_frames").max)
    return (
        f"Hard cap applied after sampling; {spec.label} uses at most {cap} frames and higher "
        "values are clamped at run time. 0 = caption the audio track only for Qwen3-Omni; "
        "7B models raise it to the family minimum with a logged warning."
    )


def _family_max_frames(spec: Any) -> int:
    return int(next(item for item in spec.param_schema if item.name == "max_frames").max)


def _context_window(spec: Any, requested: Any) -> int:
    """Clamp a requested context window to the family cap (blank means the cap)."""

    cap = int(spec.limits.context_tokens)
    try:
        value = int(float(requested)) if requested not in (None, "") else cap
    except (TypeError, ValueError):
        value = cap
    return max(1024, min(cap, value)) if value > 0 else cap


def _context_info(spec: Any) -> str:
    """Model-specific help for the context control, including its KV-cache cost."""

    cap = int(spec.limits.context_tokens)
    return (
        f"Total window for prompt, media, and reply. {spec.label} allows up to {cap:,} tokens; "
        f"its KV cache is about {spec.limits.kv_cache_gb(cap):.1f} GB at the full window "
        f"({spec.limits.kv_cache_gb(cap // 2):.1f} GB at {cap // 2:,}). GGUF servers reserve it "
        "at load, Transformers grow into it, and long chats are trimmed to fit."
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
    _folder_scan,
    _paths,
    action_button,
    log_panel,
    media_input_block,
    newest_first,
    progress_panel,
    render_progress_html,
    replace_words_editor,
)

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


_INITIAL_VARIANT = "qwen3_omni_instruct_int4"
_INITIAL_MODALITY = "video_audio"
_CAPTION_LENGTH_CHOICES = ("short", "medium", "detailed", "very detailed")
_PRIMARY_PROMPT_MODALITIES = {
    "timechat": "video_audio",
    "avocado": "video_audio",
    "qwen3_omni_instruct": "video",
    "qwen3_omni_thinking": "video",
    "qwen3_omni_captioner": "audio",
}
_COMPILE_PROBE_CACHE = TEMP_DIR / "compile_probe.json"
_COMPILE_PROBE_TTL_S = 24 * 60 * 60
_COMPILE_PROBE_LOCK = threading.Lock()
_COMPILE_PROBE_HTML: str | None = None
_COMPILE_PROBE_RUNNING = False


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


def _audio_captioner_choices() -> list[tuple[str, str]]:
    """Return Captioner variants with lightweight local download state."""

    choices = [("Auto-match the selected model precision/backend", "auto")]
    spec = MODEL_SPECS["qwen3_omni_captioner"]
    for variant in spec.variants:
        ready, _ = variant_is_ready(variant.key)
        state = " ✓ downloaded" if ready else ""
        choices.append(
            (
                f"{spec.label} — {variant.label} ({variant_size_gb(variant.key):.1f} GB){state}",
                variant.key,
            )
        )
    return choices


def validate_model_variant(
    value: str,
    previous_valid: str | None,
) -> tuple[Any, str, Any]:
    """Reject an unregistered combobox value without changing model planning."""

    labels = {key: label for label, key in _variant_choices()}
    fallback = str(previous_valid or _INITIAL_VARIANT)
    if fallback not in labels:
        fallback = _INITIAL_VARIANT
    selected = str(value or "")
    if selected in labels:
        return gr.skip(), selected, gr.skip()
    status = (
        "<span class='vc-warn'>Unknown model variant; kept "
        f"{html.escape(labels[fallback])}</span>"
    )
    return gr.update(value=fallback), fallback, status


def variant_choices_for_tier(
    selected_variant: str,
    tier: int | float,
    show_all: bool = False,
) -> list[tuple[str, str]]:
    """Filter every model family to variants supported by a VRAM tier."""

    all_choices = _variant_choices()
    if show_all:
        return all_choices
    allowed: set[str] = set()
    for family in MODEL_SPECS:
        try:
            allowed.update(allowed_variants(family, tier))
        except (KeyError, TypeError, ValueError):
            continue
    result: list[tuple[str, str]] = []
    for label, key in all_choices:
        exceeds_tier = variant_size_gb(key) > float(tier)
        if (key not in allowed or exceeds_tier) and key != selected_variant:
            continue
        if key == selected_variant and (key not in allowed or exceeds_tier):
            label = f"{label} ⚠ exceeds tier"
        result.append((label, key))
    return result


def _attention_choices() -> list[tuple[str, str]]:
    describe = getattr(attention_module, "describe_available", None)
    if callable(describe):
        try:
            descriptions = describe()
            if isinstance(descriptions, dict):
                choices: list[tuple[str, str]] = []
                for name in ATTENTION_CHOICES:
                    detail = str(descriptions.get(name) or "— unavailable").strip()
                    normalized = detail.casefold()
                    if normalized == "available":
                        label = f"{name} ✓"
                    elif normalized.startswith("available:"):
                        label = f"{name} ✓ — {detail.split(':', 1)[1].strip()}"
                    elif detail.startswith(("—", "✓")):
                        label = f"{name} {detail}"
                    elif normalized.startswith(("falls back", "unavailable")):
                        label = f"{name} — {detail}"
                    else:
                        label = detail if normalized.startswith(name.casefold()) else f"{name} — {detail}"
                    choices.append((_display(label), name))
                return choices
        except Exception:
            pass

    availability = attention_module.probe_available()
    labels = {
        "auto": "✓" if availability.get("auto") else "— unavailable",
        "flash_attention_2": "✓" if availability.get("flash_attention_2") else "— unavailable; falls back to SDPA",
        "sdpa": "✓" if availability.get("sdpa") else "— unavailable",
        "sage": "— falls back to SDPA (Sage kernel preference only)",
        "xformers": "— falls back to SDPA (efficient-kernel preference only)",
        "eager": "✓" if availability.get("eager") else "— unavailable",
    }
    return [(f"{name} {labels[name]}", name) for name in ATTENTION_CHOICES]


def _prompt_choices(family: str, modality: str) -> list[tuple[str, str]]:
    return [
        (f"{preset.group} · {_display(preset.label)}", preset.id)
        for preset in list_presets(family, modality)
    ]


def _preset_supports_family(preset: Any, family: str) -> bool:
    supported = tuple(getattr(preset, "applies_to_models", ()) or ())
    return "*" in supported or family in supported


def _resolve_prompt_preset(
    variant_key: str,
    modality: str,
    current_preset_id: str | None,
    *,
    family_union: bool = False,
) -> tuple[str, list[tuple[str, str]], Any | None]:
    """Resolve a family-safe prompt without discarding an intentional task."""

    family = variant_to_family(variant_key)
    presets = list_presets(family, None if family_union else modality)
    current: Any | None = None
    try:
        candidate = get_preset(str(current_preset_id or ""))
        if _preset_supports_family(candidate, family):
            current = candidate
    except KeyError:
        pass

    # With real inputs the menu stays focused on that modality, but a task the
    # family can run remains visible and selected. The runner deliberately
    # substitutes per item when a mixed batch needs a different prompt.
    if current is not None and current.id not in {preset.id for preset in presets}:
        family_presets = list_presets(family)
        order = {preset.id: index for index, preset in enumerate(family_presets)}
        presets = sorted(
            [*presets, current],
            key=lambda preset: order.get(preset.id, len(order)),
        )
    choices = [
        (f"{preset.group} · {_display(preset.label)}", preset.id)
        for preset in presets
    ]
    by_id = {preset.id: preset for preset in presets}
    selected = by_id.get(current.id) if current is not None else None
    if selected is None:
        try:
            selected = by_id.get(default_preset_for(family, modality).id)
        except KeyError:
            selected = None
        if selected is None:
            selected = presets[0] if presets else None
        if selected is None and not family_union:
            family_presets = list_presets(family)
            selected = family_presets[0] if family_presets else None
            if selected is not None:
                choices = [
                    (f"{preset.group} · {_display(preset.label)}", preset.id)
                    for preset in family_presets
                ]
    return family, choices, selected


def validate_prompt_preset(
    value: str,
    previous_valid: str | None,
    variant_key: str,
    modality: str,
    include_audio: bool,
) -> tuple[Any, str, str, list[str]]:
    """Keep a typed prompt combobox value on a compatible registered preset."""

    effective_modality = _effective_prompt_modality(variant_key, modality, include_audio)
    family, choices, default = _resolve_prompt_preset(
        variant_key, effective_modality, previous_valid
    )
    by_value = {preset_id: preset_id for _, preset_id in choices}
    by_value.update({label: preset_id for label, preset_id in choices})
    by_value.update(
        {
            preset.label: preset.id
            for preset in list_presets(family, effective_modality)
        }
    )
    selected = by_value.get(str(value or ""))
    context = [family, effective_modality]
    if selected is not None:
        preset = get_preset(selected)
        update = gr.update(value=selected) if selected != str(value or "") else gr.skip()
        return update, selected, _display(preset.description), context
    fallback = str(previous_valid or "")
    if fallback not in {preset_id for _, preset_id in choices}:
        fallback = default.id if default is not None else (choices[0][1] if choices else "")
    if not fallback:
        return gr.update(choices=[], value=None), "", "Unknown task preset; no compatible preset is available", context
    preset = get_preset(fallback)
    return (
        gr.update(choices=choices, value=fallback),
        fallback,
        f"Unknown task preset; kept {_display(preset.label)}",
        context,
    )


def validate_caption_length(value: str, previous_valid: str | None) -> tuple[Any, str, Any]:
    """Reject custom caption-length text while retaining the last valid choice."""

    selected = str(value or "")
    if selected in _CAPTION_LENGTH_CHOICES:
        return gr.skip(), selected, gr.skip()
    fallback = str(previous_valid or "detailed")
    if fallback not in _CAPTION_LENGTH_CHOICES:
        fallback = "detailed"
    return (
        gr.update(value=fallback),
        fallback,
        f"Unknown caption length; kept {fallback}",
    )


def _effective_prompt_modality(
    variant_key: str,
    modality: str,
    use_audio_in_video: bool,
    *,
    has_inputs: bool = True,
) -> str:
    """Match prompt filtering to the modality the pipeline will actually use."""

    family = variant_to_family(variant_key)
    if not has_inputs or not str(modality or "").strip() or modality == "unknown":
        return _PRIMARY_PROMPT_MODALITIES[family]
    if modality == "video_audio" and not use_audio_in_video:
        return "video"
    if modality == "video" and use_audio_in_video:
        try:
            if MODEL_SPECS[family].limits.requires_audio_track:
                return "video_audio"
        except KeyError:
            pass
    return modality


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
    variables = dict(zip(names, values))
    # Whisper runs after the UI has rendered the selected prompt. Preserve its
    # placeholder here so the runner can fill it with clip-local speech last.
    variables["TRANSCRIPT"] = "{{TRANSCRIPT}}"
    return variables


def render_prompt_preserving_edits(
    preset_id: str,
    variables: Mapping[str, Any],
    current_system: str,
    current_user: str,
    last_auto: Mapping[str, Any] | None,
) -> tuple[str, Any, Any, dict[str, Any]]:
    """Re-render template variables without overwriting manually edited fields."""

    preset = get_preset(str(preset_id))
    rendered_system, rendered_user = render_prompt(preset, dict(variables))
    next_system = rendered_system or ""
    next_user = rendered_user or ""
    previous = dict(last_auto or {})
    manually_loaded = bool(previous.get("manual"))
    system_is_auto = not manually_loaded and str(current_system or "") == str(previous.get("system") or "")
    user_is_auto = not manually_loaded and str(current_user or "") == str(previous.get("user") or "")
    output_system: Any = next_system if system_is_auto else gr.skip()
    output_user: Any = next_user if user_is_auto else gr.skip()
    tracked = {
        "system": next_system if system_is_auto else str(previous.get("system") or ""),
        "user": next_user if user_is_auto else str(previous.get("user") or ""),
    }
    if manually_loaded:
        tracked["manual"] = True
    description = _display(preset.description)
    if not (system_is_auto and user_is_auto):
        description += (
            "<br><span class='vc-warn'>Prompt edited manually — Reset prompts to preset "
            "re-renders it.</span>"
        )
    return description, output_system, output_user, tracked


def gguf_control_updates(
    variant_key: str,
    block_swap_auto: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return deterministic visibility/interactivity updates for a model backend."""

    is_gguf = get_variant(str(variant_key)).backend == "llamacpp"

    def disabled_info(key: str) -> str:
        base = _CONTROL_INFO[key]
        return base + (_GGUF_UNUSED_SUFFIX if is_gguf else "")

    blocks_info = _blocks_info(family_layer_count(variant_to_family(str(variant_key))))
    if is_gguf:
        blocks_info += _GGUF_UNUSED_SUFFIX
    return {
        "gguf_options": {"visible": is_gguf},
        "gguf_option": {"interactive": is_gguf},
        "attention_backend": {
            "interactive": not is_gguf,
            "info": disabled_info("attention_backend"),
        },
        "block_swap_auto": {
            "interactive": not is_gguf,
            "info": disabled_info("block_swap_auto"),
        },
        "blocks_to_swap": {
            "interactive": (not is_gguf) and (not bool(block_swap_auto)),
            "info": blocks_info,
        },
        "swap_slots": {
            "interactive": not is_gguf,
            "info": disabled_info("swap_slots"),
        },
        "offload_experts": {
            "interactive": not is_gguf,
            "info": disabled_info("offload_experts"),
        },
        "pin_cpu": {
            "interactive": not is_gguf,
            "info": disabled_info("pin_cpu"),
        },
        "pinned_ram_budget_gb": {
            "interactive": not is_gguf,
            "info": disabled_info("pinned_ram_budget_gb"),
        },
        "plan_slack_mib": {
            "interactive": not is_gguf,
            "info": disabled_info("plan_slack_mib"),
        },
        "torch_compile": {
            "interactive": not is_gguf,
            "info": disabled_info("torch_compile"),
        },
        "torch_compile_mode": {
            "interactive": not is_gguf,
            "info": disabled_info("torch_compile_mode"),
        },
        "use_cache": {
            "interactive": not is_gguf,
            "info": disabled_info("use_cache"),
        },
        "no_repeat_ngram_size": {
            "interactive": not is_gguf,
            "info": disabled_info("no_repeat_ngram_size"),
        },
    }


def request_caption_cancel(token: CancelToken | None) -> str:
    """Arm an active job and return the inline confirmation state."""

    if token is None or token.is_cancelled():
        return "inactive"
    token.arm_confirmation(window_s=8.0)
    return "confirm"


def confirm_caption_cancel(token: CancelToken | None) -> bool:
    """Confirm an armed cancellation request."""

    if token is None or token.is_cancelled() or not token.is_armed():
        return False
    token.cancel()
    return True


def keep_caption_running(token: CancelToken | None) -> bool:
    """Dismiss confirmation without touching the running job."""

    if token is None or token.is_cancelled():
        return False
    token.reset()
    return True


def sampled_frame_preview(
    paths: Iterable[str],
    variant_key: str,
    fps: Any,
    max_frames: Any,
    sampling: str,
    max_pixels: Any,
    min_pixels: Any,
    trim_start: Any = 0.0,
    trim_end: Any = None,
    adaptive_threshold: Any = 2.0,
) -> tuple[list[Any], str]:
    """Decode a CPU-only, at-most-sixteen-frame preview of the current plan."""

    first: Path | None = None
    info = None
    for raw in paths or []:
        candidate_info = probe_media(raw)
        if candidate_info.has_video:
            first = Path(raw)
            info = candidate_info
            break
    if first is None or info is None:
        return [], "Select a video first; audio and image inputs do not have a sampled-frame plan."
    requested_max = max(0, int(float(max_frames or 0)))
    if requested_max == 0:
        return [], "Maximum frames is 0, so this run uses the audio track only and has no visual-frame preview."
    start_s = max(0.0, float(trim_start or 0.0))
    end_s = float(trim_end) if trim_end not in (None, "") else float(info.duration or 0.0)
    if info.duration is not None:
        end_s = min(end_s or float(info.duration), float(info.duration))
    if end_s <= start_s:
        return [], "The selected trim range is empty; choose an end time after the start time."
    duration = end_s - start_s
    source_fps = max(0.01, float(info.fps or fps or 1.0))
    family = variant_to_family(str(variant_key))
    model = MODEL_SPECS[family]
    factor = model.limits.size_multiple
    minimum = min(requested_max, 1)
    plan_count, _, _ = plan_frame_sampling(
        duration,
        source_fps,
        FrameSamplingParams(
            strategy=str(sampling),
            fps=max(0.01, float(fps or model.limits.default_fps)),
            max_frames=requested_max,
            min_frames=minimum,
            frame_factor=1,
        ),
    )
    showing = min(16, max(1, plan_count))
    decoded = read_video_frames(
        first,
        start_s=start_s,
        end_s=end_s,
        target_fps=max(0.01, float(fps or model.limits.default_fps)),
        num_frames=showing,
        max_frames=showing,
        min_frames=1,
        max_pixels=max(1, int(float(max_pixels or model.limits.default_max_pixels))),
        min_pixels=max(1, int(float(min_pixels or model.limits.min_pixels))),
        size_multiple=factor,
        sampling=str(sampling),
        adaptive_threshold=max(0.1, min(50.0, float(adaptive_threshold or 2.0))),
    )
    width, height = decoded.resized_size
    budget_family = "qwen3-omni" if family.startswith("qwen3") else family
    visual_tokens = int(
        token_budget_estimate(budget_family, plan_count, height, width, 0.0)["video_tokens"]
    )
    gallery = [
        (frame, f"{timestamp:.2f}s")
        for frame, timestamp in zip(decoded.frames, decoded.timestamps)
    ]
    line = (
        f"Plan: {plan_count} frames at {width}x{height} (~{visual_tokens:,} visual tokens) "
        f"— showing {len(gallery)}"
    )
    return gallery, line


def unload_model_report(client: Any, gpu_index: int = 0) -> str:
    """Unload the resident model and describe the observed VRAM change."""

    before_ping = client.ping()
    if isinstance(before_ping, Mapping) and before_ping.get("busy"):
        return "<span class='vc-warn'>A job is running; the model cannot be unloaded yet.</span>"
    before_snapshot = resource_snapshot(int(gpu_index or 0))
    resident = (
        before_ping.get("loaded_variant")
        if isinstance(before_ping, Mapping)
        else None
    )
    release = getattr(client, "release_model", None)
    outcome = release(timeout_s=30.0) if callable(release) else client.unload()
    if isinstance(outcome, Mapping) and outcome.get("busy"):
        return "<span class='vc-warn'>A job is running; the model cannot be unloaded yet.</span>"
    if isinstance(outcome, Mapping) and outcome.get("error"):
        return f"<span class='vc-err'>Model unload failed: {html.escape(str(outcome['error']))}</span>"
    after_ping = client.ping()
    after_snapshot = resource_snapshot(int(gpu_index or 0))

    outer: Mapping[str, Any] = outcome if isinstance(outcome, Mapping) else {}
    payload: Mapping[str, Any] = outer
    nested = payload.get("report")
    if isinstance(nested, Mapping):
        payload = nested
    elif nested is not None and hasattr(nested, "__dataclass_fields__"):
        payload = asdict(nested)
    elif outcome is not None and hasattr(outcome, "__dataclass_fields__"):
        payload = asdict(outcome)

    released = payload.get("variant_key") or outer.get("released") or resident
    before = payload.get("vram_before_gb", before_snapshot.get("vram_used_gb", 0.0))
    after = payload.get("vram_after_gb", after_snapshot.get("vram_used_gb", 0.0))
    try:
        before_value = float(before or 0.0)
        after_value = float(after or 0.0)
    except (TypeError, ValueError):
        before_value = float(before_snapshot.get("vram_used_gb", 0.0) or 0.0)
        after_value = float(after_snapshot.get("vram_used_gb", 0.0) or 0.0)

    if isinstance(after_ping, Mapping) and after_ping.get("error"):
        suffix = f" Worker status: {html.escape(str(after_ping['error']))}."
    else:
        suffix = ""
    if released:
        return (
            f"<span class='vc-ok'>Released {html.escape(str(released))}; VRAM "
            f"{before_value:.2f} → {after_value:.2f} GB.</span>{suffix}"
        )
    return (
        f"<span class='vc-help'>No model was loaded; VRAM "
        f"{before_value:.2f} → {after_value:.2f} GB.</span>{suffix}"
    )


def failed_item_paths(result: JobResult | Mapping[str, Any] | None) -> list[str]:
    """Return unique source paths for every retryable failure status."""

    if result is None:
        return []
    raw_items = result.items if isinstance(result, JobResult) else result.get("items", [])
    paths: list[str] = []
    for item in raw_items or []:
        status = item.status if hasattr(item, "status") else item.get("status")
        path = item.path if hasattr(item, "path") else item.get("path")
        if str(status or "").casefold() in {"failed", "unsupported", "error"} and str(path or "").strip():
            text = str(path)
            if text not in paths:
                paths.append(text)
    return paths


def resolve_caption_inputs_at_start(
    settings: Mapping[str, Any],
    input_mode: str,
    cached: Sequence[str] | None,
) -> list[str]:
    """Resolve the selected tab's raw value, using its preview cache only when blank."""

    mode = str(input_mode or "upload").casefold()
    if mode == "upload":
        raw = settings.get("input_files")
        if raw:
            return _paths(raw)
    elif mode == "path":
        raw = str(settings.get("input_path") or "").strip()
        if raw:
            return _paths(raw)
    elif mode == "folder":
        raw = str(settings.get("batch_input_folder") or "").strip()
        if raw:
            selected, _summary = _folder_scan(
                raw,
                bool(settings.get("batch_recursive", False)),
                str(settings.get("batch_output_folder") or ""),
                bool(settings.get("overwrite_existing", False)),
                settings.get("batch_limit_items", 0),
                settings.get("batch_include_kinds"),
                str(settings.get("batch_name_filter") or ""),
                save_next_to_source=bool(settings.get("batch_save_next_to_source", False)),
                include_caption_coverage=True,
            )
            return selected
    return list(dict.fromkeys(str(value) for value in (cached or []) if str(value).strip()))


def _modality_for_inputs(paths: Sequence[str], fallback: str) -> str:
    if not paths:
        return fallback
    try:
        info = probe_media(paths[0])
    except Exception:
        return fallback
    return (
        "video_audio"
        if info.kind == "video"
        else "video"
        if info.kind == "video_no_audio"
        else str(info.kind or fallback)
    )


def retry_failed_inputs(state: Mapping[str, Any] | None) -> tuple[list[str], str, str | None]:
    """Derive retry paths, output kind, and prior batch destination from result state."""

    current = dict(state or {})
    paths = [str(value) for value in current.get("failed_paths", []) if str(value).strip()]
    kind = "batch" if str(current.get("output_kind") or "single") == "batch" else "single"
    batch_folder = str(current.get("batch_output_folder") or "").strip() or None
    return list(dict.fromkeys(paths)), kind, batch_folder


def results_zip_paths(
    state: Mapping[str, Any] | None,
    temp_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Resolve the finished-run source and deterministic downloads ZIP path."""

    current = dict(state or {})
    source_value = current.get("archive_source")
    if not source_value:
        if str(current.get("output_kind") or "single") == "batch":
            source_value = current.get("batch_output_folder") or current.get("editor_dir")
        else:
            source_value = current.get("run_dir")
    if not source_value:
        raise ValueError("No finished run is available to archive")
    source = normalize_path(str(source_value), must_exist=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    target = normalize_path(Path(temp_dir) / "downloads" / f"{source.name}.zip")
    return source, target


def _directory_size(path: Path) -> int:
    total = 0
    try:
        for candidate in path.rglob("*"):
            try:
                if candidate.is_file():
                    total += candidate.stat().st_size
            except OSError:
                continue
    except (OSError, PermissionError):
        pass
    return total


def _summary_value(summary: Any, name: str, default: Any = None) -> Any:
    if isinstance(summary, Mapping):
        return summary.get(name, default)
    return getattr(summary, name, default)


def run_history_rows(summaries: Iterable[Any]) -> list[list[Any]]:
    """Render backend RunSummary objects into the seven-column history table."""

    rows: list[list[Any]] = []
    for summary in summaries:
        counts = _summary_value(summary, "counts", {})
        counts = dict(counts) if isinstance(counts, Mapping) else {}
        try:
            when = datetime.fromtimestamp(float(_summary_value(summary, "created", 0.0))).astimezone()
            when_text = when.strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, TypeError, ValueError):
            when_text = "-"
        preview = re.sub(r"\s+", " ", str(_summary_value(summary, "preview", "") or "")).strip()
        rows.append(
            [
                str(_summary_value(summary, "name", "") or Path(str(_summary_value(summary, "run_dir", ""))).name),
                str(_summary_value(summary, "kind", "other") or "other"),
                str(_summary_value(summary, "model_key", "") or "-"),
                int(_summary_value(summary, "items", 0) or 0),
                f"{int(counts.get('done', 0) or 0)} / {int(counts.get('failed', 0) or 0)}",
                when_text,
                preview if len(preview) <= 90 else preview[:87].rstrip() + "...",
            ]
        )
    return rows


def run_history_records(summaries: Iterable[Any]) -> list[dict[str, Any]]:
    """Keep non-visible paths aligned with the corresponding history rows."""

    records: list[dict[str, Any]] = []
    for summary in summaries:
        raw_metadata = _summary_value(summary, "metadata_path")
        metadata_path = str(raw_metadata) if raw_metadata else None
        if metadata_path and Path(metadata_path).name.casefold() not in {
            "metadata.json",
            "editor_regeneration_metadata.json",
        }:
            metadata_path = None
        records.append(
            {
                "run_dir": str(_summary_value(summary, "run_dir", "") or ""),
                "name": str(_summary_value(summary, "name", "") or ""),
                "kind": str(_summary_value(summary, "kind", "other") or "other"),
                "metadata_path": metadata_path,
            }
        )
    return records


def _make_prompt_library(directory: Path, library_factory: Any = None) -> Any:
    if library_factory is None:
        try:
            from vcap.core.prompt_library import PromptLibrary
        except ImportError as exc:
            raise ImportError("Prompt library becomes available after the backend update") from exc
        library_factory = PromptLibrary
    return library_factory(directory) if callable(library_factory) else library_factory


def prompt_library_names(directory: str | os.PathLike[str], library_factory: Any = None) -> list[str]:
    """List saved prompt names with an injectable backend for CPU-only UI tests."""

    library = _make_prompt_library(normalize_path(directory), library_factory)
    return [str(entry.name if hasattr(entry, "name") else entry.get("name")) for entry in library.list()]


def save_prompt_library_entry(
    directory: str | os.PathLike[str],
    name: str,
    system_prompt: str,
    user_prompt: str,
    library_factory: Any = None,
) -> tuple[list[str], str, str]:
    """Save one system/user prompt pair and return choices, selection, and status."""

    library = _make_prompt_library(normalize_path(directory), library_factory)
    entry = library.save(str(name or ""), str(system_prompt or ""), str(user_prompt or ""))
    saved_name = str(entry.name if hasattr(entry, "name") else entry.get("name"))
    names = [str(item.name if hasattr(item, "name") else item.get("name")) for item in library.list()]
    return names, saved_name, f"<span class='vc-ok'>Saved prompt: {html.escape(saved_name)}</span>"


def load_prompt_library_entry(
    directory: str | os.PathLike[str],
    name: str,
    library_factory: Any = None,
) -> tuple[str, str, dict[str, Any], str]:
    """Load one prompt and mark both fields as manual for template preservation."""

    library = _make_prompt_library(normalize_path(directory), library_factory)
    entry = library.load(str(name or ""))
    loaded_name = str(entry.name if hasattr(entry, "name") else entry.get("name", name))
    raw_system = entry.system_prompt if hasattr(entry, "system_prompt") else entry.get("system_prompt", "")
    raw_user = entry.user_prompt if hasattr(entry, "user_prompt") else entry.get("user_prompt", "")
    system = str(raw_system or "")
    user = str(raw_user or "")
    manual_state = {"system": system, "user": user, "manual": True}
    return system, user, manual_state, f"<span class='vc-ok'>Loaded prompt: {html.escape(loaded_name)}</span>"


def delete_prompt_library_entry(
    directory: str | os.PathLike[str],
    name: str,
    library_factory: Any = None,
) -> tuple[list[str], str]:
    """Delete one prompt by name and return refreshed choices and status."""

    library = _make_prompt_library(normalize_path(directory), library_factory)
    requested = str(name or "")
    deleted = bool(library.delete(requested))
    names = [str(item.name if hasattr(item, "name") else item.get("name")) for item in library.list()]
    message = f"Deleted prompt: {requested}" if deleted else f"Prompt was already absent: {requested}"
    return names, f"<span class='{'vc-ok' if deleted else 'vc-warn'}'>{html.escape(message)}</span>"


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
        message = "⚠ Triton unavailable — CUDA graphs compatibility fallback; failures retry eagerly"
        css = "vc-warn"
    else:
        detail = "; ".join(str(item) for item in data.get("messages") or []) or "torch.compile is unavailable"
        message = f"✗ compile unavailable: {html.escape(detail)}"
        css = "vc-err"
    return f"<span class='{css}'>{message}</span> <span title='{html.escape(tooltip)}'>ⓘ</span>"


def _compile_probe_key() -> dict[str, str]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not-installed"
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch_version,
    }


def _read_compile_probe_cache() -> str | None:
    try:
        payload = json.loads(_COMPILE_PROBE_CACHE.read_text(encoding="utf-8"))
        age = time.time() - float(payload.get("timestamp", 0.0))
        if payload.get("key") != _compile_probe_key() or age < 0 or age > _COMPILE_PROBE_TTL_S:
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return _format_compile_report(data)
        cached_html = payload.get("html")
        return str(cached_html) if cached_html else None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_compile_probe_cache(report_html: str, data: dict[str, Any] | None) -> None:
    try:
        _COMPILE_PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = _COMPILE_PROBE_CACHE.with_suffix(".json.tmp")
        payload = {
            "key": _compile_probe_key(),
            "timestamp": time.time(),
            "data": data,
            "html": report_html,
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, _COMPILE_PROBE_CACHE)
    except OSError:
        pass


def _run_compile_probe_in_child() -> tuple[str, dict[str, Any] | None]:
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
            return f"<span class='vc-err'>✗ compile probe failed: {html.escape(detail)}</span>", None
        data = json.loads(line)
        return _format_compile_report(data), data
    except Exception as exc:
        return f"<span class='vc-err'>✗ compile probe failed: {html.escape(str(exc))}</span>", None


def _probe_compile_in_child(force: bool = False) -> str:
    """Return cached probe status and refresh stale results in a background thread."""

    global _COMPILE_PROBE_HTML, _COMPILE_PROBE_RUNNING
    if not force:
        with _COMPILE_PROBE_LOCK:
            if _COMPILE_PROBE_HTML is not None:
                return _COMPILE_PROBE_HTML
        cached = _read_compile_probe_cache()
        if cached is not None:
            with _COMPILE_PROBE_LOCK:
                _COMPILE_PROBE_HTML = cached
            return cached

    with _COMPILE_PROBE_LOCK:
        if force:
            _COMPILE_PROBE_HTML = None
        if _COMPILE_PROBE_RUNNING:
            return "<span class='vc-warn'>Probing C++ toolchain…</span>"
        _COMPILE_PROBE_RUNNING = True

    def probe() -> None:
        global _COMPILE_PROBE_HTML, _COMPILE_PROBE_RUNNING
        report_html, data = _run_compile_probe_in_child()
        _write_compile_probe_cache(report_html, data)
        with _COMPILE_PROBE_LOCK:
            _COMPILE_PROBE_HTML = report_html
            _COMPILE_PROBE_RUNNING = False

    threading.Thread(target=probe, daemon=True, name="vcap-compile-probe").start()
    return "<span class='vc-warn'>Probing C++ toolchain…</span>"


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


_GIB = float(2**30)
# A fresh worker's CUDA context is not visible to NVML until the worker exists; the
# loader measures free VRAM after creating it, so the preview budgets for it.
_CUDA_CONTEXT_ALLOWANCE_BYTES = 384 * 2**20
_BLOCK_SWAP_SLIDER_MAX = max(family_layer_count(family) for family in MODEL_SPECS)


def _blocks_info(layer_count: int) -> str:
    return (
        "Decoder layers kept in pinned RAM and streamed through the GPU each token; "
        f"0 keeps the whole decoder resident. The selected family has {int(layer_count)} layers."
    )


def _media_kinds(modality: Any) -> tuple[str, ...]:
    text = str(modality or "").strip().casefold()
    if text.startswith("video"):
        return ("video",)
    if text in {"audio", "image"}:
        return (text,)
    return ()


def _loaded_plan_text(summary: Mapping[str, Any]) -> str:
    resident = summary.get("resident_layers")
    total = summary.get("layer_count")
    swapped = summary.get("swapped_layers")
    if resident is None or total is None:
        return ""
    text = f"{int(resident)}/{int(total)} resident"
    if swapped is not None:
        text += f", {int(swapped)} swapped"
    return text


def block_swap_preview(
    variant_key: str,
    auto: bool,
    blocks: Any,
    *,
    gpu_index: int,
    reserve_gb: Any,
    plan_slack_mib: Any = 512,
    swap_slots: Any,
    offload_experts: bool,
    pin_cpu: bool,
    fps_value: Any,
    frames_value: Any,
    pixels_value: Any,
    output_tokens: Any,
    context_value: Any,
    duration: Any,
    modality: Any,
    pong: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve what the block-swap controls mean for the selected variant right now.

    Returns slider update kwargs (value, interactivity, help text) and an HTML
    status line. With ``auto`` the slider value becomes the swapped-layer count
    the loader is expected to choose, computed from the same inputs it uses:
    the safetensors header, the config, the media budget, and free VRAM. ``pong``
    is the pipeline client's ping result; when a model is resident, the free VRAM
    measured before it was placed is the figure the next load will see.
    """

    family = variant_to_family(variant_key)
    spec = MODEL_SPECS[family]
    variant = get_variant(variant_key)
    layer_count = family_layer_count(family)
    try:
        manual = int(float(blocks or 0))
    except (TypeError, ValueError, OverflowError):
        manual = 0
    manual = max(0, min(layer_count, manual))
    slider: dict[str, Any] = {"value": manual, "interactive": not bool(auto), "info": _blocks_info(layer_count)}

    if variant.scheme == "gguf":
        slider["interactive"] = False
        return slider, (
            "<span class='vc-help'>GGUF variants run inside llama-server, which fits its own GPU layer "
            "count at start (<code>--fit</code>); decoder block swap does not apply.</span>"
        )
    if offload_experts:
        slider["interactive"] = False
        return slider, (
            "<span class='vc-warn'>Legacy Accelerate expert offload is enabled; block swap is disabled "
            "and Accelerate places every decoder layer.</span>"
        )
    ready, detail = variant_is_ready(variant_key)
    if not ready:
        return slider, (
            "<span class='vc-help'>Download the checkpoint to preview the automatic plan "
            f"({html.escape(_display(detail))}).</span>"
        )

    pong_data = dict(pong or {})
    loaded_variant = pong_data.get("loaded_variant")
    raw_summary = pong_data.get("block_swap")
    loaded_summary = dict(raw_summary) if isinstance(raw_summary, Mapping) else None
    snapshot = resource_snapshot(int(gpu_index))
    free_bytes = int(float(snapshot.get("vram_free_gb", 0.0) or 0.0) * _GIB)
    total_bytes = int(float(snapshot.get("vram_total_gb", 0.0) or 0.0) * _GIB)
    ram_available = int(float(snapshot.get("ram_free_gb", 0.0) or 0.0) * _GIB) or None
    if total_bytes <= 0:
        return slider, "<span class='vc-warn'>GPU telemetry is unavailable; the loader plans at start.</span>"
    basis = "free VRAM now"
    if pong_data.get("busy"):
        basis = "free VRAM while the running job holds its model"
    elif loaded_variant and loaded_summary and loaded_summary.get("free_vram_gib"):
        before = int(float(loaded_summary["free_vram_gib"]) * _GIB)
        if before > free_bytes:
            free_bytes = before
            basis = "free VRAM measured before the resident model was placed; it is released first"
    elif not loaded_variant:
        free_bytes = max(0, free_bytes - _CUDA_CONTEXT_ALLOWANCE_BYTES)

    try:
        fps = float(fps_value or spec.limits.default_fps)
    except (TypeError, ValueError):
        fps = float(spec.limits.default_fps)
    try:
        frames = max(0, int(float(frames_value or 0)))
    except (TypeError, ValueError, OverflowError):
        frames = int(spec.limits.max_frames or 0)
    try:
        seconds = float(duration or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds > 0 and frames > 0:
        frames = min(frames, max(1, int(math.ceil(seconds * max(fps, 0.25)))))
    try:
        pixels = int(float(pixels_value or spec.limits.default_max_pixels))
    except (TypeError, ValueError, OverflowError):
        pixels = int(spec.limits.default_max_pixels)
    try:
        new_tokens = int(float(output_tokens or 0)) or None
    except (TypeError, ValueError, OverflowError):
        new_tokens = None
    hint = BudgetHint(
        max_frames=frames,
        max_pixels=pixels,
        fps=fps,
        max_new_tokens=new_tokens,
        context_tokens=_context_window(spec, context_value),
        media_kinds=_media_kinds(modality),
    )
    try:
        reserve = max(0.0, float(reserve_gb or 0.0))
    except (TypeError, ValueError):
        reserve = 2.0
    try:
        slots = max(1, min(4, int(float(swap_slots or 2))))
    except (TypeError, ValueError, OverflowError):
        slots = 2
    plan_kwargs = {
        "gpu_layers": block_swap_to_gpu_layers(bool(auto), manual, layer_count),
        "offload_experts": False,
        "max_memory": None,
        "pin_cpu": bool(pin_cpu),
        "vram_reserve_gb": reserve,
        "swap_slots": slots,
        "plan_slack_mib": max(0, min(8192, int(float(plan_slack_mib or 0)))),
    }
    try:
        plan = OffloadPlan(**plan_kwargs)
    except TypeError:
        # The fallback is only for the short F1/F2 integration window.
        plan_kwargs.pop("plan_slack_mib", None)
        plan = OffloadPlan(**plan_kwargs)
    budget = plan_model_folder(
        family,
        variant_key,
        resolve_model_dir(variant_key),
        plan,
        hint,
        free_vram_bytes=free_bytes,
        total_vram_bytes=total_bytes,
        ram_available_bytes=ram_available,
    )

    resident = int(budget.resident_layers)
    swapped = int(budget.swapped_layers)
    total_layers = int(budget.layer_count) or layer_count
    if auto:
        slider["value"] = swapped
    label = "Automatic plan (estimate)" if auto else "Manual plan"
    head = f"<strong>{label}:</strong> {resident} of {total_layers} decoder layers on GPU"
    if swapped > 0:
        head += f", <strong>{swapped} block-swapped</strong> ({budget.pinned_bytes / _GIB:.1f} GiB pinned RAM)"
    else:
        head += ", <strong>no block swap</strong>"
    parts = [
        head,
        f"expected peak {budget.expected_peak_bytes / _GIB:.1f} of {budget.free_vram_bytes / _GIB:.1f} GiB free",
        f"reserve {reserve:.1f} GB",
    ]
    if budget.stage_towers:
        parts.append("audio/vision towers staged on CPU between prefills")
    line = " · ".join(parts) + f". <span class='vc-help'>Basis: {html.escape(basis)}.</span>"
    for note in budget.notes[1:]:
        line += f"<br><span class='vc-warn'>{html.escape(str(note))}</span>"
    if swapped > 0:
        line += (
            "<br><span class='vc-help'>Swapped layers stream from RAM on every token, so decode speed "
            "falls as the swapped count rises.</span>"
        )
    if loaded_variant == variant_key and loaded_summary:
        loaded_text = _loaded_plan_text(loaded_summary)
        if loaded_text:
            line += (
                f"<br><span class='vc-help'>Loaded now: {html.escape(loaded_text)}; a job reuses the "
                "resident model while its plan still covers the media.</span>"
            )
    return slider, line


def _initial_block_swap_note(ctx: "UiContext", variant_key: str, gpu_index: int, tier: int) -> str:
    """Preview the automatic plan for the build-time defaults of the Model section."""

    try:
        family = variant_to_family(variant_key)
        preset = preset_for(family, tier)
        spec = MODEL_SPECS[family]
        ping = getattr(ctx.pipeline_client, "ping", None)
        pong = ping(timeout_s=0.3) if callable(ping) else None
        _, note = block_swap_preview(
            variant_key,
            True,
            0,
            gpu_index=int(gpu_index),
            reserve_gb=preset.offload.vram_reserve_gb,
            plan_slack_mib=getattr(preset.offload, "plan_slack_mib", 512),
            swap_slots=preset.offload.swap_slots,
            offload_experts=preset.offload.offload_experts,
            pin_cpu=preset.offload.pin_cpu,
            fps_value=preset.fps,
            frames_value=preset.max_frames,
            pixels_value=preset.max_pixels,
            output_tokens=preset.max_new_tokens,
            context_value=spec.limits.context_tokens,
            duration=0.0,
            modality="video_audio",
            pong=pong,
        )
        return note
    except Exception as exc:
        return f"<span class='vc-help'>Block swap preview unavailable: {html.escape(str(exc))}</span>"


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
    cancel_confirmation: gr.Row
    cancel_note: gr.Markdown
    cancel_yes: gr.Button
    cancel_keep: gr.Button
    hotkey_start: gr.Button
    hotkey_cancel: gr.Button
    cancel_timer: gr.Timer
    unload_model: gr.Button
    open_output: gr.Button
    open_caption: gr.Button
    reveal_clip: gr.Button
    open_editor: gr.Button
    copy_caption: gr.Button
    retry_failed: gr.Button
    results_zip: gr.Button
    results_zip_file: gr.File
    run_history: gr.Dataframe
    run_history_refresh: gr.Button
    run_history_open: gr.Button
    run_history_editor: gr.Button
    run_history_recover: gr.Button
    run_history_status: gr.Markdown
    run_history_records_state: gr.State
    run_history_selected_state: gr.State
    item_table: gr.Dataframe
    caption: gr.Textbox
    structured: gr.JSON
    srt: gr.Textbox
    reasoning: gr.Textbox
    reasoning_tab: gr.Tab
    files: gr.File
    clips: gr.Gallery
    clips_empty_hint: gr.Markdown
    last_outputs_state: gr.State
    job_done_hook: gr.HTML


def build(ctx: "UiContext") -> CaptionTabHandles:
    """Render the Caption and Processing Pipeline tabs as one linked pair.

    Both tabs share this function's local scope so the pipeline controls stay
    wired to the same handlers, registry entries, and presets as before; the
    caller supplies the surrounding ``gr.Tabs`` container. App assembly wires
    the global (queued) handlers later in :func:`wire`.
    """

    controls: dict[str, Any] = {}
    # Presets and run metadata written before v1.3.1 stored ``gpu_layers``; translate
    # it into the block-swap controls whenever such settings are coerced.
    ctx.settings_registry.add_migration(migrate_legacy_gpu_layers)
    initial_family = variant_to_family(_INITIAL_VARIANT)
    initial_spec = MODEL_SPECS[initial_family]
    initial_prompt = default_preset_for(initial_family, _INITIAL_MODALITY)
    initial_vars = {name: data["default"] for name, data in TEMPLATE_VARIABLES.items()}
    initial_system, initial_user = render_prompt(initial_prompt, initial_vars)
    initial_system, initial_user = initial_system or "", initial_user or ""
    prompt_library_dir = normalize_path(ctx.presets_dir / "prompts")
    try:
        initial_prompt_names = prompt_library_names(prompt_library_dir)
    except (ImportError, OSError, ValueError):
        initial_prompt_names = []
    gpu_choices, gpu_default, gpu_total, _ = _gpu_inventory()
    data_parallel_gpu_choices = gpu_choices if gpu_total > 0 else []
    detected_tier = auto_tier(gpu_total) if gpu_total else 32
    ctx.states["gpu_index_default"] = gpu_default

    with gr.Tab("🎬 Caption", id="caption"):
        # gr.Tabs() maps its buttons to its direct children, so every component
        # this function creates - states included - must live inside a gr.Tab.
        ctx.states["gpu_index"] = gr.State(gpu_default)
        prompt_context_state = gr.State([initial_family, _INITIAL_MODALITY])
        prompt_auto_state = gr.State(
            {"system": initial_system or "", "user": initial_user}
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=500):
                media = media_input_block(
                    ctx,
                    save_next_to_source_key="batch_save_next_to_source",
                    save_next_to_source_label="Save outputs next to the source files",
                    save_next_to_source_info=(
                        "Batch caption files, split-layout folders, transcript sidecars, segment folders, "
                        "and saved clips are written beside each source. Metadata, run_log.txt, and .work "
                        "remain in the numbered batch run directory."
                    ),
                    include_caption_coverage=True,
                )

                with gr.Accordion("Trim range", open=False):
                    with gr.Row():
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
                            description="Optional numeric trim end in seconds.", kind="float", minimum=0.0,
                        )
                    gr.Markdown(
                        "Use the player's built-in trim editor for visual trimming; its edited file becomes the first input automatically.",
                        elem_classes=["vc-help"],
                    )

                with gr.Column():
                    gr.Markdown("### Result")
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
                        with gr.Tab("Files"):
                            result_files = gr.File(
                                label="Caption outputs",
                                file_count="multiple",
                                interactive=False,
                                elem_id="vc_caption_result_files",
                            )
                        with gr.Tab("Clips"):
                            clips_empty_hint = gr.Markdown(
                                'No clips were saved for this run. Enable "Save produced clips" in '
                                "Processing Pipeline → 5. Scene detection & splitting.",
                                elem_classes=["vc-help"],
                            )
                            clips = gr.Gallery(
                                label="Produced clips",
                                columns=4,
                                rows=2,
                                height=330,
                                object_fit="cover",
                                allow_preview=True,
                                type="filepath",
                                buttons=["download", "fullscreen"],
                                visible=False,
                            )

                with gr.Row():
                    start = action_button("▶ Start Captioning", "emerald", variant="primary", scale=3)
                    cancel = action_button(
                        "⏹ Cancel",
                        "red",
                        variant="stop",
                        scale=2,
                        elem_id="vc_caption_cancel",
                        interactive=False,
                    )
                    open_output = action_button("📂 Open Output", "teal", scale=2)
                    open_caption = action_button("📝 Open Last Caption", "violet", scale=2)
                    reveal_clip = action_button("🎬 Reveal Clip", "amber", scale=2)
                    open_editor = action_button(
                        "✏️ Open in Caption Editor",
                        "cobalt",
                        scale=2,
                        elem_id="vc_open_caption_editor",
                        interactive=False,
                    )
                with gr.Row():
                    copy_caption = action_button(
                        "⧉ Copy caption",
                        "blue",
                        scale=2,
                        elem_id="vc_copy_caption",
                        interactive=False,
                    )
                    retry_failed = action_button(
                        "🔁 Retry failed",
                        "yellow",
                        scale=2,
                        elem_id="vc_retry_failed",
                        interactive=False,
                    )
                    results_zip = action_button(
                        "⬇ Results ZIP",
                        "fuchsia",
                        scale=2,
                        elem_id="vc_results_zip",
                        interactive=False,
                    )
                with gr.Row():
                    results_zip_file = gr.File(
                        label="Results ZIP download",
                        interactive=False,
                        visible=False,
                        elem_id="vc_results_zip_download",
                    )
                with gr.Row(
                    visible=False,
                    elem_id="vc_caption_cancel_confirmation",
                    elem_classes=["vc-confirm-bar"],
                ) as cancel_confirmation:
                    gr.Markdown("⚠ Cancel the running job?")
                    cancel_yes = action_button(
                        "✔ Yes, cancel",
                        "maroon",
                        variant="stop",
                        scale=0,
                        min_width=132,
                        elem_id="vc_caption_cancel_yes",
                    )
                    cancel_keep = action_button(
                        "✖ Keep running",
                        "steel",
                        scale=0,
                        min_width=148,
                        elem_id="vc_caption_cancel_keep",
                    )
                cancel_note = gr.Markdown("", elem_classes=["vc-status"])
                hotkey_start = gr.Button("Start caption hotkey", elem_id="hk_caption_start", visible="hidden")
                hotkey_cancel = gr.Button("Cancel caption hotkey", elem_id="hk_caption_cancel", visible="hidden")
                cancel_timer = gr.Timer(1.0)

                with gr.Accordion(
                    "📜 Run history",
                    open=False,
                    elem_id="vc_run_history",
                ):
                    run_history_records_state = gr.State([])
                    run_history_selected_state = gr.State({})
                    run_history = gr.Dataframe(
                        value=[],
                        headers=["Run", "Kind", "Model", "Items", "Done/Failed", "When", "Preview"],
                        datatype=["str", "str", "str", "number", "str", "str", "str"],
                        type="array",
                        interactive=False,
                        wrap=True,
                        column_widths=[160, 80, 220, 65, 100, 145, 360],
                        max_height=330,
                        buttons=["copy", "fullscreen"],
                        label="Recent runs",
                        elem_id="vc_run_history_table",
                    )
                    gr.Markdown(
                        "Newest caption, batch, and chat runs discovered below the Outputs directory.",
                        elem_classes=["vc-help"],
                    )
                    with gr.Row():
                        run_history_refresh = action_button(
                            "🔄 Refresh",
                            "pink",
                            elem_id="vc_run_history_refresh",
                        )
                        run_history_open = action_button(
                            "📂 Open folder",
                            "bronze",
                            elem_id="vc_run_history_open_folder",
                            interactive=False,
                        )
                        run_history_editor = action_button(
                            "✏️ Open in editor",
                            "mint",
                            elem_id="vc_run_history_open_editor",
                            interactive=False,
                        )
                        run_history_recover = action_button(
                            "🔁 Recover settings",
                            "coral",
                            elem_id="vc_run_history_recover",
                            interactive=False,
                        )
                    run_history_status = gr.Markdown(
                        "<span class='vc-help'>Refresh to discover recent runs.</span>",
                        elem_classes=["vc-status"],
                    )

                progress = progress_panel(ctx)
                item_table = gr.Dataframe(
                    headers=["#", "Input", "Status", "Message", "Elapsed", "Tokens"],
                    value=[],
                    type="array",
                    datatype=["number", "str", "str", "str", "str", "number"],
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
                        choices=variant_choices_for_tier(_INITIAL_VARIANT, detected_tier),
                        value=_INITIAL_VARIANT,
                        label="Model variant",
                        info="Model family, precision/backend variant, and estimated local checkpoint size.",
                        # Presets and tier filters can update the value and the
                        # choices in different events; the validator below rejects
                        # unknown values, so Gradio must not raise on a transient one.
                        allow_custom_value=True,
                    )
                    valid_model_key_state = gr.State(_INITIAL_VARIANT)
                    ctx.states["caption_valid_model_key"] = valid_model_key_state
                    controls["model_key"] = ctx.reg(
                        "model_key", model_key, _INITIAL_VARIANT, section="model",
                        description="Selected model family and checkpoint variant.", choices=[key for _, key in _variant_choices()], kind="str",
                    )
                    show_all_variants = gr.Checkbox(
                        value=False,
                        label="Show all variants (ignore VRAM tier)",
                        info="Lists variants above the active VRAM tier; they may require CPU offload or run out of memory.",
                    )
                    controls["show_all_variants"] = ctx.reg(
                        "show_all_variants", show_all_variants, False, section="model",
                        description="Show model variants that exceed the active VRAM tier.", kind="bool",
                    )
                    quant_info = gr.Markdown(_quant_line(_INITIAL_VARIANT))
                    with gr.Row():
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
                        attention = gr.Dropdown(
                            choices=_attention_choices(),
                            value="auto",
                            label="Attention backend",
                            info=_CONTROL_INFO["attention_backend"],
                        )
                        controls["attention_backend"] = ctx.reg(
                            "attention_backend", attention, "auto", section="model",
                            description="Requested attention implementation with safe runtime fallback.", choices=ATTENTION_CHOICES, kind="str",
                        )
                    vram_note = gr.Markdown(
                        f"<span class='vc-help'>Auto tier: {detected_tier} GB. The detected plan is applied at startup and on family changes.</span>"
                    )
                    with gr.Row():
                        gpu_picker = gr.Dropdown(
                            choices=gpu_choices,
                            value=gpu_default,
                            label="GPU",
                            info="The selected physical GPU is isolated for the caption worker.",
                        )
                        controls["gpu_index"] = ctx.reg(
                            "gpu_index", gpu_picker, gpu_default, section="runtime",
                            description="Physical NVIDIA GPU index used by the pipeline.", kind="int",
                            choices=[value for _, value in gpu_choices], in_preset=False,
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
                    gpu_indices = gr.CheckboxGroup(
                        choices=data_parallel_gpu_choices,
                        value=[],
                        label="Data-parallel GPUs",
                        info=(
                            "Leave empty to use the single GPU above. Selecting 2 or more GPUs splits folder batches "
                            "across workers, with one model copy loaded per GPU."
                        ),
                    )
                    controls["gpu_indices"] = ctx.reg(
                        "gpu_indices", gpu_indices, [], section="runtime",
                        description="Physical GPU indices used for data-parallel folder batch workers.",
                        choices=[value for _, value in data_parallel_gpu_choices], kind="list", in_preset=False, in_metadata=True,
                    )
                    with gr.Row():
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
                        oom_retries = gr.Number(
                            value=2,
                            minimum=0,
                            maximum=4,
                            step=1,
                            precision=0,
                            label="OOM retries",
                            info=(
                                "Automatic out-of-memory recoveries per segment; each retry lowers frames and resolution "
                                "one notch and logs it. 0 fails the segment immediately."
                            ),
                            elem_id="vc_oom_retries",
                        )
                        controls["oom_retries"] = ctx.reg(
                            "oom_retries", oom_retries, 2, section="runtime",
                            description=(
                                "Automatic out-of-memory recoveries per segment; each retry lowers frames and resolution "
                                "one notch and logs it. 0 fails the segment immediately."
                            ),
                            kind="int", minimum=0, maximum=4,
                        )
                    oom_degrade_factor = gr.Slider(
                        minimum=0.5,
                        maximum=0.95,
                        value=0.75,
                        step=0.05,
                        label="OOM degrade factor",
                        info=(
                            "Scale applied to the pixel and frame budgets on each automatic out-of-memory retry "
                            "(0.75 = reduce by 25%)."
                        ),
                        elem_id="vc_oom_degrade_factor",
                    )
                    controls["oom_degrade_factor"] = ctx.reg(
                        "oom_degrade_factor",
                        oom_degrade_factor,
                        0.75,
                        section="runtime",
                        description=(
                            "Scale applied to the pixel and frame budgets on each automatic out-of-memory retry "
                            "(0.75 = reduce by 25%)."
                        ),
                        kind="float",
                        minimum=0.5,
                        maximum=0.95,
                    )
                    with gr.Accordion("Block swap & offload plan", open=True):
                        with gr.Row():
                            block_swap_auto = gr.Checkbox(
                                value=True,
                                label="Automatic block swap",
                                info=_CONTROL_INFO["block_swap_auto"],
                                scale=2,
                            )
                            controls["block_swap_auto"] = ctx.reg(
                                "block_swap_auto", block_swap_auto, True, section="model",
                                description="Let the loader choose how many decoder layers to block-swap.", kind="bool",
                            )
                            blocks_to_swap = gr.Slider(
                                minimum=0,
                                maximum=_BLOCK_SWAP_SLIDER_MAX,
                                step=1,
                                value=0,
                                interactive=False,
                                label="Decoder layers to block-swap",
                                info=_blocks_info(family_layer_count(initial_family)),
                                scale=3,
                            )
                            controls["blocks_to_swap"] = ctx.reg(
                                "blocks_to_swap", blocks_to_swap, 0, section="model",
                                description="Decoder layers kept in pinned RAM when automatic block swap is off.",
                                kind="int", minimum=0, maximum=_BLOCK_SWAP_SLIDER_MAX,
                            )
                        block_swap_note = gr.Markdown(
                            _initial_block_swap_note(ctx, _INITIAL_VARIANT, gpu_default, detected_tier),
                            elem_classes=["vc-status"],
                        )
                        with gr.Row():
                            vram_reserve_gb = gr.Number(
                                value=2.0,
                                minimum=0,
                                maximum=24,
                                step=0.5,
                                label="VRAM to keep free (GB)",
                                info="Dedicated VRAM reserved for activations and runtime peaks.",
                            )
                            controls["vram_reserve_gb"] = ctx.reg(
                                "vram_reserve_gb", vram_reserve_gb, 2.0, section="model",
                                description="Dedicated VRAM kept free at the expected generation peak.",
                                kind="float", minimum=0, maximum=24,
                            )
                            swap_slots = gr.Dropdown(
                                choices=[1, 2, 3, 4],
                                value=2,
                                label="Swap slots",
                                info=_CONTROL_INFO["swap_slots"],
                                elem_id="vc_swap_slots",
                            )
                            controls["swap_slots"] = ctx.reg(
                                "swap_slots", swap_slots, 2, section="model",
                                description="GPU staging slots allocated for decoder block swap.",
                                choices=[1, 2, 3, 4], kind="int",
                            )
                            plan_slack_mib = gr.Number(
                                value=512,
                                minimum=0,
                                maximum=8192,
                                step=64,
                                precision=0,
                                label="Plan slack (MiB)",
                                info=_CONTROL_INFO["plan_slack_mib"],
                                elem_id="vc_plan_slack_mib",
                            )
                            controls["plan_slack_mib"] = ctx.reg(
                                "plan_slack_mib",
                                plan_slack_mib,
                                512,
                                section="model",
                                description=(
                                    "Fixed CUDA-allocator slack (MiB) added to the activation estimate when planning "
                                    "how many decoder layers stay resident."
                                ),
                                kind="int",
                                minimum=0,
                                maximum=8192,
                            )
                        with gr.Row():
                            offload_experts = gr.Checkbox(
                                value=False,
                                label="Offload MoE experts",
                                info=_CONTROL_INFO["offload_experts"],
                            )
                            controls["offload_experts"] = ctx.reg(
                                "offload_experts", offload_experts, False, section="model",
                                description="Use legacy Accelerate expert offload instead of block swap.", kind="bool",
                            )
                            pin_cpu = gr.Checkbox(
                                value=True,
                                label="Pin swapped layers in RAM",
                                info=_CONTROL_INFO["pin_cpu"],
                            )
                            controls["pin_cpu"] = ctx.reg(
                                "pin_cpu", pin_cpu, True, section="model",
                                description="Pin block-swapped decoder layers in host memory.", kind="bool",
                            )
                            pinned_ram_budget_gb = gr.Number(
                                value=0.0,
                                minimum=0,
                                maximum=1024,
                                step=0.5,
                                label="Pinned RAM budget (GB)",
                                info=_CONTROL_INFO["pinned_ram_budget_gb"],
                                elem_id="vc_pinned_ram_budget_gb",
                            )
                            controls["pinned_ram_budget_gb"] = ctx.reg(
                                "pinned_ram_budget_gb", pinned_ram_budget_gb, 0.0, section="model",
                                description=_CONTROL_INFO["pinned_ram_budget_gb"],
                                kind="float", minimum=0, maximum=1024,
                            )
                    with gr.Row():
                        compile_enabled = gr.Checkbox(
                            value=False,
                            label="torch.compile",
                            info=_CONTROL_INFO["torch_compile"],
                        )
                        controls["torch_compile"] = ctx.reg(
                            "torch_compile", compile_enabled, False, section="runtime",
                            description="Compile the language model forward pass with safe fallbacks.", kind="bool",
                        )
                        compile_mode = gr.Dropdown(
                            choices=compile_mode_choices(),
                            value=DEFAULT_COMPILE_MODE,
                            label="Compile mode",
                            info=_CONTROL_INFO["torch_compile_mode"],
                        )
                        controls["torch_compile_mode"] = ctx.reg(
                            "torch_compile_mode", compile_mode, DEFAULT_COMPILE_MODE, section="runtime",
                            description="Requested torch.compile tuning mode.",
                            choices=list(compile_mode_values()), kind="str",
                        )
                    with gr.Accordion(
                        "llama.cpp (GGUF) options",
                        open=False,
                        visible=False,
                        elem_id="vc_gguf_options",
                    ) as gguf_options:
                        with gr.Row():
                            gguf_max_frames = gr.Number(
                                value=32, minimum=1, maximum=128, step=1, precision=0,
                                label="GGUF maximum frames",
                                info=(
                                    "Upper bound on sampled video frames sent to llama-server per clip; each frame is encoded "
                                    "as an image (about 256 tokens per frame at 262,144 pixels). Frames are chosen from Sampling "
                                    "FPS, Maximum frames and Sampling strategy, then capped here and by the context window."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_max_frames",
                            )
                            controls["gguf_max_frames"] = ctx.reg(
                                "gguf_max_frames", gguf_max_frames, 32, section="runtime",
                                description=(
                                    "Upper bound on sampled video frames sent to llama-server per clip; each frame is encoded "
                                    "as an image (about 256 tokens per frame at 262,144 pixels). Frames are chosen from Sampling "
                                    "FPS, Maximum frames and Sampling strategy, then capped here and by the context window."
                                ), kind="int", minimum=1, maximum=128,
                            )
                            gguf_jpeg_quality = gr.Number(
                                value=90, minimum=50, maximum=100, step=1, precision=0,
                                label="JPEG quality",
                                info="JPEG quality of frames sent to llama-server.",
                                interactive=False,
                                elem_id="vc_gguf_jpeg_quality",
                            )
                            controls["gguf_jpeg_quality"] = ctx.reg(
                                "gguf_jpeg_quality", gguf_jpeg_quality, 90, section="runtime",
                                description="JPEG quality of frames sent to llama-server.",
                                kind="int", minimum=50, maximum=100,
                            )
                        with gr.Row():
                            gguf_threads = gr.Number(
                                value=0, minimum=0, maximum=256, step=1, precision=0,
                                label="CPU threads",
                                info="CPU threads for llama-server (--threads); 0 = llama.cpp default.",
                                interactive=False,
                                elem_id="vc_gguf_threads",
                            )
                            controls["gguf_threads"] = ctx.reg(
                                "gguf_threads", gguf_threads, 0, section="runtime",
                                description="CPU threads for llama-server (--threads); 0 = llama.cpp default.",
                                kind="int", minimum=0, maximum=256,
                            )
                            gguf_flash_attn = gr.Dropdown(
                                choices=[("Automatic", "auto"), ("On", "on"), ("Off", "off")],
                                value="auto", label="Flash attention",
                                info="llama.cpp flash attention (-fa).",
                                interactive=False,
                                elem_id="vc_gguf_flash_attn",
                            )
                            controls["gguf_flash_attn"] = ctx.reg(
                                "gguf_flash_attn", gguf_flash_attn, "auto", section="runtime",
                                description="llama.cpp flash attention (-fa).",
                                kind="str", choices=["auto", "on", "off"],
                            )
                        with gr.Row():
                            gguf_batch_size = gr.Number(
                                value=2048, minimum=64, maximum=8192, step=64, precision=0,
                                label="Logical batch size",
                                info="Logical prompt batch size (-b).",
                                interactive=False,
                                elem_id="vc_gguf_batch_size",
                            )
                            controls["gguf_batch_size"] = ctx.reg(
                                "gguf_batch_size", gguf_batch_size, 2048, section="runtime",
                                description="Logical prompt batch size (-b).",
                                kind="int", minimum=64, maximum=8192,
                            )
                            gguf_ubatch_size = gr.Number(
                                value=512, minimum=32, maximum=4096, step=32, precision=0,
                                label="Physical batch size",
                                info="Physical micro-batch size (-ub); lower values reduce VRAM during prefill.",
                                interactive=False,
                                elem_id="vc_gguf_ubatch_size",
                            )
                            controls["gguf_ubatch_size"] = ctx.reg(
                                "gguf_ubatch_size", gguf_ubatch_size, 512, section="runtime",
                                description="Physical micro-batch size (-ub); lower values reduce VRAM during prefill.",
                                kind="int", minimum=32, maximum=4096,
                            )
                        with gr.Row():
                            gguf_cache_reuse = gr.Number(
                                value=0, minimum=0, maximum=4096, step=1, precision=0,
                                label="Prompt cache reuse",
                                info="KV-cache prefix reuse chunk size (--cache-reuse); 0 disables. Speeds up repeated prompt prefixes but shifts cached positions.",
                                interactive=False,
                                elem_id="vc_gguf_cache_reuse",
                            )
                            controls["gguf_cache_reuse"] = ctx.reg(
                                "gguf_cache_reuse", gguf_cache_reuse, 0, section="runtime",
                                description="KV-cache prefix reuse chunk size (--cache-reuse); 0 disables. Speeds up repeated prompt prefixes but shifts cached positions.",
                                kind="int", minimum=0, maximum=4096,
                            )
                            gguf_ignore_tier_context = gr.Checkbox(
                                value=False, label="Ignore tier context cap",
                                info=(
                                    "Request the full Context length instead of the VRAM-tier clamp (8k for 16 GB and below, "
                                    "16k for 24 GB, 32k otherwise). Larger windows need more VRAM for the KV cache."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_ignore_tier_context",
                            )
                            controls["gguf_ignore_tier_context"] = ctx.reg(
                                "gguf_ignore_tier_context", gguf_ignore_tier_context, False, section="runtime",
                                description=(
                                    "Request the full Context length instead of the VRAM-tier clamp (8k for 16 GB and below, "
                                    "16k for 24 GB, 32k otherwise). Larger windows need more VRAM for the KV cache."
                                ),
                                kind="bool",
                            )
                        with gr.Row():
                            gguf_min_p = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.05,
                                step=0.01,
                                label="Min-p",
                                info=(
                                    "llama.cpp min-p sampling: tokens below this fraction of the top probability are "
                                    "dropped. 0.05 is the llama.cpp default; only matters when Sample tokens is on."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_min_p",
                            )
                            controls["gguf_min_p"] = ctx.reg(
                                "gguf_min_p", gguf_min_p, 0.05, section="runtime",
                                description=(
                                    "llama.cpp min-p sampling: tokens below this fraction of the top probability are "
                                    "dropped; only matters when sampling."
                                ),
                                kind="float", minimum=0.0, maximum=1.0,
                            )
                            gguf_repeat_last_n = gr.Number(
                                value=64,
                                minimum=0,
                                maximum=4096,
                                step=1,
                                precision=0,
                                label="Repeat last N",
                                info=(
                                    "Number of previous tokens the repetition penalty looks at (llama.cpp default 64; "
                                    "0 disables the window)."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_repeat_last_n",
                            )
                            controls["gguf_repeat_last_n"] = ctx.reg(
                                "gguf_repeat_last_n", gguf_repeat_last_n, 64, section="runtime",
                                description="Number of previous tokens considered by the llama.cpp repetition penalty.",
                                kind="int", minimum=0, maximum=4096,
                            )
                        with gr.Row():
                            gguf_presence_penalty = gr.Slider(
                                minimum=-2.0,
                                maximum=2.0,
                                value=0.0,
                                step=0.05,
                                label="Presence penalty",
                                info="llama.cpp presence penalty (positive values discourage tokens that already appeared).",
                                interactive=False,
                                elem_id="vc_gguf_presence_penalty",
                            )
                            controls["gguf_presence_penalty"] = ctx.reg(
                                "gguf_presence_penalty", gguf_presence_penalty, 0.0, section="runtime",
                                description="llama.cpp presence penalty; positive values discourage tokens that already appeared.",
                                kind="float", minimum=-2.0, maximum=2.0,
                            )
                            gguf_frequency_penalty = gr.Slider(
                                minimum=-2.0,
                                maximum=2.0,
                                value=0.0,
                                step=0.05,
                                label="Frequency penalty",
                                info="llama.cpp frequency penalty (scales with how often a token appeared).",
                                interactive=False,
                                elem_id="vc_gguf_frequency_penalty",
                            )
                            controls["gguf_frequency_penalty"] = ctx.reg(
                                "gguf_frequency_penalty", gguf_frequency_penalty, 0.0, section="runtime",
                                description="llama.cpp frequency penalty scaled by how often a token appeared.",
                                kind="float", minimum=-2.0, maximum=2.0,
                            )
                        with gr.Row():
                            gguf_fit_headroom_mib = gr.Number(
                                value=1536,
                                minimum=0,
                                maximum=8192,
                                step=64,
                                precision=0,
                                label="Fit headroom (MiB)",
                                info=(
                                    "Extra MiB kept free on top of VRAM to keep free for the multimodal projector's "
                                    "encoder buffers when llama.cpp fits the model to the GPU (--fit)."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_fit_headroom_mib",
                            )
                            controls["gguf_fit_headroom_mib"] = ctx.reg(
                                "gguf_fit_headroom_mib", gguf_fit_headroom_mib, 1536, section="runtime",
                                description="Extra MiB kept free for multimodal projector buffers during llama.cpp GPU fitting.",
                                kind="int", minimum=0, maximum=8192,
                            )
                            gguf_startup_timeout_s = gr.Number(
                                value=900,
                                minimum=60,
                                maximum=3600,
                                step=30,
                                precision=0,
                                label="Startup timeout (s)",
                                info=(
                                    "Seconds to wait for llama-server to become healthy after starting; large models "
                                    "on slow disks need longer."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_startup_timeout_s",
                            )
                            controls["gguf_startup_timeout_s"] = ctx.reg(
                                "gguf_startup_timeout_s", gguf_startup_timeout_s, 900, section="runtime",
                                description="Seconds to wait for llama-server to become healthy after starting.",
                                kind="int", minimum=60, maximum=3600,
                            )
                            gguf_stream_idle_timeout_s = gr.Number(
                                value=120,
                                minimum=0,
                                maximum=3600,
                                step=10,
                                precision=0,
                                label="Stream idle timeout (s)",
                                info=(
                                    "Abort a GGUF generation when no data arrives from llama-server for this many "
                                    "seconds (0 = wait forever)."
                                ),
                                interactive=False,
                                elem_id="vc_gguf_stream_idle_timeout_s",
                            )
                            controls["gguf_stream_idle_timeout_s"] = ctx.reg(
                                "gguf_stream_idle_timeout_s", gguf_stream_idle_timeout_s, 120, section="runtime",
                                description="Maximum idle seconds while waiting for streamed llama-server generation data.",
                                kind="int", minimum=0, maximum=3600,
                            )
                        gguf_extra_args = gr.Textbox(
                            value="", label="Extra llama-server arguments",
                            info="Extra llama-server command-line arguments appended verbatim (advanced; shell-split).",
                            lines=2, interactive=False,
                            elem_id="vc_gguf_extra_args",
                        )
                        controls["gguf_extra_args"] = ctx.reg(
                            "gguf_extra_args", gguf_extra_args, "", section="runtime",
                            description="Extra llama-server command-line arguments appended verbatim (advanced; shell-split).",
                            kind="str",
                        )
                    compile_status = gr.Markdown(_probe_compile_in_child(), elem_classes=["vc-status"])
                    compile_probe_timer = gr.Timer(1.0)
                    with gr.Row():
                        download = action_button("📥 Download / Verify model", "sky", scale=3)
                        refresh_ready = action_button("↻ Refresh", "lime", scale=1)
                        clear_compile = action_button("⌫ Clear compile caches", "rose", scale=2, min_width=200)
                        unload_model = action_button(
                            "⏏ Unload model", "navy", scale=2,
                            elem_id="vc_unload_model",
                        )
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
                    valid_prompt_preset_state = gr.State(initial_prompt.id)
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
                        "system_prompt", system_prompt, "", section="prompt",
                        description="Rendered or custom system instruction.", kind="str",
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
                    with gr.Row():
                        my_prompts = gr.Dropdown(
                            choices=initial_prompt_names,
                            value=None,
                            label="My prompts",
                            info="Saved system and user prompt pairs from the personal prompt library.",
                            scale=3,
                            elem_id="vc_my_prompts",
                        )
                        controls["prompt_library_selection"] = ctx.reg(
                            "prompt_library_selection",
                            my_prompts,
                            "",
                            section="prompt_library",
                            description="Currently selected personal prompt-library entry.",
                            kind="str",
                            in_preset=False,
                            in_metadata=False,
                        )
                        prompt_name = gr.Textbox(
                            value="",
                            label="Prompt name",
                            info="Name used when saving a personal prompt; Unicode names are supported.",
                            scale=3,
                            elem_id="vc_prompt_name",
                        )
                        controls["prompt_library_name"] = ctx.reg(
                            "prompt_library_name",
                            prompt_name,
                            "",
                            section="prompt_library",
                            description="Name used to save or identify a personal prompt entry.",
                            kind="str",
                            in_preset=False,
                            in_metadata=False,
                        )
                    with gr.Row():
                        save_prompt = action_button(
                            "💾 Save prompt", "green", elem_id="vc_save_prompt"
                        )
                        load_prompt = action_button(
                            "📥 Load prompt", "jade", elem_id="vc_load_prompt"
                        )
                        delete_prompt = action_button(
                            "🗑 Delete prompt", "crimson", elem_id="vc_delete_prompt"
                        )
                    prompt_library_status = gr.Markdown(
                        "<span class='vc-help'>Personal prompt library ready.</span>",
                        elem_classes=["vc-status"],
                        elem_id="vc_prompt_library_status",
                    )
                    with gr.Accordion("Template variables", open=False):
                        gr.Markdown(
                            "Prompt templates may also use `{{TRANSCRIPT}}`; it is filled with clip-local "
                            "Whisper speech after the ordinary variables are rendered.",
                            elem_classes=["vc-help"],
                        )
                        trigger_word = gr.Textbox(
                            value="ohwx",
                            label="Trigger word",
                            info="Concept token used in prompt templates and optional caption injection.",
                        )
                        controls["trigger_word"] = ctx.reg(
                            "trigger_word", trigger_word, "ohwx", section="prompt",
                            description="Concept trigger token used by templates and post-processing.", kind="str",
                        )
                        with gr.Row():
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
                        with gr.Row():
                            caption_length = gr.Dropdown(
                                choices=list(_CAPTION_LENGTH_CHOICES),
                                value="detailed",
                                allow_custom_value=True,
                                label="Caption length",
                                info="Natural-language detail target inserted into compatible templates.",
                            )
                            controls["caption_length"] = ctx.reg(
                                "caption_length", caption_length, "detailed", section="prompt",
                                description="Requested caption length or detail level.", kind="str",
                            )
                            valid_caption_length_state = gr.State("detailed")
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
                    reset_prompts = action_button("↺ Reset prompts to preset", "purple")

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
                    with gr.Row():
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
                    no_repeat_ngram_size = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=20,
                        step=1,
                        precision=0,
                        label="No-repeat n-gram size",
                        info=_CONTROL_INFO["no_repeat_ngram_size"],
                        elem_id="vc_no_repeat_ngram_size",
                    )
                    controls["no_repeat_ngram_size"] = ctx.reg(
                        "no_repeat_ngram_size",
                        no_repeat_ngram_size,
                        0,
                        section="generation",
                        description=_CONTROL_INFO["no_repeat_ngram_size"],
                        kind="int",
                        minimum=0,
                        maximum=20,
                    )
                    max_new_tokens = gr.Slider(
                        1,
                        _GLOBAL_MAX_NEW_TOKENS,
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
                    context_tokens = gr.Number(
                        value=int(initial_spec.limits.context_tokens),
                        minimum=1024,
                        maximum=_GLOBAL_MAX_CONTEXT,
                        step=256,
                        precision=0,
                        label="Context length (tokens)",
                        info=_context_info(initial_spec),
                    )
                    controls["context_tokens"] = ctx.reg(
                        "context_tokens", context_tokens, int(initial_spec.limits.context_tokens), section="generation",
                        description=(
                            "Requested context window in tokens; capped by the selected model and, for GGUF, "
                            "by the VRAM tier and llama.cpp's memory fitter."
                        ),
                        kind="int", minimum=1024, maximum=_GLOBAL_MAX_CONTEXT,
                    )
                    with gr.Row():
                        do_sample = gr.Checkbox(
                            value=bool(schema["do_sample"].default),
                            label="Sample tokens",
                            info="Prompt presets set this automatically; disable for deterministic greedy decoding.",
                        )
                        controls["do_sample"] = ctx.reg(
                            "do_sample", do_sample, bool(schema["do_sample"].default), section="generation",
                            description=schema["do_sample"].description, kind="bool",
                        )
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
                            elem_id="vc_seed",
                        )
                        controls["seed"] = ctx.reg(
                            "seed", seed, -1, section="generation",
                            description=(
                                "Seed for sampled decoding; -1 draws a fresh random seed every run. Greedy decoding "
                                "(Sample tokens off) is deterministic without it. The seed actually used is written to metadata."
                            ),
                            kind="int", minimum=-1, maximum=2147483647,
                        )
                        use_cache = gr.Checkbox(
                            value=True,
                            label="Use KV cache",
                            info=_CONTROL_INFO["use_cache"],
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

        last_outputs_state = gr.State({})
        job_done_hook = gr.HTML("", visible=False, elem_id="vcap-job-done-hook")
        ctx.states["last_outputs"] = last_outputs_state

    with gr.Tab("🎞️ Processing Pipeline", id="processing"):
        gr.Markdown(
            "Frame sampling, scene splitting, and caption text shaping for every Caption tab run. "
            "These controls are saved, loaded, and recovered with presets exactly like the Caption tab.",
            elem_classes=["vc-help"],
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=380):
                with gr.Accordion("4. Preprocessing", open=True):
                    with gr.Row():
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
                            maximum=_GLOBAL_MAX_FRAMES,
                            step=2,
                            precision=0,
                            label="Maximum frames",
                            info=_frames_info(initial_spec),
                        )
                        controls["max_frames"] = ctx.reg(
                            "max_frames", max_frames, initial_spec.limits.max_frames or 768, section="preprocessing",
                            description="Maximum decoded frames supplied to the processor; zero disables visual frames for audio workflows.", kind="int", minimum=0, maximum=_GLOBAL_MAX_FRAMES,
                        )
                    with gr.Row():
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
                            info=(
                                "Maximum resized frame area while preserving aspect ratio. TimeChat and AVoCaDO "
                                "have a 602,112-pixel per-frame safety ceiling; larger values are reduced and logged."
                            ),
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
                    with gr.Accordion(
                        "Advanced frame budget", open=False,
                        elem_id="vc_advanced_frame_budget",
                    ):
                        total_pixel_cap = gr.Number(
                            value=0,
                            minimum=0,
                            maximum=400000000,
                            step=1024,
                            precision=0,
                            label="Total pixel cap",
                            info=(
                                "Total decoded pixel budget across all sampled frames of one request; 0 uses the model "
                                "default (20,070,400). When a clip would exceed it, per-frame resolution is reduced "
                                "automatically and logged."
                            ),
                            elem_id="vc_total_pixel_cap",
                        )
                        controls["total_pixel_cap"] = ctx.reg(
                            "total_pixel_cap", total_pixel_cap, 0, section="preprocessing",
                            description=(
                                "Total decoded pixel budget across all sampled frames of one request; 0 uses the model "
                                "default (20,070,400). When a clip would exceed it, per-frame resolution is reduced "
                                "automatically and logged."
                            ),
                            kind="int", minimum=0, maximum=400000000,
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
                    adaptive_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=50.0,
                        value=2.0,
                        step=0.1,
                        label="Adaptive threshold",
                        info=(
                            "Sensitivity of adaptive sampling: minimum mean pixel difference (0-255 on a 64x64 "
                            "thumbnail) for a frame to count as visually new. Lower keeps more frames; used only "
                            "when Sampling strategy is adaptive."
                        ),
                        elem_id="vc_adaptive_threshold",
                    )
                    controls["adaptive_threshold"] = ctx.reg(
                        "adaptive_threshold", adaptive_threshold, 2.0, section="preprocessing",
                        description=(
                            "Minimum mean thumbnail-pixel difference for a frame to count as visually new during "
                            "adaptive sampling."
                        ),
                        kind="float", minimum=0.1, maximum=50.0,
                    )
                    preview_sampled_frames = action_button(
                        "🖼 Preview sampled frames", "aqua",
                        elem_id="vc_preview_sampled_frames",
                    )
                    sampled_frame_status = gr.Markdown(
                        "<span class='vc-help'>Select a video to inspect the CPU-only sampling plan.</span>",
                        elem_classes=["vc-help"],
                    )
                    sampled_frame_gallery = gr.Gallery(
                        value=[],
                        label="Sampled frames",
                        columns=4,
                        rows=2,
                        height=300,
                        object_fit="contain",
                        allow_preview=True,
                        buttons=["fullscreen"],
                        elem_id="vc_sampled_frame_gallery",
                    )
                    with gr.Row():
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
                            info=(
                                "The models always receive 16 kHz internally; this rate applies to extracted/saved "
                                "audio clips and the normalized clip."
                            ),
                        )
                        controls["audio_sample_rate"] = ctx.reg(
                            "audio_sample_rate", audio_rate, 16000, section="preprocessing",
                            description="Mono audio extraction sample rate.", choices=[16000, 24000, 48000], kind="int",
                        )
                    with gr.Accordion("Automatic rejection", open=True):
                        auto_reject = gr.Checkbox(
                            value=False,
                            label="Enable automatic rejection",
                            info="Analyze split clips and skip those failing any enabled quality rule.",
                        )
                        controls["auto_reject_enabled"] = ctx.reg(
                            "auto_reject_enabled", auto_reject, False, section="preprocessing",
                            description="Enable clip quality rejection before caption generation.", kind="bool",
                        )
                        with gr.Row():
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
                            reject_black_luma = gr.Number(
                                value=16,
                                minimum=0,
                                maximum=255,
                                step=1,
                                precision=0,
                                label="Black luma threshold",
                                info=(
                                    "Luma level at or below which a pixel counts as black for the mostly-black "
                                    "rejection rule."
                                ),
                                elem_id="vc_reject_black_luma",
                            )
                            controls["reject_black_luma"] = ctx.reg(
                                "reject_black_luma", reject_black_luma, 16, section="preprocessing",
                                description="Luma level at or below which a pixel counts as black during rejection.",
                                kind="int", minimum=0, maximum=255,
                            )
                        with gr.Row():
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
                            quality_frames = gr.Number(
                                value=8,
                                minimum=4,
                                maximum=32,
                                step=1,
                                precision=0,
                                label="Quality frames",
                                info=(
                                    "Frames analyzed per clip by the automatic rejection rules (black, motion, sharpness). "
                                    "More frames are more accurate and slower."
                                ),
                                elem_id="vc_quality_frames",
                            )
                            controls["quality_frames"] = ctx.reg(
                                "quality_frames", quality_frames, 8, section="preprocessing",
                                description=(
                                    "Frames analyzed per clip by the automatic rejection rules (black, motion, sharpness). "
                                    "More frames are more accurate and slower."
                                ),
                                kind="int", minimum=4, maximum=32,
                            )
                        with gr.Row():
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
                            reject_silence_rms = gr.Number(
                                value=0.001,
                                minimum=0.0,
                                maximum=0.1,
                                step=0.0005,
                                label="Silence RMS threshold",
                                info=(
                                    "RMS amplitude below which a 20 ms audio window counts as silent for the "
                                    "silence-ratio rejection rule."
                                ),
                                elem_id="vc_reject_silence_rms",
                            )
                            controls["reject_silence_rms"] = ctx.reg(
                                "reject_silence_rms", reject_silence_rms, 0.001, section="preprocessing",
                                description="RMS amplitude below which a 20 ms audio window counts as silent.",
                                kind="float", minimum=0.0, maximum=0.1,
                            )
                    token_budget = gr.Markdown("<span class='vc-help'>Upload media to estimate the live token budget.</span>")
            with gr.Column(scale=1, min_width=380):
                with gr.Accordion("5. Scene detection & splitting", open=True):
                    with gr.Row():
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
                    with gr.Row():
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
                    with gr.Row():
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
                    with gr.Row():
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
                        fade_threshold = gr.Number(
                            value=12.0,
                            minimum=1,
                            maximum=100,
                            step=0.5,
                            label="Fade threshold",
                            info="Fade level for the threshold detector used when Detect fades is on; lower detects softer fades.",
                            interactive=False,
                            elem_id="vc_fade_threshold",
                        )
                        controls["fade_threshold"] = ctx.reg(
                            "fade_threshold", fade_threshold, 12.0, section="splitting",
                            description="Fade level for the threshold detector used when Detect fades is on; lower detects softer fades.",
                            kind="float", minimum=1, maximum=100,
                        )
                        downscale = gr.Number(value=0, minimum=0, maximum=16, step=1, precision=0, label="Detection downscale", info="Zero lets PySceneDetect choose automatically.")
                        controls["scene_downscale"] = ctx.reg(
                            "scene_downscale", downscale, 0, section="splitting",
                            description="Explicit scene-analysis downscale factor; zero is automatic.", kind="int", minimum=0, maximum=16,
                        )
                    with gr.Row():
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
                    gr.Markdown("**Re-encode quality**")
                    gr.Markdown(
                        "Used for Precise re-encode splits and Normalize clip.",
                        elem_classes=["vc-help"],
                    )
                    with gr.Row():
                        encode_codec = gr.Dropdown(
                            choices=["libx264", "h264_nvenc", "libx265", "hevc_nvenc"],
                            value="libx264",
                            label="Video codec",
                            info=(
                                "Video encoder for precise re-encode splits and normalized clips. NVENC encoders need an "
                                "NVIDIA GPU and an FFmpeg build with NVENC; an unavailable encoder falls back to libx264 "
                                "with a logged warning."
                            ),
                            elem_id="vc_encode_codec",
                        )
                        controls["encode_codec"] = ctx.reg(
                            "encode_codec", encode_codec, "libx264", section="splitting",
                            description=(
                                "Video encoder for precise re-encode splits and normalized clips. NVENC encoders need an "
                                "NVIDIA GPU and an FFmpeg build with NVENC; an unavailable encoder falls back to libx264 "
                                "with a logged warning."
                            ),
                            kind="str", choices=["libx264", "h264_nvenc", "libx265", "hevc_nvenc"],
                        )
                        encode_crf = gr.Number(
                            value=18, minimum=0, maximum=51, step=1, precision=0,
                            label="CRF / CQ",
                            info="Constant quality for re-encoded clips (lower = higher quality, larger files). Maps to -crf for x264/x265 and -cq for NVENC.",
                            elem_id="vc_encode_crf",
                        )
                        controls["encode_crf"] = ctx.reg(
                            "encode_crf", encode_crf, 18, section="splitting",
                            description="Constant quality for re-encoded clips (lower = higher quality, larger files). Maps to -crf for x264/x265 and -cq for NVENC.",
                            kind="int", minimum=0, maximum=51,
                        )
                        encode_preset = gr.Dropdown(
                            choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"],
                            value="veryfast",
                            label="Encoder preset",
                            info="Encoder speed/efficiency preset for re-encoded clips; NVENC maps these to p1–p7.",
                            elem_id="vc_encode_preset",
                        )
                        controls["encode_preset"] = ctx.reg(
                            "encode_preset", encode_preset, "veryfast", section="splitting",
                            description="Encoder speed/efficiency preset for re-encoded clips; NVENC maps these to p1–p7.",
                            kind="str", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower"],
                        )
                        encode_audio_bitrate = gr.Dropdown(
                            choices=["96k", "128k", "192k", "256k", "320k"],
                            value="192k",
                            label="Audio bitrate",
                            info="AAC bitrate for re-encoded clip audio.",
                            elem_id="vc_encode_audio_bitrate",
                        )
                        controls["encode_audio_bitrate"] = ctx.reg(
                            "encode_audio_bitrate", encode_audio_bitrate, "192k", section="splitting",
                            description="AAC bitrate for re-encoded clip audio.",
                            kind="str", choices=["96k", "128k", "192k", "256k", "320k"],
                        )
                    with gr.Row():
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
                    with gr.Row():
                        context_carry_over = gr.Checkbox(
                            value=False,
                            label="Carry previous chunk context",
                            info=(
                                "Feeds the tail of the previous chunk's text into the next chunk prompt for long transcriptions; "
                                "not used by Captioner or TimeChat."
                            ),
                        )
                        controls["context_carry_over"] = ctx.reg(
                            "context_carry_over", context_carry_over, False, section="splitting",
                            description="Carry the previous transcription chunk's text into the next chunk prompt.", kind="bool",
                        )
                        context_carry_words = gr.Number(
                            value=60,
                            minimum=10,
                            maximum=400,
                            step=10,
                            precision=0,
                            label="Context carry words",
                            info=(
                                "Words copied from the end of the previous segment's caption into the next segment's prompt "
                                "when Carry previous chunk context is on."
                            ),
                            interactive=False,
                            elem_id="vc_context_carry_words",
                        )
                        controls["context_carry_words"] = ctx.reg(
                            "context_carry_words", context_carry_words, 60, section="splitting",
                            description=(
                                "Words copied from the end of the previous segment's caption into the next segment's prompt "
                                "when Carry previous chunk context is on."
                            ),
                            kind="int", minimum=10, maximum=400,
                        )
                    context_carry_prompt = gr.Textbox(
                        value="Context from the previous segment (do not repeat it): {{CONTEXT}}",
                        label="Context carry prompt",
                        info=(
                            "Wrapper for carried-over context; {{CONTEXT}} is replaced by the previous segment's "
                            "last words."
                        ),
                        lines=2,
                        max_lines=4,
                        interactive=False,
                        elem_id="vc_context_carry_prompt",
                    )
                    controls["context_carry_prompt"] = ctx.reg(
                        "context_carry_prompt",
                        context_carry_prompt,
                        "Context from the previous segment (do not repeat it): {{CONTEXT}}",
                        section="splitting",
                        description=(
                            "Wrapper for carried-over context; {{CONTEXT}} is replaced by the previous segment's "
                            "last words."
                        ),
                        kind="str",
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
                    detect_now = action_button("◫ Detect scenes now (preview)", "indigo")
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
            with gr.Column(scale=1, min_width=380):
                with gr.Accordion("6. Post-processing", open=True):
                    with gr.Row():
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
                            value="none",
                            label="Trigger injection",
                            info="Places the template trigger word before, after, or outside the saved caption.",
                        )
                        controls["trigger_mode"] = ctx.reg(
                            "trigger_mode", trigger_mode, "none", section="postprocessing",
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
                    with gr.Row():
                        dedupe_repeated_sentences = gr.Checkbox(
                            value=True,
                            label="Remove repeated sentences",
                            info=(
                                "Remove sentences that repeat verbatim inside one caption, keeping the first "
                                "occurrence (case- and whitespace-insensitive). Applied before prefix, suffix, and trigger."
                            ),
                            elem_id="vc_dedupe_repeated_sentences",
                        )
                        controls["dedupe_repeated_sentences"] = ctx.reg(
                            "dedupe_repeated_sentences", dedupe_repeated_sentences, True,
                            section="postprocessing",
                            description=(
                                "Remove verbatim repeated sentences inside one caption before prefix, suffix, and trigger."
                            ),
                            kind="bool",
                        )
                        caption_join_separator = gr.Textbox(
                            value=" ",
                            label="Caption join separator",
                            info=(
                                "Text placed between the prefix/trigger, caption, and suffix (default one space). "
                                "Use ', ' for tag-style captions; an empty value joins directly."
                            ),
                            max_length=16,
                            elem_id="vc_caption_join_separator",
                        )
                        controls["caption_join_separator"] = ctx.reg(
                            "caption_join_separator", caption_join_separator, " ",
                            section="postprocessing",
                            description="Text placed between caption prefixes, triggers, the model text, and suffixes.",
                            kind="str",
                        )
                    max_caption_chars = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=100000,
                        step=100,
                        precision=0,
                        label="Maximum caption characters",
                        info=(
                            "Maximum final caption length in characters; 0 keeps the full caption. Trims at the last "
                            "sentence end (or word boundary) before the limit, after prefix, suffix, trigger and replacements."
                        ),
                        elem_id="vc_max_caption_chars",
                    )
                    controls["max_caption_chars"] = ctx.reg(
                        "max_caption_chars", max_caption_chars, 0, section="postprocessing",
                        description=(
                            "Maximum final caption length in characters; 0 keeps the full caption. Trims at the last "
                            "sentence end (or word boundary) before the limit, after prefix, suffix, trigger and replacements."
                        ),
                        kind="int", minimum=0, maximum=100000,
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
                        kind="list", choices=["txt", "json", "srt", "vtt", "jsonl"],
                    )
                    with gr.Row():
                        subtitle_min_cue_s = gr.Number(
                            value=0.5,
                            minimum=0.0,
                            maximum=5.0,
                            step=0.1,
                            label="Minimum subtitle cue (s)",
                            info=(
                                "Minimum duration of one SRT/VTT cue; shorter cues are extended (still clamped to "
                                "the clip window) so players can display them."
                            ),
                            elem_id="vc_subtitle_min_cue_s",
                        )
                        controls["subtitle_min_cue_s"] = ctx.reg(
                            "subtitle_min_cue_s", subtitle_min_cue_s, 0.5, section="output",
                            description="Minimum duration of one SRT/VTT cue before clip-window clamping.",
                            kind="float", minimum=0.0, maximum=5.0,
                        )
                        subtitle_max_line_chars = gr.Number(
                            value=0,
                            minimum=0,
                            maximum=200,
                            step=1,
                            precision=0,
                            label="Subtitle line width",
                            info=(
                                "Wrap subtitle cue text at word boundaries into lines of at most this many "
                                "characters (0 = no wrapping). Typical player-friendly values are 42-60."
                            ),
                            elem_id="vc_subtitle_max_line_chars",
                        )
                        controls["subtitle_max_line_chars"] = ctx.reg(
                            "subtitle_max_line_chars", subtitle_max_line_chars, 0, section="output",
                            description="Maximum subtitle cue line width in characters; zero disables wrapping.",
                            kind="int", minimum=0, maximum=200,
                        )
                    save_reasoning = gr.Checkbox(
                        value=True,
                        label="Save reasoning",
                        info="Persist Thinking-model reasoning separately; it remains hidden from the caption.",
                    )
                    controls["save_reasoning"] = ctx.reg(
                        "save_reasoning", save_reasoning, True, section="output",
                        description="Write Thinking-model reasoning to a separate text file.", kind="bool",
                    )
                    with gr.Accordion(
                        "Long video summary & chapters", open=False,
                        elem_id="vc_long_video_summary",
                    ):
                        summarize_segments = gr.Checkbox(
                            value=False,
                            label="Summarize segments",
                            info=(
                                "After a multi-segment item finishes, feed all segment captions back to the model to write "
                                "a whole-video summary with chapters (<stem>_summary.txt, also stored under \"summary\" in the "
                                "JSON output). Needs a text-capable model (Qwen3-Omni Instruct/Thinking); other models log a "
                                "warning and skip."
                            ),
                            elem_id="vc_summarize_segments",
                        )
                        controls["summarize_segments"] = ctx.reg(
                            "summarize_segments", summarize_segments, False, section="output",
                            description=(
                                "After a multi-segment item finishes, feed all segment captions back to the model to write "
                                "a whole-video summary with chapters (<stem>_summary.txt, also stored under \"summary\" in the "
                                "JSON output). Needs a text-capable model (Qwen3-Omni Instruct/Thinking); other models log a "
                                "warning and skip."
                            ), kind="bool",
                        )
                        summary_prompt = gr.Textbox(
                            value=DEFAULT_SUMMARY_PROMPT,
                            label="Summary prompt",
                            info="Prompt used for the summary stage; {{LANGUAGE}} is rendered.",
                            lines=7,
                            max_lines=12,
                            elem_classes=["vc-mono"],
                            elem_id="vc_summary_prompt",
                        )
                        controls["summary_prompt"] = ctx.reg(
                            "summary_prompt", summary_prompt, DEFAULT_SUMMARY_PROMPT, section="output",
                            description="Prompt used for the summary stage; {{LANGUAGE}} is rendered.", kind="str",
                        )
                        summary_max_new_tokens = gr.Number(
                            value=1024,
                            minimum=64,
                            maximum=8192,
                            step=64,
                            precision=0,
                            label="Summary maximum new tokens",
                            info="Maximum tokens generated by the long-video summary and chapters stage.",
                            elem_id="vc_summary_max_new_tokens",
                        )
                        controls["summary_max_new_tokens"] = ctx.reg(
                            "summary_max_new_tokens", summary_max_new_tokens, 1024, section="output",
                            description="Maximum tokens generated by the long-video summary and chapters stage.",
                            kind="int", minimum=64, maximum=8192,
                        )

                with gr.Accordion("7. Speech transcript (Whisper)", open=False):
                    transcript_enabled = gr.Checkbox(
                        value=False,
                        label="Also transcribe speech with Whisper during caption runs",
                        info=(
                            "Runs Whisper once on each video or audio item before captioning and writes "
                            "transcript sidecars beside the caption outputs."
                        ),
                        elem_id="vc_transcript_enabled",
                    )
                    controls["transcript_enabled"] = ctx.reg(
                        "transcript_enabled",
                        transcript_enabled,
                        False,
                        section="transcript",
                        description="Also transcribe speech with Whisper during caption runs.",
                        kind="bool",
                    )
                    transcript_formats = gr.CheckboxGroup(
                        choices=[
                            ("SubRip", "srt"),
                            ("WebVTT", "vtt"),
                            ("Plain text", "txt"),
                            ("LRC", "lrc"),
                            ("TSV", "tsv"),
                            ("JSON", "json"),
                        ],
                        value=["srt", "txt"],
                        label="Transcript formats",
                        info="File formats written beside each caption output.",
                        show_select_all=True,
                    )
                    controls["transcript_formats"] = ctx.reg(
                        "transcript_formats",
                        transcript_formats,
                        ["srt", "txt"],
                        section="transcript",
                        description="Transcript sidecar formats written during caption runs.",
                        kind="list",
                        choices=["srt", "vtt", "txt", "lrc", "tsv", "json"],
                    )
                    transcript_inject_prompt = gr.Checkbox(
                        value=True,
                        label="Inject transcript into the caption prompt",
                        info=(
                            "Adds the overlapping speech to each clip prompt when the prompt does not "
                            "already contain {{TRANSCRIPT}}."
                        ),
                    )
                    controls["transcript_inject_prompt"] = ctx.reg(
                        "transcript_inject_prompt",
                        transcript_inject_prompt,
                        True,
                        section="transcript",
                        description="Inject the overlapping Whisper transcript into each caption prompt.",
                        kind="bool",
                    )
                    transcript_prompt_wrapper = gr.Textbox(
                        value=(
                            "Exact speech transcript for this clip (use it verbatim for dialogue, do not invent speech):\n"
                            "{{TRANSCRIPT}}"
                        ),
                        label="Transcript prompt wrapper",
                        info=(
                            "Block appended when prompt injection is on and the user prompt has no "
                            "{{TRANSCRIPT}} token."
                        ),
                        lines=3,
                        max_lines=6,
                        elem_classes=["vc-mono"],
                    )
                    controls["transcript_prompt_wrapper"] = ctx.reg(
                        "transcript_prompt_wrapper",
                        transcript_prompt_wrapper,
                        (
                            "Exact speech transcript for this clip (use it verbatim for dialogue, do not invent speech):\n"
                            "{{TRANSCRIPT}}"
                        ),
                        section="transcript",
                        description="Wrapper appended around an injected clip-local transcript.",
                        kind="str",
                    )
                    transcript_file_suffix = gr.Textbox(
                        value="_transcript",
                        label="Transcript file suffix",
                        info="Suffix inserted between the source stem and transcript extension.",
                        max_length=80,
                    )
                    controls["transcript_file_suffix"] = ctx.reg(
                        "transcript_file_suffix",
                        transcript_file_suffix,
                        "_transcript",
                        section="transcript",
                        description="Filename suffix used for transcript sidecars.",
                        kind="str",
                    )
                    gr.Markdown(
                        "Uses the model and decoding settings from the Transcribe tab. The Whisper model "
                        "downloads automatically on first use.",
                        elem_classes=["vc-help"],
                    )

                with gr.Accordion(
                    "8. Audio captions & dataset clip layout",
                    open=False,
                    elem_id="vc_audio_caption_layout",
                ):
                    audio_caption_layout_hint = gr.Markdown(
                        "<span class='vc-help'>Audio captions are off; the current single-caption layout is unchanged.</span>",
                        elem_classes=["vc-status"],
                        elem_id="vc_audio_caption_layout_hint",
                    )
                    audio_caption_source = gr.Dropdown(
                        choices=[
                            ("Off (single caption file, current behaviour)", "none"),
                            ("Whisper speech transcript", "whisper"),
                            ("Qwen3-Omni Captioner sound description", "captioner"),
                            ("Whisper transcript + Captioner description", "both"),
                        ],
                        value="none",
                        label="Audio caption source",
                        info=(
                            "Select what produces each clip's audio caption. Any choice except Off writes "
                            "video_caption/ and audio_caption/ dataset folders."
                        ),
                        elem_id="vc_audio_caption_source",
                    )
                    controls["audio_caption_source"] = ctx.reg(
                        "audio_caption_source",
                        audio_caption_source,
                        "none",
                        section="audio_captions",
                        description="Source used to produce the audio caption for each clip.",
                        choices=["none", "whisper", "captioner", "both"],
                        kind="str",
                    )
                    video_caption_source = gr.Radio(
                        choices=[
                            ("Generate with the selected model", "generate"),
                            ("Reuse existing caption files (skip the caption model)", "existing"),
                        ],
                        value="generate",
                        label="Video caption source",
                        info=(
                            "Existing mode reads a clean caption from video_caption/<name>.txt, then <name>.txt, "
                            "then beside the source, and never loads the selected main caption model."
                        ),
                        interactive=False,
                        elem_id="vc_video_caption_source",
                    )
                    controls["video_caption_source"] = ctx.reg(
                        "video_caption_source",
                        video_caption_source,
                        "generate",
                        section="audio_captions",
                        description="Generate video captions or reuse existing caption sidecars without loading the main model.",
                        choices=["generate", "existing"],
                        kind="str",
                    )
                    audio_caption_model_key = gr.Dropdown(
                        choices=_audio_captioner_choices(),
                        value="auto",
                        label="Sound-caption model",
                        info=(
                            "Qwen3-Omni Captioner variant used for sound descriptions. Auto matches the selected "
                            "main model precision/backend and uses the VRAM tier for 7B BF16 models."
                        ),
                        interactive=False,
                        elem_id="vc_audio_caption_model_key",
                    )
                    controls["audio_caption_model_key"] = ctx.reg(
                        "audio_caption_model_key",
                        audio_caption_model_key,
                        "auto",
                        section="audio_captions",
                        description="Qwen3-Omni Captioner variant for prompt-free sound descriptions; auto matches the main model.",
                        choices=[value for _, value in _audio_captioner_choices()],
                        kind="str",
                    )
                    audio_caption_transcript_style = gr.Radio(
                        choices=[
                            ("Plain text (one paragraph)", "plain"),
                            ("One segment per line", "lines"),
                            ("Timestamped lines [mm:ss.s - mm:ss.s]", "timestamped"),
                        ],
                        value="plain",
                        label="Whisper transcript style",
                        info="Controls how clip-local Whisper segments are rendered in audio_caption/<name>.txt.",
                        interactive=False,
                        elem_id="vc_audio_caption_transcript_style",
                    )
                    controls["audio_caption_transcript_style"] = ctx.reg(
                        "audio_caption_transcript_style",
                        audio_caption_transcript_style,
                        "plain",
                        section="audio_captions",
                        description="Rendering style for the clip-local Whisper transcript in audio captions.",
                        choices=["plain", "lines", "timestamped"],
                        kind="str",
                    )
                    audio_caption_template = gr.Textbox(
                        value=DEFAULT_AUDIO_CAPTION_TEMPLATE,
                        label="Audio caption template",
                        info="Tokens: {{TRANSCRIPT}}, {{SOUND_CAPTION}}, {{FILENAME}}. Empty parts collapse cleanly.",
                        lines=3,
                        max_lines=4,
                        interactive=False,
                        elem_classes=["vc-mono"],
                        elem_id="vc_audio_caption_template",
                    )
                    controls["audio_caption_template"] = ctx.reg(
                        "audio_caption_template",
                        audio_caption_template,
                        DEFAULT_AUDIO_CAPTION_TEMPLATE,
                        section="audio_captions",
                        description="Template used to compose each audio_caption text file.",
                        kind="str",
                    )
                    caption_write_merged = gr.Checkbox(
                        value=True,
                        label="Write merged caption beside each clip",
                        info="Write <stem>.txt beside the clip; off leaves only video_caption/ and audio_caption/ parts.",
                        interactive=False,
                        elem_id="vc_caption_write_merged",
                    )
                    controls["caption_write_merged"] = ctx.reg(
                        "caption_write_merged",
                        caption_write_merged,
                        True,
                        section="audio_captions",
                        description="Write a merged caption beside every clip in split-layout runs.",
                        kind="bool",
                    )
                    caption_merge_template = gr.Textbox(
                        value=DEFAULT_CAPTION_MERGE_TEMPLATE,
                        label="Merged caption template",
                        info=(
                            "Tokens: {{VIDEO_CAPTION}}, {{AUDIO_CAPTION}}, {{TRANSCRIPT}}, "
                            "{{SOUND_CAPTION}}, {{FILENAME}}."
                        ),
                        lines=3,
                        max_lines=6,
                        interactive=False,
                        elem_classes=["vc-mono"],
                        elem_id="vc_caption_merge_template",
                    )
                    controls["caption_merge_template"] = ctx.reg(
                        "caption_merge_template",
                        caption_merge_template,
                        DEFAULT_CAPTION_MERGE_TEMPLATE,
                        section="audio_captions",
                        description="Template used to compose merged per-clip captions.",
                        kind="str",
                    )
                    with gr.Row():
                        audio_caption_empty_policy = gr.Radio(
                            choices=[
                                ("Skip the audio caption (merged file = video caption only)", "skip"),
                                ("Write a placeholder", "placeholder"),
                            ],
                            value="skip",
                            label="Empty-audio policy",
                            info="Controls output when Whisper and the sound Captioner both return no text.",
                            interactive=False,
                            scale=3,
                            elem_id="vc_audio_caption_empty_policy",
                        )
                        controls["audio_caption_empty_policy"] = ctx.reg(
                            "audio_caption_empty_policy",
                            audio_caption_empty_policy,
                            "skip",
                            section="audio_captions",
                            description="Skip an empty audio caption or write a placeholder.",
                            choices=["skip", "placeholder"],
                            kind="str",
                        )
                        audio_caption_empty_text = gr.Textbox(
                            value="No speech.",
                            label="Empty-audio placeholder",
                            info="Text written only when the empty-audio policy is Write a placeholder.",
                            interactive=False,
                            scale=2,
                            elem_id="vc_audio_caption_empty_text",
                        )
                        controls["audio_caption_empty_text"] = ctx.reg(
                            "audio_caption_empty_text",
                            audio_caption_empty_text,
                            "No speech.",
                            section="audio_captions",
                            description="Placeholder written when an audio caption is empty.",
                            kind="str",
                        )

    handles = CaptionTabHandles(
        media=media,
        progress=progress,
        logs=logs,
        controls=controls,
        start=start,
        cancel=cancel,
        cancel_confirmation=cancel_confirmation,
        cancel_note=cancel_note,
        cancel_yes=cancel_yes,
        cancel_keep=cancel_keep,
        hotkey_start=hotkey_start,
        hotkey_cancel=hotkey_cancel,
        cancel_timer=cancel_timer,
        unload_model=unload_model,
        open_output=open_output,
        open_caption=open_caption,
        reveal_clip=reveal_clip,
        open_editor=open_editor,
        copy_caption=copy_caption,
        retry_failed=retry_failed,
        results_zip=results_zip,
        results_zip_file=results_zip_file,
        run_history=run_history,
        run_history_refresh=run_history_refresh,
        run_history_open=run_history_open,
        run_history_editor=run_history_editor,
        run_history_recover=run_history_recover,
        run_history_status=run_history_status,
        run_history_records_state=run_history_records_state,
        run_history_selected_state=run_history_selected_state,
        item_table=item_table,
        caption=caption,
        structured=structured,
        srt=srt,
        reasoning=reasoning,
        reasoning_tab=reasoning_tab,
        files=result_files,
        clips=clips,
        clips_empty_hint=clips_empty_hint,
        last_outputs_state=last_outputs_state,
        job_done_hook=job_done_hook,
    )
    ctx.caption_handles = handles

    # Lightweight local interactions can be wired immediately.
    last_outputs_state.change(
        lambda value: [
            path
            for path in dict(value or {}).get("files", [])
            if path and Path(str(path)).is_file()
        ],
        inputs=last_outputs_state,
        outputs=result_files,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    gpu_picker.change(
        lambda value: int(value or 0),
        inputs=gpu_picker,
        outputs=ctx.states["gpu_index"],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def set_subprocess_mode(enabled: bool) -> None:
        try:
            setter = getattr(ctx.pipeline_client, "set_subprocess_mode", None)
            if callable(setter):
                setter(bool(enabled))
            else:
                ctx.pipeline_client.subprocess_mode = bool(enabled)
        except Exception as exc:
            ctx.app_log.warn(f"Could not change subprocess mode: {exc}", scope="runtime")

    ctx.states["set_subprocess_mode"] = set_subprocess_mode
    subprocess_mode.change(
        set_subprocess_mode,
        inputs=subprocess_mode,
        outputs=[],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def audio_caption_control_state(
        source: str,
        write_merged: bool = True,
        empty_policy: str = "skip",
    ) -> tuple[Any, ...]:
        selected = str(source or "none").casefold()
        active = selected != "none"
        uses_captioner = selected in {"captioner", "both"}
        uses_whisper = selected in {"whisper", "both"}
        if active:
            merged = "merged `<name>.txt` next to the clip" if write_merged else "separate parts only"
            hint = (
                "<span class='vc-ok'>Per clip: `video_caption/<name>.txt`, "
                f"`audio_caption/<name>.txt`, {merged}.</span>"
            )
        else:
            hint = "<span class='vc-help'>Audio captions are off; the current single-caption layout is unchanged.</span>"
        return (
            gr.update(interactive=active),
            gr.update(interactive=uses_captioner, choices=_audio_captioner_choices()),
            gr.update(interactive=uses_whisper),
            gr.update(interactive=active),
            gr.update(interactive=active),
            gr.update(interactive=active and bool(write_merged)),
            gr.update(interactive=active),
            gr.update(interactive=active and str(empty_policy) == "placeholder"),
            hint,
        )

    audio_caption_state_inputs = [
        audio_caption_source,
        caption_write_merged,
        audio_caption_empty_policy,
    ]
    audio_caption_state_outputs = [
        video_caption_source,
        audio_caption_model_key,
        audio_caption_transcript_style,
        audio_caption_template,
        caption_write_merged,
        caption_merge_template,
        audio_caption_empty_policy,
        audio_caption_empty_text,
        audio_caption_layout_hint,
    ]
    for event in (
        audio_caption_source.change,
        caption_write_merged.change,
        audio_caption_empty_policy.change,
    ):
        event(
            audio_caption_control_state,
            inputs=audio_caption_state_inputs,
            outputs=audio_caption_state_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    ctx.states["audio_caption_control_handler"] = audio_caption_control_state

    audio_adapters = ctx.states.setdefault("preset_value_adapters", {})
    audio_adapters["video_caption_source"] = lambda settings: gr.update(
        value=settings.get("video_caption_source", "generate"),
        interactive=str(settings.get("audio_caption_source", "none")) != "none",
    )
    audio_adapters["audio_caption_model_key"] = lambda settings: gr.update(
        value=settings.get("audio_caption_model_key", "auto"),
        choices=_audio_captioner_choices(),
        interactive=str(settings.get("audio_caption_source", "none")) in {"captioner", "both"},
    )
    audio_adapters["audio_caption_transcript_style"] = lambda settings: gr.update(
        value=settings.get("audio_caption_transcript_style", "plain"),
        interactive=str(settings.get("audio_caption_source", "none")) in {"whisper", "both"},
    )
    for key in (
        "audio_caption_template",
        "caption_write_merged",
        "audio_caption_empty_policy",
    ):
        audio_adapters[key] = lambda settings, selected_key=key: gr.update(
            value=settings.get(selected_key),
            interactive=str(settings.get("audio_caption_source", "none")) != "none",
        )
    audio_adapters["caption_merge_template"] = lambda settings: gr.update(
        value=settings.get("caption_merge_template", DEFAULT_CAPTION_MERGE_TEMPLATE),
        interactive=(
            str(settings.get("audio_caption_source", "none")) != "none"
            and bool(settings.get("caption_write_merged", True))
        ),
    )
    audio_adapters["audio_caption_empty_text"] = lambda settings: gr.update(
        value=settings.get("audio_caption_empty_text", "No speech."),
        interactive=(
            str(settings.get("audio_caption_source", "none")) != "none"
            and str(settings.get("audio_caption_empty_policy", "skip")) == "placeholder"
        ),
    )
    compile_probe_timer.tick(
        _probe_compile_in_child,
        outputs=compile_status,
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
        validate_model_variant,
        inputs=[model_key, valid_model_key_state],
        outputs=[model_key, valid_model_key_state, vram_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    def model_identity_lines(value: str) -> tuple[Any, Any]:
        if str(value) not in {key for _, key in _variant_choices()}:
            return gr.skip(), gr.skip()
        return _quant_line(value), _ready_line(value)

    model_key.change(
        model_identity_lines,
        inputs=model_key,
        outputs=[quant_info, ready_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    # This is the first user-selection chain, so the inexpensive identity line
    # changes in the same browser turn as the dropdown instead of waiting behind
    # model-default and VRAM recalculation handlers.
    model_key.select(
        model_identity_lines,
        inputs=model_key,
        outputs=[quant_info, ready_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    gguf_children = [
        gguf_max_frames,
        gguf_jpeg_quality,
        gguf_threads,
        gguf_batch_size,
        gguf_ubatch_size,
        gguf_flash_attn,
        gguf_cache_reuse,
        gguf_ignore_tier_context,
        gguf_min_p,
        gguf_repeat_last_n,
        gguf_presence_penalty,
        gguf_frequency_penalty,
        gguf_fit_headroom_mib,
        gguf_startup_timeout_s,
        gguf_stream_idle_timeout_s,
        gguf_extra_args,
    ]
    backend_controls = [
        attention,
        block_swap_auto,
        blocks_to_swap,
        swap_slots,
        offload_experts,
        pin_cpu,
        pinned_ram_budget_gb,
        plan_slack_mib,
        compile_enabled,
        compile_mode,
        use_cache,
        no_repeat_ngram_size,
    ]
    backend_output_names = [
        "attention_backend",
        "block_swap_auto",
        "blocks_to_swap",
        "swap_slots",
        "offload_experts",
        "pin_cpu",
        "pinned_ram_budget_gb",
        "plan_slack_mib",
        "torch_compile",
        "torch_compile_mode",
        "use_cache",
        "no_repeat_ngram_size",
    ]

    def apply_backend_control_state(variant_key: str, automatic_swap: bool = True) -> tuple[Any, ...]:
        if str(variant_key) not in {key for _, key in _variant_choices()}:
            return tuple(
                gr.skip()
                for _ in range(1 + len(gguf_children) + len(backend_output_names))
            )
        updates = gguf_control_updates(str(variant_key), bool(automatic_swap))
        return (
            gr.update(**updates["gguf_options"]),
            *[gr.update(**updates["gguf_option"]) for _ in gguf_children],
            *[gr.update(**updates[name]) for name in backend_output_names],
        )

    ctx.states["gguf_control_handler"] = apply_backend_control_state
    backend_outputs = [gguf_options, *gguf_children, *backend_controls]
    model_key.change(
        apply_backend_control_state,
        inputs=[model_key, block_swap_auto],
        outputs=backend_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    block_swap_auto.change(
        apply_backend_control_state,
        inputs=[model_key, block_swap_auto],
        outputs=backend_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def set_resolution(value: str) -> Any:
        return gr.skip() if value == "custom" else int(value)

    # Gradio 6 dropdowns fire ``input`` on every blur; ``select`` fires only
    # when the user actually picks an option, which is what these handlers mean.
    resolution_preset.select(
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

    def gpu_total_for(selected_gpu: int) -> float:
        return next(
            (
                device.total_gb
                for device in __import__("vcap.core.gpu", fromlist=["list_gpus"]).list_gpus()
                if device.index == int(selected_gpu)
            ),
            gpu_total or 32,
        )

    def tier_warning(selected_variant: str, physical_tier: int) -> str:
        family = variant_to_family(selected_variant)
        if selected_variant in allowed_variants(family, physical_tier):
            return ""
        required = int(math.ceil(variant_size_gb(selected_variant)))
        return (
            f" <span class='vc-warn'>Selected variant needs ~{required} GB; detected tier "
            f"{physical_tier} GB — expect offload/slow or OOM.</span>"
        )

    vram_outputs = [
        model_key,
        attention,
        fps,
        max_frames,
        max_pixels,
        max_new_tokens,
        block_swap_auto,
        vram_reserve_gb,
        swap_slots,
        offload_experts,
        pin_cpu,
        vram_note,
    ]

    def apply_vram_plan(
        selected_variant: str,
        selected_tier: str,
        selected_gpu: int,
        show_all: bool,
        *,
        switch_variant: bool,
    ) -> tuple[Any, ...]:
        try:
            family = variant_to_family(selected_variant)
            total = gpu_total_for(selected_gpu)
            physical_tier = auto_tier(total)
            resolved_tier = auto_tier(total) if selected_tier == "auto" else int(selected_tier)
            preset = preset_for(family, resolved_tier)
            applied = apply_preset({}, preset)
            candidates = [
                variant.key
                for variant in MODEL_SPECS[family].variants
                if variant.scheme == applied["variant_scheme"]
                and variant.key in allowed_variants(family, resolved_tier)
            ]
            if switch_variant and get_variant(selected_variant).scheme == "gguf":
                # A GGUF preset that does not fit steps down to a smaller GGUF
                # (Q8 -> Q4) so it stays on the fast llama.cpp path instead of
                # jumping to a Transformers precision.
                gguf_candidates = [
                    variant.key
                    for variant in MODEL_SPECS[family].variants
                    if variant.scheme == "gguf" and variant.key in allowed_variants(family, resolved_tier)
                ]
                if gguf_candidates:
                    candidates = gguf_candidates
            next_variant = candidates[0] if switch_variant and candidates else selected_variant
            plan_tier = resolved_tier
            kept = ""
            kept_scheme = get_variant(next_variant).scheme
            if kept_scheme != applied["variant_scheme"]:
                # The tier plan is tuned for another precision; reuse the richest
                # tier plan that targets the kept scheme so offload/frame budgets
                # match the model actually selected (e.g. INT4 resident instead of
                # an INT8 plan with a CPU-offloaded decoder tail).
                for tier_option in sorted((t for t in VRAM_TIERS if t <= resolved_tier), reverse=True):
                    try:
                        candidate_preset = preset_for(family, tier_option)
                    except ValueError:
                        continue  # tier not offered for this family (e.g. Qwen3 below 8 GB)
                    if candidate_preset.variant_scheme == kept_scheme:
                        preset = candidate_preset
                        plan_tier = tier_option
                        break
                kept = f" Keeping the selected variant ({html.escape(get_variant(next_variant).label)})."
                if plan_tier != resolved_tier:
                    kept += f" Using the {plan_tier} GB plan tuned for this precision."
            offload = preset.offload
            choices = variant_choices_for_tier(next_variant, physical_tier, bool(show_all))
            next_spec = MODEL_SPECS[variant_to_family(next_variant)]
            token_cap = int(next_spec.limits.max_new_tokens_cap)
            token_schema = next(
                item for item in next_spec.param_schema if item.name == "max_new_tokens"
            )
            token_update = gr.update(
                value=max(1, min(int(preset.max_new_tokens), token_cap)),
                minimum=1,
                maximum=_GLOBAL_MAX_NEW_TOKENS,
                step=1,
                info=f"{token_schema.description} Selected-family maximum: {token_cap:,}.",
            )
            note = (
                f"<span class='vc-ok'>{resolved_tier} GB plan applied.</span> {html.escape(preset.notes)}{kept}"
                + tier_warning(next_variant, physical_tier)
            )
            return (
                gr.update(choices=choices, value=next_variant),
                preset.attention,
                preset.fps,
                preset.max_frames,
                preset.max_pixels,
                token_update,
                offload.gpu_layers == "auto",
                offload.vram_reserve_gb,
                offload.swap_slots,
                offload.offload_experts,
                offload.pin_cpu,
                note,
            )
        except Exception as exc:
            return (*[gr.skip() for _ in range(11)], f"<span class='vc-err'>{html.escape(str(exc))}</span>")

    def apply_vram(
        selected_variant: str,
        selected_tier: str,
        selected_gpu: int,
        show_all: bool,
    ) -> tuple[Any, ...]:
        return apply_vram_plan(
            selected_variant,
            selected_tier,
            selected_gpu,
            show_all,
            switch_variant=True,
        )

    def apply_auto_vram(
        selected_variant: str,
        selected_tier: str,
        selected_gpu: int,
        show_all: bool,
        *,
        keep_variant: bool = True,
    ) -> tuple[Any, ...]:
        """Apply the automatic tier plan.

        ``keep_variant=True`` (user-driven changes) never replaces the chosen
        variant; the startup path passes ``False`` so a preset variant that
        cannot fit the detected tier is swapped for one that does.
        """

        if selected_tier != "auto":
            physical_tier = auto_tier(gpu_total_for(selected_gpu))
            choices = variant_choices_for_tier(selected_variant, physical_tier, bool(show_all))
            note = (
                f"<span class='vc-help'>Manual {html.escape(str(selected_tier))} GB plan retained.</span>"
                + tier_warning(selected_variant, physical_tier)
            )
            return gr.update(choices=choices, value=selected_variant), *[gr.skip() for _ in range(10)], note
        physical_tier = auto_tier(gpu_total_for(selected_gpu))
        selected_fits = selected_variant in allowed_variants(
            variant_to_family(selected_variant),
            physical_tier,
        )
        return apply_vram_plan(
            selected_variant,
            selected_tier,
            selected_gpu,
            show_all,
            switch_variant=(not selected_fits) and not keep_variant,
        )

    def apply_auto_vram_startup(
        selected_variant: str,
        selected_tier: str,
        selected_gpu: int,
        show_all: bool,
    ) -> tuple[Any, ...]:
        return apply_auto_vram(selected_variant, selected_tier, selected_gpu, show_all, keep_variant=False)

    vram_preset.select(
        apply_vram,
        inputs=[model_key, vram_preset, gpu_picker, show_all_variants],
        outputs=vram_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def filter_variant_choices(selected_variant: str, selected_gpu: int, show_all: bool) -> Any:
        physical_tier = auto_tier(gpu_total_for(selected_gpu))
        return gr.update(
            choices=variant_choices_for_tier(selected_variant, physical_tier, bool(show_all)),
            value=selected_variant,
        )

    show_all_variants.change(
        filter_variant_choices,
        inputs=[model_key, gpu_picker, show_all_variants],
        outputs=model_key,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    gpu_picker.change(
        apply_auto_vram,
        inputs=[model_key, vram_preset, gpu_picker, show_all_variants],
        outputs=vram_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    ctx.states["caption_auto_vram_binding"] = {
        "fn": apply_auto_vram_startup,
        "inputs": [model_key, vram_preset, gpu_picker, show_all_variants],
        # Preset-owned inputs come from the settings a preset load just applied;
        # None keeps the live component value (the GPU choice is not in presets).
        "input_keys": ["model_key", "vram_preset", None, "show_all_variants"],
        "outputs": vram_outputs,
    }

    def preset_max_frames(settings: dict[str, Any]) -> Any:
        """Ship a preset's frame cap together with the family bound it must satisfy.

        A plain value would reach the frontend before ``model_constraints``
        raises the bound, and Gradio rejects inputs above the old maximum.
        """

        try:
            spec = MODEL_SPECS[variant_to_family(str(settings.get("model_key") or _INITIAL_VARIANT))]
        except KeyError:
            return settings.get("max_frames")
        cap = _family_max_frames(spec)
        try:
            value = int(settings.get("max_frames") or 0)
        except (TypeError, ValueError):
            value = cap
        return gr.update(
            value=max(0, min(value, cap)),
            minimum=0,
            maximum=_GLOBAL_MAX_FRAMES,
            step=2,
            info=_frames_info(spec),
        )

    ctx.states.setdefault("preset_value_adapters", {})["max_frames"] = preset_max_frames

    def preset_context_tokens(settings: dict[str, Any]) -> Any:
        """Clamp a preset's context window to the family cap and ship the bound with it."""

        try:
            spec = MODEL_SPECS[variant_to_family(str(settings.get("model_key") or _INITIAL_VARIANT))]
        except KeyError:
            return settings.get("context_tokens")
        return gr.update(
            value=_context_window(spec, settings.get("context_tokens")),
            minimum=1024,
            maximum=_GLOBAL_MAX_CONTEXT,
            info=_context_info(spec),
        )

    ctx.states["preset_value_adapters"]["context_tokens"] = preset_context_tokens

    def preset_prompt_preset(settings: dict[str, Any]) -> Any:
        variant_key = str(settings.get("model_key") or _INITIAL_VARIANT)
        preset_id = str(settings.get("prompt_preset_id") or "")
        try:
            family = variant_to_family(variant_key)
            preset = get_preset(preset_id)
        except (KeyError, TypeError, ValueError):
            family = variant_to_family(_INITIAL_VARIANT)
            modality = _PRIMARY_PROMPT_MODALITIES[family]
        else:
            modality = next(
                (
                    candidate
                    for candidate in preset.modalities
                    if preset_id in {item.id for item in list_presets(family, candidate)}
                ),
                _PRIMARY_PROMPT_MODALITIES[family],
            )
        choices = _prompt_choices(family, modality)
        if preset_id and preset_id not in {value for _, value in choices}:
            try:
                preset = get_preset(preset_id)
            except KeyError:
                pass
            else:
                choices.append((f"{preset.group} · {_display(preset.label)}", preset.id))
        return gr.update(choices=choices, value=preset_id or None)

    ctx.states["preset_value_adapters"]["prompt_preset_id"] = preset_prompt_preset

    def preset_model_key(settings: dict[str, Any]) -> Any:
        """Ship the preset's variant together with choices that contain it.

        A preset may select a variant the current VRAM-tier filter hides (for
        example a GGUF Q8 build on a 32 GB card). Updating only the value would
        make Gradio reject it on the next event, so the choices are rebuilt with
        the preset variant included (tagged when it exceeds the tier).
        """

        value = str(settings.get("model_key") or _INITIAL_VARIANT)
        known = {key for _, key in _variant_choices()}
        if value not in known:
            value = _INITIAL_VARIANT
        try:
            physical_tier = auto_tier(gpu_total_for(_default_gpu_index()))
        except Exception:
            physical_tier = 32
        show_all = bool(settings.get("show_all_variants", False))
        return gr.update(choices=variant_choices_for_tier(value, physical_tier, show_all), value=value)

    ctx.states["preset_value_adapters"]["model_key"] = preset_model_key

    def preset_max_new_tokens(settings: dict[str, Any]) -> Any:
        try:
            spec = MODEL_SPECS[variant_to_family(str(settings.get("model_key") or _INITIAL_VARIANT))]
        except KeyError:
            return settings.get("max_new_tokens")
        cap = int(spec.limits.max_new_tokens_cap)
        schema_values = {item.name: item for item in spec.param_schema}
        try:
            value = int(settings.get("max_new_tokens"))
        except (TypeError, ValueError):
            value = int(schema_values["max_new_tokens"].default)
        return gr.update(
            value=max(1, min(value, cap)),
            minimum=1,
            maximum=_GLOBAL_MAX_NEW_TOKENS,
            step=1,
            info=(
                f"{schema_values['max_new_tokens'].description} Selected-family maximum: {cap:,}."
            ),
        )

    ctx.states["preset_value_adapters"]["max_new_tokens"] = preset_max_new_tokens

    def model_defaults(variant_key: str, current_max_tokens: Any = None) -> tuple[Any, ...]:
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        values = {item.name: item for item in spec.param_schema}
        thinking = "enable_thinking" in values
        token_cap = int(spec.limits.max_new_tokens_cap)
        try:
            kept_tokens = int(current_max_tokens)
        except (TypeError, ValueError):
            kept_tokens = int(values["max_new_tokens"].default)
        kept_tokens = max(1, min(kept_tokens, token_cap))
        return (
            gr.update(value=values["temperature"].default, minimum=0, maximum=2, step=0.01),
            gr.update(value=values["top_p"].default, minimum=0, maximum=1, step=0.01),
            gr.update(value=values["top_k"].default, minimum=0, maximum=200, step=1),
            gr.update(value=values["repetition_penalty"].default, minimum=0.5, maximum=2, step=0.01),
            gr.update(
                value=kept_tokens,
                minimum=1,
                maximum=_GLOBAL_MAX_NEW_TOKENS,
                step=1,
                info=f"{values['max_new_tokens'].description} Selected-family maximum: {token_cap:,}.",
            ),
            gr.update(value=values["do_sample"].default),
            gr.update(value=bool(values.get("enable_thinking") and values["enable_thinking"].default), interactive=thinking),
            gr.update(value=values["fps"].default, minimum=_GLOBAL_FPS_MIN, maximum=_GLOBAL_FPS_MAX, step=values["fps"].step),
            gr.update(
                value=min(int(values["max_frames"].default), _family_max_frames(spec)),
                minimum=0,
                maximum=_GLOBAL_MAX_FRAMES,
                step=values["max_frames"].step,
                info=_frames_info(spec),
            ),
            gr.update(value=values["max_pixels"].default, minimum=_GLOBAL_MIN_PIXELS, maximum=_GLOBAL_MAX_PIXELS, step=values["max_pixels"].step),
            gr.update(value=spec.limits.min_pixels, minimum=_GLOBAL_MIN_PIXELS),
            gr.update(value=values["use_audio_in_video"].default, interactive="video_audio" in spec.capabilities),
            gr.update(
                value=int(spec.limits.context_tokens),
                minimum=1024,
                maximum=_GLOBAL_MAX_CONTEXT,
                info=_context_info(spec),
            ),
        )

    def model_constraints(variant_key: str, current_max_tokens: Any = None) -> tuple[Any, ...]:
        if str(variant_key) not in {key for _, key in _variant_choices()}:
            return tuple(gr.skip() for _ in range(13))
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        values = {item.name: item for item in spec.param_schema}
        thinking = values.get("enable_thinking")
        token_cap = int(spec.limits.max_new_tokens_cap)
        try:
            kept_tokens = int(current_max_tokens)
        except (TypeError, ValueError):
            kept_tokens = int(values["max_new_tokens"].default)
        kept_tokens = max(1, min(kept_tokens, token_cap))
        return (
            gr.update(minimum=0, maximum=2, step=0.01),
            gr.update(minimum=0, maximum=1, step=0.01),
            gr.update(minimum=0, maximum=200, step=1),
            gr.update(minimum=0.5, maximum=2, step=0.01),
            gr.update(
                value=kept_tokens,
                minimum=1,
                maximum=_GLOBAL_MAX_NEW_TOKENS,
                step=1,
                info=f"{values['max_new_tokens'].description} Selected-family maximum: {token_cap:,}.",
            ),
            gr.skip(),
            gr.update(
                value=bool(thinking.default) if thinking is not None else False,
                interactive=thinking is not None,
            ),
            gr.update(minimum=_GLOBAL_FPS_MIN, maximum=_GLOBAL_FPS_MAX, step=values["fps"].step),
            gr.update(minimum=0, maximum=_GLOBAL_MAX_FRAMES, step=values["max_frames"].step, info=_frames_info(spec)),
            gr.update(minimum=_GLOBAL_MIN_PIXELS, maximum=_GLOBAL_MAX_PIXELS, step=values["max_pixels"].step),
            gr.update(minimum=_GLOBAL_MIN_PIXELS, maximum=_GLOBAL_MAX_PIXELS),
            gr.update(interactive="video_audio" in spec.capabilities),
            gr.update(minimum=1024, maximum=_GLOBAL_MAX_CONTEXT, info=_context_info(spec)),
        )

    model_key.change(
        model_constraints,
        inputs=[model_key, max_new_tokens],
        outputs=[temperature, top_p, top_k, repetition, max_new_tokens, do_sample, enable_thinking, fps, max_frames, max_pixels, min_pixels, use_audio, context_tokens],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    model_defaults_event = model_key.select(
        model_defaults,
        inputs=[model_key, max_new_tokens],
        outputs=[temperature, top_p, top_k, repetition, max_new_tokens, do_sample, enable_thinking, fps, max_frames, max_pixels, min_pixels, use_audio, context_tokens],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    model_defaults_event.then(
        apply_auto_vram,
        inputs=[model_key, vram_preset, gpu_picker, show_all_variants],
        outputs=vram_outputs,
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

    def render_selected(preset_id: str, *values: Any) -> tuple[Any, Any, Any, Any]:
        try:
            preset = get_preset(str(preset_id))
            system, user = render_prompt(preset, _prompt_variables(values))
            system, user = system or "", user or ""
            tracked = {"system": system, "user": user}
            return _display(preset.description), system, user, tracked
        except Exception as exc:
            return (
                f"<span class='vc-warn'>{html.escape(str(exc))}</span>",
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

    def select_prompt(preset_id: str, variant_key: str, *values: Any) -> tuple[Any, ...]:
        description, system, user, tracked = render_selected(preset_id, *values)
        try:
            preset = get_preset(str(preset_id))
            family_spec = MODEL_SPECS[variant_to_family(variant_key)]
            schema_values = {item.name: item.default for item in family_spec.param_schema}
            merged = {**schema_values, **preset.generation_overrides}
            token_update: Any = gr.skip()
            if "max_new_tokens" in merged:
                # Prompt presets carry family-agnostic token budgets; clamp them to
                # the selected family's cap in the same update as the slider bound
                # so the component never holds a value above its maximum.
                token_cap = int(family_spec.limits.max_new_tokens_cap)
                try:
                    requested = int(merged["max_new_tokens"])
                except (TypeError, ValueError):
                    requested = token_cap
                token_update = gr.update(value=max(1, min(requested, token_cap)), maximum=_GLOBAL_MAX_NEW_TOKENS)
            return (
                description,
                system,
                user,
                tracked,
                merged.get("temperature", gr.skip()),
                merged.get("top_p", gr.skip()),
                merged.get("top_k", gr.skip()),
                merged.get("repetition_penalty", gr.skip()),
                token_update,
                merged.get("do_sample", gr.skip()),
            )
        except Exception:
            return description, system, user, tracked, *[gr.skip() for _ in range(6)]

    ctx.states["caption_prompt_select_handler"] = select_prompt
    prompt_preset.select(
        select_prompt,
        inputs=[prompt_preset, model_key, *variable_components],
        outputs=[prompt_description, system_prompt, user_prompt, prompt_auto_state, temperature, top_p, top_k, repetition, max_new_tokens, do_sample],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    reset_prompts.click(
        render_selected,
        inputs=[prompt_preset, *variable_components],
        outputs=[prompt_description, system_prompt, user_prompt, prompt_auto_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    my_prompts.change(
        lambda selected: str(selected or ""),
        inputs=my_prompts,
        outputs=prompt_name,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def save_personal_prompt(
        name: str,
        system_text: str,
        user_text: str,
    ) -> tuple[Any, str, str]:
        try:
            names, selected, message = save_prompt_library_entry(
                prompt_library_dir, name, system_text, user_text
            )
            return gr.update(choices=names, value=selected), selected, message
        except ImportError:
            return (
                gr.skip(),
                str(name or ""),
                "<span class='vc-warn'>Prompt library becomes available after the backend update.</span>",
            )
        except Exception as exc:
            return gr.skip(), str(name or ""), f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    def load_personal_prompt(selected: str, typed_name: str) -> tuple[Any, ...]:
        name = str(selected or typed_name or "")
        try:
            system_text, user_text, manual_state, message = load_prompt_library_entry(
                prompt_library_dir, name
            )
            description = (
                "<span class='vc-warn'>Prompt edited manually — Reset prompts to preset re-renders it.</span>"
            )
            return system_text, user_text, manual_state, description, name, message
        except ImportError:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                name,
                "<span class='vc-warn'>Prompt library becomes available after the backend update.</span>",
            )
        except Exception as exc:
            return (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), name,
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
            )

    def delete_personal_prompt(selected: str, typed_name: str) -> tuple[Any, str, str]:
        name = str(selected or typed_name or "")
        try:
            names, message = delete_prompt_library_entry(prompt_library_dir, name)
            return gr.update(choices=names, value=None), "", message
        except ImportError:
            return (
                gr.skip(),
                name,
                "<span class='vc-warn'>Prompt library becomes available after the backend update.</span>",
            )
        except Exception as exc:
            return gr.skip(), name, f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    save_prompt.click(
        save_personal_prompt,
        inputs=[prompt_name, system_prompt, user_prompt],
        outputs=[my_prompts, prompt_name, prompt_library_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    load_prompt.click(
        load_personal_prompt,
        inputs=[my_prompts, prompt_name],
        outputs=[
            system_prompt,
            user_prompt,
            prompt_auto_state,
            prompt_description,
            prompt_name,
            prompt_library_status,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    delete_prompt.click(
        delete_personal_prompt,
        inputs=[my_prompts, prompt_name],
        outputs=[my_prompts, prompt_name, prompt_library_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    prompt_preset.change(
        validate_prompt_preset,
        inputs=[
            prompt_preset,
            valid_prompt_preset_state,
            model_key,
            media.modality_state,
            use_audio,
        ],
        outputs=[
            prompt_preset,
            valid_prompt_preset_state,
            prompt_description,
            prompt_context_state,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    caption_length.change(
        validate_caption_length,
        inputs=[caption_length, valid_caption_length_state],
        outputs=[caption_length, valid_caption_length_state, prompt_description],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    def render_variable_change(preset_id: str, *values: Any) -> tuple[str, Any, Any, dict[str, str]]:
        variable_values = values[: len(variable_components)]
        current_system, current_user, tracked = values[len(variable_components) :]
        try:
            return render_prompt_preserving_edits(
                str(preset_id),
                _prompt_variables(variable_values),
                str(current_system or ""),
                str(current_user or ""),
                tracked if isinstance(tracked, Mapping) else None,
            )
        except Exception as exc:
            return f"<span class='vc-err'>{html.escape(str(exc))}</span>", gr.skip(), gr.skip(), dict(tracked or {})

    for component in variable_components:
        component.input(
            render_variable_change,
            inputs=[prompt_preset, *variable_components, system_prompt, user_prompt, prompt_auto_state],
            outputs=[prompt_description, system_prompt, user_prompt, prompt_auto_state],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def sync_prompt_context(
        variant_key: str,
        modality: str,
        include_audio: bool,
        selected_inputs: list[str] | None,
        current_preset_id: str,
        previous_context: list[str] | tuple[str, str] | None,
        *values: Any,
    ) -> tuple[Any, str, Any, Any, list[str], Any]:
        has_inputs = bool(selected_inputs)
        resolved_modality = (
            _modality_for_inputs(selected_inputs or [], modality)
            if has_inputs
            else modality
        )
        effective_modality = _effective_prompt_modality(
            variant_key,
            resolved_modality,
            include_audio,
            has_inputs=has_inputs,
        )
        family, choices, preset = _resolve_prompt_preset(
            variant_key,
            effective_modality,
            current_preset_id,
            family_union=(
                not has_inputs
                or not str(resolved_modality or "").strip()
                or str(resolved_modality).casefold() == "unknown"
            ),
        )
        context_modality = "family" if not has_inputs else effective_modality
        context = [family, context_modality]
        if preset is None:
            ctx.app_log.warn(
                f"No task preset supports {family} for {effective_modality} input.",
                scope="prompts",
            )
            return (
                gr.update(choices=[], value=None),
                "<span class='vc-warn'>No compatible task preset for this model and modality.</span>",
                "",
                "",
                context,
                {"system": "", "user": ""},
            )
        description = _display(preset.description)
        current: Any | None = None
        try:
            current = get_preset(str(current_preset_id or ""))
        except KeyError:
            pass
        if current is not None and not _preset_supports_family(current, family):
            family_label = _display(MODEL_SPECS[family].label)
            decision = (
                f"Preset {_display(current.label)} does not support {family_label}; "
                f"using {_display(preset.label)}."
            )
            description = f"{description}<br><span class='vc-warn'>{decision}</span>"
            ctx.app_log.warn(decision, scope="prompts")
        elif (
            has_inputs
            and current is not None
            and current.id == preset.id
            and current.id not in {
                candidate.id for candidate in list_presets(family, effective_modality)
            }
        ):
            targets = ", ".join(current.modalities)
            hint = (
                f"{_display(current.label)} targets {targets}; it is kept for this "
                f"{effective_modality} input and the runner will substitute per item when needed."
            )
            description = f"{description}<br><span class='vc-help'>{html.escape(hint)}</span>"
            ctx.app_log.log(hint, scope="prompts")
        same_context = list(previous_context or []) == context
        if same_context and preset.id == str(current_preset_id or ""):
            system: Any = gr.skip()
            user: Any = gr.skip()
            tracked: Any = gr.skip()
        else:
            rendered_system, rendered_user = render_prompt(preset, _prompt_variables(values))
            system = rendered_system or ""
            user = rendered_user or ""
            tracked = {"system": system, "user": user}
        return (
            gr.update(choices=choices, value=preset.id),
            description,
            system,
            user,
            context,
            tracked,
        )

    prompt_sync_inputs = [
        model_key,
        media.modality_state,
        use_audio,
        media.resolved_state,
        prompt_preset,
        prompt_context_state,
        *variable_components,
    ]
    prompt_sync_outputs = [
        prompt_preset,
        prompt_description,
        system_prompt,
        user_prompt,
        prompt_context_state,
        prompt_auto_state,
    ]
    ctx.states["caption_prompt_context_handler"] = sync_prompt_context
    model_key.select(
        sync_prompt_context,
        inputs=prompt_sync_inputs,
        outputs=prompt_sync_outputs,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    media.modality_state.change(
        sync_prompt_context,
        inputs=prompt_sync_inputs,
        outputs=prompt_sync_outputs,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    use_audio.change(
        sync_prompt_context,
        inputs=prompt_sync_inputs,
        outputs=prompt_sync_outputs,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )

    def budget_line(
        variant_key: str,
        fps_value: float,
        frames_value: int,
        pixels_value: int,
        duration: float,
        output_tokens: int,
        include_audio: bool,
        context_value: Any = None,
    ) -> str:
        if float(duration or 0) <= 0:
            return "<span class='vc-help'>Upload media to estimate the live token budget.</span>"
        try:
            family = variant_to_family(variant_key)
            spec = MODEL_SPECS[family]
            frame_count = min(
                max(1, int(frames_value or 1)),
                _family_max_frames(spec),
                max(1, int(math.ceil(float(duration) * float(fps_value)))),
            )
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
            window = _context_window(spec, context_value)
            ok = fits_context(estimate, window, int(output_tokens or 0))
            css, word = ("vc-ok", "OK") if ok else ("vc-err", "OVER BUDGET")
            return (
                f"≈ {total:,} input tokens of {window:,} — "
                f"<span class='{css}'>{word}</span> · {frame_count} frames · reserves {int(output_tokens or 0):,} output tokens"
                f" · KV cache ≈ {spec.limits.kv_cache_gb(window):.1f} GB at this window"
            )
        except Exception as exc:
            return f"<span class='vc-warn'>Token estimate unavailable: {html.escape(str(exc))}</span>"

    budget_inputs = [model_key, fps, max_frames, max_pixels, media.duration_state, max_new_tokens, use_audio, context_tokens]
    for event_component in budget_inputs:
        event_component.change(
            budget_line,
            inputs=budget_inputs,
            outputs=token_budget,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    block_swap_inputs = [
        model_key,
        block_swap_auto,
        blocks_to_swap,
        gpu_picker,
        vram_reserve_gb,
        plan_slack_mib,
        swap_slots,
        offload_experts,
        pin_cpu,
        fps,
        max_frames,
        max_pixels,
        max_new_tokens,
        context_tokens,
        media.duration_state,
        media.modality_state,
    ]

    def refresh_block_swap(
        variant_key: str,
        auto: bool,
        blocks: Any,
        selected_gpu: int,
        reserve: Any,
        plan_slack: Any,
        slots: Any,
        experts: bool,
        pin: bool,
        fps_value: Any,
        frames_value: Any,
        pixels_value: Any,
        output_tokens: Any,
        context_value: Any,
        duration: Any,
        modality: Any,
    ) -> tuple[Any, str]:
        """Show what the block-swap controls resolve to and keep the slider in step."""

        try:
            ping = getattr(ctx.pipeline_client, "ping", None)
            pong = ping(timeout_s=0.3) if callable(ping) else None
            slider, note = block_swap_preview(
                str(variant_key),
                bool(auto),
                blocks,
                gpu_index=int(selected_gpu or 0),
                reserve_gb=reserve,
                plan_slack_mib=plan_slack,
                swap_slots=slots,
                offload_experts=bool(experts),
                pin_cpu=bool(pin),
                fps_value=fps_value,
                frames_value=frames_value,
                pixels_value=pixels_value,
                output_tokens=output_tokens,
                context_value=context_value,
                duration=duration,
                modality=modality,
                pong=pong,
            )
            return gr.update(**slider), note
        except Exception as exc:
            return gr.skip(), f"<span class='vc-warn'>Block swap preview unavailable: {html.escape(str(exc))}</span>"

    block_swap_outputs = [blocks_to_swap, block_swap_note]
    # The slider is an output here, so it listens on ``input`` (user edits only)
    # while every other dependency listens on ``change`` so preset loads refresh
    # the preview too; the programmatic slider update never re-triggers itself.
    for event_component in block_swap_inputs:
        if event_component is blocks_to_swap:
            continue
        event_component.change(
            refresh_block_swap,
            inputs=block_swap_inputs,
            outputs=block_swap_outputs,
            queue=False,
            trigger_mode="always_last",
            show_progress="hidden",
            api_visibility="private",
        )
    blocks_to_swap.input(
        refresh_block_swap,
        inputs=block_swap_inputs,
        outputs=block_swap_outputs,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    context_carry_over.change(
        lambda enabled: (
            gr.update(interactive=bool(enabled)),
            gr.update(interactive=bool(enabled)),
        ),
        inputs=context_carry_over,
        outputs=[context_carry_words, context_carry_prompt],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    fade.change(
        lambda enabled: gr.update(interactive=bool(enabled)),
        inputs=fade,
        outputs=fade_threshold,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    preview_sampled_frames.click(
        sampled_frame_preview,
        inputs=[
            media.resolved_state,
            model_key,
            fps,
            max_frames,
            sampling,
            max_pixels,
            min_pixels,
            trim_start,
            trim_end,
            adaptive_threshold,
        ],
        outputs=[sampled_frame_gallery, sampled_frame_status],
        concurrency_id="vc-frame-preview",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    model_release_lock = threading.Lock()
    pending_model_releases = 0

    def model_release_pending() -> bool:
        with model_release_lock:
            return pending_model_releases > 0

    ctx.states["model_release_pending"] = model_release_pending

    def release_previous_model(variant_key: str) -> None:
        """Record a selection immediately and release its predecessor off-request."""

        nonlocal pending_model_releases
        selected = str(variant_key)
        if selected not in {key for _, key in _variant_choices()}:
            return
        select_variant = getattr(ctx.pipeline_client, "select_variant", None)
        if not callable(select_variant):
            return

        record_selection = getattr(ctx.pipeline_client, "record_variant_selection", None)
        release_recorded = getattr(ctx.pipeline_client, "release_recorded_variant", None)
        if callable(record_selection) and callable(release_recorded):
            record_selection(selected)
            release_selected = release_recorded
        else:
            release_selected = select_variant

        with model_release_lock:
            pending_model_releases += 1

        def release() -> None:
            nonlocal pending_model_releases
            try:
                outcome = release_selected(selected)
                if not isinstance(outcome, Mapping):
                    return
                if outcome.get("busy"):
                    ctx.app_log.log(
                        f"Model selection changed to {selected}; the resident model will be "
                        "released when the current job finishes.",
                        scope="models",
                    )
                    return
                released = outcome.get("released")
                if released is None:
                    return
                report = outcome.get("report")
                try:
                    freed = (
                        float(report.get("freed_vram_gb", 0.0) or 0.0)
                        if isinstance(report, Mapping)
                        else 0.0
                    )
                except (TypeError, ValueError):
                    freed = 0.0
                ctx.app_log.log(
                    f"Unloaded {released} after the model selection changed to {selected} "
                    f"(freed {freed:.2f} GiB of VRAM).",
                    scope="models",
                )
            except Exception as exc:
                ctx.app_log.warn(
                    f"Could not release the previous model after selecting {selected}: {exc}",
                    scope="models",
                )
            finally:
                with model_release_lock:
                    pending_model_releases = max(0, pending_model_releases - 1)

        threading.Thread(
            target=release,
            daemon=True,
            name="vcap-model-selection-release",
        ).start()

    ctx.states["caption_model_change_handler"] = release_previous_model

    model_key.change(
        release_previous_model,
        inputs=model_key,
        outputs=[],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    ).then(
        refresh_block_swap,
        inputs=block_swap_inputs,
        outputs=block_swap_outputs,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )

    def limit_line(
        variant_key: str,
        fps_value: float,
        pixels_value: int,
        reserve: int,
        include_audio: bool,
        context_value: Any = None,
    ) -> str:
        if str(variant_key) not in {key for _, key in _variant_choices()}:
            return gr.skip()
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        limit = spec.limits.compute_max_duration(
            fps=float(fps_value or spec.limits.default_fps),
            max_pixels=int(pixels_value or spec.limits.default_max_pixels),
            context=_context_window(spec, context_value),
            reserve_tokens=int(reserve or 0),
            include_audio=bool(include_audio),
        )
        return f"Model-limit auto-split ceiling: **{limit:.1f} s** at the current FPS, pixel, audio, and output-token budget."

    limit_inputs = [model_key, fps, max_pixels, max_new_tokens, use_audio, context_tokens]
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
        except CancelledError:
            yield [], "<span class='vc-warn'>Scene preview cancelled.</span>"
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
        return _probe_compile_in_child(force=True)

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
            ready, detail = result.get("value", (False, "Model action returned no result"))
            if not ready:
                css = "vc-warn" if "cancel" in str(detail).casefold() else "vc-err"
                yield f"<span class='{css}'>{html.escape(str(detail))}</span>"
            else:
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
    unload_model.click(
        lambda selected_gpu: unload_model_report(ctx.pipeline_client, int(selected_gpu or 0)),
        inputs=gpu_picker,
        outputs=ready_status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def refresh_run_history() -> tuple[Any, ...]:
        try:
            from vcap.core.outputs import list_recent_runs
        except ImportError:
            return (
                [],
                [],
                {},
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "<span class='vc-warn'>Run history becomes available after the backend update.</span>",
            )
        try:
            summaries = list_recent_runs(ctx.outputs_dir, 40)
            records = run_history_records(summaries)
            message = (
                f"<span class='vc-ok'>Found {len(records)} recent run(s).</span>"
                if records
                else "<span class='vc-help'>No finished runs were found.</span>"
            )
            return (
                run_history_rows(summaries),
                records,
                {},
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                message,
            )
        except Exception as exc:
            return (
                gr.skip(),
                gr.skip(),
                {},
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                f"<span class='vc-err'>Could not refresh run history: {html.escape(str(exc))}</span>",
            )

    history_refresh_outputs = [
        run_history,
        run_history_records_state,
        run_history_selected_state,
        run_history_open,
        run_history_editor,
        run_history_recover,
        run_history_status,
    ]
    run_history_refresh.click(
        refresh_run_history,
        outputs=history_refresh_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def select_run_history(
        records: list[Mapping[str, Any]] | None,
        evt: gr.SelectData,
    ) -> tuple[Any, ...]:
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        try:
            selected = dict((records or [])[int(index)])
        except (IndexError, TypeError, ValueError):
            return (
                {},
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                "<span class='vc-warn'>Select a run row first.</span>",
            )
        destination = str(selected.get("run_dir") or "")
        metadata = str(selected.get("metadata_path") or "")
        return (
            selected,
            gr.update(interactive=bool(destination)),
            gr.update(interactive=bool(destination)),
            gr.update(interactive=bool(metadata)),
            f"<span class='vc-ok'>Selected {html.escape(selected.get('name') or Path(destination).name)}.</span>",
        )

    run_history.select(
        select_run_history,
        inputs=run_history_records_state,
        outputs=[
            run_history_selected_state,
            run_history_open,
            run_history_editor,
            run_history_recover,
            run_history_status,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def open_history_folder(selected: Mapping[str, Any] | None) -> str:
        destination = str((selected or {}).get("run_dir") or "")
        if not destination:
            return "<span class='vc-warn'>Select a run row first.</span>"
        ok, message = open_in_file_manager(destination)
        return f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"

    run_history_open.click(
        open_history_folder,
        inputs=run_history_selected_state,
        outputs=run_history_status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    ctx.states["run_history_binding"] = {
        "refresh_fn": refresh_run_history,
        "refresh_outputs": history_refresh_outputs,
        "selected": run_history_selected_state,
        "open_editor": run_history_editor,
        "recover": run_history_recover,
        "status": run_history_status,
    }

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


def _merge_live_outputs(state: dict[str, Any], data: Mapping[str, Any] | None, status: str | None) -> bool:
    """Fold run folder and finished-item artifacts from a progress/item event into the live output state."""

    payload = dict(data or {})
    changed = False
    run_dir = payload.get("run_dir")
    if run_dir and state.get("run_dir") != str(run_dir):
        state["run_dir"] = str(run_dir)
        changed = True
    if str(status or "").casefold() == "done":
        outputs = payload.get("outputs") or {}
        caption_path = (
            outputs.get("merged_caption") or outputs.get("txt") or outputs.get("video_caption")
            if isinstance(outputs, Mapping)
            else None
        )
        if caption_path and state.get("caption_path") != str(caption_path):
            state["caption_path"] = str(caption_path)
            changed = True
        clip_path = payload.get("clip_path")
        if clip_path and state.get("clip_path") != str(clip_path):
            state["clip_path"] = str(clip_path)
            changed = True
    return changed


def _result_payload(result: JobResult) -> tuple[str, Any, str, str, list[Any], dict[str, Any]]:
    completed = [item for item in result.items if item.status == "done"]
    displayable = completed or [item for item in result.items if item.outputs]
    if not displayable:
        return "", None, "", "", [], {"run_dir": result.run_dir, "files": []}
    item = displayable[-1]
    caption_path = (
        item.outputs.get("merged_caption")
        or item.outputs.get("txt")
        or item.outputs.get("video_caption")
    )
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
    output_files: list[str] = []
    for result_item in result.items:
        candidates = list(result_item.outputs.values())
        for record in result_item.segments:
            record_outputs = record.get("outputs") if isinstance(record, Mapping) else None
            if isinstance(record_outputs, Mapping):
                candidates.extend(record_outputs.values())
        for candidate in candidates:
            path = str(candidate or "")
            if path and Path(path).is_file() and path not in output_files:
                output_files.append(path)
    state = {
        "run_dir": result.run_dir,
        "metadata_path": result.metadata_path,
        "caption_path": caption_path,
        "clip_path": clip_paths[-1] if clip_paths else None,
        "files": output_files,
    }
    return caption, structured, srt, reasoning, gallery, state


def _result_summary(result: JobResult) -> tuple[str, str, str, str]:
    """Return terminal label, message, status class, and ETA text."""

    part_detail = ""
    # Naming the last item's files only helps single runs; batches list them in
    # the Items table and the run folder instead.
    for item in (reversed(result.items) if len(result.items) == 1 else ()):
        parts: list[str] = []
        for label, raw in (
            ("video", item.video_caption_path),
            ("audio", item.audio_caption_path),
            ("merged", item.merged_caption_path),
        ):
            if not raw:
                continue
            path = Path(raw)
            display = f"{path.parent.name}/{path.name}" if label != "merged" else path.name
            parts.append(f"{label}: {display}")
        if parts:
            part_detail = "; files: " + ", ".join(parts)
            break
    counts = result.counts
    done = int(counts.get("done", 0))
    skipped = int(counts.get("skipped", 0))
    failed = int(counts.get("failed", 0))
    unsupported = int(counts.get("unsupported", 0))
    cancelled = int(counts.get("cancelled", 0))
    audio_captions = int(counts.get("audio_captions", 0))
    no_speech = int(counts.get("no_speech", 0))
    audio_detail = (
        f", audio captions: {audio_captions}"
        if "audio_captions" in counts
        else ""
    )
    if no_speech:
        audio_detail += f", no speech: {no_speech}"
    if cancelled:
        message = (
            f"Cancelled: {cancelled} cancelled, {done} done, {skipped} skipped, "
            f"{failed} failed"
        )
        if unsupported:
            message += f", {unsupported} unsupported"
        message += audio_detail + part_detail
        return (
            "Cancelled",
            f"{message} in {result.elapsed:.1f}s",
            "vc-warn",
            "cancelled",
        )
    message = f"Complete: {done} done, {skipped} skipped, {failed} failed"
    if unsupported:
        message += f", {unsupported} unsupported"
    message += audio_detail + part_detail
    return (
        "Complete",
        f"{message} in {result.elapsed:.1f}s",
        "vc-ok",
        "done",
    )


def _job_done_message(result: JobResult) -> str:
    """Build the concise browser notification for a terminal caption job."""

    counts = result.counts
    done = int(counts.get("done", 0))
    failed = int(counts.get("failed", 0)) + int(counts.get("unsupported", 0))
    cancelled = int(counts.get("cancelled", 0))
    if cancelled:
        return f"Job cancelled: {done} done, {failed} failed"
    if not done and failed:
        return f"Job failed: {done} done, {failed} failed"
    return f"Caption job finished: {done} done, {failed} failed"


def _job_done_payload(message: str, settings: dict[str, Any]) -> str:
    """Serialize one completion hook payload without firing it on page load."""

    return json.dumps(
        {
            "message": message,
            "desktop": bool(settings.get("desktop_notification_on_finish", False)),
            "sound": bool(settings.get("play_sound_on_finish", False)),
        },
        ensure_ascii=False,
    )


def _should_auto_open_output(
    result: JobResult,
    items: list[InputItem],
    output_kind: str,
    enabled: bool,
) -> bool:
    """Return whether a successful single-file run should reveal its folder."""

    counts = result.counts
    return (
        bool(enabled)
        and output_kind == "single"
        and len(items) == 1
        and bool(items[0].path)
        and int(counts.get("done", 0)) >= 1
        and int(counts.get("failed", 0)) == 0
        and int(counts.get("cancelled", 0)) == 0
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
        handles.clips_empty_hint,
        handles.last_outputs_state,
        handles.job_done_hook,
        handles.cancel,
        handles.cancel_confirmation,
        handles.unload_model,
        handles.open_editor,
        handles.copy_caption,
        handles.retry_failed,
        handles.results_zip,
        handles.results_zip_file,
    ]

    def run_caption(*args: Any):
        value_count = len(registry_components)
        settings = registry.values_to_dict(args[:value_count])
        cached_resolved = [str(value) for value in (args[value_count] or [])]
        input_mode = str(args[value_count + 1] or "upload")
        input_modality = str(args[value_count + 2] or "unknown")
        retry_state = (
            dict(args[value_count + 3] or {})
            if len(args) > value_count + 3 and isinstance(args[value_count + 3], Mapping)
            else None
        )
        if retry_state is None:
            resolved = resolve_caption_inputs_at_start(
                settings,
                input_mode,
                cached_resolved,
            )
            input_modality = _modality_for_inputs(resolved, input_modality)
        else:
            retry_paths, retry_kind, prior_batch_folder = retry_failed_inputs(retry_state)
            resolved = retry_paths
            input_mode = "folder" if retry_kind == "batch" else "retry"
            input_modality = str(retry_state.get("input_modality") or input_modality)
            settings["overwrite_existing"] = True
            if retry_kind == "batch" and prior_batch_folder:
                settings["batch_output_folder"] = prior_batch_folder
            if retry_kind == "batch" and retry_state.get("source_root"):
                settings["batch_input_folder"] = str(retry_state["source_root"])
            if retry_kind == "batch":
                settings["batch_save_next_to_source"] = bool(
                    retry_state.get("batch_save_next_to_source", False)
                )
        token = CancelToken()
        ctx.activate_cancel(token)
        ctx.states["caption_job_token"] = token
        items: list[InputItem] = [InputItem(path=value) for value in resolved]
        if not items:
            family = variant_to_family(str(settings.get("model_key", _INITIAL_VARIANT)))
            if (
                retry_state is not None
                or "text" not in MODEL_SPECS[family].capabilities
                or (
                    str(settings.get("audio_caption_source", "none")) != "none"
                    and str(settings.get("video_caption_source", "generate")) == "existing"
                )
            ):
                message = (
                    "No failed items are available to retry."
                    if retry_state is not None
                    else "Select at least one input. Text-only queries require a Qwen3-Omni Instruct or Thinking model."
                )
                yield (
                    render_progress_html(0, "Input required", message),
                    f"<span class='vc-err'>{message}</span>",
                    "**ETA:** —",
                    "**Speed:** — · **Context:** —",
                    [],
                    *[gr.skip() for _ in range(8)],
                    "",
                    gr.update(value="⏹ Cancel", interactive=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value=None, visible=False),
                )
                ctx.clear_active_cancel(token)
                if ctx.states.get("caption_job_token") is token:
                    ctx.states["caption_job_token"] = None
                return
            items = [InputItem(path="", kind="text", text_prompt_only=True, text=str(settings.get("user_prompt") or ""))]

        prompt_modality = (
            "text"
            if items[0].text_prompt_only
            else _effective_prompt_modality(
                str(settings.get("model_key", _INITIAL_VARIANT)),
                input_modality,
                bool(settings.get("use_audio_in_video", True)),
            )
        )
        _, _, resolved_prompt = _resolve_prompt_preset(
            str(settings.get("model_key", _INITIAL_VARIANT)),
            prompt_modality,
            str(settings.get("prompt_preset_id") or ""),
        )
        if (
            resolved_prompt is not None
            and resolved_prompt.id != str(settings.get("prompt_preset_id") or "")
        ):
            settings["prompt_preset_id"] = resolved_prompt.id
            prompt_values = (
                settings.get("trigger_word"),
                settings.get("language"),
                settings.get("source_language"),
                settings.get("target_language"),
                settings.get("caption_length"),
                settings.get("avoid_list"),
                settings.get("subject_class"),
                settings.get("extra_instructions"),
            )
            rendered_system, rendered_user = render_prompt(
                resolved_prompt,
                _prompt_variables(prompt_values),
            )
            settings["system_prompt"] = rendered_system or ""
            settings["user_prompt"] = rendered_user or ""

        segment_mode = str(settings.get("segment_mode") or "whole")
        if segment_mode == "scenes" and not bool(settings.get("scene_detect_enabled")):
            segment_mode = "whole"
        settings["segment_mode"] = segment_mode
        settings["compile_mode"] = settings.get("torch_compile_mode", DEFAULT_COMPILE_MODE)
        settings["recursive"] = bool(settings.get("batch_recursive", settings.get("scan_subfolders", False)))
        output_kind = "batch" if input_mode == "folder" else "single"
        output_kwargs: dict[str, Any] = {
            "kind": output_kind,
            "outputs_root": str(settings.get("outputs_dir") or ctx.outputs_dir),
            "batch_output_dir": (
                str(settings.get("batch_output_folder") or ctx.outputs_dir / "batch_captions")
                if output_kind == "batch"
                else None
            ),
            "mirror_names": True,
            "overwrite": bool(settings.get("overwrite_existing", False)),
            "save_processed_files": bool(settings.get("save_processed_files", False)),
            "save_clips": bool(settings.get("save_clips", False)),
            "recursive": bool(settings["recursive"]),
            "limit_items": max(0, int(settings.get("batch_limit_items", 0) or 0)),
            "include_kinds": tuple(
                ("video", "audio", "image", "text")
                if settings.get("batch_include_kinds") is None
                else settings.get("batch_include_kinds")
            ),
            "name_filter": str(settings.get("batch_name_filter") or ""),
            "save_next_to_source": bool(settings.get("batch_save_next_to_source", False)),
        }
        if output_kind == "batch" and str(settings.get("batch_input_folder") or "").strip():
            output_kwargs["source_root"] = str(normalize_path(settings["batch_input_folder"]))
        while True:
            try:
                output = OutputSpec(**output_kwargs)
                break
            except TypeError as exc:
                # Task A owns these backend fields; keep this UI usable while its
                # branch is landing by dropping only the reported unknown key.
                unknown = next(
                    (key for key in ("source_root", "include_kinds", "name_filter") if key in str(exc)),
                    None,
                )
                if unknown is None or unknown not in output_kwargs:
                    raise
                output_kwargs.pop(unknown, None)
        spec = JobSpec.from_settings(settings, items, output)
        set_mode = ctx.states.get("set_subprocess_mode")
        if callable(set_mode):
            set_mode(bool(settings.get("subprocess_mode", True)))
        else:
            ctx.pipeline_client.subprocess_mode = bool(settings.get("subprocess_mode", True))
        event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        sink = _UiSink(
            ctx,
            event_queue,
            mirror_logs=False,
        )
        terminal: dict[str, Any] = {}
        release_pending = ctx.states.get("model_release_pending")
        waiting_for_model_release = bool(
            callable(release_pending) and release_pending()
        )

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
            [index, Path(item.path).name if item.path else "Text prompt", "queued", "Waiting", "—", None]
            for index, item in enumerate(items, start=1)
        ]
        last_emit = 0.0
        last_fraction = 0.0
        last_message = "Starting caption job"
        last_eta = "—"
        last_speed = "—"
        last_context = "—"
        processed_count = 0
        total_count = len(items)
        remaining_count = total_count
        item_started_at: dict[int, float] = {}
        terminal_items: set[int] = set()
        live_outputs: dict[str, Any] = {}
        live_dirty = False
        if waiting_for_model_release:
            starting_label = "Waiting"
            starting_status = "Waiting for the previous model to unload..."
        elif retry_state is not None:
            starting_label = "Retrying"
            starting_status = f"Retrying {len(items)} item(s)"
        else:
            starting_label = "Preparing…"
            starting_status = "Preparing caption job…"
        yield (
            render_progress_html(
                0,
                starting_label,
                f"0/{len(items)} processed · {len(items)} remaining · ETA —",
            ),
            f"**Status:** {starting_status}",
            "**ETA:** —",
            "**Speed:** — · **Context:** —",
            item_rows,
            "",
            None,
            "",
            "",
            gr.update(visible=False),
            gr.update(value=[], visible=False),
            gr.update(
                value='No clips were saved for this run. Enable "Save produced clips" in '
                "Processing Pipeline → 5. Scene detection & splitting.",
                visible=True,
            ),
            {},
            "",
            gr.update(value="⏹ Cancel", interactive=True),
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value=None, visible=False),
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
                    live_dirty = _merge_live_outputs(live_outputs, data, None) or live_dirty
                    eta_seconds = data.get("eta_s", data.get("eta_seconds"))
                    last_eta = format_eta(eta_seconds) if eta_seconds is not None else "—"
                    speed = data.get("tok_per_s") or data.get("tokens_per_second")
                    if speed is not None:
                        last_speed = f"{float(speed):.2f} tok/s"
                    if data.get("prompt_tokens") is not None:
                        last_context = context_usage_text(data.get("prompt_tokens"), data.get("context_limit"))
                    raw_index = data.get("item_index", event.item_index)
                    index = int(raw_index or 0)
                    if 0 <= index < len(item_rows):
                        item_started_at.setdefault(index, now)
                        item_rows[index][2] = "running"
                        item_rows[index][3] = event.message
                        elapsed_value = data.get("item_elapsed_s")
                        if elapsed_value is None:
                            elapsed_value = now - item_started_at[index]
                        item_rows[index][4] = f"{max(0.0, float(elapsed_value)):.1f}s"
                        live_tokens = data.get("new_tokens", data.get("tokens"))
                        if live_tokens is not None:
                            item_rows[index][5] = int(live_tokens or 0)
                    processed_count = int(data.get("processed", processed_count) or 0)
                    total_count = int(data.get("total", event.total_items or total_count) or total_count)
                    remaining_count = int(
                        data.get("remaining", max(0, total_count - processed_count))
                        or 0
                    )
                elif kind == "item":
                    event = payload
                    data = event.data or {}
                    live_dirty = _merge_live_outputs(live_outputs, data, event.status) or live_dirty
                    raw_index = data.get("item_index", event.item_index)
                    index = int(raw_index or 0)
                    if 0 <= index < len(item_rows):
                        item_rows[index][2] = str(event.status or "done")
                        item_rows[index][3] = event.message
                        elapsed_value = data.get(
                            "item_elapsed_s",
                            data.get("elapsed_s", data.get("elapsed")),
                        )
                        if elapsed_value is None:
                            started = item_started_at.get(index, now)
                            elapsed_value = now - started
                        item_rows[index][4] = f"{max(0.0, float(elapsed_value)):.1f}s"
                        item_tokens = data.get("new_tokens", data.get("tokens"))
                        if item_tokens is not None:
                            item_rows[index][5] = int(item_tokens or 0)
                        if str(event.status or "").casefold() not in {"running", "queued"}:
                            terminal_items.add(index)
                    processed_count = int(data.get("processed", len(terminal_items)) or 0)
                    total_count = int(data.get("total", event.total_items or total_count) or total_count)
                    remaining_count = int(
                        data.get("remaining", max(0, total_count - processed_count))
                        or 0
                    )
                    eta_seconds = data.get("eta_s", data.get("eta_seconds"))
                    last_eta = format_eta(eta_seconds) if eta_seconds is not None else "\u2014"
                    last_message = event.message
                    last_fraction = float(event.fraction or last_fraction)
                if now - last_emit < 0.12 and kind == "progress":
                    continue
                last_emit = now
                progress_detail = (
                    f"{processed_count}/{total_count} processed · {remaining_count} remaining · ETA {last_eta}"
                )
                yield (
                    render_progress_html(last_fraction, last_message, progress_detail),
                    f"**Status:** {html.escape(last_message)}",
                    f"**ETA:** {last_eta}",
                    f"**Speed:** {last_speed} · **Context:** {last_context}",
                    item_rows,
                    *[gr.skip() for _ in range(7)],
                    dict(live_outputs) if live_dirty else gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )
                live_dirty = False
            thread.join()
            if "error" in terminal:
                raise terminal["error"]
            result: JobResult = terminal["result"]
            final_caption, structured, srt_text, reasoning_text, gallery, state = _result_payload(result)
            state = {**state, **{key: value for key, value in live_outputs.items() if not state.get(key)}}
            state["editor_dir"] = (
                str(output.source_root)
                if output_kind == "batch" and output.save_next_to_source and output.source_root
                else
                str(output.batch_output_dir)
                if output_kind == "batch" and output.batch_output_dir
                else str(result.run_dir)
            )
            state["editor_recursive"] = output_kind == "batch"
            state["failed_paths"] = failed_item_paths(result)
            state["output_kind"] = output_kind
            state["input_mode"] = input_mode
            state["input_modality"] = input_modality
            state["batch_output_folder"] = (
                str(output.batch_output_dir) if output.batch_output_dir else None
            )
            state["source_root"] = str(output.source_root) if getattr(output, "source_root", None) else None
            state["batch_save_next_to_source"] = bool(output.save_next_to_source)
            state["archive_source"] = (
                str(output.source_root)
                if output_kind == "batch" and output.save_next_to_source and output.source_root
                else
                str(output.batch_output_dir)
                if output_kind == "batch" and output.batch_output_dir
                else str(result.run_dir)
            )
            state["job_finished"] = True
            terminal_label, message, status_class, eta_text = _result_summary(result)
            for index, result_item in enumerate(result.items):
                if index >= len(item_rows):
                    break
                usage_tokens = 0
                for segment in result_item.segments:
                    usage = segment.get("usage") if isinstance(segment, Mapping) else None
                    if isinstance(usage, Mapping):
                        usage_tokens += int(usage.get("new_tokens", usage.get("completion_tokens", 0)) or 0)
                if isinstance(result_item.summary_usage, Mapping):
                    usage_tokens += int(result_item.summary_usage.get("new_tokens", 0) or 0)
                item_rows[index][5] = usage_tokens
            if _should_auto_open_output(
                result,
                items,
                output_kind,
                bool(settings.get("open_output_folder_on_single_finish", False)),
            ):
                opened, open_message = open_in_file_manager(result.run_dir)
                if not opened:
                    ctx.app_log.warn(open_message, scope="outputs")
            yield (
                render_progress_html(1, terminal_label, message),
                f"<span class='{status_class}'>**Status:** {html.escape(message)}</span>",
                f"**ETA:** {eta_text}",
                f"**Speed:** {last_speed} · **Context:** {last_context}",
                item_rows,
                final_caption,
                structured,
                srt_text,
                reasoning_text,
                gr.update(visible=bool(reasoning_text)),
                gr.update(value=gallery, visible=bool(gallery)),
                gr.update(visible=not bool(gallery)),
                state,
                _job_done_payload(_job_done_message(result), settings),
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=bool(state.get("run_dir"))),
                gr.update(interactive=bool(final_caption.strip())),
                gr.update(interactive=bool(state.get("failed_paths"))),
                gr.update(interactive=bool(state.get("archive_source"))),
                gr.update(value=None, visible=False),
            )
        except (CancelledError, KeyboardInterrupt) as exc:
            ctx.app_log.warn(str(exc), scope="cancel")
            yield (
                render_progress_html(last_fraction, "Cancelled", str(exc)),
                f"<span class='vc-warn'>**Status:** {html.escape(str(exc))}</span>",
                "**ETA:** cancelled",
                f"**Speed:** {last_speed} · **Context:** {last_context}",
                item_rows,
                *[gr.skip() for _ in range(7)],
                gr.skip(),
                _job_done_payload("Job cancelled", settings),
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=None, visible=False),
            )
        except BaseException as exc:
            ctx.app_log.exception(f"Caption job failed: {exc}", scope="ui")
            yield (
                render_progress_html(last_fraction, "Failed", str(exc)),
                f"<span class='vc-err'>**Status:** {html.escape(str(exc))}</span>",
                "**ETA:** failed",
                f"**Speed:** {last_speed} · **Context:** {last_context}",
                item_rows,
                *[gr.skip() for _ in range(8)],
                _job_done_payload("Job failed", settings),
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=None, visible=False),
            )
        finally:
            ctx.clear_active_cancel(token)
            if ctx.states.get("caption_job_token") is token:
                ctx.states["caption_job_token"] = None

    ctx.states["caption_run_handler"] = run_caption

    run_inputs = [
        *registry_components,
        handles.media.resolved_state,
        handles.media.mode_state,
        handles.media.modality_state,
    ]
    start_event = handles.start.click(
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
    hotkey_start_event = handles.hotkey_start.click(
        run_caption,
        inputs=run_inputs,
        outputs=output_components,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )
    retry_event = handles.retry_failed.click(
        run_caption,
        inputs=[*run_inputs, handles.last_outputs_state],
        outputs=output_components,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )
    notify_js = """
    (payload) => {
      if (!payload) return [];
      let data = payload;
      if (typeof payload === 'string') {
        try { data = JSON.parse(payload); } catch (_error) { return []; }
      }
      if (data && data.message && window.__vcapNotifyJobDone) {
        window.__vcapNotifyJobDone(data.message, Boolean(data.desktop), Boolean(data.sound));
      }
      return [];
    }
    """
    for event in (start_event, hotkey_start_event, retry_event):
        event.then(
            fn=None,
            inputs=handles.job_done_hook,
            outputs=[],
            js=notify_js,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        history_binding = ctx.states.get("run_history_binding")
        if isinstance(history_binding, Mapping):
            event.then(
                history_binding["refresh_fn"],
                outputs=history_binding["refresh_outputs"],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

        recover_recent_binding = ctx.states.get("recover_recent_binding")
        if isinstance(recover_recent_binding, Mapping):
            event.then(
                recover_recent_binding["refresh_fn"],
                outputs=recover_recent_binding["refresh_outputs"],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

    def request_cancel_job() -> tuple[Any, Any, str]:
        token = ctx.states.get("caption_job_token") or ctx.get_active_cancel()
        state = request_caption_cancel(token)
        if state != "confirm":
            return (
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                "No active caption job to cancel.",
            )
        return (
            gr.update(value="⏹ Cancel", interactive=True),
            gr.update(visible=True),
            "<span class='vc-warn'>Waiting for cancel confirmation.</span>",
        )

    def confirm_cancel_job() -> tuple[Any, Any, str]:
        caption_token = ctx.states.get("caption_job_token")
        token = caption_token or ctx.get_active_cancel()
        if not confirm_caption_cancel(token):
            return (
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                "No armed caption cancellation request.",
            )
        if token is caption_token:
            ctx.pipeline_client.cancel(force=False)
        return (
            gr.update(value="Cancelling…", interactive=False),
            gr.update(visible=False),
            "<span class='vc-warn'>Cooperative cancellation requested.</span>",
        )

    def keep_running_job() -> tuple[Any, Any, str]:
        token = ctx.states.get("caption_job_token") or ctx.get_active_cancel()
        kept = keep_caption_running(token)
        return (
            gr.update(value="⏹ Cancel", interactive=bool(kept)),
            gr.update(visible=False),
            (
                "<span class='vc-ok'>Caption job continues.</span>"
                if kept
                else "No active caption job."
            ),
        )

    def escape_cancel_job() -> tuple[Any, Any, str]:
        token = ctx.states.get("caption_job_token") or ctx.get_active_cancel()
        if token is not None and token.is_armed():
            return confirm_cancel_job()
        return request_cancel_job()

    handles.cancel.click(
        request_cancel_job,
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.hotkey_cancel.click(
        escape_cancel_job,
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    confirm_cancel_event = handles.cancel_yes.click(
        confirm_cancel_job,
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def resync_live_log() -> tuple[str, int]:
        lines, revision = ctx.app_log.tail_snapshot(300)
        return newest_first("\n".join(lines)), revision

    confirm_cancel_event.then(
        resync_live_log,
        outputs=[handles.logs.log, handles.logs.revision],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.cancel_keep.click(
        keep_running_job,
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    ctx.states["caption_cancel_handlers"] = {
        "request": request_cancel_job,
        "confirm": confirm_cancel_job,
        "keep": keep_running_job,
        "escape": escape_cancel_job,
    }

    def refresh_cancel_button(
        last_state: Mapping[str, Any] | None,
        current_caption: str,
    ) -> tuple[Any, ...]:
        token = ctx.states.get("caption_job_token") or ctx.get_active_cancel()
        any_job_running = (
            ctx.get_active_cancel() is not None
            or bool(getattr(ctx.pipeline_client, "_busy", False))
        )
        state = dict(last_state or {})
        if token is None or token.is_cancelled():
            return (
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                gr.update(interactive=not any_job_running),
                gr.update(interactive=bool(state.get("failed_paths")) and not any_job_running),
                gr.update(interactive=bool(state.get("job_finished")) and not any_job_running),
                gr.update(interactive=bool(str(current_caption or "").strip()) and not any_job_running),
                "",
            )
        if token.is_armed():
            return (
                gr.skip(),
                gr.skip(),
                gr.update(interactive=False),
                gr.skip(),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.skip(),
            )
        return (
            gr.update(value="⏹ Cancel", interactive=True),
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
        )

    ctx.states["caption_cancel_timer_handler"] = refresh_cancel_button
    handles.cancel_timer.tick(
        refresh_cancel_button,
        inputs=[handles.last_outputs_state, handles.caption],
        outputs=[
            handles.cancel,
            handles.cancel_confirmation,
            handles.unload_model,
            handles.retry_failed,
            handles.results_zip,
            handles.copy_caption,
            handles.cancel_note,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    handles.copy_caption.click(
        lambda text: (
            "<span class='vc-ok'>Copied.</span>"
            if str(text or "")
            else "<span class='vc-warn'>No caption is available to copy.</span>"
        ),
        inputs=handles.caption,
        outputs=handles.progress.status,
        js=(
            "(text) => { if (text) navigator.clipboard.writeText(String(text)); "
            "return [text]; }"
        ),
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def create_results_zip(state: Mapping[str, Any] | None):
        try:
            source, target = results_zip_paths(state, ctx.temp_dir)
        except Exception as exc:
            yield gr.update(value=None, visible=False), f"<span class='vc-err'>{html.escape(str(exc))}</span>"
            return
        size_bytes = _directory_size(source)
        size_gb = size_bytes / float(1024 ** 3)
        if size_bytes > 2 * 1024 ** 3:
            yield (
                gr.update(value=None, visible=False),
                f"<span class='vc-warn'>Source is {size_gb:.2f} GB; creating ZIP now.</span>",
            )
        else:
            yield (
                gr.update(value=None, visible=False),
                f"<span class='vc-help'>Creating results ZIP from {html.escape(str(source))}…</span>",
            )
        try:
            from vcap.core.archive import zip_directory
        except ImportError:
            yield (
                gr.update(value=None, visible=False),
                "<span class='vc-warn'>Results ZIP becomes available after the backend update.</span>",
            )
            return
        try:
            archive = zip_directory(source, target)
            archive_size = archive.stat().st_size
            yield (
                gr.update(value=str(archive), visible=True),
                f"<span class='vc-ok'>Results ZIP: {archive_size / (1024 ** 2):.2f} MB at "
                f"{html.escape(str(archive))}</span>",
            )
        except Exception as exc:
            yield gr.update(value=None, visible=False), f"<span class='vc-err'>Could not create ZIP: {html.escape(str(exc))}</span>"

    handles.results_zip.click(
        create_results_zip,
        inputs=handles.last_outputs_state,
        outputs=[handles.results_zip_file, handles.progress.status],
        concurrency_id="vc-results-zip",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    open_labels = {"run_dir": "run folder", "caption_path": "caption file", "clip_path": "clip"}

    def open_last(state: dict[str, Any], key: str, reveal: bool = False) -> str:
        current = dict(state or {})
        value = current.get(key)
        if not value and key == "clip_path" and current.get("run_dir"):
            note = (
                "No saved clip to reveal: clips are kept only when 'Save produced clips' is enabled. "
                "Opening the run folder instead."
            )
            gr.Warning(note)
            ok, message = open_in_file_manager(current["run_dir"])
            css = "vc-warn" if ok else "vc-err"
            return f"<span class='{css}'>{html.escape(note + ' ' + message)}</span>"
        if not value:
            label = open_labels.get(key, key.replace("_", " "))
            hint = "The running job has not produced it yet." if current else "Run a caption job first."
            note = f"No {label} is available yet. {hint}"
            gr.Warning(note)
            return f"<span class='vc-warn'>{html.escape(note)}</span>"
        ok, message = reveal_in_file_manager(value) if reveal else open_in_file_manager(value)
        (gr.Info if ok else gr.Warning)(message)
        css = "vc-ok" if ok else "vc-err"
        return f"<span class='{css}'>{html.escape(message)}</span>"

    handles.open_output.click(
        lambda state: open_last(state, "run_dir"),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        concurrency_id="vc-open-buttons",
        concurrency_limit=8,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.open_caption.click(
        lambda state: open_last(state, "caption_path"),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        concurrency_id="vc-open-buttons",
        concurrency_limit=8,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.reveal_clip.click(
        lambda state: open_last(state, "clip_path", True),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        concurrency_id="vc-open-buttons",
        concurrency_limit=8,
        show_progress="hidden",
        api_visibility="private",
    )

    editor_binding = ctx.states.get("editor_open_binding")
    main_tabs = ctx.states.get("main_tabs")
    if isinstance(editor_binding, Mapping) and main_tabs is not None:
        def prepare_editor_open(state: Mapping[str, Any] | None) -> tuple[Any, Any, str]:
            current = dict(state or {})
            destination = current.get("editor_dir") or current.get("run_dir")
            if not destination:
                return gr.skip(), gr.skip(), "<span class='vc-warn'>No finished run is available for the editor.</span>"
            return (
                str(destination),
                bool(current.get("editor_recursive", str(current.get("kind") or "") == "batch")),
                f"<span class='vc-ok'>Opening {html.escape(str(destination))} in Caption Editor.</span>",
            )

        open_editor_event = handles.open_editor.click(
            prepare_editor_open,
            inputs=handles.last_outputs_state,
            outputs=[editor_binding["folder"], editor_binding["recursive"], handles.progress.status],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        scan_editor_event = open_editor_event.then(
            editor_binding["scan_fn"],
            inputs=editor_binding["inputs"],
            outputs=editor_binding["outputs"],
            show_progress="minimal",
            api_visibility="private",
        )
        scan_editor_event.then(
            lambda: gr.Tabs(selected="editor"),
            outputs=main_tabs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

        history_editor_event = handles.run_history_editor.click(
            prepare_editor_open,
            inputs=handles.run_history_selected_state,
            outputs=[
                editor_binding["folder"],
                editor_binding["recursive"],
                handles.run_history_status,
            ],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        history_scan_event = history_editor_event.then(
            editor_binding["scan_fn"],
            inputs=editor_binding["inputs"],
            outputs=editor_binding["outputs"],
            show_progress="minimal",
            api_visibility="private",
        )
        history_scan_event.then(
            lambda: gr.Tabs(selected="editor"),
            outputs=main_tabs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    recover_binding = ctx.states.get("recover_load_binding")
    if isinstance(recover_binding, Mapping) and main_tabs is not None:
        def prepare_history_recovery(selected: Mapping[str, Any] | None) -> tuple[Any, str]:
            metadata_path = str((selected or {}).get("metadata_path") or "")
            if not metadata_path:
                return gr.skip(), "<span class='vc-warn'>The selected run has no metadata.json.</span>"
            return (
                metadata_path,
                f"<span class='vc-ok'>Loading settings from {html.escape(metadata_path)}.</span>",
            )

        recover_path_event = handles.run_history_recover.click(
            prepare_history_recovery,
            inputs=handles.run_history_selected_state,
            outputs=[recover_binding["path"], handles.run_history_status],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        recover_load_event = recover_path_event.then(
            recover_binding["load_fn"],
            inputs=recover_binding["inputs"],
            outputs=recover_binding["outputs"],
            show_progress="minimal",
            api_visibility="private",
        )
        recover_load_event.then(
            lambda: gr.Tabs(selected="recover"),
            outputs=main_tabs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )


__all__ = [
    "CaptionTabHandles",
    "DEFAULT_SUMMARY_PROMPT",
    "build",
    "confirm_caption_cancel",
    "delete_prompt_library_entry",
    "failed_item_paths",
    "gguf_control_updates",
    "keep_caption_running",
    "load_prompt_library_entry",
    "prompt_library_names",
    "render_prompt_preserving_edits",
    "request_caption_cancel",
    "resolve_caption_inputs_at_start",
    "results_zip_paths",
    "retry_failed_inputs",
    "run_history_records",
    "run_history_rows",
    "sampled_frame_preview",
    "save_prompt_library_entry",
    "unload_model_report",
    "validate_model_variant",
    "variant_choices_for_tier",
    "wire",
]
