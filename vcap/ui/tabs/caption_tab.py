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
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Iterable

import gradio as gr

from vcap import TEMP_DIR
from vcap.core.clip_fitness import TRAINER_TARGETS
from vcap.core.captions_post import to_srt
from vcap.core.media import probe_media
from vcap.core.paths import normalize_path, open_in_file_manager, reveal_in_file_manager
from vcap.core.preprocess import fits_context, token_budget_estimate
from vcap.ui.components import context_usage_text
from vcap.core.progress import ProgressEvent, format_eta
from vcap.core.scene_split import (
    SceneDetectParams,
    cap_scene_lengths,
    detect_scenes,
    merge_short_scenes,
)
from vcap.core.subprocess_runner import CancelToken, CancelledError, build_child_env
from vcap.core.gpu import resource_snapshot
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


def _frames_info(spec: Any) -> str:
    cap = int(next(item for item in spec.param_schema if item.name == "max_frames").max)
    return (
        f"Hard cap applied after sampling; {spec.label} uses at most {cap} frames and higher "
        "values are clamped at run time. Zero is valid for audio-only presets."
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


def _resolve_prompt_preset(
    variant_key: str,
    modality: str,
    current_preset_id: str | None,
) -> tuple[str, list[tuple[str, str]], Any | None]:
    """Resolve one compatible prompt selection for a model and modality."""

    family = variant_to_family(variant_key)
    presets = list_presets(family, modality)
    choices = _prompt_choices(family, modality)
    by_id = {preset.id: preset for preset in presets}
    selected = by_id.get(str(current_preset_id or ""))
    if selected is None:
        try:
            selected = by_id.get(default_preset_for(family, modality).id)
        except KeyError:
            selected = None
        if selected is None:
            selected = presets[0] if presets else None
    return family, choices, selected


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
    plan = OffloadPlan(
        gpu_layers=block_swap_to_gpu_layers(bool(auto), manual, layer_count),
        offload_experts=False,
        max_memory=None,
        pin_cpu=bool(pin_cpu),
        vram_reserve_gb=reserve,
        swap_slots=slots,
    )
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
    hotkey_start: gr.Button
    hotkey_cancel: gr.Button
    cancel_timer: gr.Timer
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
    gpu_choices, gpu_default, gpu_total, _ = _gpu_inventory()
    data_parallel_gpu_choices = gpu_choices if gpu_total > 0 else []
    detected_tier = auto_tier(gpu_total) if gpu_total else 32
    ctx.states["gpu_index_default"] = gpu_default

    with gr.Tab("🎬 Caption", id="caption"):
        # gr.Tabs() maps its buttons to its direct children, so every component
        # this function creates - states included - must live inside a gr.Tab.
        ctx.states["gpu_index"] = gr.State(gpu_default)
        prompt_context_state = gr.State([initial_family, _INITIAL_MODALITY])

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
                    cancel = action_button(
                        "⏹ Cancel",
                        "red",
                        variant="stop",
                        size="lg",
                        scale=2,
                        elem_id="vc_caption_cancel",
                        interactive=False,
                    )
                    open_output = action_button("📂 Open Output", "teal", size="md", scale=2)
                    open_caption = action_button("📝 Open Last Caption", "violet", size="md", scale=2)
                    reveal_clip = action_button("🎬 Reveal Clip", "amber", size="md", scale=2)
                hotkey_start = gr.Button("Start caption hotkey", elem_id="hk_caption_start", visible="hidden")
                hotkey_cancel = gr.Button("Cancel caption hotkey", elem_id="hk_caption_cancel", visible="hidden")
                cancel_timer = gr.Timer(1.0)

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
                        choices=variant_choices_for_tier(_INITIAL_VARIANT, detected_tier),
                        value=_INITIAL_VARIANT,
                        allow_custom_value=True,
                        label="Model variant",
                        info="Model family, precision/backend variant, and estimated local checkpoint size.",
                    )
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
                    with gr.Row(elem_classes=["vc-compact-row"]):
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
                            info="Unavailable optimized backends fall back safely to PyTorch SDPA.",
                        )
                        controls["attention_backend"] = ctx.reg(
                            "attention_backend", attention, "auto", section="model",
                            description="Requested attention implementation with safe runtime fallback.", choices=ATTENTION_CHOICES, kind="str",
                        )
                    vram_note = gr.Markdown(
                        f"<span class='vc-help'>Auto tier: {detected_tier} GB. The detected plan is applied at startup and on family changes.</span>"
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
                    with gr.Accordion("Block swap & offload plan", open=True):
                        with gr.Row(elem_classes=["vc-compact-row"]):
                            block_swap_auto = gr.Checkbox(
                                value=True,
                                label="Automatic block swap",
                                info=(
                                    "Fits the decoder to free VRAM minus the reserve at load time and shows the "
                                    "resulting swapped-layer count; uncheck to set the count yourself."
                                ),
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
                        with gr.Row(elem_classes=["vc-compact-row"]):
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
                                choices=[2, 3],
                                value=2,
                                label="Swap slots",
                                info="Advanced: GPU staging slots used to prefetch swapped decoder layers.",
                            )
                            controls["swap_slots"] = ctx.reg(
                                "swap_slots", swap_slots, 2, section="model",
                                description="GPU staging slots allocated for decoder block swap.",
                                choices=[2, 3], kind="int",
                            )
                        with gr.Row(elem_classes=["vc-compact-row"]):
                            offload_experts = gr.Checkbox(
                                value=False,
                                label="Offload MoE experts",
                                info="Legacy Accelerate expert offload; disables block swap",
                            )
                            controls["offload_experts"] = ctx.reg(
                                "offload_experts", offload_experts, False, section="model",
                                description="Use legacy Accelerate expert offload instead of block swap.", kind="bool",
                            )
                            pin_cpu = gr.Checkbox(
                                value=True,
                                label="Pin swapped layers in RAM",
                                info="Use pinned host memory for faster decoder-layer transfers.",
                            )
                            controls["pin_cpu"] = ctx.reg(
                                "pin_cpu", pin_cpu, True, section="model",
                                description="Pin block-swapped decoder layers in host memory.", kind="bool",
                            )
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        compile_enabled = gr.Checkbox(
                            value=False,
                            label="torch.compile",
                            info=(
                                "The first generation can spend 1–5 minutes compiling kernels; later runs reuse them. "
                                "A compile runtime failure restores the loaded model and retries that segment eagerly."
                            ),
                        )
                        controls["torch_compile"] = ctx.reg(
                            "torch_compile", compile_enabled, False, section="runtime",
                            description="Compile the language model forward pass with safe fallbacks.", kind="bool",
                        )
                        compile_mode = gr.Dropdown(
                            choices=compile_mode_choices(),
                            value=DEFAULT_COMPILE_MODE,
                            label="Compile mode",
                            info=(
                                "Both choices avoid explicit CUDA graph replay, which is incompatible with the "
                                "DynamicCache used by these decoders. Max autotune has a longer first run."
                            ),
                        )
                        controls["torch_compile_mode"] = ctx.reg(
                            "torch_compile_mode", compile_mode, DEFAULT_COMPILE_MODE, section="runtime",
                            description="Requested torch.compile tuning mode.",
                            choices=list(compile_mode_values()), kind="str",
                        )
                    compile_status = gr.Markdown(_probe_compile_in_child(), elem_classes=["vc-status"])
                    compile_probe_timer = gr.Timer(1.0)
                    with gr.Row(elem_classes=["vc-compact-row"]):
                        download = action_button("📥 Download / Verify model", "sky", size="md", scale=3)
                        refresh_ready = action_button("↻ Refresh", "lime", size="md", scale=1)
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
            with gr.Column(scale=1, min_width=380):
                with gr.Accordion("5. Scene detection & splitting", open=True):
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
            with gr.Column(scale=1, min_width=380):
                with gr.Accordion("6. Post-processing", open=True):
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
                        value=True,
                        label="Save reasoning",
                        info="Persist Thinking-model reasoning separately; it remains hidden from the caption.",
                    )
                    controls["save_reasoning"] = ctx.reg(
                        "save_reasoning", save_reasoning, True, section="output",
                        description="Write Thinking-model reasoning to a separate text file.", kind="bool",
                    )

    handles = CaptionTabHandles(
        media=media,
        progress=progress,
        logs=logs,
        controls=controls,
        start=start,
        cancel=cancel,
        hotkey_start=hotkey_start,
        hotkey_cancel=hotkey_cancel,
        cancel_timer=cancel_timer,
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
        job_done_hook=job_done_hook,
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
        lambda value: (_quant_line(value), _ready_line(value)),
        inputs=model_key,
        outputs=[quant_info, ready_status],
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
                preset.max_new_tokens,
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

    def model_defaults(variant_key: str) -> tuple[Any, ...]:
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        values = {item.name: item for item in spec.param_schema}
        thinking = "enable_thinking" in values
        return (
            gr.update(value=values["temperature"].default, minimum=0, maximum=2, step=0.01),
            gr.update(value=values["top_p"].default, minimum=0, maximum=1, step=0.01),
            gr.update(value=values["top_k"].default, minimum=0, maximum=200, step=1),
            gr.update(value=values["repetition_penalty"].default, minimum=0.5, maximum=2, step=0.01),
            gr.update(value=values["max_new_tokens"].default, minimum=1, maximum=_GLOBAL_MAX_NEW_TOKENS, step=1),
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

    def model_constraints(variant_key: str) -> tuple[Any, ...]:
        spec = MODEL_SPECS[variant_to_family(variant_key)]
        values = {item.name: item for item in spec.param_schema}
        thinking = values.get("enable_thinking")
        return (
            gr.update(minimum=0, maximum=2, step=0.01),
            gr.update(minimum=0, maximum=1, step=0.01),
            gr.update(minimum=0, maximum=200, step=1),
            gr.update(minimum=0.5, maximum=2, step=0.01),
            gr.update(minimum=1, maximum=_GLOBAL_MAX_NEW_TOKENS, step=1),
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
        inputs=model_key,
        outputs=[temperature, top_p, top_k, repetition, max_new_tokens, do_sample, enable_thinking, fps, max_frames, max_pixels, min_pixels, use_audio, context_tokens],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    model_defaults_event = model_key.select(
        model_defaults,
        inputs=model_key,
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

    prompt_preset.select(
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

    def describe_prompt(
        preset_id: str,
        variant_key: str,
        modality: str,
        include_audio: bool,
    ) -> tuple[str, list[str]]:
        description = _display(get_preset(str(preset_id)).description) if preset_id else ""
        effective_modality = _effective_prompt_modality(variant_key, modality, include_audio)
        return description, [variant_to_family(variant_key), effective_modality]

    prompt_preset.change(
        describe_prompt,
        inputs=[prompt_preset, model_key, media.modality_state, use_audio],
        outputs=[prompt_description, prompt_context_state],
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

    def sync_prompt_context(
        variant_key: str,
        modality: str,
        include_audio: bool,
        selected_inputs: list[str] | None,
        current_preset_id: str,
        previous_context: list[str] | tuple[str, str] | None,
        *values: Any,
    ) -> tuple[Any, str, Any, Any, list[str]]:
        effective_modality = _effective_prompt_modality(
            variant_key,
            modality,
            include_audio,
            has_inputs=bool(selected_inputs),
        )
        family, choices, preset = _resolve_prompt_preset(
            variant_key,
            effective_modality,
            current_preset_id,
        )
        context = [family, effective_modality]
        if preset is None:
            return (
                gr.update(choices=[], value=None),
                "<span class='vc-warn'>No compatible task preset for this model and modality.</span>",
                "",
                "",
                context,
            )
        same_context = list(previous_context or []) == context
        if same_context and preset.id == str(current_preset_id or ""):
            system: Any = gr.skip()
            user: Any = gr.skip()
        else:
            rendered_system, rendered_user = render_prompt(preset, _prompt_variables(values))
            system = rendered_system or ""
            user = rendered_user
        return (
            gr.update(choices=choices, value=preset.id),
            _display(preset.description),
            system,
            user,
            context,
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
    ]
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

    def release_previous_model(variant_key: str) -> None:
        selected = str(variant_key)
        try:
            select_variant = getattr(ctx.pipeline_client, "select_variant", None)
            if not callable(select_variant):
                return
            outcome = select_variant(selected)
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
        caption_path = outputs.get("txt") if isinstance(outputs, Mapping) else None
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
        handles.last_outputs_state,
        handles.job_done_hook,
        handles.cancel,
    ]

    def run_caption(*args: Any):
        value_count = len(registry_components)
        settings = registry.values_to_dict(args[:value_count])
        resolved = [str(value) for value in (args[value_count] or [])]
        input_mode = str(args[value_count + 1] or "upload")
        input_modality = str(args[value_count + 2] or "unknown")
        token = CancelToken()
        ctx.activate_cancel(token)
        ctx.states["caption_job_token"] = token
        items: list[InputItem] = [InputItem(path=value) for value in resolved]
        if not items:
            family = variant_to_family(str(settings.get("model_key", _INITIAL_VARIANT)))
            if "text" not in MODEL_SPECS[family].capabilities:
                message = "Select at least one input. Text-only queries require a Qwen3-Omni Instruct or Thinking model."
                yield (
                    render_progress_html(0, "Input required", message),
                    f"<span class='vc-err'>{message}</span>",
                    "**ETA:** —",
                    "**Speed:** — · **Context:** —",
                    [],
                    *[gr.skip() for _ in range(7)],
                    "",
                    gr.update(value="⏹ Cancel", interactive=False),
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
            settings["user_prompt"] = rendered_user

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
        }
        if output_kind == "batch" and str(settings.get("batch_input_folder") or "").strip():
            output_kwargs["source_root"] = str(normalize_path(settings["batch_input_folder"]))
        try:
            output = OutputSpec(**output_kwargs)
        except TypeError as exc:
            if "source_root" not in output_kwargs or "source_root" not in str(exc):
                raise
            output_kwargs.pop("source_root", None)
            output = OutputSpec(**output_kwargs)
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
            mirror_logs=bool(getattr(ctx.pipeline_client, "subprocess_mode", settings.get("subprocess_mode", True))),
        )
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
        last_context = "—"
        processed_count = 0
        total_count = len(items)
        remaining_count = total_count
        item_started_at: dict[int, float] = {}
        terminal_items: set[int] = set()
        live_outputs: dict[str, Any] = {}
        live_dirty = False
        yield (
            render_progress_html(
                0,
                "Starting",
                f"0/{len(items)} processed · {len(items)} remaining · ETA —",
            ),
            "**Status:** Starting caption worker",
            "**ETA:** —",
            "**Speed:** — · **Context:** —",
            item_rows,
            "",
            None,
            "",
            "",
            gr.update(visible=False),
            [],
            {},
            "",
            gr.update(value="⏹ Cancel", interactive=True),
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
                        if str(event.status or "").casefold() not in {"running", "queued"}:
                            terminal_items.add(index)
                    processed_count = int(data.get("processed", len(terminal_items)) or 0)
                    total_count = int(data.get("total", event.total_items or total_count) or total_count)
                    remaining_count = int(
                        data.get("remaining", max(0, total_count - processed_count))
                        or 0
                    )
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
                    *[gr.skip() for _ in range(6)],
                    dict(live_outputs) if live_dirty else gr.skip(),
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
            terminal_label, message, status_class, eta_text = _result_summary(result)
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
                gallery,
                state,
                _job_done_payload(_job_done_message(result), settings),
                gr.update(value="⏹ Cancel", interactive=False),
            )
        except (CancelledError, KeyboardInterrupt) as exc:
            ctx.app_log.warn(str(exc), scope="cancel")
            yield (
                render_progress_html(last_fraction, "Cancelled", str(exc)),
                f"<span class='vc-warn'>**Status:** {html.escape(str(exc))}</span>",
                "**ETA:** cancelled",
                f"**Speed:** {last_speed} · **Context:** {last_context}",
                item_rows,
                *[gr.skip() for _ in range(6)],
                gr.skip(),
                _job_done_payload("Job cancelled", settings),
                gr.update(value="⏹ Cancel", interactive=False),
            )
        except BaseException as exc:
            ctx.app_log.exception(f"Caption job failed: {exc}", scope="ui")
            yield (
                render_progress_html(last_fraction, "Failed", str(exc)),
                f"<span class='vc-err'>**Status:** {html.escape(str(exc))}</span>",
                "**ETA:** failed",
                f"**Speed:** {last_speed} · **Context:** {last_context}",
                item_rows,
                *[gr.skip() for _ in range(7)],
                _job_done_payload("Job failed", settings),
                gr.update(value="⏹ Cancel", interactive=False),
            )
        finally:
            ctx.clear_active_cancel(token)
            if ctx.states.get("caption_job_token") is token:
                ctx.states["caption_job_token"] = None

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
    for event in (start_event, hotkey_start_event):
        event.then(
            fn=None,
            inputs=handles.job_done_hook,
            outputs=[],
            js=notify_js,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def cancel_job() -> tuple[Any, str]:
        token = ctx.states.get("caption_job_token")
        if token is None or token.is_cancelled():
            return gr.update(value="⏹ Cancel", interactive=False), "**Status:** No active caption job to cancel."
        if not token.is_armed():
            token.arm_confirmation(window_s=6)
            return (
                gr.update(value="⚠ Click again to confirm cancel", interactive=True),
                "<span class='vc-warn'>**Status:** Click Cancel again within 6 seconds to stop the running job.</span>",
            )
        token.cancel()
        ctx.pipeline_client.cancel(force=False)
        return (
            gr.update(value="Cancelling…", interactive=False),
            "<span class='vc-warn'>**Status:** Cooperative cancellation requested.</span>",
        )

    for trigger in (handles.cancel.click, handles.hotkey_cancel.click):
        trigger(
            cancel_job,
            outputs=[handles.cancel, handles.progress.status],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def refresh_cancel_button() -> Any:
        token = ctx.states.get("caption_job_token")
        if token is None or token.is_cancelled():
            return gr.update(value="⏹ Cancel", interactive=False)
        if token.is_armed():
            return gr.skip()
        return gr.update(value="⏹ Cancel", interactive=True)

    handles.cancel_timer.tick(
        refresh_cancel_button,
        outputs=handles.cancel,
        queue=False,
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


__all__ = ["CaptionTabHandles", "build", "variant_choices_for_tier", "wire"]
