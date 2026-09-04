"""Standalone Whisper transcription UI and its lightweight orchestration helpers."""

from __future__ import annotations

import html
import json
import math
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gradio as gr

from vcap import VERSION
from vcap.core.media import guess_kind_by_extension
from vcap.core.outputs import MetadataBuilder, OutputWriter, RunLog, allocate_run_dir
from vcap.core.paths import list_media_files, normalize_path, open_in_file_manager
from vcap.core.subprocess_runner import CancelToken
from vcap.ui.components import (
    LogPanelHandles,
    MediaInputHandles,
    ProgressPanelHandles,
    action_button,
    media_input_block,
    progress_panel,
    render_progress_html,
    log_panel,
)

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


TRANSCRIPT_FORMATS = ("srt", "vtt", "txt", "lrc", "tsv", "json")
LIVE_UPDATE_INTERVAL_S = 0.5
LIVE_SEGMENT_LIMIT = 100
LIVE_TRANSCRIPT_LIMIT_BYTES = 4 * 1024
FINAL_SEGMENT_LIMIT = 500
FINAL_TRANSCRIPT_LIMIT_BYTES = 200 * 1024
SRT_PREVIEW_CUE_LIMIT = 300
SRT_PREVIEW_LIMIT_BYTES = 256 * 1024


@dataclass
class TranscriptionPlan:
    run_dir: Path
    output_kind: str
    source_root: Path | None
    batch_output_dir: Path | None
    items: list[dict[str, Any]]
    skipped: list[dict[str, Any]]


@dataclass
class TranscribeTabHandles:
    media: MediaInputHandles
    progress: ProgressPanelHandles
    logs: LogPanelHandles
    controls: dict[str, Any]
    transcript: gr.Textbox
    srt: gr.Textbox
    segments: gr.Dataframe
    files: gr.File
    json_result: gr.JSON
    item_table: gr.Dataframe
    start: gr.Button
    cancel: gr.Button
    cancel_confirmation: gr.Row
    cancel_note: gr.Markdown
    confirm_cancel: gr.Button
    keep_running: gr.Button
    open_output: gr.Button
    open_transcript: gr.Button
    open_editor: gr.Button
    copy_transcript: gr.Button
    results_zip: gr.Button
    results_zip_file: gr.File
    retry_failed: gr.Button
    hotkey_start: gr.Button
    hotkey_cancel: gr.Button
    cancel_timer: gr.Timer
    last_outputs_state: gr.State
    job_done_hook: gr.HTML
    model_info: gr.Markdown
    download_model: gr.Button
    delete_model: gr.Button
    delete_confirmation: gr.Row
    confirm_delete: gr.Button
    keep_model: gr.Button
    refresh_models: gr.Button


def _path_values(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    resolved: list[str] = []
    for item in values:
        raw = getattr(item, "name", item)
        if not str(raw or "").strip():
            continue
        try:
            path = normalize_path(str(raw))
        except Exception:
            continue
        if path.is_file() and guess_kind_by_extension(path) in {"video", "audio"}:
            resolved.append(str(path))
    return list(dict.fromkeys(resolved))


def resolve_transcribe_inputs_at_start(
    settings: Mapping[str, Any],
    input_mode: str,
    cached: Sequence[str] | None,
) -> list[str]:
    """Resolve the active input mode, restricting standalone transcription to AV media."""

    mode = str(input_mode or "upload").casefold()
    if mode == "upload" and settings.get("whisper_input_files"):
        return _path_values(settings.get("whisper_input_files"))
    if mode == "path" and str(settings.get("whisper_input_path") or "").strip():
        return _path_values(settings.get("whisper_input_path"))
    if mode == "folder":
        raw = str(settings.get("whisper_batch_input_folder") or "").strip()
        if raw:
            try:
                root = normalize_path(raw, must_exist=True)
                if root.is_dir():
                    return [
                        str(path)
                        for path in list_media_files(
                            root,
                            recursive=bool(settings.get("whisper_batch_recursive", False)),
                            kinds=("video", "audio"),
                        )
                    ]
            except (OSError, ValueError):
                return []
    return _path_values(list(cached or []))


def _format_extension(settings: Mapping[str, Any]) -> str:
    formats = [
        str(value).casefold().lstrip(".")
        for value in settings.get("whisper_formats", TRANSCRIPT_FORMATS) or []
        if str(value).casefold().lstrip(".") in TRANSCRIPT_FORMATS
    ]
    return formats[0] if formats else "txt"


def prepare_transcription_plan(
    paths: Sequence[str],
    settings: Mapping[str, Any],
    input_mode: str,
    outputs_root: str | Path,
) -> TranscriptionPlan:
    """Allocate a collision-safe bookkeeping directory and map per-item outputs."""

    unique = _path_values(list(paths))
    mode = str(input_mode or "upload").casefold()
    output_kind = "batch" if mode == "folder" or len(unique) > 1 else "single"
    run_dir = allocate_run_dir(outputs_root, "whisper", output_kind)
    source_root: Path | None = None
    if mode == "folder" and str(settings.get("whisper_batch_input_folder") or "").strip():
        try:
            source_root = normalize_path(str(settings["whisper_batch_input_folder"]))
        except Exception:
            source_root = None
    configured = str(settings.get("whisper_batch_output_dir") or "").strip()
    batch_output = (
        normalize_path(configured)
        if configured
        else normalize_path(Path(outputs_root) / "batch_transcripts")
    ) if output_kind == "batch" else None
    save_next = bool(settings.get("whisper_batch_save_next_to_source", False))
    overwrite = bool(settings.get("whisper_batch_overwrite", False))
    extension = _format_extension(settings)
    limit = max(0, int(settings.get("whisper_batch_limit_items", 0) or 0))
    trim_start = max(0.0, float(settings.get("whisper_trim_start_s", 0.0) or 0.0))
    raw_end = settings.get("whisper_trim_end_s")
    trim_end = float(raw_end) if raw_end not in {None, "", 0, 0.0} else None
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    selected = 0
    for index, raw in enumerate(unique):
        source = normalize_path(raw)
        if output_kind == "single":
            out_dir = run_dir
        elif save_next:
            out_dir = source.parent
        else:
            relative_parent = Path()
            if source_root is not None:
                try:
                    relative_parent = source.parent.relative_to(source_root)
                except ValueError:
                    relative_parent = Path()
            assert batch_output is not None
            out_dir = batch_output / relative_parent
        expected = out_dir / f"{source.stem}.{extension}"
        reason = ""
        if expected.is_file() and not overwrite:
            reason = f"Transcript already exists: {expected}"
        elif output_kind == "batch" and limit and selected >= limit:
            reason = f"Excluded by batch limit of {limit} item(s)."
        if reason:
            skipped.append(
                {
                    "index": index,
                    "path": str(source),
                    "status": "skipped",
                    "message": reason,
                    "elapsed_s": 0.0,
                    "segments": 0,
                    "files": [str(expected)] if expected.is_file() else [],
                }
            )
            continue
        selected += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        items.append(
            {
                "index": index,
                "path": str(source),
                "out_dir": str(out_dir),
                "stem": source.stem,
                "trim_start_s": trim_start if output_kind == "single" else 0.0,
                "trim_end_s": trim_end if output_kind == "single" else None,
            }
        )
    return TranscriptionPlan(run_dir, output_kind, source_root, batch_output, items, skipped)


def transcription_item_rows(items: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Render stable rows for the standalone tab's item tracker."""

    return [
        [
            int(item.get("index", index)) + 1,
            Path(str(item.get("path") or "")).name,
            str(item.get("status") or "queued").title(),
            str(item.get("message") or ""),
            f"{float(item.get('elapsed_s', 0.0) or 0.0):.1f}s",
            int(item.get("segments", 0) or 0),
        ]
        for index, item in enumerate(items)
    ]


def _model_info_markdown(alias: str, models_dir: Path) -> str:
    try:
        from vcap.whisper.models import format_size, get_model, is_model_ready, model_dir

        info = get_model(alias)
        ready = is_model_ready(alias, models_dir)
        location = model_dir(alias, models_dir)
        if info is None:
            if "/" not in str(alias):
                return "<span class='vc-err'>Custom models must be a Hugging Face repo ID containing `/`.</span>"
            repo_id, size, note = str(alias), "size reported during download", "Custom Hugging Face model"
        else:
            repo_id = info.repo_id
            size = format_size(info.size_bytes)
            note = info.note
        state = "✓ downloaded" if ready else "not downloaded"
        return (
            f"**{html.escape(repo_id)}** · {html.escape(size)} · {state}<br>"
            f"<span class='vc-help'>{html.escape(note)} · `{html.escape(str(location))}`</span>"
        )
    except Exception as exc:
        return f"<span class='vc-warn'>Whisper model inventory unavailable: {html.escape(str(exc))}</span>"


def _model_choices(models_dir: Path) -> list[tuple[str, str]]:
    try:
        from vcap.whisper.models import model_choices

        return model_choices(models_dir)
    except Exception:
        return [("large-v1", "large-v1")]


def _run_transcription_client(request: dict[str, Any], **kwargs: Any) -> Any:
    """Lazy seam used by UI tests so no worker or model needs to load."""

    from vcap.whisper.client import run_transcription

    return run_transcription(request, **kwargs)


def write_transcription_metadata(
    ctx: "UiContext",
    plan: TranscriptionPlan,
    settings: Mapping[str, Any],
    params: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    elapsed_s: float,
    *,
    cancelled: bool = False,
) -> Path:
    """Persist the same recoverable metadata shape used by caption runs."""

    model_alias = str(settings.get("whisper_model") or "large-v1")
    try:
        from vcap.whisper.models import get_model

        model = get_model(model_alias)
        model_info = {
            "alias": model_alias,
            "repo_id": model.repo_id if model is not None else model_alias,
            "size_bytes": model.size_bytes if model is not None else 0,
        }
    except Exception:
        model_info = {"alias": model_alias, "repo_id": model_alias}
    done = sum(str(item.get("status")) == "done" for item in items)
    skipped = sum(str(item.get("status")) == "skipped" for item in items)
    failed = sum(str(item.get("status")) in {"failed", "error"} for item in items)
    builder = MetadataBuilder()
    builder.build(
        VERSION,
        model_info,
        ctx.settings_registry.metadata_subset(dict(settings)),
        [dict(item) for item in items],
        {"elapsed_s": max(0.0, float(elapsed_s))},
        {
            "device": params.get("device"),
            "gpu_index": params.get("gpu_index"),
        },
        {
            "kind": "whisper_transcription",
            "output_kind": plan.output_kind,
            "params": dict(params),
            "counts": {"done": done, "skipped": skipped, "failed": failed},
            "cancelled": bool(cancelled),
            "source_root": str(plan.source_root) if plan.source_root else None,
            "batch_output_dir": str(plan.batch_output_dir) if plan.batch_output_dir else None,
        },
    )
    metadata = builder.write(plan.run_dir / "metadata.json")
    if plan.output_kind == "batch":
        OutputWriter().write_json(
            plan.run_dir / "summary.json",
            {
                "counts": {"done": done, "skipped": skipped, "failed": failed},
                "processing_time_seconds": max(0.0, float(elapsed_s)),
                "items": [dict(item) for item in items],
            },
        )
    return metadata


def request_transcription_cancel(ctx: "UiContext") -> tuple[Any, Any, str]:
    token = ctx.states.get("transcribe_job_token")
    if isinstance(token, CancelToken) and not token.is_cancelled():
        token.arm_confirmation(8.0)
        return (
            gr.update(value="⏹ Cancel", interactive=True),
            gr.update(visible=True),
            "<span class='vc-warn'>Waiting for cancel confirmation.</span>",
        )
    return (
        gr.update(value="⏹ Cancel", interactive=False),
        gr.update(visible=False),
        "<span class='vc-help'>No transcription is running.</span>",
    )


def confirm_transcription_cancel(ctx: "UiContext") -> tuple[Any, Any, str]:
    token = ctx.states.get("transcribe_job_token")
    if isinstance(token, CancelToken) and token.is_armed():
        token.cancel()
        return (
            gr.update(value="Cancelling…", interactive=False),
            gr.update(visible=False),
            "<span class='vc-warn'>Cancellation requested. Finishing the current worker step.</span>",
        )
    return (
        gr.update(value="⏹ Cancel", interactive=isinstance(token, CancelToken)),
        gr.update(visible=False),
        "<span class='vc-help'>No armed transcription cancellation request.</span>",
    )


def keep_transcription_running(ctx: "UiContext") -> tuple[Any, Any, str]:
    token = ctx.states.get("transcribe_job_token")
    if isinstance(token, CancelToken) and not token.is_cancelled():
        token.reset()
        return (
            gr.update(value="⏹ Cancel", interactive=True),
            gr.update(visible=False),
            "<span class='vc-ok'>Transcription continues.</span>",
        )
    return (
        gr.update(value="⏹ Cancel", interactive=False),
        gr.update(visible=False),
        "<span class='vc-help'>No transcription is running.</span>",
    )


def _registered(
    ctx: "UiContext",
    key: str,
    component: Any,
    default: Any,
    description: str,
    kind: str,
    **metadata: Any,
) -> Any:
    return ctx.reg(
        key,
        component,
        default,
        section="whisper",
        description=description,
        kind=kind,
        **metadata,
    )


def build(ctx: "UiContext") -> TranscribeTabHandles:
    """Build the standalone transcription tab without importing model runtimes."""

    model_choices = _model_choices(ctx.models_dir)
    try:
        from vcap.whisper.params import COMPUTE_TYPE_CHOICES, DEVICE_CHOICES, LANGUAGE_CHOICES
    except Exception:
        COMPUTE_TYPE_CHOICES = ["float16", "bfloat16", "float32", "int8", "int8_float16", "int8_bfloat16"]
        DEVICE_CHOICES = ["auto", "cuda", "cpu"]
        LANGUAGE_CHOICES = ["Automatic Detection", "english"]
    try:
        from vcap.ui.tabs.caption_tab import _gpu_inventory

        gpu_choices, _gpu_default, _gpu_total, _gpu_text = _gpu_inventory()
        gpu_choices = [
            (
                str(label)
                .replace("NVIDIA GeForce ", "")
                .replace("NVIDIA RTX ", "RTX ")
                .replace("AMD Radeon ", "Radeon "),
                value,
            )
            for label, value in gpu_choices
        ]
    except Exception:
        gpu_choices = [("GPU 0", 0)]

    controls: dict[str, Any] = {}
    last_outputs_state = gr.State({})
    job_done_hook = gr.HTML("", visible=False, elem_id="vcap-transcribe-done-hook")

    with gr.Row(equal_height=False):
        with gr.Column(scale=7, min_width=520):
            media = media_input_block(
                ctx,
                registry_keys={
                    "input_files": "whisper_input_files",
                    "input_path": "whisper_input_path",
                    "batch_input_folder": "whisper_batch_input_folder",
                    "batch_output_folder": "whisper_batch_output_dir",
                    "batch_recursive": "whisper_batch_recursive",
                    "overwrite_existing": "whisper_batch_overwrite",
                    "batch_limit_items": "whisper_batch_limit_items",
                },
                state_prefix="whisper_",
                allowed_kinds=("video", "audio"),
                settings_section="whisper",
                output_folder_default=ctx.outputs_dir / "batch_transcripts",
                output_folder_registry_default="",
                output_folder_label="Batch output folder",
                output_folder_info="Videos and audio keep their relative source folders below this directory.",
                overwrite_label="Overwrite existing transcripts",
                overwrite_info="Off skips media whose first selected transcript format already exists.",
                limit_info="Process only the first N items not already transcribed; 0 processes all.",
                folder_placeholder="Folder containing video or audio files",
                upload_description="Upload one or more video or audio files for speech transcription.",
                show_archive_upload=False,
                show_kind_filters=False,
                save_next_to_source_key="whisper_batch_save_next_to_source",
                default_existing_extension=".srt",
                existing_item_noun="transcribed",
                existing_files_label="transcripts",
                input_tabs_elem_id="vc-transcribe-input-tabs",
            )
            with gr.Accordion("Trim range", open=False):
                with gr.Row():
                    description = "Start transcription at this second for a single file; 0 starts at the beginning."
                    controls["whisper_trim_start_s"] = _registered(
                        ctx,
                        "whisper_trim_start_s",
                        gr.Number(value=0.0, minimum=0.0, label="Start seconds", info=description),
                        0.0,
                        description,
                        "float",
                        minimum=0.0,
                    )
                    description = "Stop transcription at this second for a single file; 0 uses the end."
                    controls["whisper_trim_end_s"] = _registered(
                        ctx,
                        "whisper_trim_end_s",
                        gr.Number(value=0.0, minimum=0.0, label="End seconds (0 = end)", info=description),
                        0.0,
                        description,
                        "float",
                        minimum=0.0,
                    )

            gr.Markdown("### Result")
            transcript = gr.Textbox(
                label="Transcript",
                lines=12,
                max_lines=20,
                interactive=False,
                buttons=["copy"],
                info="Completed segments stream here with timestamps; the final view is plain text.",
            )
            with gr.Tabs():
                with gr.Tab("SRT preview"):
                    srt = gr.Textbox(
                        label="SubRip transcript",
                        lines=12,
                        max_lines=16,
                        interactive=False,
                        buttons=["copy"],
                        elem_classes=["vc-mono"],
                    )
                with gr.Tab("Segments"):
                    segments = gr.Dataframe(
                        headers=["#", "Start", "End", "Text", "Prob"],
                        datatype=["number", "number", "number", "str", "number"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Files"):
                    files = gr.File(label="Produced transcript files", file_count="multiple", interactive=False)
                with gr.Tab("JSON"):
                    json_result = gr.JSON(label="Transcript result", value=None)

            with gr.Row():
                start = action_button("▶ Start Transcription", "emerald", elem_id="vc_transcribe_start")
                cancel = action_button(
                    "⏹ Cancel",
                    "red",
                    interactive=False,
                    elem_id="vc_transcribe_cancel",
                )
                open_output = action_button("📂 Open Output", "cyan", interactive=False)
                open_transcript = action_button("📝 Open Last Transcript", "blue", interactive=False)
            with gr.Row():
                open_editor = action_button(
                    "✏️ Open in Caption Editor",
                    "violet",
                    interactive=False,
                    min_width=210,
                )
                copy_transcript = action_button("⧉ Copy transcript", "violet", interactive=False)
                results_zip = action_button("⬇ Results ZIP", "orange", interactive=False)
                retry_failed = action_button("🔁 Retry failed", "cyan", interactive=False)
            with gr.Row(visible=False, elem_classes=["vc-confirm-bar"]) as cancel_confirmation:
                gr.Markdown("⚠ Cancel the running transcription?")
                confirm_cancel = action_button("✔ Yes, cancel", "red")
                keep_running = action_button("✖ Keep running", "gray")
            cancel_note = gr.Markdown("", elem_classes=["vc-status"])
            with gr.Row():
                results_zip_file = gr.File(
                    label="Results ZIP download",
                    visible=False,
                    interactive=False,
                    elem_id="vc_transcribe_zip_download",
                )
            hotkey_start = gr.Button("Start transcription hotkey", visible=False, elem_id="hk_transcribe_start")
            hotkey_cancel = gr.Button("Cancel transcription hotkey", visible=False, elem_id="hk_transcribe_cancel")
            cancel_timer = gr.Timer(1.0)
            progress = progress_panel(
                ctx,
                waiting_detail="Waiting for a transcription job.",
                throughput_text="**Speed:** —",
            )
            item_table = gr.Dataframe(
                headers=["#", "Input", "Status", "Message", "Elapsed", "Segments"],
                datatype=["number", "str", "str", "str", "str", "number"],
                value=[],
                interactive=False,
                wrap=True,
                label="Items",
            )
            logs = log_panel(ctx)

        with gr.Column(scale=5, min_width=430):
            with gr.Accordion("1. Model", open=True):
                description = "Whisper alias or a custom Hugging Face repository ID containing a slash."
                controls["whisper_model"] = _registered(
                    ctx,
                    "whisper_model",
                    gr.Dropdown(
                        choices=model_choices,
                        value="large-v1",
                        label="Whisper model",
                        info=description,
                        allow_custom_value=True,
                    ),
                    "large-v1",
                    description,
                    "str",
                )
                gr.Markdown(
                    "Models download automatically on first use; the first run of a model waits for the download.",
                    elem_classes=["vc-help"],
                )
                model_info = gr.Markdown(_model_info_markdown("large-v1", ctx.models_dir))
                description = "Numeric precision used by CTranslate2; float16 is the proven CUDA default."
                controls["whisper_compute_type"] = _registered(
                    ctx,
                    "whisper_compute_type",
                    gr.Dropdown(COMPUTE_TYPE_CHOICES, value="float16", label="Compute type", info=description),
                    "float16",
                    description,
                    "str",
                    choices=COMPUTE_TYPE_CHOICES,
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=120):
                        description = "Auto uses CUDA when available and otherwise falls back to CPU."
                        controls["whisper_device"] = _registered(
                            ctx,
                            "whisper_device",
                            gr.Dropdown(DEVICE_CHOICES, value="auto", label="Device", info=description),
                            "auto",
                            description,
                            "str",
                            choices=DEVICE_CHOICES,
                        )
                    with gr.Column(scale=2, min_width=240):
                        description = "Physical GPU index exposed to the isolated Whisper worker."
                        controls["whisper_gpu_index"] = _registered(
                            ctx,
                            "whisper_gpu_index",
                            gr.Dropdown(gpu_choices, value=0, label="GPU", info=description),
                            0,
                            description,
                            "int",
                            choices=gpu_choices,
                        )
                    with gr.Column(scale=1, min_width=120):
                        description = "CPU inference threads; 0 lets CTranslate2 choose."
                        controls["whisper_cpu_threads"] = _registered(
                            ctx,
                            "whisper_cpu_threads",
                            gr.Number(value=0, minimum=0, step=1, precision=0, label="CPU threads", info=description),
                            0,
                            description,
                            "int",
                            minimum=0,
                            maximum=1024,
                        )
                with gr.Row():
                    download_model = action_button("📥 Download / Verify model", "emerald")
                    delete_model = action_button("🗑 Delete model files", "red")
                    refresh_models = action_button("↻ Refresh", "cyan")
                with gr.Row(visible=False, elem_classes=["vc-confirm-bar"]) as delete_confirmation:
                    gr.Markdown("⚠ Delete the selected Whisper model files?")
                    confirm_delete = action_button("✔ Yes, delete", "red")
                    keep_model = action_button("✖ Keep model", "gray")

            with gr.Accordion("2. Language & task", open=False):
                description = "Spoken language name, or Automatic Detection when the language is unknown."
                controls["whisper_language"] = _registered(
                    ctx,
                    "whisper_language",
                    gr.Dropdown(LANGUAGE_CHOICES, value="english", label="Language", info=description),
                    "english",
                    description,
                    "str",
                    choices=LANGUAGE_CHOICES,
                )
                description = "Translate recognized speech into English instead of transcribing in its source language."
                controls["whisper_translate"] = _registered(
                    ctx,
                    "whisper_translate",
                    gr.Checkbox(value=False, label="Translate to English", info=description),
                    False,
                    description,
                    "bool",
                )
                description = "Text that guides vocabulary, names, punctuation, and the opening style."
                controls["whisper_initial_prompt"] = _registered(
                    ctx,
                    "whisper_initial_prompt",
                    gr.Textbox(value="", label="Initial prompt", lines=2, info=description),
                    "",
                    description,
                    "str",
                )
                description = "Reapply the initial prompt at every long-form decoding window."
                controls["whisper_repeat_initial_prompt"] = _registered(
                    ctx,
                    "whisper_repeat_initial_prompt",
                    gr.Checkbox(value=False, label="Repeat initial prompt every window", info=description),
                    False,
                    description,
                    "bool",
                )
                description = "Force this text at the beginning of the first decoded window."
                controls["whisper_prefix"] = _registered(
                    ctx,
                    "whisper_prefix",
                    gr.Textbox(value="", label="Prefix", info=description),
                    "",
                    description,
                    "str",
                )
                description = "Words and phrases to bias during recognition."
                controls["whisper_hotwords"] = _registered(
                    ctx,
                    "whisper_hotwords",
                    gr.Textbox(value="", label="Hotwords", info=description),
                    "",
                    description,
                    "str",
                )

            with gr.Accordion("3. Decoding", open=False):
                with gr.Row():
                    description = "Number of candidate beams retained during beam search."
                    controls["whisper_beam_size"] = _registered(
                        ctx, "whisper_beam_size",
                        gr.Number(value=5, minimum=1, maximum=20, step=1, precision=0, label="Beam size", info=description),
                        5, description, "int", minimum=1, maximum=20,
                    )
                    description = "Candidate samples used when decoding with a nonzero temperature."
                    controls["whisper_best_of"] = _registered(
                        ctx, "whisper_best_of",
                        gr.Number(value=5, minimum=1, maximum=20, step=1, precision=0, label="Best of", info=description),
                        5, description, "int", minimum=1, maximum=20,
                    )
                    description = "Beam-search patience multiplier; 1 uses standard beam search."
                    controls["whisper_patience"] = _registered(
                        ctx, "whisper_patience",
                        gr.Number(value=1.0, minimum=0.01, maximum=10.0, step=0.05, label="Patience", info=description),
                        1.0, description, "float", minimum=0.01, maximum=10.0,
                    )
                description = "Sampling temperature; 0 uses deterministic decoding."
                controls["whisper_temperature"] = _registered(
                    ctx, "whisper_temperature",
                    gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Temperature", info=description),
                    0.0, description, "float", minimum=0.0, maximum=1.0,
                )
                with gr.Row():
                    description = "Penalty applied to unusually short or long hypotheses."
                    controls["whisper_length_penalty"] = _registered(
                        ctx, "whisper_length_penalty",
                        gr.Number(value=1.0, minimum=0.01, maximum=10.0, step=0.05, label="Length penalty", info=description),
                        1.0, description, "float", minimum=0.01, maximum=10.0,
                    )
                    description = "Penalty discouraging repeated tokens; values above 1 reduce repetition."
                    controls["whisper_repetition_penalty"] = _registered(
                        ctx, "whisper_repetition_penalty",
                        gr.Number(value=1.2, minimum=0.01, maximum=10.0, step=0.05, label="Repetition penalty", info=description),
                        1.2, description, "float", minimum=0.01, maximum=10.0,
                    )
                    description = "Prevent repetition of token n-grams; 0 disables the rule."
                    controls["whisper_no_repeat_ngram_size"] = _registered(
                        ctx, "whisper_no_repeat_ngram_size",
                        gr.Number(value=0, minimum=0, maximum=100, step=1, precision=0, label="No-repeat n-gram size", info=description),
                        0, description, "int", minimum=0, maximum=100,
                    )
                with gr.Row():
                    description = "Reject outputs whose gzip compression ratio suggests repetition."
                    controls["whisper_compression_ratio_threshold"] = _registered(
                        ctx, "whisper_compression_ratio_threshold",
                        gr.Number(value=2.4, minimum=0.01, maximum=100.0, step=0.1, label="Compression ratio threshold", info=description),
                        2.4, description, "float", minimum=0.01, maximum=100.0,
                    )
                    description = "Reject a segment when its average log probability is below this value."
                    controls["whisper_log_prob_threshold"] = _registered(
                        ctx, "whisper_log_prob_threshold",
                        gr.Number(value=-1.0, minimum=-100.0, maximum=0.0, step=0.1, label="Log-prob threshold", info=description),
                        -1.0, description, "float", minimum=-100.0, maximum=0.0,
                    )
                    description = "Treat a window as silence when the no-speech probability exceeds this value."
                    controls["whisper_no_speech_threshold"] = _registered(
                        ctx, "whisper_no_speech_threshold",
                        gr.Slider(0.0, 1.0, value=0.6, step=0.01, label="No-speech threshold", info=description),
                        0.6, description, "float", minimum=0.0, maximum=1.0,
                    )
                description = "Use decoded text from the previous window as context for the next window."
                controls["whisper_condition_on_previous_text"] = _registered(
                    ctx, "whisper_condition_on_previous_text",
                    gr.Checkbox(value=True, label="Condition on previous text", info=description),
                    True, description, "bool",
                )
                description = "Reset previous-text context when decoding temperature reaches this value."
                controls["whisper_prompt_reset_on_temperature"] = _registered(
                    ctx, "whisper_prompt_reset_on_temperature",
                    gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Prompt reset on temperature", info=description),
                    0.5, description, "float", minimum=0.0, maximum=1.0,
                )
                with gr.Row():
                    description = "Suppress blank output at the beginning of a sampling window."
                    controls["whisper_suppress_blank"] = _registered(
                        ctx, "whisper_suppress_blank",
                        gr.Checkbox(value=True, label="Suppress blank", info=description),
                        True, description, "bool",
                    )
                    description = "Python-style list of token IDs to suppress; [-1] uses Whisper's default list."
                    controls["whisper_suppress_tokens"] = _registered(
                        ctx, "whisper_suppress_tokens",
                        gr.Textbox(value="[-1]", label="Suppress tokens", info=description),
                        "[-1]", description, "str",
                    )
                with gr.Row():
                    description = "Maximum allowed initial timestamp in seconds."
                    controls["whisper_max_initial_timestamp"] = _registered(
                        ctx, "whisper_max_initial_timestamp",
                        gr.Number(value=1.0, minimum=0.0, maximum=60.0, step=0.1, label="Max initial timestamp", info=description),
                        1.0, description, "float", minimum=0.0, maximum=60.0,
                    )
                    description = "Maximum tokens per decoding window; 0 uses the model default."
                    controls["whisper_max_new_tokens"] = _registered(
                        ctx, "whisper_max_new_tokens",
                        gr.Number(value=0, minimum=0, maximum=65536, step=1, precision=0, label="Max new tokens", info=description),
                        0, description, "int", minimum=0, maximum=65536,
                    )
                    description = "Audio decoding window length in seconds."
                    controls["whisper_chunk_length"] = _registered(
                        ctx, "whisper_chunk_length",
                        gr.Slider(1, 30, value=30, step=1, label="Chunk length", info=description),
                        30, description, "int", minimum=1, maximum=30,
                    )
                description = "Skip long silent spans around possible hallucinations; 0 disables this safeguard."
                controls["whisper_hallucination_silence_threshold"] = _registered(
                    ctx, "whisper_hallucination_silence_threshold",
                    gr.Number(value=0.0, minimum=0.0, maximum=60.0, step=0.1, label="Hallucination silence threshold", info=description),
                    0.0, description, "float", minimum=0.0, maximum=60.0,
                )
                with gr.Row():
                    description = "Minimum probability accepted for automatic language detection."
                    controls["whisper_language_detection_threshold"] = _registered(
                        ctx, "whisper_language_detection_threshold",
                        gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Language detection threshold", info=description),
                        0.5, description, "float", minimum=0.0, maximum=1.0,
                    )
                    description = "Number of opening segments considered during language detection."
                    controls["whisper_language_detection_segments"] = _registered(
                        ctx, "whisper_language_detection_segments",
                        gr.Number(value=1, minimum=1, maximum=100, step=1, precision=0, label="Language detection segments", info=description),
                        1, description, "int", minimum=1, maximum=100,
                    )
                with gr.Row():
                    description = "Use faster-whisper's batched inference pipeline."
                    controls["whisper_use_batched_inference"] = _registered(
                        ctx, "whisper_use_batched_inference",
                        gr.Checkbox(value=False, label="Use batched inference", info=description),
                        False, description, "bool",
                    )
                    description = "Number of chunks decoded together by batched inference."
                    controls["whisper_batch_size"] = _registered(
                        ctx, "whisper_batch_size",
                        gr.Slider(1, 64, value=1, step=1, label="Batch size", info=description),
                        1, description, "int", minimum=1, maximum=64,
                    )

            with gr.Accordion("4. Word timestamps & output", open=False):
                description = "Generate word-level timestamps in addition to segment timestamps."
                controls["whisper_word_timestamps"] = _registered(
                    ctx, "whisper_word_timestamps",
                    gr.Checkbox(value=True, label="Word timestamps", info=description),
                    True, description, "bool",
                )
                description = "Rebuild word timestamps into sentence-aware subtitle cues."
                controls["whisper_normalize_word_timestamps"] = _registered(
                    ctx, "whisper_normalize_word_timestamps",
                    gr.Checkbox(value=True, label="Normalize word timestamp output", info=description),
                    True, description, "bool",
                )
                description = "Create word-by-word underlined subtitle cues; unavailable while normalization is enabled."
                controls["whisper_highlight_words"] = _registered(
                    ctx, "whisper_highlight_words",
                    gr.Checkbox(value=False, label="Highlight words", info=description, interactive=False),
                    False, description, "bool",
                )
                with gr.Row():
                    description = "Punctuation characters attached to the beginning of the following word."
                    controls["whisper_prepend_punctuations"] = _registered(
                        ctx, "whisper_prepend_punctuations",
                        gr.Textbox(value="\"'([{-", label="Prepend punctuations", info=description),
                        "\"'([{-", description, "str",
                    )
                    description = "Punctuation characters attached to the end of the preceding word."
                    controls["whisper_append_punctuations"] = _registered(
                        ctx, "whisper_append_punctuations",
                        gr.Textbox(value="\"'.,!?:)]}", label="Append punctuations", info=description),
                        "\"'.,!?:)]}", description, "str",
                    )
                description = "Transcript file formats written for each item."
                controls["whisper_formats"] = _registered(
                    ctx, "whisper_formats",
                    gr.CheckboxGroup(
                        choices=list(TRANSCRIPT_FORMATS),
                        value=list(TRANSCRIPT_FORMATS),
                        label="Output formats",
                        info=description,
                    ),
                    list(TRANSCRIPT_FORMATS), description, "list", choices=list(TRANSCRIPT_FORMATS),
                )
                description = "Append a high-resolution timestamp to each transcript filename."
                controls["whisper_add_timestamp"] = _registered(
                    ctx, "whisper_add_timestamp",
                    gr.Checkbox(value=False, label="Add timestamp to file names", info=description),
                    False, description, "bool",
                )

            with gr.Accordion("5. Voice activity detection", open=False):
                description = "Filter nonspeech with Silero VAD before Whisper decoding."
                controls["whisper_vad_filter"] = _registered(
                    ctx, "whisper_vad_filter",
                    gr.Checkbox(value=False, label="Enable Silero VAD filter", info=description),
                    False, description, "bool",
                )
                description = "Silero probability threshold used to classify speech."
                controls["whisper_vad_threshold"] = _registered(
                    ctx, "whisper_vad_threshold",
                    gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="VAD threshold", info=description),
                    0.5, description, "float", minimum=0.0, maximum=1.0,
                )
                with gr.Row():
                    description = "Discard speech regions shorter than this many milliseconds."
                    controls["whisper_vad_min_speech_ms"] = _registered(
                        ctx, "whisper_vad_min_speech_ms",
                        gr.Number(value=250, minimum=0, maximum=60000, step=10, precision=0, label="Min speech ms", info=description),
                        250, description, "int", minimum=0, maximum=60000,
                    )
                    description = "Split speech after this many seconds; 9999 means unlimited."
                    controls["whisper_vad_max_speech_s"] = _registered(
                        ctx, "whisper_vad_max_speech_s",
                        gr.Number(value=9999.0, minimum=0.001, maximum=9999.0, step=1, label="Max speech s", info=description),
                        9999.0, description, "float", minimum=0.001, maximum=9999.0,
                    )
                with gr.Row():
                    description = "Silence duration required before splitting a speech region."
                    controls["whisper_vad_min_silence_ms"] = _registered(
                        ctx, "whisper_vad_min_silence_ms",
                        gr.Number(value=2000, minimum=0, maximum=60000, step=10, precision=0, label="Min silence ms", info=description),
                        2000, description, "int", minimum=0, maximum=60000,
                    )
                    description = "Audio padding retained on both sides of each detected speech region."
                    controls["whisper_vad_speech_pad_ms"] = _registered(
                        ctx, "whisper_vad_speech_pad_ms",
                        gr.Number(value=400, minimum=0, maximum=10000, step=10, precision=0, label="Speech pad ms", info=description),
                        400, description, "int", minimum=0, maximum=10000,
                    )

    controls["whisper_model"].change(
        lambda alias: _model_info_markdown(str(alias or "large-v1"), ctx.models_dir),
        inputs=controls["whisper_model"],
        outputs=model_info,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def highlight_state(normalize: bool, words: bool) -> Any:
        enabled = bool(words) and not bool(normalize)
        return gr.update(interactive=True) if enabled else gr.update(value=False, interactive=False)

    for trigger in (
        controls["whisper_normalize_word_timestamps"].change,
        controls["whisper_word_timestamps"].change,
    ):
        trigger(
            highlight_state,
            inputs=[
                controls["whisper_normalize_word_timestamps"],
                controls["whisper_word_timestamps"],
            ],
            outputs=controls["whisper_highlight_words"],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    format_event = controls["whisper_formats"].change(
        lambda values: "." + str((values or ["txt"])[0]).casefold().lstrip("."),
        inputs=controls["whisper_formats"],
        outputs=media.existing_extension_state,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    if media.scan_fn is not None and media.scan_inputs and media.scan_outputs:
        format_event.then(
            media.scan_fn,
            inputs=media.scan_inputs,
            outputs=media.scan_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    ctx.states["whisper_existing_extension"] = media.existing_extension_state
    handles = TranscribeTabHandles(
        media=media,
        progress=progress,
        logs=logs,
        controls=controls,
        transcript=transcript,
        srt=srt,
        segments=segments,
        files=files,
        json_result=json_result,
        item_table=item_table,
        start=start,
        cancel=cancel,
        cancel_confirmation=cancel_confirmation,
        cancel_note=cancel_note,
        confirm_cancel=confirm_cancel,
        keep_running=keep_running,
        open_output=open_output,
        open_transcript=open_transcript,
        open_editor=open_editor,
        copy_transcript=copy_transcript,
        results_zip=results_zip,
        results_zip_file=results_zip_file,
        retry_failed=retry_failed,
        hotkey_start=hotkey_start,
        hotkey_cancel=hotkey_cancel,
        cancel_timer=cancel_timer,
        last_outputs_state=last_outputs_state,
        job_done_hook=job_done_hook,
        model_info=model_info,
        download_model=download_model,
        delete_model=delete_model,
        delete_confirmation=delete_confirmation,
        confirm_delete=confirm_delete,
        keep_model=keep_model,
        refresh_models=refresh_models,
    )
    ctx.transcribe_handles = handles
    return handles


class _WhisperUiSink:
    def __init__(self, events: "queue.Queue[tuple[str, Any]]") -> None:
        self.events = events

    def on_log(self, message: str, level: str = "info") -> None:
        self.events.put(("log", {"message": message, "level": level}))

    def on_download(self, payload: dict[str, Any]) -> None:
        self.events.put(("download", dict(payload)))

    def on_progress(self, payload: dict[str, Any]) -> None:
        self.events.put(("progress", dict(payload)))

    def on_segment(self, payload: dict[str, Any]) -> None:
        self.events.put(("segment", dict(payload)))

    def on_item_done(self, payload: dict[str, Any]) -> None:
        self.events.put(("item_done", dict(payload)))

    def on_item_error(self, payload: dict[str, Any]) -> None:
        self.events.put(("item_error", dict(payload)))


def _segment_probability(segment: Any) -> float | None:
    words = list(_field(segment, "words", []) or [])
    probabilities = [
        float(_field(word, "probability"))
        for word in words
        if _field(word, "probability") is not None
    ]
    if probabilities:
        return round(sum(probabilities) / len(probabilities), 4)
    average = _field(segment, "avg_logprob")
    return round(math.exp(float(average)), 4) if average is not None else None


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _utf8_prefix(value: str, limit_bytes: int) -> str:
    raw = str(value or "").encode("utf-8")
    if len(raw) <= limit_bytes:
        return str(value or "")
    return raw[: max(0, int(limit_bytes))].decode("utf-8", errors="ignore")


def _utf8_tail(value: str, limit_bytes: int) -> str:
    raw = str(value or "").encode("utf-8")
    if limit_bytes <= 0:
        return ""
    if len(raw) <= limit_bytes:
        return str(value or "")
    return raw[-int(limit_bytes) :].decode("utf-8", errors="ignore")


def _cap_text_with_note(value: str, limit_bytes: int, note: str) -> str:
    text = str(value or "").strip()
    if len(text.encode("utf-8")) <= limit_bytes:
        return text
    suffix = f"\n\n{note}"
    suffix_bytes = len(suffix.encode("utf-8"))
    if suffix_bytes >= limit_bytes:
        return _utf8_tail(suffix, limit_bytes)
    return _utf8_prefix(text, max(0, limit_bytes - suffix_bytes)).rstrip() + suffix


def _live_transcript_view(lines: Sequence[str], segment_count: int) -> str:
    """Return complete recent segment lines without exceeding the live payload budget."""

    recent = list(lines)[-LIVE_SEGMENT_LIMIT:]
    selected: list[str] = []
    for line in reversed(recent):
        candidate = [str(line), *selected]
        earlier = max(0, int(segment_count) - len(candidate))
        prefix = f"… ({earlier} earlier segments)\n" if earlier else ""
        rendered = prefix + "\n".join(candidate)
        if len(rendered.encode("utf-8")) > LIVE_TRANSCRIPT_LIMIT_BYTES:
            break
        selected = candidate
    if selected:
        earlier = max(0, int(segment_count) - len(selected))
        prefix = f"… ({earlier} earlier segments)\n" if earlier else ""
        return prefix + "\n".join(selected)
    if not recent:
        return ""
    earlier = max(0, int(segment_count) - 1)
    prefix = f"… ({earlier} earlier segments)\n" if earlier else ""
    room = max(0, LIVE_TRANSCRIPT_LIMIT_BYTES - len(prefix.encode("utf-8")))
    return prefix + _utf8_tail(recent[-1], room)


def _srt_file_preview(path: Path, cue_limit: int = SRT_PREVIEW_CUE_LIMIT) -> str:
    cues: list[str] = []
    current: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    current.append(line.rstrip("\r\n"))
                    continue
                if current:
                    cues.append("\n".join(current))
                    current = []
                    if len(cues) >= cue_limit:
                        break
            if current and len(cues) < cue_limit:
                cues.append("\n".join(current))
    except (OSError, UnicodeError):
        return ""
    body = "\n\n".join(cues).rstrip()
    note = f"… full file: {path}"
    rendered = f"{body}\n\n{note}" if body else note
    if len(rendered.encode("utf-8")) <= SRT_PREVIEW_LIMIT_BYTES:
        return rendered
    return _cap_text_with_note(body, SRT_PREVIEW_LIMIT_BYTES, note)


def _result_views(result: Any, produced_files: Sequence[str]) -> tuple[str, str, list[list[Any]], dict[str, Any]]:
    if result is None:
        return "", "", [], {}
    result_segments = list(_field(result, "segments", []) or [])
    rows = [
        [
            index,
            round(float(_field(segment, "start", 0.0)), 3),
            round(float(_field(segment, "end", 0.0)), 3),
            str(_field(segment, "text", "")).strip(),
            _segment_probability(segment),
        ]
        for index, segment in enumerate(result_segments[:FINAL_SEGMENT_LIMIT], start=1)
    ]
    if len(result_segments) > FINAL_SEGMENT_LIMIT:
        rows.append(
            [
                None,
                None,
                None,
                f"showing {FINAL_SEGMENT_LIMIT} of {len(result_segments)} (open the JSON/TSV for all)",
                None,
            ]
        )
    available_paths = [Path(str(raw)) for raw in produced_files]
    srt_path = next(
        (
            path
            for path in reversed(available_paths)
            if path.suffix.casefold() == ".srt" and path.is_file()
        ),
        None,
    )
    srt_text = _srt_file_preview(srt_path) if srt_path is not None else ""
    transcript_path = next(
        (
            path
            for path in reversed(available_paths)
            if path.suffix.casefold() == ".txt" and path.is_file()
        ),
        None,
    )
    plain = str(_field(result, "text", "") or "").strip()
    full_text_note = (
        f"… transcript capped at 200 KB; full file: {transcript_path}"
        if transcript_path is not None
        else "… transcript capped at 200 KB; open a produced TXT/JSON file for the full text."
    )
    plain = _cap_text_with_note(plain, FINAL_TRANSCRIPT_LIMIT_BYTES, full_text_note)
    word_count = sum(
        len(list(_field(segment, "words", []) or []))
        or len(re.findall(r"\S+", str(_field(segment, "text", "") or "")))
        for segment in result_segments
    )
    summary = {
        "language": _field(result, "language"),
        "language_probability": _field(result, "language_probability"),
        "duration_s": float(_field(result, "duration_s", 0.0) or 0.0),
        "elapsed_s": float(_field(result, "elapsed_s", 0.0) or 0.0),
        "model": str(_field(result, "model", "") or ""),
        "compute_type": str(_field(result, "compute_type", "") or ""),
        "device": str(_field(result, "device", "") or ""),
        "segment_count": len(result_segments),
        "word_count": word_count,
        "files": [str(value) for value in produced_files],
    }
    return plain, srt_text, rows, summary


def _completion_payload(message: str, settings: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "message": message,
            "desktop": bool(settings.get("desktop_notification_on_finish", False)),
            "sound": bool(settings.get("play_sound_on_finish", False)),
        },
        ensure_ascii=False,
    )


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _segment_clock(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}" if hours else f"{minutes:02d}:{secs:02d}.{millis:03d}"


def _safe_fraction(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def wire(ctx: "UiContext") -> None:
    """Wire transcription, model actions, cancellation, and output utilities."""

    handles = ctx.transcribe_handles
    if handles is None:
        raise RuntimeError("transcribe_tab.build() must run before wire()")
    registry = ctx.settings_registry
    registry_components = registry.components()
    output_components = [
        handles.progress.bars,
        handles.progress.status,
        handles.progress.eta,
        handles.progress.tokens,
        handles.item_table,
        handles.transcript,
        handles.srt,
        handles.segments,
        handles.files,
        handles.json_result,
        handles.last_outputs_state,
        handles.job_done_hook,
        handles.cancel,
        handles.cancel_confirmation,
        handles.start,
        handles.open_output,
        handles.open_transcript,
        handles.open_editor,
        handles.copy_transcript,
        handles.results_zip,
        handles.retry_failed,
        handles.controls["whisper_model"],
        handles.model_info,
    ]

    def run_transcription(*args: Any):
        value_count = len(registry_components)
        settings = registry.values_to_dict(args[:value_count])
        cached = list(args[value_count] or [])
        input_mode = str(args[value_count + 1] or "upload")
        retry_state = (
            dict(args[value_count + 2] or {})
            if len(args) > value_count + 2 and isinstance(args[value_count + 2], Mapping)
            else None
        )
        if retry_state is not None:
            paths = [str(value) for value in retry_state.get("failed_paths", []) if str(value).strip()]
            input_mode = str(retry_state.get("input_mode") or input_mode)
            settings["whisper_batch_overwrite"] = True
            if retry_state.get("source_root"):
                settings["whisper_batch_input_folder"] = retry_state["source_root"]
            if retry_state.get("batch_output_dir"):
                settings["whisper_batch_output_dir"] = retry_state["batch_output_dir"]
        else:
            paths = resolve_transcribe_inputs_at_start(settings, input_mode, cached)

        model_name = str(settings.get("whisper_model") or "large-v1").strip()
        try:
            from vcap.whisper.models import get_model

            known_model = get_model(model_name) is not None
        except Exception:
            known_model = model_name == "large-v1"
        error = ""
        if not paths:
            error = "Select at least one video or audio input."
        elif not known_model and "/" not in model_name:
            error = "Custom Whisper models must be a Hugging Face repo ID containing `/`."
        elif not list(settings.get("whisper_formats") or []):
            error = "Select at least one transcript output format."
        try:
            trim_start = float(settings.get("whisper_trim_start_s", 0.0) or 0.0)
            trim_end = float(settings.get("whisper_trim_end_s", 0.0) or 0.0)
            if trim_end and trim_end <= trim_start:
                error = "Trim end must be later than trim start."
        except (TypeError, ValueError):
            error = "Trim values must be valid seconds."

        if error:
            yield (
                render_progress_html(0.0, "Input required", error),
                f"<span class='vc-err'>{html.escape(error)}</span>",
                "**ETA:** —",
                "**Speed:** —",
                [],
                "",
                "",
                [],
                [],
                None,
                {},
                "",
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.skip(),
                gr.skip(),
            )
            return

        token = CancelToken()
        ctx.states["transcribe_job_token"] = token
        ctx.activate_cancel(token)
        plan = prepare_transcription_plan(paths, settings, input_mode, ctx.outputs_dir)
        from vcap.whisper.client import build_request
        from vcap.whisper.params import TranscriptOutputOptions, WhisperParams

        params = WhisperParams.from_settings(settings)
        options = TranscriptOutputOptions(
            formats=tuple(settings.get("whisper_formats") or TRANSCRIPT_FORMATS),
            add_timestamp=bool(settings.get("whisper_add_timestamp", False)),
        )
        request = build_request(params, options, plan.items, models_dir=ctx.models_dir)
        item_state: dict[int, dict[str, Any]] = {
            int(item["index"]): {
                "index": int(item["index"]),
                "path": str(item["path"]),
                "status": "queued",
                "message": "Waiting",
                "elapsed_s": 0.0,
                "segments": 0,
                "files": [],
            }
            for item in plan.items
        }
        item_state.update({int(item["index"]): dict(item) for item in plan.skipped})
        produced_files: list[str] = [
            str(path)
            for item in plan.skipped
            for path in item.get("files", [])
            if Path(str(path)).is_file()
        ]
        live_lines: deque[str] = deque(maxlen=LIVE_SEGMENT_LIMIT)
        live_segment_rows: deque[list[Any]] = deque(maxlen=LIVE_SEGMENT_LIMIT)
        live_segment_count = 0
        latest_segment_end = 0.0
        active_item: int | None = None
        last_result: Any | None = None
        last_result_index: int | None = None
        last_srt = ""
        last_json: dict[str, Any] | None = None
        last_plain = ""
        started = time.perf_counter()
        event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        sink = _WhisperUiSink(event_queue)

        def target() -> None:
            try:
                if not plan.items:
                    from vcap.whisper.client import TranscriptionOutcome

                    event_queue.put(
                        (
                            "complete",
                            TranscriptionOutcome(True, [], {}, 0.0, False, None),
                        )
                    )
                    return
                outcome = _run_transcription_client(
                    request,
                    sink=sink,
                    cancel=token,
                    request_dir=plan.run_dir / ".work" / "whisper",
                )
                event_queue.put(("complete", outcome))
            except Exception as exc:
                event_queue.put(("fatal", exc))

        def output(
            fraction: float,
            label: str,
            detail: str,
            status: str,
            eta_s: float | None,
            speed: float | None,
            done_hook: str = "",
            terminal: bool = False,
            include_views: bool = True,
            refresh_model: bool = False,
        ) -> tuple[Any, ...]:
            rows = transcription_item_rows([item_state[key] for key in sorted(item_state)])
            if plan.output_kind == "batch":
                if bool(settings.get("whisper_batch_save_next_to_source", False)):
                    output_directory = plan.source_root or Path(paths[0]).parent
                else:
                    output_directory = plan.batch_output_dir or plan.run_dir
            else:
                output_directory = plan.run_dir
            state = {
                "run_dir": str(plan.run_dir),
                "metadata_path": str(plan.run_dir / "metadata.json"),
                "produced_files": list(produced_files),
                "transcript_path": next(
                    (path for path in reversed(produced_files) if Path(path).suffix.casefold() == ".txt"),
                    produced_files[-1] if produced_files else None,
                ),
                "output_dir": str(output_directory),
                "editor_dir": str(output_directory),
                "archive_source": str(plan.run_dir),
                "output_kind": plan.output_kind,
                "input_mode": input_mode,
                "source_root": str(plan.source_root) if plan.source_root else None,
                "batch_output_dir": str(plan.batch_output_dir) if plan.batch_output_dir else None,
                "failed_paths": [
                    item["path"]
                    for item in item_state.values()
                    if str(item.get("status")) in {"failed", "error"}
                ],
            }
            if include_views:
                view_values: tuple[Any, ...] = (
                    last_plain if terminal else _live_transcript_view(live_lines, live_segment_count),
                    last_srt,
                    list(live_segment_rows),
                    list(produced_files),
                    last_json,
                )
            else:
                view_values = tuple(gr.skip() for _ in range(5))
            if refresh_model:
                model_update: Any = gr.update(
                    choices=_model_choices(ctx.models_dir),
                    value=model_name,
                )
                model_info_update: Any = _model_info_markdown(model_name, ctx.models_dir)
            else:
                model_update = gr.skip()
                model_info_update = gr.skip()
            return (
                render_progress_html(fraction, label, detail),
                status,
                f"**ETA:** {_clock(eta_s)}",
                f"**Speed:** {speed:.1f}× realtime" if speed is not None else "**Speed:** —",
                rows,
                *view_values,
                state,
                done_hook,
                gr.update(value="⏹ Cancel", interactive=not terminal),
                gr.update(visible=False),
                gr.update(interactive=terminal),
                gr.update(interactive=terminal and bool(produced_files or plan.run_dir.exists())),
                gr.update(interactive=terminal and bool(produced_files)),
                gr.update(interactive=terminal),
                gr.update(interactive=terminal and bool(last_plain)),
                gr.update(interactive=terminal and bool(produced_files)),
                gr.update(
                    interactive=terminal
                    and any(str(item.get("status")) in {"failed", "error"} for item in item_state.values())
                ),
                model_update,
                model_info_update,
            )

        worker = threading.Thread(target=target, name="whisper-ui-client", daemon=True)
        outcome: Any | None = None
        fatal: Exception | None = None
        with RunLog(plan.run_dir, ctx.app_log):
            ctx.app_log.log(
                f"Whisper run started: {len(plan.items)} queued, {len(plan.skipped)} skipped",
                scope="whisper",
            )
            worker.start()
            yield output(
                0.0,
                "Starting transcription",
                f"{len(plan.items)} item(s) queued",
                "<span class='vc-ok'>Starting Whisper worker…</span>",
                None,
                None,
            )
            completed = len(plan.skipped)
            pending_update: tuple[float, str, str, str, float | None, float | None] | None = None
            last_live_yield_at = time.monotonic()
            last_overall_fraction = completed / max(1, len(item_state))
            while outcome is None and fatal is None:
                try:
                    event, payload = event_queue.get(timeout=0.15)
                except queue.Empty:
                    if not worker.is_alive():
                        fatal = RuntimeError("Whisper client stopped without a result")
                    if (
                        pending_update is not None
                        and time.monotonic() - last_live_yield_at >= LIVE_UPDATE_INTERVAL_S
                    ):
                        yield output(*pending_update)
                        pending_update = None
                        last_live_yield_at = time.monotonic()
                    continue
                if event == "complete":
                    outcome = payload
                    break
                if event == "fatal":
                    fatal = payload
                    break
                if (
                    pending_update is not None
                    and time.monotonic() - last_live_yield_at >= LIVE_UPDATE_INTERVAL_S
                ):
                    yield output(*pending_update)
                    pending_update = None
                    last_live_yield_at = time.monotonic()
                if event == "download":
                    fraction = _safe_fraction(payload.get("fraction"))
                    pending_update = (
                        fraction,
                        "Downloading Whisper model",
                        str(payload.get("message") or "Downloading model files"),
                        "<span class='vc-ok'>Downloading / verifying the selected model…</span>",
                        None,
                        None,
                    )
                elif event != "log":
                    item_index = int(payload.get("item_index", plan.items[0]["index"] if plan.items else 0))
                    current = item_state.setdefault(
                        item_index,
                        {
                            "index": item_index,
                            "path": "",
                            "status": "running",
                            "message": "",
                            "elapsed_s": 0.0,
                            "segments": 0,
                            "files": [],
                        },
                    )
                    if event in {"progress", "segment"}:
                        current["status"] = "running"
                    if event == "progress":
                        fraction = _safe_fraction(payload.get("fraction"))
                        current["message"] = str(payload.get("message") or "Transcribing")
                        current["elapsed_s"] = float(payload.get("elapsed_s") or 0.0)
                        current["segments"] = max(
                            int(current.get("segments") or 0),
                            int(payload.get("segments") or 0),
                        )
                        elapsed_item = float(payload.get("elapsed_s") or 0.0)
                        trim_offset = float(settings.get("whisper_trim_start_s", 0.0) or 0.0)
                        processed_audio = max(0.0, latest_segment_end - trim_offset)
                        realtime = (
                            processed_audio / elapsed_item
                            if processed_audio > 0 and elapsed_item > 0
                            else None
                        )
                        last_overall_fraction = (completed + fraction) / max(1, len(item_state))
                        pending_update = (
                            last_overall_fraction,
                            "Transcribing speech",
                            current["message"],
                            f"<span class='vc-ok'>{html.escape(current['message'])}</span>",
                            payload.get("eta_s"),
                            realtime,
                        )
                    elif event == "segment":
                        if active_item != item_index:
                            active_item = item_index
                            live_lines.clear()
                            live_segment_rows.clear()
                            live_segment_count = 0
                            latest_segment_end = 0.0
                        start_s = float(payload.get("start") or 0.0)
                        end_s = float(payload.get("end") or 0.0)
                        text_value = str(payload.get("text") or "").strip()
                        live_segment_count += 1
                        latest_segment_end = max(latest_segment_end, end_s)
                        live_lines.append(
                            f"[{_segment_clock(start_s)} → {_segment_clock(end_s)}] {text_value}"
                        )
                        live_segment_rows.append(
                            [
                                live_segment_count,
                                round(start_s, 3),
                                round(end_s, 3),
                                text_value,
                                None,
                            ]
                        )
                        current["segments"] = live_segment_count
                        detail = f"Segment {live_segment_count}: {text_value[-160:]}"
                        pending_update = (
                            last_overall_fraction,
                            "Transcribing speech",
                            detail,
                            f"<span class='vc-ok'>{html.escape(detail)}</span>",
                            None,
                            None,
                        )
                    elif event == "item_done":
                        is_skipped = bool(payload.get("skipped", False))
                        current.update(
                            status="skipped" if is_skipped else "done",
                            message=(
                                str(payload.get("message") or "Existing transcript kept")
                                if is_skipped
                                else "Transcribed"
                            ),
                            files=[str(value) for value in payload.get("files", []) or []],
                        )
                        produced_files.extend(
                            value
                            for value in current["files"]
                            if value not in produced_files and Path(value).is_file()
                        )
                        completed += 1
                    elif event == "item_error":
                        current.update(
                            status="failed",
                            message=str(payload.get("message") or "Transcription failed"),
                        )
                        completed += 1

                    if event in {"item_done", "item_error"}:
                        last_overall_fraction = completed / max(1, len(item_state))
                        pending_update = (
                            last_overall_fraction,
                            "Processing items",
                            str(current.get("message") or ""),
                            f"<span class='vc-ok'>{completed} of {len(item_state)} item(s) processed.</span>",
                            None,
                            None,
                        )

                if (
                    pending_update is not None
                    and time.monotonic() - last_live_yield_at >= LIVE_UPDATE_INTERVAL_S
                ):
                    yield output(*pending_update)
                    pending_update = None
                    last_live_yield_at = time.monotonic()

            worker.join(timeout=5.0)
            elapsed = time.perf_counter() - started
            if outcome is not None:
                for payload in list(getattr(outcome, "items", []) or []):
                    item_index = int(payload.get("item_index", 0))
                    current = item_state.setdefault(item_index, {"index": item_index, "path": ""})
                    if payload.get("skipped"):
                        current.update(
                            status="skipped",
                            message=str(payload.get("message") or "Existing transcript kept"),
                            files=[str(value) for value in payload.get("files", []) or []],
                        )
                        produced_files.extend(
                            value
                            for value in current.get("files", [])
                            if value not in produced_files and Path(value).is_file()
                        )
                    elif payload.get("event") == "item_error" or payload.get("message") and not payload.get("files"):
                        current.update(status="failed", message=str(payload.get("message") or "Transcription failed"))
                    else:
                        current.update(
                            status="skipped" if payload.get("skipped") else "done",
                            message="Existing transcript kept" if payload.get("skipped") else "Transcribed",
                            files=[str(value) for value in payload.get("files", []) or []],
                        )
                        produced_files.extend(
                            value for value in current.get("files", []) if value not in produced_files and Path(value).is_file()
                        )
                results = dict(getattr(outcome, "results", {}) or {})
                if results:
                    last_index = sorted(results)[-1]
                    last_result_index = last_index
                    last_result = results[last_index]
                    item_state.setdefault(last_index, {})["segments"] = len(
                        list(_field(last_result, "segments", []) or [])
                    )
                    item_state[last_index]["elapsed_s"] = float(
                        _field(last_result, "elapsed_s", 0.0) or 0.0
                    )
                if getattr(outcome, "error", None):
                    for current in item_state.values():
                        if current.get("status") in {"queued", "running"}:
                            current.update(status="failed", message=str(outcome.error))
            if fatal is not None:
                for current in item_state.values():
                    if current.get("status") in {"queued", "running"}:
                        current.update(status="failed", message=f"{type(fatal).__name__}: {fatal}")
                ctx.app_log.error(f"Whisper run failed: {type(fatal).__name__}: {fatal}", scope="whisper")
            cancelled = token.is_cancelled() or bool(getattr(outcome, "cancelled", False))
            for current in item_state.values():
                if current.get("status") in {"queued", "running"}:
                    current.update(
                        status="failed" if not cancelled else "cancelled",
                        message="Cancelled" if cancelled else "No worker result",
                    )
            all_items = [item_state[key] for key in sorted(item_state)]
            try:
                metadata_path = write_transcription_metadata(
                    ctx,
                    plan,
                    settings,
                    params.to_dict(),
                    all_items,
                    elapsed,
                    cancelled=cancelled,
                )
            except Exception as exc:
                metadata_path = plan.run_dir / "metadata.json"
                ctx.app_log.error(f"Could not write Whisper metadata: {exc}", scope="whisper")
            done = sum(str(item.get("status")) == "done" for item in all_items)
            skipped = sum(str(item.get("status")) == "skipped" for item in all_items)
            failed = sum(str(item.get("status")) in {"failed", "error"} for item in all_items)
            label = "Cancelled" if cancelled else "Complete"
            status_text = (
                f"Cancelled: {done} transcribed · {skipped} skipped · {failed} failed in {elapsed:.1f}s"
                if cancelled
                else f"Complete: {done} transcribed · {skipped} skipped · {failed} failed in {elapsed:.1f}s"
            )
            log_text = (
                status_text
                if cancelled
                else f"Done: {done} transcribed · {skipped} skipped · {failed} failed in {elapsed:.1f}s"
            )
            ctx.app_log.log(log_text, level="warning" if failed or cancelled else "info", scope="whisper")
            if (
                not cancelled
                and plan.output_kind == "single"
                and done > 0
                and bool(settings.get("open_output_folder_on_single_finish", False))
            ):
                open_in_file_manager(plan.run_dir)
            completion = _completion_payload(
                f"Transcription finished: {done} transcribed, {failed} failed",
                settings,
            )
            final_fraction = 1.0 if not cancelled else completed / max(1, len(item_state))
            final_speed = (
                float(_field(last_result, "duration_s", 0.0) or 0.0)
                / float(_field(last_result, "elapsed_s", 0.0) or 1.0)
                if last_result is not None
                else None
            )
            final_args = (
                final_fraction,
                label,
                status_text,
                f"<span class='{'vc-warn' if failed or cancelled else 'vc-ok'}'>{status_text}</span>",
                0.0,
                final_speed,
            )
            ctx.clear_active_cancel(token)
            if ctx.states.get("transcribe_job_token") is token:
                ctx.states["transcribe_job_token"] = None
            status_output = list(
                output(
                    *final_args,
                    done_hook=completion,
                    terminal=True,
                    include_views=False,
                    refresh_model=False,
                )
            )
            status_state = dict(status_output[10])
            status_state["metadata_path"] = str(metadata_path)
            status_output[10] = status_state
            yield tuple(status_output)

            try:
                if last_result is not None:
                    last_plain, last_srt, live_segment_rows, last_json = _result_views(
                        last_result,
                        produced_files,
                    )
                    if last_result_index is not None:
                        item_state.setdefault(last_result_index, {})["segments"] = int(
                            (last_json or {}).get("segment_count", len(live_segment_rows))
                        )
                elif produced_files:
                    for path_text in reversed(produced_files):
                        path = Path(path_text)
                        if path.suffix.casefold() == ".txt" and path.is_file() and not last_plain:
                            text_value = path.read_text(encoding="utf-8").strip()
                            last_plain = _cap_text_with_note(
                                text_value,
                                FINAL_TRANSCRIPT_LIMIT_BYTES,
                                f"… transcript capped at 200 KB; full file: {path}",
                            )
                        elif path.suffix.casefold() == ".srt" and path.is_file() and not last_srt:
                            last_srt = _srt_file_preview(path)
            except Exception as exc:
                ctx.app_log.error(f"Could not prepare Whisper result previews: {exc}", scope="whisper")
                last_plain = _cap_text_with_note(
                    last_plain,
                    FINAL_TRANSCRIPT_LIMIT_BYTES,
                    "… result preview was limited; use the produced transcript files.",
                )
            result_output = list(
                output(
                    *final_args,
                    done_hook=completion,
                    terminal=True,
                    include_views=True,
                    refresh_model=True,
                )
            )
            result_state = dict(result_output[10])
            result_state["metadata_path"] = str(metadata_path)
            result_output[10] = result_state
            yield tuple(result_output)
        ctx.clear_active_cancel(token)
        if ctx.states.get("transcribe_job_token") is token:
            ctx.states["transcribe_job_token"] = None

    ctx.states["transcribe_run_handler"] = run_transcription
    run_inputs = [*registry_components, handles.media.resolved_state, handles.media.mode_state]
    start_event = handles.start.click(
        run_transcription,
        inputs=run_inputs,
        outputs=output_components,
        api_name="transcribe",
        api_description="Transcribe selected video or audio inputs with Whisper.",
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="public",
    )
    hotkey_event = handles.hotkey_start.click(
        run_transcription,
        inputs=run_inputs,
        outputs=output_components,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )
    retry_event = handles.retry_failed.click(
        run_transcription,
        inputs=[*run_inputs, handles.last_outputs_state],
        outputs=output_components,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    handles.cancel.click(
        lambda: request_transcription_cancel(ctx),
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def escape_cancel() -> tuple[Any, Any, str]:
        token = ctx.states.get("transcribe_job_token")
        if isinstance(token, CancelToken) and token.is_armed():
            return confirm_transcription_cancel(ctx)
        return request_transcription_cancel(ctx)

    handles.hotkey_cancel.click(
        escape_cancel,
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.keep_running.click(
        lambda: keep_transcription_running(ctx),
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.confirm_cancel.click(
        lambda: confirm_transcription_cancel(ctx),
        outputs=[handles.cancel, handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    cancel_timer_enabled = False

    def refresh_cancel_confirmation() -> Any:
        nonlocal cancel_timer_enabled
        token = ctx.states.get("transcribe_job_token")
        if isinstance(token, CancelToken) and not token.is_cancelled() and token.is_armed():
            return gr.skip()
        should_enable = isinstance(token, CancelToken) and not token.is_cancelled()
        if should_enable == cancel_timer_enabled:
            return gr.skip()
        cancel_timer_enabled = should_enable
        return gr.update(value="⏹ Cancel", interactive=should_enable)

    handles.cancel_timer.tick(
        refresh_cancel_confirmation,
        outputs=handles.cancel,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def refresh_cancel_note() -> tuple[Any, Any]:
        token = ctx.states.get("transcribe_job_token")
        if isinstance(token, CancelToken) and not token.is_cancelled() and token.is_armed():
            return gr.skip(), gr.skip()
        return gr.update(visible=False), ""

    handles.cancel_timer.tick(
        refresh_cancel_note,
        outputs=[handles.cancel_confirmation, handles.cancel_note],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    ctx.states["transcribe_cancel_handlers"] = {
        "request": lambda: request_transcription_cancel(ctx),
        "confirm": lambda: confirm_transcription_cancel(ctx),
        "keep": lambda: keep_transcription_running(ctx),
        "escape": escape_cancel,
        "refresh": refresh_cancel_confirmation,
    }

    def refresh_models(selected: str) -> tuple[Any, str]:
        choices = _model_choices(ctx.models_dir)
        values = [str(choice[1]) for choice in choices]
        value = str(selected or "large-v1")
        if value not in values and "/" not in value:
            value = "large-v1"
        return gr.update(choices=choices, value=value), _model_info_markdown(value, ctx.models_dir)

    handles.refresh_models.click(
        refresh_models,
        inputs=handles.controls["whisper_model"],
        outputs=[handles.controls["whisper_model"], handles.model_info],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.delete_model.click(
        lambda: gr.update(visible=True),
        outputs=handles.delete_confirmation,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.keep_model.click(
        lambda: gr.update(visible=False),
        outputs=handles.delete_confirmation,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def delete_selected_model(alias: str) -> tuple[Any, str, Any, str]:
        value = str(alias or "large-v1")
        if ctx.get_active_cancel() is not None:
            choices_update, info = refresh_models(value)
            return (
                choices_update,
                info,
                gr.update(visible=False),
                "<span class='vc-warn'>Wait for the active job before deleting model files.</span>",
            )
        try:
            from vcap.whisper.models import delete_model as delete_files

            removed = delete_files(value, ctx.models_dir)
            message = f"<span class='vc-ok'>Deleted {removed / (1024 ** 2):.1f} MB for {html.escape(value)}.</span>"
        except Exception as exc:
            message = f"<span class='vc-err'>Could not delete model: {html.escape(str(exc))}</span>"
        choices_update, info = refresh_models(value)
        return choices_update, info, gr.update(visible=False), message

    handles.confirm_delete.click(
        delete_selected_model,
        inputs=handles.controls["whisper_model"],
        outputs=[
            handles.controls["whisper_model"],
            handles.model_info,
            handles.delete_confirmation,
            handles.progress.status,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    model_action_outputs = [
        handles.progress.bars,
        handles.progress.status,
        handles.progress.eta,
        handles.progress.tokens,
        handles.model_info,
        handles.cancel,
        handles.cancel_confirmation,
        handles.download_model,
    ]

    def ensure_selected_model(*values: Any):
        settings = registry.values_to_dict(values)
        alias = str(settings.get("whisper_model") or "large-v1").strip()
        try:
            from vcap.whisper.client import build_request
            from vcap.whisper.models import get_model
            from vcap.whisper.params import TranscriptOutputOptions, WhisperParams

            if get_model(alias) is None and "/" not in alias:
                raise ValueError("Custom Whisper models must be a Hugging Face repo ID containing `/`.")
            params = WhisperParams.from_settings(settings)
            request = build_request(
                params,
                TranscriptOutputOptions(formats=()),
                [],
                models_dir=ctx.models_dir,
                action="ensure_model",
            )
        except Exception as exc:
            message = f"<span class='vc-err'>{html.escape(str(exc))}</span>"
            yield (
                render_progress_html(0.0, "Model error", str(exc)),
                message,
                "**ETA:** —",
                "**Speed:** —",
                _model_info_markdown(alias, ctx.models_dir),
                gr.update(value="⏹ Cancel", interactive=False),
                gr.update(visible=False),
                gr.update(interactive=True),
            )
            return
        token = CancelToken()
        ctx.states["transcribe_job_token"] = token
        ctx.activate_cancel(token)
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        sink = _WhisperUiSink(events)

        def target() -> None:
            try:
                events.put(
                    (
                        "complete",
                        _run_transcription_client(
                            request,
                            sink=sink,
                            cancel=token,
                            request_dir=ctx.temp_dir / "whisper",
                        ),
                    )
                )
            except Exception as exc:
                events.put(("fatal", exc))

        worker = threading.Thread(target=target, name="whisper-model-download", daemon=True)
        worker.start()
        yield (
            render_progress_html(0.0, "Checking model", alias),
            "<span class='vc-ok'>Checking local model files…</span>",
            "**ETA:** —",
            "**Speed:** —",
            _model_info_markdown(alias, ctx.models_dir),
            gr.update(value="⏹ Cancel", interactive=True),
            gr.update(visible=False),
            gr.update(interactive=False),
        )
        terminal: tuple[str, Any] | None = None
        while terminal is None:
            try:
                event, payload = events.get(timeout=0.15)
            except queue.Empty:
                if not worker.is_alive():
                    terminal = ("fatal", RuntimeError("Whisper model worker stopped unexpectedly"))
                continue
            if event in {"complete", "fatal"}:
                terminal = event, payload
                continue
            if event == "download":
                fraction = _safe_fraction(payload.get("fraction"))
                speed_bps = float(payload.get("speed_bps") or 0.0)
                total = float(payload.get("total") or 0.0)
                downloaded = float(payload.get("bytes") or 0.0)
                eta = (total - downloaded) / speed_bps if total > downloaded and speed_bps > 0 else None
                yield (
                    render_progress_html(fraction, "Downloading model", str(payload.get("message") or alias)),
                    f"<span class='vc-ok'>{html.escape(str(payload.get('message') or 'Downloading model files'))}</span>",
                    f"**ETA:** {_clock(eta)}",
                    f"**Speed:** {speed_bps / (1024 ** 2):.1f} MB/s" if speed_bps else "**Speed:** —",
                    _model_info_markdown(alias, ctx.models_dir),
                    gr.update(value="⏹ Cancel", interactive=True),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                )
        worker.join(timeout=5.0)
        event, payload = terminal
        cancelled = token.is_cancelled() or bool(getattr(payload, "cancelled", False))
        failed = event == "fatal" or bool(getattr(payload, "error", None))
        if cancelled:
            label, message, css = "Cancelled", "Model download cancelled; partial files remain resumable.", "vc-warn"
        elif failed:
            detail = str(payload if event == "fatal" else getattr(payload, "error", "Model action failed"))
            label, message, css = "Model error", detail, "vc-err"
        else:
            label, message, css = "Model ready", f"{alias} is downloaded and verified.", "vc-ok"
        ctx.clear_active_cancel(token)
        if ctx.states.get("transcribe_job_token") is token:
            ctx.states["transcribe_job_token"] = None
        yield (
            render_progress_html(1.0 if not cancelled and not failed else 0.0, label, message),
            f"<span class='{css}'>{html.escape(message)}</span>",
            "**ETA:** 00:00:00" if not cancelled and not failed else "**ETA:** —",
            "**Speed:** —",
            _model_info_markdown(alias, ctx.models_dir),
            gr.update(value="⏹ Cancel", interactive=False),
            gr.update(visible=False),
            gr.update(interactive=True),
        )

    handles.download_model.click(
        ensure_selected_model,
        inputs=registry_components,
        outputs=model_action_outputs,
        concurrency_id="gpu_queue",
        concurrency_limit=1,
        show_progress="hidden",
        api_visibility="private",
    )

    def open_state_target(state: Mapping[str, Any] | None, key: str) -> str:
        target = dict(state or {}).get(key)
        if not target:
            return "<span class='vc-warn'>No finished transcription output is available.</span>"
        ok, message = open_in_file_manager(str(target))
        css = "vc-ok" if ok else "vc-err"
        return f"<span class='{css}'>{html.escape(message)}</span>"

    handles.open_output.click(
        lambda state: open_state_target(state, "output_dir"),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.open_transcript.click(
        lambda state: open_state_target(state, "transcript_path"),
        inputs=handles.last_outputs_state,
        outputs=handles.progress.status,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.copy_transcript.click(
        fn=None,
        inputs=handles.transcript,
        outputs=[],
        js="(text) => { if (text) navigator.clipboard.writeText(text); return []; }",
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    def create_results_zip(state: Mapping[str, Any] | None) -> tuple[Any, str]:
        import zipfile

        current = dict(state or {})
        candidates = [str(value) for value in current.get("produced_files", []) if Path(str(value)).is_file()]
        for key in ("metadata_path",):
            value = current.get(key)
            if value and Path(str(value)).is_file():
                candidates.append(str(value))
        run_dir = Path(str(current.get("run_dir") or ""))
        run_log = run_dir / "run_log.txt"
        if run_log.is_file():
            candidates.append(str(run_log))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return gr.update(value=None, visible=False), "<span class='vc-warn'>No result files are available to archive.</span>"
        target_dir = normalize_path(ctx.temp_dir / "downloads")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{run_dir.name or 'whisper_results'}.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            used: set[str] = set()
            for index, raw in enumerate(candidates, 1):
                path = normalize_path(raw, must_exist=True)
                name = path.name
                if name in used:
                    name = f"{index:04d}_{name}"
                used.add(name)
                archive.write(path, arcname=name)
        return gr.update(value=str(target), visible=True), f"<span class='vc-ok'>Created {html.escape(target.name)}.</span>"

    handles.results_zip.click(
        create_results_zip,
        inputs=handles.last_outputs_state,
        outputs=[handles.results_zip_file, handles.progress.status],
        concurrency_id="vc-transcribe-zip",
        concurrency_limit=1,
        show_progress="minimal",
        api_visibility="private",
    )

    editor_binding = ctx.states.get("editor_open_binding")
    main_tabs = ctx.states.get("main_tabs")
    if isinstance(editor_binding, Mapping) and main_tabs is not None:
        def prepare_editor(state: Mapping[str, Any] | None) -> tuple[Any, bool, str]:
            current = dict(state or {})
            destination = current.get("editor_dir")
            if not destination:
                return gr.skip(), gr.skip(), "<span class='vc-warn'>No transcript folder is available for the editor.</span>"
            return (
                str(destination),
                bool(current.get("output_kind") == "batch"),
                f"<span class='vc-ok'>Opening {html.escape(str(destination))} in Caption Editor.</span>",
            )

        editor_event = handles.open_editor.click(
            prepare_editor,
            inputs=handles.last_outputs_state,
            outputs=[editor_binding["folder"], editor_binding["recursive"], handles.progress.status],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        scan_event = editor_event.then(
            editor_binding["scan_fn"],
            inputs=editor_binding["inputs"],
            outputs=editor_binding["outputs"],
            show_progress="minimal",
            api_visibility="private",
        )
        scan_event.then(
            lambda: gr.Tabs(selected="editor"),
            outputs=main_tabs,
            queue=False,
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
    for event in (start_event, hotkey_event, retry_event):
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
        recover_binding = ctx.states.get("recover_recent_binding")
        if isinstance(recover_binding, Mapping):
            event.then(
                recover_binding["refresh_fn"],
                outputs=recover_binding["refresh_outputs"],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )


__all__ = [
    "TranscribeTabHandles",
    "TranscriptionPlan",
    "build",
    "confirm_transcription_cancel",
    "keep_transcription_running",
    "prepare_transcription_plan",
    "request_transcription_cancel",
    "resolve_transcribe_inputs_at_start",
    "transcription_item_rows",
    "wire",
    "write_transcription_metadata",
]
