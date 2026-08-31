"""Trainer fitness analysis, Musubi dataset export, and fixed sub-splitting."""

from __future__ import annotations

import html
import json
import math
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import gradio as gr

from vcap.core.clip_fitness import (
    TRAINER_TARGETS,
    evaluate_clip,
    resolution_bucket_preview,
    suggest_clip_length,
)
from vcap.core.export import discover_dataset_folders, write_kohya_musubi_toml
from vcap.core.media import probe_media
from vcap.core.outputs import OutputWriter
from vcap.core.paths import (
    collision_safe_path,
    list_media_files,
    normalize_path,
    open_in_file_manager,
    sanitize_filename,
)
from vcap.core.scene_split import fixed_length_segments, split_video
from vcap.ui.components import action_button, render_progress_html

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


_FITNESS_HEADERS = ["File", "Duration", "FPS", "Frames", "Verdict", "Bucket preview"]


def _trainer_config(target: str, custom_fps: float, custom_frames: int) -> str | dict[str, Any]:
    if str(target).casefold() != "custom":
        return str(target).casefold()
    frames = max(1, int(custom_frames or 1))
    fps = max(0.01, float(custom_fps or 24.0))
    return {
        "name": "custom",
        "default_frames": frames,
        "frames": frames,
        "fps": fps,
        "default_fps": fps,
        "min_frames": frames,
        "max_frames": max(frames, 100_000),
        "recommended_frames": [frames],
    }


def trainer_clip_suggestion(
    target: str,
    custom_fps: float = 24.0,
    custom_frames: int = 81,
) -> tuple[str, float]:
    """Return the trainer suggestion line and matching sub-split duration."""

    selected = _trainer_config(target, custom_fps, custom_frames)
    if isinstance(selected, str):
        config = dict(TRAINER_TARGETS[selected])
    else:
        config = dict(TRAINER_TARGETS["custom"])
        config.update(selected)
    fps = max(0.01, float(config.get("default_fps", 24.0)))
    frames = max(1, int(config.get("default_frames", 1)))
    valid = suggest_clip_length(config, fps)
    valid_text = ", ".join(str(frame_count) for frame_count, _ in valid)
    seconds = frames / fps
    return (
        f"**Suggested clip length: {frames} frames = {seconds:.2f} s @ {fps:g} fps "
        f"(valid: {valid_text})**",
        seconds,
    )


def _bucket_geometry(
    width: int | None,
    height: int | None,
    target: str | Mapping[str, Any],
    bucket: str,
    policy: str,
) -> tuple[str, dict[str, Any]]:
    if not width or not height:
        return "geometry unavailable", {}
    if isinstance(target, Mapping):
        config = dict(TRAINER_TARGETS["custom"])
        config.update(target)
    else:
        config = dict(TRAINER_TARGETS[str(target)])
    configured = config.get("buckets", {}).get(str(bucket).casefold())
    bucket_value: str | dict[str, Any] = bucket
    if configured:
        bucket_value = {
            "width": configured[0],
            "height": configured[1],
            "multiple": config.get("resolution_multiple", 2),
        }
    out_w, out_h, geometry = resolution_bucket_preview(
        int(width), int(height), bucket_value, str(policy)  # type: ignore[arg-type]
    )
    crop = "/".join(str(value) for value in geometry.get("crop", (0, 0, 0, 0)))
    pad = "/".join(str(value) for value in geometry.get("pad", (0, 0, 0, 0)))
    operation = str(geometry.get("operation") or policy).replace("_", " ")
    return f"{out_w}×{out_h} · {operation} · crop L/T/R/B {crop} · pad L/T/R/B {pad}", geometry


def analyze_clip_fitness(
    folder: str | Path,
    target: str,
    bucket: str,
    policy: str,
    *,
    custom_fps: float = 24.0,
    custom_frames: int = 81,
) -> tuple[list[list[Any]], str, dict[str, Any]]:
    """Probe a folder and build table rows plus a serializable trainer plan."""

    root = normalize_path(folder, must_exist=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    selected_target = _trainer_config(target, custom_fps, custom_frames)
    files = list_media_files(root, recursive=True, kinds=("video",))
    rows: list[list[Any]] = []
    planned: list[dict[str, Any]] = []
    ok_count = dropped = split_count = 0
    for path in files:
        info = probe_media(path)
        report = evaluate_clip(info, selected_target, bucket, policy)  # type: ignore[arg-type]
        if report.frames_available < report.frames_needed:
            verdict = f"will be dropped: {report.frames_available}f < {report.frames_needed}f"
            dropped += 1
        else:
            pieces = report.frames_available // max(1, report.frames_needed)
            if pieces >= 2:
                verdict = f"sub-split ×{pieces} suggested"
                split_count += 1
            else:
                verdict = f"OK {report.suggested_frames or report.frames_needed}f"
                ok_count += 1
        bucket_text, geometry = _bucket_geometry(
            info.width,
            info.height,
            selected_target,
            bucket,
            policy,
        )
        relative = path.relative_to(root).as_posix()
        rows.append([
            relative,
            round(float(info.duration or 0.0), 3),
            round(float(info.fps or 0.0), 3),
            report.frames_available,
            verdict,
            bucket_text,
        ])
        planned.append({
            "file": relative,
            "duration_s": info.duration,
            "source_fps": info.fps,
            "frames_available": report.frames_available,
            "frames_needed": report.frames_needed,
            "suggested_frames": report.suggested_frames,
            "verdict": verdict,
            "bucket": list(report.bucket),
            "geometry": geometry,
            "warnings": report.warnings,
        })
    summary = (
        f"**{len(rows)} clips** · {ok_count} ready · {dropped} will be dropped · "
        f"{split_count} sub-split suggestions"
    )
    plan = {
        "format": "secourses_vcap_clip_fitness_plan",
        "version": 1,
        "source_folder": str(root),
        "trainer_target": selected_target,
        "resolution_bucket": bucket,
        "resize_policy": policy,
        "items": planned,
    }
    return rows, summary, plan


def parse_target_frames(value: str) -> list[int]:
    """Parse comma, semicolon, or whitespace separated positive frame counts."""

    result: list[int] = []
    for token in re.split(r"[,;\s]+", str(value or "").strip()):
        if not token:
            continue
        frames = int(token)
        if frames <= 0:
            raise ValueError("Target frame counts must be positive")
        if frames not in result:
            result.append(frames)
    if not result:
        raise ValueError("Enter at least one target frame count")
    return result


def write_clip_fitness_plan(
    plan: Mapping[str, Any],
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
) -> Path:
    """Write a collision-safe plan outside the scanned source directory."""

    raw_source = str(plan.get("source_folder") or "").strip()
    raw_output = str(output_dir or "").strip()
    if not raw_source:
        raise ValueError("Clip fitness plan has no source folder")
    if not raw_output:
        raise ValueError("Choose a plan output directory")
    source = normalize_path(raw_source, must_exist=True)
    destination = normalize_path(raw_output)
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Plan output directory must be outside the scanned source folder")
    destination.mkdir(parents=True, exist_ok=True)
    source_name = sanitize_filename(source.name or "clips")
    stamp = sanitize_filename(timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    target = collision_safe_path(destination / f"{source_name}_{stamp}.json")
    return OutputWriter().write_json(target, dict(plan), pretty=True)


def append_deduped_progress_line(lines: list[str], message: str, limit: int = 250) -> bool:
    """Append a progress message unless it repeats the immediately preceding line."""

    text = str(message).strip()
    if not text or (lines and lines[-1] == text):
        return False
    lines.append(text)
    if len(lines) > max(1, int(limit)):
        del lines[: len(lines) - max(1, int(limit))]
    return True


def _auto_dataset_kind(root: Path) -> str:
    folders = discover_dataset_folders(root)
    if any(folder.kind in {"video", "mixed"} for folder in folders):
        return "video"
    if any(folder.kind == "image" for folder in folders):
        return "image"
    raise ValueError(f"No video or image dataset folders found in {root}")


def build(ctx: "UiContext") -> None:
    """Render Dataset & Export controls and connect backend operations."""

    fitness_plan = gr.State({})
    initial_suggestion, initial_target_seconds = trainer_clip_suggestion("wan", 16.0, 81)

    with gr.Accordion("Clip fitness checker", open=True):
        with gr.Row(elem_classes=["vc-compact-row"]):
            fitness_folder = gr.Textbox(
                value=str(ctx.outputs_dir), label="Source folder",
                info="Video clips are scanned recursively for trainer compatibility.", scale=5,
            )
            target = gr.Dropdown(
                choices=[
                    (TRAINER_TARGETS[key]["label"], key)
                    for key in ("wan", "hunyuan", "ltx2", "minimax_h3", "custom")
                ],
                value="wan", label="Trainer target",
                info="Applies the trainer's valid frame-count rule.", scale=2,
            )
            custom_fps = gr.Number(
                value=16.0, minimum=0.01, step=0.01, label="FPS",
                info="Used only by the Custom target.", visible=False,
            )
            custom_frames = gr.Number(
                value=81, minimum=1, precision=0, label="Frames",
                info="Minimum frame count used only by the Custom target.", visible=False,
            )
        fitness_suggestion = gr.Markdown(initial_suggestion, elem_classes=["vc-status"])
        with gr.Row(elem_classes=["vc-compact-row"]):
            plan_output_dir = gr.Textbox(
                value=str(ctx.outputs_dir / "clip_fitness"),
                label="Plan output directory",
                info="Plans are timestamped here and never written into the scanned source folder.",
                scale=4,
            )
            bucket = gr.Radio(
                choices=["480p", "720p", "1080p"], value="720p",
                label="Resolution bucket", info="Preview output dimensions without resizing source files.",
            )
            policy = gr.Radio(
                choices=[("Keep AR", "keep_ar"), ("Letterbox", "letterbox"), ("Crop", "crop"), ("Area", "area")],
                value="keep_ar", label="Resize policy",
                info="Choose aspect-preserving resize, padding, crop, or area matching.",
            )
            analyze = action_button("Analyze", "blue", size="md")
            write_plan = action_button("Write plan JSON", "amber", size="md")
        fitness_table = gr.Dataframe(
            value=[], headers=_FITNESS_HEADERS,
            datatype=["str", "number", "number", "number", "str", "str"],
            type="array", interactive=False, show_search="filter", max_height=480,
            pinned_columns=1, static_columns=list(range(6)), buttons=["copy", "fullscreen"],
            label="Trainer fitness",
        )
        fitness_summary = gr.Markdown("No clips analyzed.", elem_classes=["vc-status"])

    with gr.Accordion("Kohya / Musubi TOML generator", open=True):
        with gr.Row(elem_classes=["vc-compact-row"]):
            dataset_root = gr.Textbox(
                value=str(ctx.outputs_dir), label="Dataset root",
                info="Root or parent of direct-media dataset folders.", scale=5,
            )
            dataset_kind = gr.Dropdown(
                choices=[("Auto", "auto"), ("Video", "video"), ("Image", "image")],
                value="auto", label="Kind", info="Auto selects video when any video dataset folder is present.",
            )
            caption_extension = gr.Dropdown(
                choices=[".txt", ".json", ".srt"], value=".txt", label="Caption extension",
                info="Sidecar extension consumed by the trainer.",
            )
            num_repeats = gr.Number(
                value=1, minimum=1, precision=0, label="Repeats",
                info="Default repeats unless the folder has an N_name prefix.",
            )
        with gr.Row(elem_classes=["vc-compact-row"]):
            resolution_w = gr.Number(value=1280, minimum=16, precision=0, label="Width", info="Training resolution width.")
            resolution_h = gr.Number(value=720, minimum=16, precision=0, label="Height", info="Training resolution height.")
            batch_size = gr.Number(value=1, minimum=1, precision=0, label="Batch size", info="Dataset batch size in the TOML.")
            enable_bucket = gr.Checkbox(value=True, label="Enable bucket", info="Enable Musubi aspect-ratio buckets.")
            no_upscale = gr.Checkbox(value=False, label="No upscale", info="Prevent smaller media from being enlarged.")
        with gr.Row(elem_classes=["vc-compact-row"]):
            target_frames = gr.Textbox(value="81", label="Target frames", info="Comma-separated video frame buckets.")
            frame_extraction = gr.Dropdown(
                choices=["head", "chunk", "slide", "uniform", "full"], value="head",
                label="Frame extraction", info="Musubi video frame extraction strategy.",
            )
            frame_stride = gr.Number(value=1, minimum=1, precision=0, label="Frame stride", info="Stride between decoded source frames.")
            frame_sample = gr.Number(value=1, minimum=1, precision=0, label="Frame sample", info="Sampling interval passed to Musubi.")
            max_frames = gr.Number(value=129, minimum=1, precision=0, label="Max frames", info="Maximum decoded frames per source video.")
            source_fps = gr.Number(value=16.0, minimum=0.01, step=0.01, label="Source FPS", info="Expected source/training frame rate.")
        with gr.Row(elem_classes=["vc-compact-row"]):
            wan_defaults = action_button("Wan 81f defaults", "cyan", size="sm")
            ltx_defaults = action_button("LTX 121f 25fps", "violet", size="sm")
            minimax_defaults = action_button("MiniMax H3 124f", "orange", size="sm")
        with gr.Row(elem_classes=["vc-compact-row"]):
            toml_output = gr.Textbox(
                value=str(ctx.outputs_dir / "dataset_config.toml"), label="Output path",
                info="Destination for the generated UTF-8 TOML.", scale=5,
            )
            generate_toml = action_button("Generate", "emerald", size="md")
            open_toml_folder = action_button("Open folder", "teal", size="md")
        # Gradio 6.26 rejects language="toml"; plain Code preserves formatting.
        toml_code = gr.Code(value="", language=None, lines=18, label="Generated TOML", buttons=["copy", "download"])
        toml_status = gr.Markdown("", elem_classes=["vc-status"])

    with gr.Accordion("Sub-split tool", open=False):
        with gr.Row(elem_classes=["vc-compact-row"]):
            split_folder = gr.Textbox(
                value=str(ctx.outputs_dir), label="Video folder",
                info="Videos are scanned recursively and written into separate clip folders.", scale=5,
            )
            split_output = gr.Textbox(
                value=str(ctx.outputs_dir / "sub_split"), label="Output root",
                info="Each source uses a collision-safe <stem> directory.", scale=5,
            )
        with gr.Row(elem_classes=["vc-compact-row"]):
            target_seconds = gr.Number(
                value=initial_target_seconds, minimum=0.1, step=0.1, label="Target seconds",
                info="Maximum duration of each fixed-length segment.",
            )
            overlap = gr.Radio(
                choices=[("0 s", 0.0), ("0.5 s", 0.5), ("1.0 s", 1.0)],
                value=0.0, label="Overlap", info="Temporal overlap between adjacent clips.",
            )
            split_mode = gr.Radio(
                choices=[("Stream copy", "copy"), ("Precise", "precise")],
                value="copy", label="Cut mode",
                info="Stream copy falls back to precise encoding when timing is inexact.",
            )
            run_split = action_button("Run sub-split", "rose", size="md")
        split_progress = gr.HTML(render_progress_html(0.0, "Ready", "Waiting for a video folder."))
        split_log = gr.Textbox(
            value="", label="Sub-split log", lines=10, max_lines=14,
            interactive=False, autoscroll=True, elem_classes=["vc-log"],
        )

    def target_changed(value: str, fps_value: float, frames_value: int) -> tuple[Any, Any, str, float]:
        config = TRAINER_TARGETS.get(str(value), TRAINER_TARGETS["custom"])
        custom = str(value) == "custom"
        selected_fps = float(fps_value or config["default_fps"]) if custom else float(config["default_fps"])
        selected_frames = int(frames_value or config["default_frames"]) if custom else int(config["default_frames"])
        suggestion, seconds = trainer_clip_suggestion(value, selected_fps, selected_frames)
        return (
            gr.update(visible=custom),
            gr.update(visible=custom),
            suggestion,
            seconds,
        )

    target.change(
        target_changed,
        inputs=[target, custom_fps, custom_frames],
        outputs=[custom_fps, custom_frames, fitness_suggestion, target_seconds],
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def custom_target_changed(value: str, fps_value: float, frames_value: int) -> tuple[Any, Any]:
        if str(value) != "custom":
            return gr.skip(), gr.skip()
        suggestion, seconds = trainer_clip_suggestion(value, fps_value, frames_value)
        return suggestion, seconds

    for custom_control in (custom_fps, custom_frames):
        custom_control.change(
            custom_target_changed,
            inputs=[target, custom_fps, custom_frames],
            outputs=[fitness_suggestion, target_seconds],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def analyze_handler(folder_value: str, target_value: str, bucket_value: str, policy_value: str, fps_value: float, frames_value: int) -> tuple[Any, ...]:
        try:
            rows, summary, plan = analyze_clip_fitness(
                folder_value, target_value, bucket_value, policy_value,
                custom_fps=float(fps_value or 24.0), custom_frames=int(frames_value or 1),
            )
            ctx.app_log.log(f"Clip fitness analyzed {len(rows)} file(s).", scope="dataset")
            return rows, summary, plan
        except Exception as exc:
            ctx.app_log.error(f"Clip fitness analysis failed: {exc}", scope="dataset")
            return [], f"<span class='vc-err'>{html.escape(str(exc))}</span>", {}

    analyze.click(
        analyze_handler,
        inputs=[fitness_folder, target, bucket, policy, custom_fps, custom_frames],
        outputs=[fitness_table, fitness_summary, fitness_plan],
        show_progress="minimal", api_visibility="private",
    )

    def write_plan_handler(plan: dict[str, Any], output_dir: str) -> str:
        if not plan:
            return "<span class='vc-warn'>Analyze clips before writing a plan.</span>"
        try:
            target_path = write_clip_fitness_plan(plan, output_dir)
            ctx.app_log.log(f"Wrote clip fitness plan: {target_path}", scope="dataset")
            return f"Wrote `{target_path}`."
        except Exception as exc:
            return f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    write_plan.click(
        write_plan_handler, inputs=[fitness_plan, plan_output_dir], outputs=fitness_summary,
        queue=False, show_progress="hidden", api_visibility="private",
    )

    preset_outputs = [target_frames, source_fps, max_frames, resolution_w, resolution_h, frame_extraction, frame_stride, frame_sample]
    wan_defaults.click(lambda: ("81", 16.0, 81, 832, 480, "head", 1, 1), outputs=preset_outputs, queue=False, show_progress="hidden", api_visibility="private")
    ltx_defaults.click(lambda: ("121", 25.0, 121, 832, 480, "head", 1, 1), outputs=preset_outputs, queue=False, show_progress="hidden", api_visibility="private")
    minimax_defaults.click(lambda: ("124", 24.0, 124, 832, 480, "head", 1, 1), outputs=preset_outputs, queue=False, show_progress="hidden", api_visibility="private")

    def generate_handler(*values: Any) -> tuple[str, str, Any]:
        (
            root_value, kind_value, width, height, extension, batch, buckets_enabled,
            upscale_disabled, repeats, frames_text, extraction, stride, sample,
            maximum_frames, fps_value, output_value,
        ) = values
        try:
            root_path = normalize_path(root_value, must_exist=True)
            selected_kind = _auto_dataset_kind(root_path) if kind_value == "auto" else str(kind_value)
            path = write_kohya_musubi_toml(
                root_path, output_value, kind=selected_kind,
                resolution=(int(width), int(height)), caption_extension=str(extension),
                batch_size=int(batch), enable_bucket=bool(buckets_enabled),
                bucket_no_upscale=bool(upscale_disabled), num_repeats=int(repeats),
                target_frames=parse_target_frames(frames_text), frame_extraction=str(extraction),
                frame_stride=int(stride), frame_sample=int(sample), max_frames=int(maximum_frames),
                source_fps=float(fps_value) if fps_value is not None else None,
            )
            text = path.read_text(encoding="utf-8")
            message = f"Generated {selected_kind} dataset TOML at `{path}`."
            ctx.app_log.log(message, scope="dataset")
            return text, message, gr.update(value=str(path))
        except Exception as exc:
            ctx.app_log.error(f"TOML generation failed: {exc}", scope="dataset")
            return gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", gr.skip()

    generate_toml.click(
        generate_handler,
        inputs=[
            dataset_root, dataset_kind, resolution_w, resolution_h, caption_extension,
            batch_size, enable_bucket, no_upscale, num_repeats, target_frames,
            frame_extraction, frame_stride, frame_sample, max_frames, source_fps, toml_output,
        ],
        outputs=[toml_code, toml_status, toml_output],
        show_progress="minimal", api_visibility="private",
    )

    def open_handler(path_value: str) -> str:
        if not str(path_value or "").strip():
            return "<span class='vc-warn'>Generate a TOML or enter an output path first.</span>"
        target_path = normalize_path(path_value)
        directory = target_path if target_path.is_dir() else target_path.parent
        ok, message = open_in_file_manager(directory)
        return f"<span class='{'vc-ok' if ok else 'vc-err'}'>{html.escape(message)}</span>"

    open_toml_folder.click(
        open_handler, inputs=toml_output, outputs=toml_status,
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def split_handler(folder_value: str, output_value: str, seconds: float, overlap_value: float, mode_value: str):
        events: "queue.Queue[tuple[str, float, str]]" = queue.Queue()
        terminal: dict[str, Any] = {}

        def work() -> None:
            try:
                root = normalize_path(folder_value, must_exist=True)
                out_root = normalize_path(output_value)
                out_root.mkdir(parents=True, exist_ok=True)
                files = list_media_files(root, recursive=True, kinds=("video",))
                total = max(1, len(files))
                written = 0
                for file_index, path in enumerate(files):
                    info = probe_media(path)
                    if not info.duration or info.duration <= 0:
                        events.put(("log", file_index / total, f"Skipped {path.name}: duration unavailable"))
                        continue
                    segments = fixed_length_segments(float(info.duration), float(seconds), float(overlap_value))
                    destination = collision_safe_path(out_root / path.stem)
                    destination.mkdir(parents=True, exist_ok=False)

                    def progress(local: float, message: str, base: int = file_index) -> None:
                        events.put(("progress", (base + local) / total, f"{path.name}: {message}"))

                    clips = split_video(
                        path, segments, destination, mode=str(mode_value), keep_audio=True,
                        name_template="clip_{index:04d}", progress_cb=progress,
                    )
                    written += len(clips)
                    events.put(("log", (file_index + 1) / total, f"{path.name}: wrote {len(clips)} clips to {destination}"))
                terminal["message"] = f"Complete: wrote {written} clips from {len(files)} video(s)."
            except BaseException as exc:
                terminal["error"] = exc
            finally:
                events.put(("terminal", 1.0, ""))

        threading.Thread(target=work, daemon=True, name="vcap-dataset-subsplit").start()
        lines: list[str] = []
        yield render_progress_html(0.0, "Starting", "Preparing fixed-length split plans."), ""
        while True:
            kind, fraction, message = events.get()
            if kind == "terminal":
                break
            append_deduped_progress_line(lines, message)
            yield render_progress_html(fraction, "Sub-splitting", message), "\n".join(lines)
        if "error" in terminal:
            exc = terminal["error"]
            ctx.app_log.error(f"Sub-split failed: {exc}", scope="dataset")
            yield render_progress_html(0.0, "Failed", str(exc)), "\n".join([*lines, f"ERROR: {exc}"])
            return
        message = str(terminal.get("message") or "Complete.")
        ctx.app_log.log(message, scope="dataset")
        yield render_progress_html(1.0, "Complete", message), "\n".join([*lines, message])

    run_split.click(
        split_handler, inputs=[split_folder, split_output, target_seconds, overlap, split_mode],
        outputs=[split_progress, split_log], show_progress="hidden", api_visibility="private",
    )


__all__ = [
    "analyze_clip_fitness",
    "append_deduped_progress_line",
    "build",
    "parse_target_frames",
    "trainer_clip_suggestion",
    "write_clip_fitness_plan",
]
