"""Reusable Gradio controls shared by the application tabs."""

from __future__ import annotations

import html
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import gradio as gr

from vcap.core import gpu
from vcap.core.captions_post import parse_replace_pairs, replace_pairs_to_html_chips
from vcap.core.media import MediaInfo, probe_media
from vcap.core.paths import guess_kind_by_extension, list_media_files, normalize_path
from vcap.core.presets import PresetError

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


_KIND_ICONS = {
    "video": "🎬",
    "video_no_audio": "🎬",
    "audio": "🎵",
    "image": "🖼️",
    "text": "📄",
    "unknown": "❔",
}


def action_button(label: str, hue: str, **kwargs: Any) -> gr.Button:
    """Create a consistent product action button with a generated hue class."""

    classes = list(kwargs.pop("elem_classes", []) or [])
    classes.extend(["vc-btn", f"vc-btn-{hue}"])
    return gr.Button(label, elem_classes=classes, **kwargs)


@dataclass
class PresetBarHandles:
    dropdown: gr.Dropdown
    save_as: gr.Textbox
    save: gr.Button
    load: gr.Button
    delete: gr.Button
    reset: gr.Button
    status: gr.Markdown


def _preset_choices(ctx: "UiContext") -> list[tuple[str, str]]:
    entries = ctx.preset_store.list_presets()
    width = max(2, len(str(max(1, len(entries)))))
    return [
        (f"{index:0{width}d}. {'★ ' if entry.is_default else ''}{entry.name}", entry.name)
        for index, entry in enumerate(entries, start=1)
    ]


def preset_bar(ctx: "UiContext") -> PresetBarHandles:
    """Render the universal preset strip. Event wiring is finalized after all tabs."""

    choices = _preset_choices(ctx)
    initial = ctx.preset_store.startup_preset_name()
    with gr.Row(elem_classes=["vc-preset-bar", "vc-compact-row"]):
        dropdown = gr.Dropdown(
            choices=choices,
            value=initial,
            label="Universal preset",
            info="Defaults are marked with a star and are read-only.",
            scale=4,
            min_width=260,
        )
        save_as = gr.Textbox(
            label="Save as",
            placeholder="My caption workflow",
            info="Name for a new or existing user preset.",
            scale=3,
            min_width=220,
        )
        save = action_button("💾 Save", "green", size="md", scale=1, min_width=92)
        load = action_button("📥 Load", "blue", size="md", scale=1, min_width=92)
        delete = action_button("🗑️ Delete", "rose", size="md", scale=1, min_width=96)
        reset = action_button("↺ Reset", "gold", size="md", scale=1, min_width=92)
        status = gr.Markdown(
            "<span class='vc-ok'>Ready.</span>",
            elem_classes=["vc-status"],
            scale=3,
            min_width=320,
        )
    handles = PresetBarHandles(dropdown, save_as, save, load, delete, reset, status)
    ctx.preset_handles = handles
    return handles


def wire_preset_bar(ctx: "UiContext", demo: gr.Blocks) -> None:
    """Wire universal preset operations after the registry is complete."""

    handles = ctx.preset_handles
    if handles is None:
        raise RuntimeError("preset_bar() must be built before it is wired")
    registry = ctx.settings_registry
    components = registry.components()

    def preset_values(settings: dict[str, Any]) -> list[Any]:
        """Apply presets only to controls that participate in preset storage."""

        values = registry.dict_to_values(settings)
        return [
            value if entry.in_preset else gr.skip()
            for entry, value in zip(registry.entries(), values)
        ]

    def save_preset(name: str, *values: Any) -> tuple[Any, str, str]:
        try:
            requested = str(name or "").strip()
            if not requested:
                raise PresetError("Enter a name in Save as")
            settings = registry.values_to_dict(values)
            saved = ctx.preset_store.save(requested, registry.preset_subset(settings))
            return (
                gr.update(choices=_preset_choices(ctx), value=saved),
                saved,
                f"<span class='vc-ok'>Saved preset: {html.escape(saved)}</span>",
            )
        except Exception as exc:
            return gr.skip(), str(name or ""), f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    def load_preset(name: str) -> tuple[Any, ...]:
        try:
            if not name:
                raise PresetError("Choose a preset to load")
            loaded = ctx.preset_store.load(str(name))
            coerced, warnings = registry.coerce(loaded)
            suffix = f" ({len(warnings)} value adjustment(s))" if warnings else ""
            return (*preset_values(coerced), f"<span class='vc-ok'>Loaded {html.escape(str(name))}{suffix}</span>")
        except Exception as exc:
            return (*[gr.skip() for _ in components], f"<span class='vc-err'>{html.escape(str(exc))}</span>")

    def delete_preset(name: str) -> tuple[Any, str]:
        try:
            if not name:
                raise PresetError("Choose a preset to delete")
            deleted = ctx.preset_store.delete(str(name))
            choices = _preset_choices(ctx)
            selected = choices[0][1] if choices else None
            message = f"Deleted {html.escape(str(name))}." if deleted else "Preset was already absent."
            return gr.update(choices=choices, value=selected), f"<span class='vc-ok'>{message}</span>"
        except Exception as exc:
            return gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>"

    def reset_settings() -> tuple[Any, ...]:
        return (*preset_values(registry.defaults()), "<span class='vc-ok'>Restored preset defaults.</span>")

    def startup_preset() -> tuple[Any, ...]:
        startup_name = ctx.preset_store.startup_preset_name()
        if not startup_name:
            return (
                *preset_values(registry.defaults()),
                gr.update(choices=_preset_choices(ctx)),
                "<span class='vc-ok'>Ready with application defaults.</span>",
            )
        try:
            loaded = ctx.preset_store.load(startup_name)
            coerced, warnings = registry.coerce(loaded)
            suffix = f"; adjusted {len(warnings)} value(s)" if warnings else ""
            return (
                *preset_values(coerced),
                gr.update(choices=_preset_choices(ctx), value=startup_name),
                f"<span class='vc-ok'>Auto-loaded {html.escape(startup_name)}{suffix}.</span>",
            )
        except Exception as exc:
            return (
                *preset_values(registry.defaults()),
                gr.update(choices=_preset_choices(ctx)),
                f"<span class='vc-warn'>Last preset could not load: {html.escape(str(exc))}</span>",
            )

    handles.save.click(
        save_preset,
        inputs=[handles.save_as, *components],
        outputs=[handles.dropdown, handles.save_as, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    load_event = handles.load.click(
        load_preset,
        inputs=handles.dropdown,
        outputs=[*components, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.delete.click(
        delete_preset,
        inputs=handles.dropdown,
        outputs=[handles.dropdown, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    reset_event = handles.reset.click(
        reset_settings,
        outputs=[*components, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    startup_event = demo.load(
        startup_preset,
        outputs=[*components, handles.dropdown, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    auto_vram = ctx.states.get("caption_auto_vram_binding")
    if isinstance(auto_vram, dict):
        for dependency in (load_event, reset_event, startup_event):
            dependency.then(
                auto_vram["fn"],
                inputs=auto_vram["inputs"],
                outputs=auto_vram["outputs"],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )


@dataclass
class MediaInputHandles:
    files: gr.File
    path: gr.Textbox
    folder: gr.Textbox
    output_folder: gr.Textbox
    recursive: gr.Checkbox
    overwrite: gr.Checkbox
    limit_items: gr.Number
    rescan: gr.Button
    scan_summary: gr.Markdown
    video: gr.Video
    audio: gr.Audio
    image: gr.Image
    info: gr.Markdown
    gallery: gr.HTML
    resolved_state: gr.State
    mode_state: gr.State
    modality_state: gr.State
    duration_state: gr.State


def _paths(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result: list[str] = []
    for item in values:
        raw = getattr(item, "name", item)
        if raw:
            try:
                path = normalize_path(str(raw))
            except Exception:
                continue
            if path.is_file():
                result.append(str(path))
    return list(dict.fromkeys(result))


def _duration(value: float | None) -> str:
    if value is None:
        return "unknown"
    seconds = max(0.0, float(value))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{remainder:05.2f}" if hours else f"{minutes:02d}:{remainder:05.2f}"


def _media_info_markdown(info: MediaInfo) -> str:
    if info.kind == "unknown":
        return f"<span class='vc-err'>Could not inspect media: {html.escape(info.error or 'unknown error')}</span>"
    # ASCII separators keep compact dimensions visually unambiguous.
    geometry = f"{int(info.width)} x {int(info.height)}" if info.width and info.height else "n/a"
    frame_rate = f"{info.fps:.3g} fps" if info.fps else "n/a"
    audio = (
        f"{info.audio_codec or 'audio'}, {info.audio_sample_rate or '?'} Hz, {info.audio_channels or '?'} ch"
        if info.has_audio
        else "no audio"
    )
    codec = info.video_codec or info.audio_codec or info.container or "n/a"
    return (
        f"**{html.escape(info.path.name)}** · {html.escape(info.kind)} · {_duration(info.duration)} · "
        f"{geometry} · {frame_rate} · {html.escape(audio)} · codec `{html.escape(codec)}`"
    )


def _input_gallery(paths: Iterable[str]) -> str:
    tiles: list[str] = []
    for raw in paths:
        path = Path(raw)
        kind = guess_kind_by_extension(path)
        icon = _KIND_ICONS.get(kind, _KIND_ICONS["unknown"])
        tiles.append(
            '<div class="vc-input-tile" title="'
            + html.escape(str(path))
            + '"><span class="vc-input-icon">'
            + icon
            + '</span><span class="vc-input-name">'
            + html.escape(path.name)
            + "</span></div>"
        )
    if not tiles:
        return '<div class="vc-help">No inputs selected.</div>'
    return '<div class="vc-input-list">' + "".join(tiles) + "</div>"


def _preview_updates(paths: list[str]) -> tuple[Any, ...]:
    if not paths:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            "<span class='vc-help'>Choose a file to see its media details.</span>",
            _input_gallery([]),
            "unknown",
            0.0,
        )
    info = probe_media(paths[0])
    video = gr.update(value=None, visible=False)
    audio = gr.update(value=None, visible=False)
    image = gr.update(value=None, visible=False)
    if info.kind in {"video", "video_no_audio"}:
        video = gr.update(value=str(info.path), visible=True)
    elif info.kind == "audio":
        audio = gr.update(value=str(info.path), visible=True)
    elif info.kind == "image":
        image = gr.update(value=str(info.path), visible=True)
    modality = "video_audio" if info.kind == "video" else "video" if info.kind == "video_no_audio" else info.kind
    return video, audio, image, _media_info_markdown(info), _input_gallery(paths), modality, float(info.duration or 0.0)


def _folder_scan(
    folder: str,
    recursive: bool,
    output_folder: str = "",
    overwrite: bool = False,
    limit_items: int | float = 0,
) -> tuple[list[str], str]:
    raw = str(folder or "").strip()
    if not raw:
        return [], "<span class='vc-help'>Choose a folder for a light extension scan.</span>"
    try:
        root = normalize_path(raw)
    except Exception as exc:
        return [], f"<span class='vc-err'>{html.escape(str(exc))}</span>"
    if not root.is_dir():
        return [], f"<span class='vc-err'>Folder does not exist: {html.escape(str(root))}</span>"
    found = list_media_files(root, recursive=bool(recursive), kinds=("video", "audio", "image"))
    try:
        output_root = normalize_path(output_folder) if str(output_folder or "").strip() else None
    except Exception as exc:
        return [], f"<span class='vc-err'>Invalid output folder: {html.escape(str(exc))}</span>"
    counts = {"video": 0, "audio": 0, "image": 0}
    existing = 0
    for path in found:
        kind = guess_kind_by_extension(path)
        if kind in counts:
            counts[kind] += 1
        relative = path.relative_to(root)
        caption_path = output_root / relative.with_suffix(".txt") if output_root is not None else None
        if caption_path is not None and caption_path.is_file():
            existing += 1
    overwrite_hint = (
        "Overwrite is on; existing captions will be replaced."
        if overwrite
        else "Overwrite is off; existing captions will be skipped."
    )
    summary = (
        f"🎬 {counts['video']} videos · 🎵 {counts['audio']} audios · "
        f"🖼️ {counts['image']} images · {existing} already captioned in output folder. "
        f"{overwrite_hint}"
    )
    limit = max(0, int(limit_items or 0))
    if limit:
        summary += f" · limiting to first {limit}"
    return [str(path) for path in found], summary


def _resolved_after_preview_edit(
    value: str | None,
    current: list[str] | None,
    input_mode: str,
) -> list[str]:
    """Keep folder scans authoritative while accepting single-file edits."""

    selected = list(current or [])
    if str(input_mode).casefold() == "folder" or not value:
        return selected
    resolved = str(normalize_path(value))
    if selected:
        cache_root = os.environ.get("GRADIO_TEMP_DIR")
        if cache_root:
            try:
                current_path = normalize_path(selected[0])
                cached_path = normalize_path(resolved)
                root = normalize_path(cache_root)
                cached_path.relative_to(root)
                try:
                    current_path.relative_to(root)
                except ValueError:
                    return selected
            except (OSError, TypeError, ValueError):
                pass
    return [resolved, *selected[1:]] if selected else [resolved]


def media_input_block(ctx: "UiContext") -> MediaInputHandles:
    """Build the single upload/path/folder input surface and auto-preview wiring."""

    resolved_state = gr.State([])
    mode_state = gr.State("upload")
    modality_state = gr.State("video_audio")
    duration_state = gr.State(0.0)
    ctx.states.update(
        resolved_inputs=resolved_state,
        input_mode=mode_state,
        input_modality=modality_state,
        input_duration=duration_state,
    )

    with gr.Group(elem_classes=["vc-card"]):
        gr.Markdown("### Input", elem_classes=["vc-section-title"])
        with gr.Tabs(selected="upload", elem_id="vc-input-tabs"):
            with gr.Tab("📤 Upload files", id="upload") as upload_tab:
                files = gr.File(
                    file_count="multiple",
                    file_types=["video", "audio", "image", ".txt"],
                    type="filepath",
                    label="Files",
                    height=118,
                )
                ctx.reg(
                    "input_files",
                    files,
                    [],
                    section="input",
                    description="Uploaded video, audio, image, or text files.",
                    in_preset=False,
                )
            with gr.Tab("📄 File path", id="path") as path_tab:
                path = gr.Textbox(
                    label="File path",
                    placeholder="Paste a file path — same as uploading",
                    info="Quoted, mixed-separator, and non-ASCII paths are supported.",
                )
                ctx.reg(
                    "input_path",
                    path,
                    "",
                    section="input",
                    description="Local path treated exactly like an uploaded file.",
                    in_preset=False,
                    kind="str",
                )
            with gr.Tab("📁 Folder batch", id="folder") as folder_tab:
                folder = gr.Textbox(
                    label="Input folder",
                    placeholder="Folder containing videos, audio, or images",
                    info="The light scan reads filenames only; media is probed during processing.",
                )
                ctx.reg(
                    "batch_input_folder",
                    folder,
                    "",
                    section="input",
                    description="Batch source folder.",
                    in_preset=False,
                    kind="str",
                )
                output_folder = gr.Textbox(
                    value=str(ctx.outputs_dir / "batch_captions"),
                    label="Batch output folder",
                    info="Input names and relative folders are mirrored here.",
                )
                ctx.reg(
                    "batch_output_folder",
                    output_folder,
                    str(ctx.outputs_dir / "batch_captions"),
                    section="input",
                    description="Destination folder for mirrored batch captions.",
                    in_preset=False,
                    kind="str",
                )
                with gr.Row(elem_classes=["vc-compact-row"]):
                    recursive = gr.Checkbox(
                        value=False,
                        label="Scan subfolders",
                        info="Include supported files below the selected folder.",
                    )
                    ctx.reg(
                        "batch_recursive",
                        recursive,
                        False,
                        section="input",
                        description="Recursively scan the batch input folder.",
                        kind="bool",
                    )
                    overwrite = gr.Checkbox(
                        value=False,
                        label="Overwrite existing captions",
                        info="Off skips media that already has the requested output.",
                    )
                    ctx.reg(
                        "overwrite_existing",
                        overwrite,
                        False,
                        section="output",
                        description="Replace existing batch captions instead of skipping them.",
                        kind="bool",
                    )
                    limit_items = gr.Number(
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                        label="Limit items (0 = all)",
                        info="Process only the first N items that are not already captioned.",
                    )
                    ctx.reg(
                        "batch_limit_items",
                        limit_items,
                        0,
                        section="input",
                        description="Maximum processable batch items after existing-caption skips.",
                        kind="int",
                        minimum=0,
                    )
                    rescan = action_button("↻ Rescan", "cyan", size="md")
                scan_summary = gr.Markdown(
                    "<span class='vc-help'>Choose a folder for a light extension scan.</span>",
                    elem_classes=["vc-status"],
                )

        with gr.Group():
            video = gr.Video(
                label="Video preview and trim",
                format=None,
                interactive=True,
                visible=False,
                height=390,
                buttons=["download"],
                elem_classes=["vc-preview"],
            )
            audio = gr.Audio(
                label="Audio preview and trim",
                type="filepath",
                format=None,
                interactive=True,
                editable=True,
                visible=False,
                buttons=["download"],
                elem_classes=["vc-preview"],
            )
            image = gr.Image(
                label="Image preview",
                type="filepath",
                interactive=True,
                visible=False,
                buttons=["download", "fullscreen"],
                elem_classes=["vc-preview"],
            )
        info = gr.Markdown(
            "<span class='vc-help'>Choose a file to see its media details.</span>",
            elem_classes=["vc-status"],
        )
        gallery = gr.HTML(_input_gallery([]), elem_classes=["vc-scroll-result"])

    preview_outputs = [video, audio, image, info, gallery, modality_state, duration_state]

    def choose_upload(value: Any) -> tuple[Any, ...]:
        selected = _paths(value)
        return (*_preview_updates(selected), selected, "upload")

    def choose_path(value: str) -> tuple[Any, ...]:
        selected = _paths(value)
        return (*_preview_updates(selected), selected, "path")

    def scan_folder(
        value: str,
        recursive_value: bool,
        output_value: str,
        overwrite_value: bool,
        limit_value: int | float,
    ) -> tuple[Any, ...]:
        selected, summary = _folder_scan(
            value,
            recursive_value,
            output_value,
            overwrite_value,
            limit_value,
        )
        return (*_preview_updates(selected), selected, "folder", summary)

    upload_outputs = [*preview_outputs, resolved_state, mode_state]
    files.change(
        choose_upload,
        inputs=files,
        outputs=upload_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    path.change(
        choose_path,
        inputs=path,
        outputs=upload_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    upload_tab.select(
        choose_upload,
        inputs=files,
        outputs=upload_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    path_tab.select(
        choose_path,
        inputs=path,
        outputs=upload_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    folder_outputs = [*preview_outputs, resolved_state, mode_state, scan_summary]
    folder_inputs = [folder, recursive, output_folder, overwrite, limit_items]
    folder_tab.select(
        scan_folder,
        inputs=folder_inputs,
        outputs=folder_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    for trigger in (
        folder.change,
        recursive.change,
        output_folder.change,
        overwrite.change,
        limit_items.change,
        rescan.click,
    ):
        trigger(
            scan_folder,
            inputs=folder_inputs,
            outputs=folder_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def accept_editor_value(
        value: str | None,
        current: list[str] | None,
        input_mode: str,
    ) -> tuple[Any, ...]:
        selected = _resolved_after_preview_edit(value, current, input_mode)
        updates = _preview_updates(selected)
        return selected, updates[3], updates[4], updates[5], updates[6]

    editor_outputs = [resolved_state, info, gallery, modality_state, duration_state]
    video.change(
        accept_editor_value,
        inputs=[video, resolved_state, mode_state],
        outputs=editor_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    audio.change(
        accept_editor_value,
        inputs=[audio, resolved_state, mode_state],
        outputs=editor_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    image.change(
        accept_editor_value,
        inputs=[image, resolved_state, mode_state],
        outputs=editor_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    return MediaInputHandles(
        files,
        path,
        folder,
        output_folder,
        recursive,
        overwrite,
        limit_items,
        rescan,
        scan_summary,
        video,
        audio,
        image,
        info,
        gallery,
        resolved_state,
        mode_state,
        modality_state,
        duration_state,
    )


@dataclass
class LogPanelHandles:
    log: gr.Textbox
    revision: gr.State
    meter: gr.HTML
    clear: gr.Button
    log_timer: gr.Timer
    meter_timer: gr.Timer


LOG_PANEL_CHAR_LIMIT = 120_000


def newest_first(text: str) -> str:
    """Reverse a chronological newline-delimited log block so the latest line is on top."""

    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(reversed(lines))


def merge_log_newest_first(current: str, new_lines: list[str], limit: int = LOG_PANEL_CHAR_LIMIT) -> str:
    """Prepend chronological ``new_lines`` above ``current`` (newest first) and trim the oldest tail."""

    fresh = "\n".join(line for line in reversed([str(item) for item in new_lines]) if line.strip())
    combined = "\n".join(part for part in (fresh, str(current or "").strip()) if part)
    if len(combined) > limit:
        combined = combined[:limit]
    return combined


def log_panel(ctx: "UiContext") -> LogPanelHandles:
    """Build revision-polled live logs and a Torch-free resource meter."""

    with gr.Group(elem_classes=["vc-card"]):
        with gr.Row(elem_classes=["vc-compact-row"]):
            gr.Markdown("### Live log", elem_classes=["vc-section-title"])
            clear = action_button("⌫ Clear", "orange", size="md", min_width=86)
        log = gr.Textbox(
            value=newest_first(ctx.app_log.tail(300)),
            label="Application log (newest first)",
            lines=14,
            max_lines=14,
            autoscroll=False,
            interactive=False,
            buttons=["copy"],
            show_label=True,
            elem_classes=["vc-log"],
        )
        gpu_index = int(ctx.states.get("gpu_index_default", 0) or 0)
        meter = gr.HTML(gpu.render_resource_meter_html(gpu.resource_snapshot(gpu_index)))
    revision = gr.State(ctx.app_log.revision)
    log_timer = gr.Timer(0.5)
    meter_timer = gr.Timer(1.0)

    def poll_log(cursor: int, current: str) -> tuple[Any, Any]:
        lines, new_revision = ctx.app_log.snapshot(int(cursor or 0))
        if new_revision == int(cursor or 0):
            return gr.skip(), gr.skip()
        return merge_log_newest_first(str(current or ""), lines), new_revision

    def poll_resources(selected_gpu: int | str | None) -> str:
        try:
            index = int(selected_gpu or 0)
            return gpu.render_resource_meter_html(gpu.resource_snapshot(index))
        except Exception as exc:
            return f"<div class='vc-err'>Resource meter unavailable: {html.escape(str(exc))}</div>"

    log_timer.tick(
        poll_log,
        inputs=[revision, log],
        outputs=[log, revision],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    meter_timer.tick(
        poll_resources,
        inputs=ctx.states["gpu_index"],
        outputs=meter,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    clear.click(
        lambda: ("", ctx.app_log.revision),
        outputs=[log, revision],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    return LogPanelHandles(log, revision, meter, clear, log_timer, meter_timer)


@dataclass
class ProgressPanelHandles:
    bars: gr.HTML
    status: gr.Markdown
    eta: gr.Markdown
    tokens: gr.Markdown


def render_progress_html(fraction: float, label: str, detail: str = "") -> str:
    percent = min(100.0, max(0.0, float(fraction) * 100.0))
    return (
        '<div class="vc-progress"><div class="vc-progress__labels">'
        f"<span>{html.escape(label)}</span><span>{percent:.1f}%</span></div>"
        '<div class="vc-progress__track" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{percent:.1f}"><span class="vc-progress__fill" style="width:{percent:.1f}%"></span></div>'
        + (f'<div class="vc-help" style="margin-top:6px">{html.escape(detail)}</div>' if detail else "")
        + "</div>"
    )


def progress_panel(ctx: "UiContext") -> ProgressPanelHandles:
    """Build stable progress, status, ETA, and throughput outputs."""

    del ctx
    with gr.Group(elem_classes=["vc-card"]):
        bars = gr.HTML(render_progress_html(0.0, "Ready", "Waiting for a caption job."))
        with gr.Row(elem_classes=["vc-compact-row"]):
            status = gr.Markdown("**Status:** Ready", scale=5)
            eta = gr.Markdown("**ETA:** —", scale=2)
            tokens = gr.Markdown("**Speed:** —", scale=2)
    return ProgressPanelHandles(bars, status, eta, tokens)


@dataclass
class ReplaceWordsHandles:
    text: gr.Textbox
    chips: gr.HTML
    case_insensitive: gr.Checkbox
    whole_words: gr.Checkbox
    regex: gr.Checkbox


def replace_words_editor() -> ReplaceWordsHandles:
    """Create a live replacement-pair editor with escaped visual chips."""

    text = gr.Textbox(
        value="",
        label="Word replacements",
        placeholder="find;replace\none phrase;another phrase",
        info="Use one find;replace pair per line. A pipe may also separate pairs.",
        lines=4,
        max_lines=8,
        elem_classes=["vc-mono"],
    )
    chips = gr.HTML(replace_pairs_to_html_chips([]))
    with gr.Row(elem_classes=["vc-compact-row"]):
        case_insensitive = gr.Checkbox(
            value=True,
            label="Ignore case",
            info="Match uppercase and lowercase forms together.",
        )
        whole_words = gr.Checkbox(
            value=True,
            label="Whole words",
            info="Do not replace matches embedded inside longer words.",
        )
        regex = gr.Checkbox(
            value=False,
            label="Regex",
            info="Interpret each find value as a regular expression.",
        )
    def render_chips(value: str) -> str:
        return replace_pairs_to_html_chips(parse_replace_pairs(value))

    text.input(
        render_chips,
        inputs=text,
        outputs=chips,
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
        api_visibility="private",
    )
    text.change(
        render_chips,
        inputs=text,
        outputs=chips,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    return ReplaceWordsHandles(text, chips, case_insensitive, whole_words, regex)


__all__ = [
    "LogPanelHandles",
    "MediaInputHandles",
    "PresetBarHandles",
    "ProgressPanelHandles",
    "ReplaceWordsHandles",
    "action_button",
    "log_panel",
    "media_input_block",
    "preset_bar",
    "progress_panel",
    "render_progress_html",
    "replace_words_editor",
    "wire_preset_bar",
]
