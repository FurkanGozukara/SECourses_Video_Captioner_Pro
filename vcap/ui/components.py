"""Reusable Gradio controls shared by the application tabs."""

from __future__ import annotations

import hashlib
import html
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import gradio as gr
from gradio import processing_utils

from vcap import TEMP_DIR
from vcap.core import gpu
from vcap.core.captions_post import parse_replace_pairs, replace_pairs_to_html_chips
from vcap.core.media import MediaInfo, make_thumbnail, probe_media
from vcap.core.logs import get_log
from vcap.core.paths import (
    guess_kind_by_extension,
    list_media_files,
    normalize_path,
    sanitize_filename,
)
from vcap.core.presets import PresetError
from vcap.ui.theme import TOGGLE_ACCORDIONS_JS

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


def context_usage_text(used: Any, limit: Any) -> str:
    """Format used/total context tokens for a status line ('—' when unknown)."""

    try:
        used_value = int(used) if used is not None else 0
    except (TypeError, ValueError):
        used_value = 0
    if used_value <= 0:
        return "—"
    try:
        limit_value = int(limit) if limit is not None else 0
    except (TypeError, ValueError):
        limit_value = 0
    if limit_value > 0:
        return f"{used_value:,} / {limit_value:,} ({100.0 * used_value / limit_value:.0f}%)"
    return f"{used_value:,}"


@dataclass
class PresetBarHandles:
    dropdown: gr.Dropdown
    save_as: gr.Textbox
    save: gr.Button
    load: gr.Button
    delete: gr.Button
    delete_confirmation: gr.Row
    delete_question: gr.Markdown
    delete_yes: gr.Button
    delete_keep: gr.Button
    reset: gr.Button
    status: gr.Markdown
    load_last: gr.Button
    toggle_accordions: gr.Button


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
    # The shipped default, never the last-used marker: a launch always starts
    # from the same configuration, and Load last values restores the other one.
    initial = ctx.preset_store.default_startup_name()
    # The status line sits under the strip on its own full-width line rather
    # than squeezed into the last flex slot of an eight-control row.
    with gr.Row(elem_classes=["vc-preset-bar"]):
        dropdown = gr.Dropdown(
            choices=choices,
            value=initial,
            label="Universal preset",
            info="Selecting a preset applies it immediately. Defaults are marked with a star and are read-only.",
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
        save = action_button("💾 Save", "green", scale=1, min_width=92)
        load = action_button("📥 Load", "blue", scale=1, min_width=92)
        delete = action_button("🗑️ Delete", "rose", scale=1, min_width=96)
        reset = action_button("↺ Reset", "gold", scale=1, min_width=92)
        load_last = action_button(
            "⟲ Load Last Values",
            "teal",
            scale=2,
            min_width=186,
            elem_id="vc_load_last_values",
        )
        toggle_accordions = action_button(
            "⇕ Open / Close All",
            "indigo",
            scale=2,
            min_width=168,
            elem_id="vc_toggle_accordions",
        )
    status = gr.Markdown(
        "<span class='vc-ok'>Ready.</span>",
        elem_classes=["vc-status"],
    )
    with gr.Row(
        visible=False,
        elem_id="vc_preset_delete_confirmation",
        elem_classes=["vc-confirm-bar"],
    ) as delete_confirmation:
        delete_question = gr.Markdown("Delete the selected user preset?")
        # scale=0 sizes each button to its label so the question keeps the
        # rest of the bar instead of being squeezed into a third of it.
        delete_yes = action_button(
            "✔ Yes, delete",
            "red",
            scale=0,
            min_width=132,
            variant="stop",
            elem_id="vc_preset_delete_yes",
        )
        delete_keep = action_button(
            "✖ Keep preset",
            "slate",
            scale=0,
            min_width=140,
            elem_id="vc_preset_delete_keep",
        )
    handles = PresetBarHandles(
        dropdown=dropdown,
        save_as=save_as,
        save=save,
        load=load,
        delete=delete,
        delete_confirmation=delete_confirmation,
        delete_question=delete_question,
        delete_yes=delete_yes,
        delete_keep=delete_keep,
        reset=reset,
        status=status,
        load_last=load_last,
        toggle_accordions=toggle_accordions,
    )
    # Purely client-side: the accordion header is a plain button, so opening or
    # closing every section on the visible tab never touches the server.
    toggle_accordions.click(
        fn=None,
        inputs=None,
        outputs=[],
        js=TOGGLE_ACCORDIONS_JS,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    ctx.preset_handles = handles
    return handles


def wire_preset_bar(ctx: "UiContext", demo: gr.Blocks) -> None:
    """Wire universal preset operations after the registry is complete."""

    handles = ctx.preset_handles
    if handles is None:
        raise RuntimeError("preset_bar() must be built before it is wired")
    registry = ctx.settings_registry
    components = registry.components()
    adapters: dict[str, Callable[[dict[str, Any]], Any]] = dict(
        ctx.states.get("preset_value_adapters") or {}
    )
    # What the latest load, reset, or startup pushed into the UI: the preset
    # name guards the dropdown's change event against the programmatic updates
    # made after save/delete/startup, and the coerced settings let follow-ups
    # read the loaded values instead of component values that the browser may
    # not have applied yet.
    applied_state = gr.State({"name": "", "settings": {}})
    delete_state = gr.State({})

    def applied(name: str | None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"name": str(name or ""), "settings": dict(settings or {})}

    def skipped() -> list[Any]:
        return [gr.skip() for _ in components]

    def preset_values(settings: dict[str, Any]) -> list[Any]:
        """Apply presets only to controls that participate in preset storage.

        Tabs may register adapters (``ctx.states["preset_value_adapters"]``) for
        controls whose bounds depend on other settings; those ship the value
        together with its bounds so no handler sees a value outside them.
        """

        values = registry.dict_to_values(settings)
        result: list[Any] = []
        for entry, value in zip(registry.entries(), values):
            if not entry.in_preset:
                result.append(gr.skip())
            elif entry.key in adapters:
                result.append(adapters[entry.key](settings))
            else:
                result.append(value)
        return result

    def save_preset(name: str, *values: Any) -> tuple[Any, str, str, Any]:
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
                applied(saved),
            )
        except Exception as exc:
            return gr.skip(), str(name or ""), f"<span class='vc-err'>{html.escape(str(exc))}</span>", gr.skip()

    def load_preset(name: str) -> tuple[Any, ...]:
        try:
            if not name:
                raise PresetError("Choose a preset to load")
            loaded = ctx.preset_store.load(str(name))
            coerced, warnings = registry.coerce(loaded)
            suffix = f" ({len(warnings)} value adjustment(s))" if warnings else ""
            return (
                *preset_values(coerced),
                f"<span class='vc-ok'>Loaded {html.escape(str(name))}{suffix}</span>",
                applied(name, coerced),
            )
        except Exception as exc:
            return (*skipped(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", applied(""))

    def select_preset(name: str, state: dict[str, Any] | None) -> tuple[Any, ...]:
        """Apply a preset the moment it is picked; ignore re-selection of the applied one."""

        current = str((state or {}).get("name") or "")
        if not name or str(name) == current:
            return (*skipped(), gr.skip(), applied(current))
        return load_preset(str(name))

    def request_delete_preset(name: str) -> tuple[Any, Any, Any, str]:
        try:
            if not name:
                raise PresetError("Choose a preset to delete")
            selected = str(name)
            entry = next(
                (item for item in ctx.preset_store.list_presets() if item.name == selected),
                None,
            )
            if entry is None:
                raise PresetError(f"Preset not found: {selected}")
            if entry.is_default:
                # Delegate to the store so its established protected-preset
                # refusal remains the exact message shown to the user.
                ctx.preset_store.delete(selected)
            question = f'⚠ Delete preset "{html.escape(selected)}"?'
            return (
                {"name": selected},
                gr.update(visible=True),
                question,
                f"<span class='vc-warn'>{question}</span>",
            )
        except Exception as exc:
            return (
                {},
                gr.update(visible=False),
                gr.skip(),
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
            )

    def keep_preset(state: dict[str, Any] | None) -> tuple[dict[str, Any], Any, str]:
        selected = str((state or {}).get("name") or "")
        message = (
            f"Kept preset {html.escape(selected)}."
            if selected
            else "No preset deletion was pending."
        )
        return {}, gr.update(visible=False), f"<span class='vc-help'>{message}</span>"

    def confirm_delete_preset(state: dict[str, Any] | None) -> tuple[Any, ...]:
        selected = str((state or {}).get("name") or "")
        if not selected:
            return (
                *skipped(),
                gr.skip(),
                gr.skip(),
                "<span class='vc-warn'>No preset deletion was pending.</span>",
                gr.skip(),
                gr.update(visible=False),
                {},
            )
        try:
            if not ctx.preset_store.delete(selected):
                raise PresetError("Preset was already absent")
            default_name = str(
                getattr(ctx.preset_store, "default_preset_name", None) or ""
            ).strip()
            if not default_name:
                from vcap.ui.app import DEFAULT_PRESET_NAME

                default_name = DEFAULT_PRESET_NAME
            loaded = ctx.preset_store.load(default_name)
            coerced, _warnings = registry.coerce(loaded)
            return (
                *preset_values(coerced),
                gr.update(choices=_preset_choices(ctx), value=default_name),
                "",
                (
                    "<span class='vc-ok'>Deleted "
                    f"{html.escape(selected)}; loaded {html.escape(default_name)}.</span>"
                ),
                applied(default_name, coerced),
                gr.update(visible=False),
                {},
            )
        except Exception as exc:
            return (
                *skipped(),
                gr.update(choices=_preset_choices(ctx)),
                gr.skip(),
                f"<span class='vc-err'>{html.escape(str(exc))}</span>",
                gr.skip(),
                gr.update(visible=False),
                {},
            )

    def reset_settings() -> tuple[Any, ...]:
        defaults = registry.defaults()
        return (
            *preset_values(defaults),
            gr.update(value=None),
            "<span class='vc-ok'>Restored application defaults (defaults).</span>",
            applied("", defaults),
        )

    def apply_named_preset(name: str | None, *, remember: bool, lead: str) -> tuple[Any, ...]:
        """Apply one preset by name, falling back to application defaults."""

        if not name:
            defaults = registry.defaults()
            return (
                *preset_values(defaults),
                gr.update(choices=_preset_choices(ctx)),
                "<span class='vc-ok'>Ready with application defaults.</span>",
                applied("", defaults),
            )
        try:
            loaded = ctx.preset_store.load(name, mark_last_used=remember)
            coerced, warnings = registry.coerce(loaded)
            suffix = f"; adjusted {len(warnings)} value(s)" if warnings else ""
            return (
                *preset_values(coerced),
                gr.update(choices=_preset_choices(ctx), value=name),
                f"<span class='vc-ok'>{lead} {html.escape(name)}{suffix}.</span>",
                applied(name, coerced),
            )
        except Exception as exc:
            defaults = registry.defaults()
            return (
                *preset_values(defaults),
                gr.update(choices=_preset_choices(ctx)),
                f"<span class='vc-warn'>{html.escape(name)} could not load: {html.escape(str(exc))}</span>",
                applied("", defaults),
            )

    def startup_preset() -> tuple[Any, ...]:
        """Launch with the shipped default so every start looks the same.

        The last-used marker is neither read nor written here; restoring it is
        what the Load last values button is for.
        """

        return apply_named_preset(
            ctx.preset_store.default_startup_name(),
            remember=False,
            lead="Started with",
        )

    def load_last_values() -> tuple[Any, ...]:
        """Apply the preset this browser profile used last, on request."""

        remembered = ctx.preset_store.get_last_used()
        return apply_named_preset(
            remembered or ctx.preset_store.default_startup_name(),
            remember=True,
            lead="Loaded last used:" if remembered else "No preset used yet; loaded",
        )

    ctx.states["preset_bar_handlers"] = {
        "confirm_delete": confirm_delete_preset,
        "reset": reset_settings,
        "component_count": len(components),
    }

    handles.save.click(
        save_preset,
        inputs=[handles.save_as, *components],
        outputs=[handles.dropdown, handles.save_as, handles.status, applied_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    load_event = handles.load.click(
        load_preset,
        inputs=handles.dropdown,
        outputs=[*components, handles.status, applied_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    # Selecting a preset applies it at once; Load re-applies the selection after
    # manual edits. Gradio 6 dropdowns fire ``input`` on every blur (twice per
    # mouse pick), so the binding uses ``change`` guarded by the applied name.
    select_event = handles.dropdown.change(
        select_preset,
        inputs=[handles.dropdown, applied_state],
        outputs=[*components, handles.status, applied_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.delete.click(
        request_delete_preset,
        inputs=handles.dropdown,
        outputs=[delete_state, handles.delete_confirmation, handles.delete_question, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    delete_confirm_event = handles.delete_yes.click(
        confirm_delete_preset,
        inputs=delete_state,
        outputs=[
            *components,
            handles.dropdown,
            handles.save_as,
            handles.status,
            applied_state,
            handles.delete_confirmation,
            delete_state,
        ],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    handles.delete_keep.click(
        keep_preset,
        inputs=delete_state,
        outputs=[delete_state, handles.delete_confirmation, handles.status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    reset_event = handles.reset.click(
        reset_settings,
        outputs=[*components, handles.dropdown, handles.status, applied_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    startup_event = demo.load(
        startup_preset,
        outputs=[*components, handles.dropdown, handles.status, applied_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    load_last_event = handles.load_last.click(
        load_last_values,
        outputs=[*components, handles.dropdown, handles.status, applied_state],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    auto_vram = ctx.states.get("caption_auto_vram_binding")
    if isinstance(auto_vram, dict):
        input_keys = list(auto_vram.get("input_keys") or [])
        output_count = len(auto_vram["outputs"])

        def follow_up(state: dict[str, Any] | None, *values: Any) -> tuple[Any, ...]:
            """Run the tier plan on the settings that were just applied, or skip."""

            settings = dict((state or {}).get("settings") or {})
            if not settings:
                return tuple(gr.skip() for _ in range(output_count))
            keys = input_keys + [None] * (len(values) - len(input_keys))
            resolved = [
                settings[key] if key and key in settings else value
                for key, value in zip(keys, values)
            ]
            return tuple(auto_vram["fn"](*resolved))

        for dependency in (
            load_event,
            select_event,
            reset_event,
            startup_event,
            load_last_event,
            delete_confirm_event,
        ):
            dependency.then(
                follow_up,
                inputs=[applied_state, *auto_vram["inputs"]],
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
    zip_upload: gr.File
    output_folder: gr.Textbox
    recursive: gr.Checkbox
    overwrite: gr.Checkbox
    limit_items: gr.Number
    include_kinds: gr.CheckboxGroup
    name_filter: gr.Textbox
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
    save_next_to_source: Any = None
    existing_extension_state: Any = None
    scan_fn: Callable[..., tuple[Any, ...]] | None = None
    scan_inputs: list[Any] | None = None
    scan_outputs: list[Any] | None = None


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
        raw = str(info.error or "unknown ffprobe error")
        get_log().warn(f"{info.path.name}: {raw}", scope="inputs")
        return (
            "<span class='vc-warn'>1 unreadable file: "
            f"{html.escape(info.path.name)} (skipped)</span>"
        )
    audio = (
        f"{info.audio_codec or 'audio'}, {info.audio_sample_rate or '?'} Hz, {info.audio_channels or '?'} ch"
        if info.has_audio
        else "no audio"
    )
    codec = info.video_codec or info.audio_codec or info.container or "n/a"
    if info.kind == "audio":
        return (
            f"**{html.escape(info.path.name)}** · audio · {_duration(info.duration)} · "
            f"{html.escape(audio)} · codec `{html.escape(codec)}`"
        )
    # ASCII separators keep compact dimensions visually unambiguous.
    geometry = f"{int(info.width)} x {int(info.height)}" if info.width and info.height else "n/a"
    frame_rate = f"{info.fps:.3g} fps" if info.fps else "n/a"
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
            "<span class='vc-help'>No inputs selected.</span>",
            _input_gallery([]),
            "unknown",
            0.0,
        )
    info = probe_media(paths[0])
    video = gr.update(value=None, visible=False)
    audio = gr.update(value=None, visible=False)
    image = gr.update(value=None, visible=False)
    details = _media_info_markdown(info)
    modality = "video_audio" if info.kind == "video" else "video" if info.kind == "video_no_audio" else info.kind
    if info.kind in {"video", "video_no_audio"}:
        try:
            playable = bool(processing_utils.video_is_playable(str(info.path)))
        except Exception:
            playable = False
        if playable:
            video = gr.update(value=str(info.path), visible=True)
        else:
            container = info.container or info.path.suffix.lstrip(".") or "unknown"
            video_codec = info.video_codec or "unknown video"
            audio_codec = info.audio_codec or "no audio"
            preview_note = (
                "Preview shows the first frame: "
                f"{container}/{video_codec}/{audio_codec} is not browser-playable. "
                "Trim range still works."
            )
            details += f"<br><span class='vc-help'>{html.escape(preview_note)}</span>"
            try:
                stat = info.path.stat()
                identity = f"{info.path.resolve(strict=False)}\0{stat.st_size}\0{stat.st_mtime_ns}"
                digest = hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()[:28]
                poster = normalize_path(TEMP_DIR / "preview_posters" / f"{digest}.png")
                if not poster.is_file() or not poster.stat().st_size:
                    poster = make_thumbnail(info.path, poster, at_seconds=0.0, width=960)
                image = gr.update(value=str(poster), visible=True)
            except Exception as exc:
                details += (
                    "<br><span class='vc-warn'>First-frame preview unavailable: "
                    f"{html.escape(str(exc))}</span>"
                )
    elif info.kind == "audio":
        audio = gr.update(value=str(info.path), visible=True)
    elif info.kind == "image":
        image = gr.update(value=str(info.path), visible=True)
    return video, audio, image, details, _input_gallery(paths), modality, float(info.duration or 0.0)


def _folder_scan(
    folder: str,
    recursive: bool,
    output_folder: str = "",
    overwrite: bool = False,
    limit_items: int | float = 0,
    include_kinds: Iterable[str] | None = None,
    name_filter: str = "",
    allowed_kinds: Iterable[str] = ("video", "audio", "image", "text"),
    existing_extension: str = ".txt",
    save_next_to_source: bool = False,
    existing_item_noun: str = "captioned",
    existing_files_label: str = "captions",
    include_caption_coverage: bool = False,
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
    selected_kinds = tuple(
        value
        for value in (str(item).casefold() for item in allowed_kinds)
        if value in {"video", "audio", "image", "text"}
    ) or ("video", "audio", "image", "text")
    found = list_media_files(
        root,
        recursive=bool(recursive),
        kinds=selected_kinds,
    )
    try:
        # Task A owns this helper. The local import keeps the UI buildable while
        # the backend branch is still landing in the shared working tree.
        from vcap.core.media import filter_media_paths
    except ImportError:
        pass
    else:
        found = filter_media_paths(
            found,
            (
                list(selected_kinds)
                if include_kinds is None
                else list(include_kinds)
            ),
            str(name_filter or ""),
        )
    try:
        output_root = normalize_path(output_folder) if str(output_folder or "").strip() else None
    except Exception as exc:
        return [], f"<span class='vc-err'>Invalid output folder: {html.escape(str(exc))}</span>"
    counts = {"video": 0, "audio": 0, "image": 0, "text": 0}
    existing = 0
    audio_captioned = 0
    extension = str(existing_extension or ".txt").strip().casefold()
    if not extension.startswith("."):
        extension = "." + extension
    for path in found:
        kind = guess_kind_by_extension(path)
        if kind in counts:
            counts[kind] += 1
        relative = path.relative_to(root)
        caption_path = (
            path.with_suffix(extension)
            if save_next_to_source
            else output_root / relative.with_suffix(extension)
            if output_root is not None
            else None
        )
        if caption_path is not None and caption_path.is_file():
            existing += 1
        if include_caption_coverage:
            caption_parent = (
                path.parent
                if save_next_to_source
                else output_root / relative.parent
                if output_root is not None
                else None
            )
            if (
                caption_parent is not None
                and (caption_parent / "audio_caption" / f"{path.stem}.txt").is_file()
            ):
                audio_captioned += 1
    overwrite_hint = (
        f"Overwrite is on; existing {existing_files_label} will be replaced."
        if overwrite
        else f"Overwrite is off; existing {existing_files_label} will be skipped."
    )
    count_labels = {
        "video": f"🎬 {counts['video']} videos",
        "audio": f"🎵 {counts['audio']} audios",
        "image": f"🖼️ {counts['image']} images",
        "text": f"📄 {counts['text']} texts",
    }
    summary = " · ".join(count_labels[kind] for kind in selected_kinds)
    if include_caption_coverage:
        summary += (
            f" · {len(found)} media files · {existing} already captioned (<stem>.txt)"
            f" · {audio_captioned} with audio captions (audio_caption/)"
        )
    location = "next to source files" if save_next_to_source else "in output folder"
    if include_caption_coverage:
        summary += f" · outputs {location}. {overwrite_hint}"
    else:
        summary += f" · {existing} already {existing_item_noun} {location}. {overwrite_hint}"
    limit = max(0, int(limit_items or 0))
    if limit:
        summary += f" · limiting to first {limit}"
    return [str(path) for path in found], summary


def input_mode_from_tab(event: gr.SelectData) -> str:
    """Map a Gradio Tabs selection payload to the authoritative input mode."""

    values = [getattr(event, "value", event), getattr(event, "index", None)]
    direct = {"upload": "upload", "path": "path", "folder": "folder"}
    for value in values:
        normalized = str(value or "").strip().casefold()
        if normalized in direct:
            return direct[normalized]
        if "upload" in normalized:
            return "upload"
        if "file path" in normalized or normalized.endswith("path"):
            return "path"
        if "folder" in normalized:
            return "folder"
    index = getattr(event, "index", None)
    if isinstance(index, int) and 0 <= index <= 2:
        return ("upload", "path", "folder")[index]
    return "upload"


def _descend_single_top_level_folder(root: Path, max_depth: int = 5) -> Path:
    """Descend through archive wrapper folders without hiding mixed roots."""

    selected = normalize_path(root)
    for _ in range(max(0, int(max_depth))):
        try:
            entries = list(selected.iterdir())
        except OSError:
            break
        if len(entries) != 1 or not entries[0].is_dir():
            break
        selected = normalize_path(entries[0])
    return selected


def _filtered_light_media(
    root: Path,
    *,
    recursive: bool,
    include_kinds: Iterable[str] | None,
    name_filter: str,
) -> list[Path]:
    found = list_media_files(
        root,
        recursive=recursive,
        kinds=("video", "audio", "image", "text"),
    )
    try:
        from vcap.core.media import filter_media_paths
    except ImportError:
        return found
    return filter_media_paths(
        found,
        (
            ["video", "audio", "image", "text"]
            if include_kinds is None
            else list(include_kinds)
        ),
        str(name_filter or ""),
    )


def extracted_batch_folder(
    extraction_root: str | os.PathLike[str],
    include_kinds: Iterable[str] | None = None,
    name_filter: str = "",
) -> tuple[Path, bool]:
    """Choose an archive's useful root and whether media require recursion."""

    selected = _descend_single_top_level_folder(normalize_path(extraction_root))
    top_level = _filtered_light_media(
        selected,
        recursive=False,
        include_kinds=include_kinds,
        name_filter=name_filter,
    )
    recursive = _filtered_light_media(
        selected,
        recursive=True,
        include_kinds=include_kinds,
        name_filter=name_filter,
    )
    return selected, not top_level and bool(recursive)


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
        try:
            cached_path = normalize_path(resolved)
            cached_path.relative_to(normalize_path(TEMP_DIR / "preview_posters"))
            return selected
        except (OSError, TypeError, ValueError):
            pass
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


def media_input_block(
    ctx: "UiContext",
    *,
    registry_keys: Mapping[str, str] | None = None,
    state_prefix: str = "",
    allowed_kinds: Iterable[str] = ("video", "audio", "image", "text"),
    settings_section: str | None = None,
    output_folder_default: str | os.PathLike[str] | None = None,
    output_folder_registry_default: str | None = None,
    output_folder_label: str = "Batch output folder",
    output_folder_info: str = "Input names and relative folders are mirrored here.",
    overwrite_label: str = "Overwrite existing captions",
    overwrite_info: str = "Off skips media that already has the requested output.",
    limit_info: str = "Process only the first N items that are not already captioned.",
    folder_placeholder: str = "Folder containing videos, audio, or images",
    upload_description: str = "Uploaded video, audio, image, or text files.",
    show_archive_upload: bool = True,
    show_kind_filters: bool = True,
    save_next_to_source_key: str | None = None,
    save_next_to_source_label: str = "Save transcripts next to the source files",
    save_next_to_source_info: str = (
        "Write each transcript beside its source instead of mirroring it below the batch output folder."
    ),
    include_caption_coverage: bool = False,
    default_existing_extension: str = ".txt",
    existing_item_noun: str = "captioned",
    existing_files_label: str = "captions",
    input_tabs_elem_id: str = "vc-input-tabs",
) -> MediaInputHandles:
    """Build the shared upload/path/folder surface with optional namespacing.

    Caption keeps the historical defaults. Other tabs can restrict media kinds
    and provide their own registry/state keys without duplicating preview and
    folder-scan behaviour.
    """

    keys = dict(registry_keys or {})

    def setting_key(name: str) -> str:
        return str(keys.get(name) or name)

    def state_key(name: str) -> str:
        return f"{state_prefix}{name}" if state_prefix else name

    section = str(settings_section or "input")
    selected_kinds = tuple(
        value
        for value in (str(item).casefold() for item in allowed_kinds)
        if value in {"video", "audio", "image", "text"}
    ) or ("video", "audio", "image", "text")

    resolved_state = gr.State([])
    mode_state = gr.State("upload")
    modality_state = gr.State("video_audio")
    duration_state = gr.State(0.0)
    existing_extension_state = gr.State(str(default_existing_extension or ".txt"))
    ctx.states.update(
        {
            state_key("resolved_inputs"): resolved_state,
            state_key("input_mode"): mode_state,
            state_key("input_modality"): modality_state,
            state_key("input_duration"): duration_state,
        }
    )

    with gr.Column():
        gr.Markdown("### Input")
        with gr.Tabs(selected="upload", elem_id=input_tabs_elem_id) as input_tabs:
            with gr.Tab("📤 Upload files", id="upload") as upload_tab:
                # Inside a group Gradio draws the drop zone with its bold dashed
                # outline, the same look as the reference-voice field in IndexTTS.
                with gr.Group():
                    files = gr.File(
                        file_count="multiple",
                        file_types=[".txt" if kind == "text" else kind for kind in selected_kinds],
                        type="filepath",
                        label="Files",
                        height=118,
                    )
                    gr.Markdown(upload_description, elem_classes=["vc-help"])
                ctx.reg(
                    setting_key("input_files"),
                    files,
                    [],
                    section=section,
                    description=upload_description,
                    in_preset=False,
                    kind="list",
                )
            with gr.Tab("📄 File path", id="path") as path_tab:
                path = gr.Textbox(
                    label="File path",
                    placeholder="Paste a file path — same as uploading",
                    info="Quoted, mixed-separator, and non-ASCII paths are supported.",
                )
                ctx.reg(
                    setting_key("input_path"),
                    path,
                    "",
                    section=section,
                    description="Local path treated exactly like an uploaded file.",
                    in_preset=False,
                    kind="str",
                )
            with gr.Tab("📁 Folder batch", id="folder") as folder_tab:
                folder = gr.Textbox(
                    label="Input folder",
                    placeholder=folder_placeholder,
                    info="The light scan reads filenames only; media is probed during processing.",
                )
                ctx.reg(
                    setting_key("batch_input_folder"),
                    folder,
                    "",
                    section=section,
                    description="Batch source folder.",
                    in_preset=False,
                    kind="str",
                )
                if show_archive_upload:
                    with gr.Group():
                        zip_upload = gr.File(
                            label="…or upload a ZIP archive of media",
                            file_types=[".zip"],
                            type="filepath",
                            elem_id="vc_batch_zip_upload",
                        )
                    gr.Markdown(
                        "The archive is extracted safely below Outputs/uploaded_batches, then scanned with "
                        "the folder options below.",
                        elem_classes=["vc-help"],
                    )
                else:
                    zip_upload = gr.State(None)
                default_output = str(output_folder_default or (ctx.outputs_dir / "batch_captions"))
                output_folder = gr.Textbox(
                    value=default_output,
                    label=output_folder_label,
                    info=output_folder_info,
                )
                ctx.reg(
                    setting_key("batch_output_folder"),
                    output_folder,
                    default_output
                    if output_folder_registry_default is None
                    else output_folder_registry_default,
                    section=section,
                    description=output_folder_info,
                    in_preset=False,
                    kind="str",
                )
                if show_kind_filters:
                    with gr.Row():
                        kind_choices = [(kind.title(), kind) for kind in selected_kinds]
                        include_kinds = gr.CheckboxGroup(
                            choices=kind_choices,
                            value=list(selected_kinds),
                            label="Include media kinds",
                            info="Media kinds included when scanning the batch folder.",
                            scale=3,
                            elem_id="vc_batch_include_kinds",
                        )
                        ctx.reg(
                            setting_key("batch_include_kinds"),
                            include_kinds,
                            list(selected_kinds),
                            section=section,
                            description="Media kinds included when scanning the batch folder.",
                            kind="list",
                            choices=list(selected_kinds),
                        )
                        name_filter = gr.Textbox(
                            value="",
                            label="File name filter",
                            placeholder="*.mp4;clip_*",
                            info=(
                                "Optional glob on file names, for example *.mp4 or clip_*; "
                                "separate several patterns with ;. Empty includes every file."
                            ),
                            scale=3,
                            elem_id="vc_batch_name_filter",
                        )
                        ctx.reg(
                            setting_key("batch_name_filter"),
                            name_filter,
                            "",
                            section=section,
                            description=(
                                "Optional glob on file names, for example *.mp4 or clip_*; "
                                "separate several patterns with ;. Empty includes every file."
                            ),
                            kind="str",
                        )
                else:
                    include_kinds = gr.State(list(selected_kinds))
                    name_filter = gr.State("")
                with gr.Row():
                    recursive = gr.Checkbox(
                        value=False,
                        label="Scan subfolders",
                        info="Include supported files below the selected folder.",
                    )
                    ctx.reg(
                        setting_key("batch_recursive"),
                        recursive,
                        False,
                        section=section,
                        description="Recursively scan the batch input folder.",
                        kind="bool",
                    )
                    overwrite = gr.Checkbox(
                        value=False,
                        label=overwrite_label,
                        info=overwrite_info,
                    )
                    ctx.reg(
                        setting_key("overwrite_existing"),
                        overwrite,
                        False,
                        section=section if settings_section else "output",
                        description=overwrite_info,
                        kind="bool",
                    )
                    limit_items = gr.Number(
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                        label="Limit items (0 = all)",
                        info=limit_info,
                    )
                    ctx.reg(
                        setting_key("batch_limit_items"),
                        limit_items,
                        0,
                        section=section,
                        description=limit_info,
                        kind="int",
                        minimum=0,
                    )
                    rescan = action_button("↻ Rescan", "cyan")
                save_next_to_source: Any = None
                if save_next_to_source_key:
                    save_next_to_source = gr.Checkbox(
                        value=False,
                        label=save_next_to_source_label,
                        info=save_next_to_source_info,
                    )
                    ctx.reg(
                        save_next_to_source_key,
                        save_next_to_source,
                        False,
                        section=section,
                        description=save_next_to_source_info,
                        kind="bool",
                    )
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
            )
            audio = gr.Audio(
                label="Audio preview and trim",
                type="filepath",
                format=None,
                interactive=True,
                editable=True,
                visible=False,
                buttons=["download"],
            )
            image = gr.Image(
                label="Image preview",
                type="filepath",
                interactive=True,
                visible=False,
                height=390,
                buttons=["download", "fullscreen"],
            )
        info = gr.Markdown(
            "<span class='vc-help'>Choose a file to see its media details.</span>",
            elem_classes=["vc-status"],
        )
        gallery = gr.HTML(_input_gallery([]))

    preview_outputs = [video, audio, image, info, gallery, modality_state, duration_state]

    # Gradio can finish a light folder scan after another tab has already been
    # selected. This server-side generation guard prevents that late callback
    # from replacing the newly active tab's preview and resolved-input state.
    mode_guard = {"mode": "upload", "revision": 0}
    mode_guard_lock = threading.Lock()

    def activate_mode(selected_mode: str) -> int:
        with mode_guard_lock:
            mode_guard["mode"] = selected_mode
            mode_guard["revision"] += 1
            return int(mode_guard["revision"])

    def mode_snapshot() -> tuple[str, int]:
        with mode_guard_lock:
            return str(mode_guard["mode"]), int(mode_guard["revision"])

    def choose_upload(value: Any) -> tuple[Any, ...]:
        activate_mode("upload")
        selected = _paths(value)
        return (*_preview_updates(selected), selected)

    def choose_path(value: str) -> tuple[Any, ...]:
        activate_mode("path")
        selected = _paths(value)
        return (*_preview_updates(selected), selected)

    def scan_folder(
        value: str,
        recursive_value: bool,
        output_value: str,
        overwrite_value: bool,
        limit_value: int | float,
        kinds_value: list[str] | None,
        name_filter_value: str,
        existing_extension_value: str = ".txt",
        save_next_value: bool = False,
    ) -> tuple[Any, ...]:
        active_mode, revision = mode_snapshot()
        if active_mode != "folder":
            return tuple(gr.skip() for _ in folder_outputs)
        selected, summary = _folder_scan(
            value,
            recursive_value,
            output_value,
            overwrite_value,
            limit_value,
            kinds_value,
            name_filter_value,
            selected_kinds,
            existing_extension_value,
            save_next_value,
            existing_item_noun,
            existing_files_label,
            include_caption_coverage,
        )
        if mode_snapshot() != ("folder", revision):
            return tuple(gr.skip() for _ in folder_outputs)
        return (*_preview_updates(selected), selected, summary)

    def select_input_mode(event: gr.SelectData) -> str:
        selected_mode = input_mode_from_tab(event)
        activate_mode(selected_mode)
        return selected_mode

    def choose_folder_tab(*values: Any) -> tuple[Any, ...]:
        activate_mode("folder")
        return scan_folder(*values)

    input_tabs.select(
        select_input_mode,
        outputs=mode_state,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    upload_outputs = [*preview_outputs, resolved_state]
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
    folder_outputs = [*preview_outputs, resolved_state, scan_summary]
    folder_inputs = [
        folder,
        recursive,
        output_folder,
        overwrite,
        limit_items,
        include_kinds,
        name_filter,
        existing_extension_state,
    ]
    if save_next_to_source is not None:
        folder_inputs.append(save_next_to_source)
    folder_tab.select(
        choose_folder_tab,
        inputs=folder_inputs,
        outputs=folder_outputs,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    scan_triggers = [
        folder.change,
        recursive.change,
        output_folder.change,
        overwrite.change,
        limit_items.change,
        rescan.click,
    ]
    if show_kind_filters:
        scan_triggers.extend([include_kinds.change, name_filter.change])
    if save_next_to_source is not None:
        scan_triggers.append(save_next_to_source.change)
        save_next_to_source.change(
            lambda enabled: gr.update(
                interactive=not bool(enabled),
                info=(
                    "Disabled while outputs are saved beside each source file."
                    if enabled
                    else output_folder_info
                ),
            ),
            inputs=save_next_to_source,
            outputs=output_folder,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    for trigger in scan_triggers:
        trigger(
            scan_folder,
            inputs=folder_inputs,
            outputs=folder_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    def extract_batch_zip(
        uploaded: Any,
        recursive_value: bool,
        output_value: str,
        overwrite_value: bool,
        limit_value: int | float,
        kinds_value: list[str] | None,
        name_filter_value: str,
        save_next_value: bool = False,
    ) -> tuple[Any, ...]:
        activate_mode("folder")
        raw = getattr(uploaded, "name", uploaded)
        if not raw:
            return (
                gr.skip(),
                gr.skip(),
                *[gr.skip() for _ in preview_outputs],
                gr.skip(),
                "<span class='vc-help'>Choose a ZIP archive to extract.</span>",
            )
        try:
            # Task F1 owns the extractor. Keep the UI importable while that
            # backend update is landing in the shared working tree.
            from vcap.core.archive import extract_zip
        except ImportError:
            return (
                gr.skip(),
                gr.skip(),
                *[gr.skip() for _ in preview_outputs],
                gr.skip(),
                "<span class='vc-warn'>ZIP upload becomes available after the backend update.</span>",
            )
        try:
            source = normalize_path(str(raw), must_exist=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            stem = sanitize_filename(source.stem) or "batch"
            destination = normalize_path(
                ctx.outputs_dir / "uploaded_batches" / f"{stem}_{stamp}"
            )
            suffix = 2
            while destination.exists():
                destination = normalize_path(
                    ctx.outputs_dir
                    / "uploaded_batches"
                    / f"{stem}_{stamp}_{suffix}"
                )
                suffix += 1
            report = extract_zip(source, destination)
            report_destination = normalize_path(
                getattr(report, "destination", destination)
            )
            selected_folder, nested_media_only = extracted_batch_folder(
                report_destination,
                kinds_value,
                name_filter_value,
            )
            enabled_for_upload = nested_media_only and not bool(recursive_value)
            effective_recursive = bool(recursive_value) or nested_media_only
            selected, scan_text = _folder_scan(
                str(selected_folder),
                effective_recursive,
                output_value,
                overwrite_value,
                limit_value,
                kinds_value,
                name_filter_value,
                save_next_to_source=save_next_value,
                include_caption_coverage=include_caption_coverage,
            )
            files = int(getattr(report, "files", 0) or 0)
            total_bytes = int(getattr(report, "total_bytes", 0) or 0)
            skipped = [str(value) for value in (getattr(report, "skipped", []) or [])]
            skipped_text = ""
            if skipped:
                preview = ", ".join(html.escape(value) for value in skipped[:4])
                extra = f" (+{len(skipped) - 4} more)" if len(skipped) > 4 else ""
                skipped_text = (
                    f" Skipped {len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'}: "
                    f"{preview}{extra}."
                )
            extraction_line = (
                f"Extracted {files} files ({total_bytes / (1024 ** 2):.2f} MB); "
                f"using {html.escape(str(selected_folder))}"
            )
            if enabled_for_upload:
                extraction_line += (
                    "; Scan subfolders enabled because the media sit in subfolders."
                )
            else:
                extraction_line += "."
            message = f"<span class='vc-ok'>{extraction_line}{skipped_text}</span><br>{scan_text}"
            return (
                str(selected_folder),
                gr.update(value=effective_recursive),
                *_preview_updates(selected),
                selected,
                message,
            )
        except Exception as exc:
            return (
                gr.skip(),
                gr.skip(),
                *[gr.skip() for _ in preview_outputs],
                gr.skip(),
                f"<span class='vc-err'>Could not extract ZIP: {html.escape(str(exc))}</span>",
            )

    if show_archive_upload:
        zip_upload.upload(
            extract_batch_zip,
            inputs=[
                zip_upload,
                recursive,
                output_folder,
                overwrite,
                limit_items,
                include_kinds,
                name_filter,
                *([save_next_to_source] if save_next_to_source is not None else []),
            ],
            outputs=[folder, recursive, *folder_outputs],
            show_progress="minimal",
            api_visibility="private",
        )

    ctx.states[state_key("media_mode_guard")] = mode_guard
    ctx.states[state_key("media_mode_handlers")] = {
        "upload": choose_upload,
        "path": choose_path,
        "folder": choose_folder_tab,
        "select": select_input_mode,
    }

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
        zip_upload,
        output_folder,
        recursive,
        overwrite,
        limit_items,
        include_kinds,
        name_filter,
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
        save_next_to_source,
        existing_extension_state,
        scan_folder,
        folder_inputs,
        folder_outputs,
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


def poll_log_value(app_log: Any, cursor: int, current: str) -> tuple[Any, Any]:
    """Return one live-log poll update, recovering an invalid future cursor."""

    cursor_value = max(0, int(cursor or 0))
    lines, new_revision, cursor_reset = app_log.snapshot_for_poll(
        cursor_value,
        recovery_limit=300,
    )
    if cursor_reset:
        return newest_first("\n".join(lines)), new_revision
    if new_revision == cursor_value:
        return gr.skip(), gr.skip()
    return merge_log_newest_first(str(current or ""), lines), new_revision


def log_panel(ctx: "UiContext") -> LogPanelHandles:
    """Build revision-polled live logs and a Torch-free resource meter."""

    initial_lines, initial_revision = ctx.app_log.tail_snapshot(300)
    with gr.Column():
        with gr.Row():
            gr.Markdown("### Live log")
            # scale=0 keeps the button at its label width instead of taking the
            # half of the row that Gradio's default flex share would hand it.
            clear = action_button("⌫ Clear", "orange", scale=0, min_width=112)
        log = gr.Textbox(
            value=newest_first("\n".join(initial_lines)),
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
        gr.Markdown(
            "Updates pause while this browser tab is hidden and catch up when it is visible again.",
            elem_classes=["vc-help"],
        )
    revision = gr.State(initial_revision)
    log_timer = gr.Timer(0.5)
    meter_timer = gr.Timer(1.0)

    def poll_log(cursor: int, current: str) -> tuple[Any, Any]:
        return poll_log_value(ctx.app_log, cursor, current)

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


def progress_panel(
    ctx: "UiContext",
    *,
    waiting_detail: str = "Waiting for a caption job.",
    throughput_text: str = "**Speed:** — · **Context:** —",
) -> ProgressPanelHandles:
    """Build stable progress, status, ETA, and throughput outputs."""

    del ctx
    with gr.Column():
        bars = gr.HTML(render_progress_html(0.0, "Ready", waiting_detail))
        with gr.Row():
            status = gr.Markdown("**Status:** Ready", scale=5)
            eta = gr.Markdown("**ETA:** —", scale=2)
            tokens = gr.Markdown(throughput_text, scale=2)
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
    with gr.Row():
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
    "extracted_batch_folder",
    "input_mode_from_tab",
    "log_panel",
    "media_input_block",
    "poll_log_value",
    "preset_bar",
    "progress_panel",
    "render_progress_html",
    "replace_words_editor",
    "wire_preset_bar",
]
