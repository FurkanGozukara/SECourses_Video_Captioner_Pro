"""Recover registered UI settings from single-run or batch metadata."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gradio as gr

from vcap.core.outputs import load_metadata
from vcap.core.paths import natural_sort_key, normalize_path
from vcap.ui.components import action_button

if TYPE_CHECKING:
    from vcap.core.registry import SettingsRegistry
    from vcap.ui.app import UiContext


_MODEL_PROMPT_KEYS = {
    "model_key",
    "vram_preset",
    "attention_backend",
    "prompt_preset_id",
    "system_prompt",
    "user_prompt",
    "trigger_word",
    "language",
    "source_language",
    "target_language",
    "caption_length",
    "subject_class",
    "avoid_list",
    "extra_instructions",
}


def recent_metadata_paths(outputs_dir: str | Path, limit: int = 40) -> list[Path]:
    """Return recent run metadata files, newest first and path-stable on ties."""

    root = normalize_path(outputs_dir)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    try:
        for path in root.rglob("metadata.json"):
            try:
                if path.is_file():
                    found.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
    except (OSError, PermissionError):
        pass
    found.sort(key=lambda item: (-item[0], natural_sort_key(item[1]), str(item[1]).casefold()))
    return [path for _, path in found[: max(0, int(limit))]]


def _metadata_document(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return load_metadata(value)


def extract_metadata_settings(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract flat UI settings and fill the model key from model_info."""

    settings = dict(metadata.get("settings") or {})
    model_info = metadata.get("model_info")
    if isinstance(model_info, Mapping) and "model_key" not in settings:
        variant = model_info.get("variant_key")
        if variant:
            settings["model_key"] = variant
    return settings


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_recovery_diff_table(
    metadata: str | Path | Mapping[str, Any],
    registry: "SettingsRegistry",
    current_values: Mapping[str, Any] | Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a pure key/saved/current/difference table for tests and UI styling."""

    document = _metadata_document(metadata)
    saved, _ = registry.coerce(extract_metadata_settings(document))
    if current_values is None:
        current = registry.defaults()
    elif isinstance(current_values, Mapping):
        current = dict(current_values)
    else:
        current = registry.values_to_dict(list(current_values))
    rows: list[dict[str, Any]] = []
    for entry in registry.entries():
        saved_value = saved.get(entry.key, entry.default)
        current_value = current.get(entry.key, entry.default)
        rows.append(
            {
                "key": entry.key,
                "saved_value": saved_value,
                "current_value": current_value,
                "different": saved_value != current_value,
            }
        )
    return rows


build_recover_diff_table = build_recovery_diff_table


def _styled_diff(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    frame = pd.DataFrame(
        [[row["key"], _display_value(row["saved_value"]), _display_value(row["current_value"])] for row in rows],
        columns=["Key", "Saved value", "Current value"],
    )
    differences = [bool(row["different"]) for row in rows]

    def style_row(row: Any) -> list[str]:
        changed = differences[int(row.name)] if int(row.name) < len(differences) else False
        highlight = "background-color: rgba(251, 191, 36, 0.22); font-weight: 650" if changed else ""
        return ["", highlight, highlight]

    return frame.style.apply(style_row, axis=1)


def _recent_choices(outputs_dir: Path) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for path in recent_metadata_paths(outputs_dir):
        try:
            label = path.parent.relative_to(outputs_dir).as_posix()
        except ValueError:
            label = path.parent.name
        choices.append((label, str(path)))
    return choices


def build(ctx: "UiContext") -> None:
    """Render metadata selection, diff inspection, and registry-wide recovery."""

    recovered_state = gr.State({})
    choices = _recent_choices(ctx.outputs_dir)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            metadata_file = gr.File(
                label="Metadata file", file_types=[".json"], type="filepath",
            )
            metadata_path = gr.Textbox(
                value=choices[0][1] if choices else "", label="Metadata path",
                info="A local path can be used instead of uploading a file.",
            )
            recent = gr.Dropdown(
                choices=choices, value=choices[0][1] if choices else None,
                label="Recent run", info="Recent metadata discovered below the outputs directory.",
            )
            with gr.Row(elem_classes=["vc-compact-row"]):
                refresh = action_button("Refresh", "blue", size="md")
                load = action_button("Load", "cyan", size="md")
            with gr.Row(elem_classes=["vc-compact-row"]):
                apply_all = action_button("Apply to UI", "emerald", size="md")
                apply_model_prompt = action_button("Apply model + prompt only", "violet", size="md")
            recovery_status = gr.Markdown("Choose metadata to compare.", elem_classes=["vc-status"])

        with gr.Column(scale=6, min_width=520):
            diff_table = gr.Dataframe(
                value=[], headers=["Key", "Saved value", "Current value"],
                datatype=["str", "str", "str"], type="pandas", interactive=False,
                show_search="filter", max_height=620, pinned_columns=1,
                column_widths=[180, 390, 390],
                static_columns=[0, 1, 2], wrap=True, buttons=["copy", "fullscreen"],
                label="Saved settings comparison",
            )

    registry = ctx.registry
    registry_components = registry.components()

    recent.change(
        lambda value: value or "", inputs=recent, outputs=metadata_path,
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def refresh_handler() -> Any:
        updated = _recent_choices(ctx.outputs_dir)
        value = updated[0][1] if updated else None
        return gr.update(choices=updated, value=value)

    refresh.click(
        refresh_handler, outputs=recent,
        queue=False, show_progress="hidden", api_visibility="private",
    )

    def resolve_path(upload: Any, typed: str, selected: str) -> str:
        if upload:
            if isinstance(upload, Mapping):
                return str(upload.get("path") or upload.get("name") or "")
            return str(getattr(upload, "name", upload))
        return str(typed or selected or "")

    def load_handler(upload: Any, typed: str, selected: str, *current_values: Any) -> tuple[Any, str, dict[str, Any], Any]:
        source = resolve_path(upload, typed, selected)
        try:
            document = load_metadata(source)
            raw_settings = extract_metadata_settings(document)
            coerced, warnings = registry.coerce(raw_settings)
            current = registry.values_to_dict(current_values)
            rows = build_recovery_diff_table(document, registry, current)
            differences = sum(bool(row["different"]) for row in rows)
            warning_text = " ".join(warnings[:5])
            message = f"Loaded {source}. {differences} setting(s) differ."
            if warning_text:
                message += f" Warnings: {warning_text}"
            ctx.app_log.log(message, scope="recover")
            recovered = {"path": source, "settings": coerced, "warnings": warnings}
            return _styled_diff(rows), message, recovered, gr.update(value=source)
        except Exception as exc:
            ctx.app_log.error(f"Metadata recovery load failed: {exc}", scope="recover")
            return gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", {}, gr.skip()

    load.click(
        load_handler,
        inputs=[metadata_file, metadata_path, recent, *registry_components],
        outputs=[diff_table, recovery_status, recovered_state, metadata_path],
        show_progress="minimal", api_visibility="private",
    )

    def apply_handler(recovered: dict[str, Any], model_prompt_only: bool) -> tuple[Any, ...]:
        settings = recovered.get("settings") if isinstance(recovered, dict) else None
        if not isinstance(settings, dict):
            return (*[gr.skip() for _ in registry_components], "<span class='vc-warn'>Load metadata first.</span>")
        values = registry.dict_to_values(settings)
        if model_prompt_only:
            values = [
                gr.update(value=value)
                if entry.section in {"model", "prompt"} or entry.key in _MODEL_PROMPT_KEYS
                else gr.skip()
                for entry, value in zip(registry.entries(), values)
            ]
            message = "Applied recovered model and prompt settings."
        else:
            values = [gr.update(value=value) for value in values]
            message = f"Applied {len(values)} recovered settings to the UI."
        warnings = recovered.get("warnings") or []
        if warnings:
            message += " " + " ".join(str(value) for value in warnings[:5])
        ctx.app_log.log(message, scope="recover")
        return *values, message

    apply_all.click(
        lambda recovered: apply_handler(recovered, False), inputs=recovered_state,
        outputs=[*registry_components, recovery_status],
        queue=False, show_progress="hidden", api_visibility="private",
    )
    apply_model_prompt.click(
        lambda recovered: apply_handler(recovered, True), inputs=recovered_state,
        outputs=[*registry_components, recovery_status],
        queue=False, show_progress="hidden", api_visibility="private",
    )


__all__ = [
    "build",
    "build_recover_diff_table",
    "build_recovery_diff_table",
    "extract_metadata_settings",
    "recent_metadata_paths",
]
