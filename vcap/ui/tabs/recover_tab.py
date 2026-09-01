"""Recover registered UI settings from single-run or batch metadata."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gradio as gr

from vcap.core import gpu
from vcap.core.outputs import load_metadata
from vcap.core.paths import natural_sort_key, normalize_path
from vcap.ui.components import action_button

if TYPE_CHECKING:
    from vcap.core.registry import SettingsRegistry
    from vcap.ui.app import UiContext


_MODEL_PROMPT_KEYS = {
    "model_key",
    "vram_preset",
    "vram_reserve_gb",
    "swap_slots",
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
_ALWAYS_SKIPPED_KEYS = {"theme", "theme_mode", "outputs_dir", "temp_dir", "models_dir"}
_OPTIONAL_PATH_KEYS = {
    "input_files",
    "input_path",
    "batch_input_folder",
    "batch_output_folder",
}
_GPU_KEYS = {"gpu_index", "gpu_indices"}
_RECOVERY_KEY_MAP = {
    "compile_mode": "torch_compile_mode",
    "recursive": "batch_recursive",
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
    return load_metadata(resolve_metadata_path(value))


def resolve_metadata_path(value: str | Path) -> Path:
    """Resolve either a metadata file or a single/batch run directory."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Choose a metadata.json file or run folder")
    path = normalize_path(raw)
    if path.is_dir():
        path = path / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    return path


def extract_metadata_settings(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract flat UI settings and fill the model key from model_info."""

    settings = dict(metadata.get("settings") or {})
    model_info = metadata.get("model_info")
    if isinstance(model_info, Mapping) and "model_key" not in settings:
        variant = model_info.get("variant_key")
        if variant:
            settings["model_key"] = variant
    return settings


def _map_recovery_keys(
    settings: Mapping[str, Any],
    registry: "SettingsRegistry",
) -> dict[str, Any]:
    """Map runtime metadata aliases back to their registered UI controls."""

    mapped = dict(settings)
    registered = set(registry.keys())
    for stored_key, ui_key in _RECOVERY_KEY_MAP.items():
        if stored_key in mapped and stored_key not in registered and ui_key in registered:
            mapped.setdefault(ui_key, mapped[stored_key])
            del mapped[stored_key]
    return mapped


def present_recovery_settings(
    metadata: str | Path | Mapping[str, Any],
    registry: "SettingsRegistry",
    *,
    model_prompt_only: bool = False,
    restore_paths: bool = False,
    available_gpu_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Coerce stored settings while excluding unsafe machine and path values."""

    selected, warnings, skipped = _recovery_settings_details(
        metadata,
        registry,
        model_prompt_only=model_prompt_only,
        restore_paths=restore_paths,
        available_gpu_indices=available_gpu_indices,
    )
    if skipped:
        warnings.append(f"Skipped keys: {', '.join(skipped)}.")
    return selected, warnings


def _recovery_settings_details(
    metadata: str | Path | Mapping[str, Any],
    registry: "SettingsRegistry",
    *,
    model_prompt_only: bool = False,
    restore_paths: bool = False,
    available_gpu_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return selected values, coercion warnings, and explicitly skipped keys."""

    document = _metadata_document(metadata)
    source = _map_recovery_keys(extract_metadata_settings(document), registry)
    coerced, warnings = registry.coerce(source)
    allowed = {
        entry.key
        for entry in registry.entries()
        if not model_prompt_only
        or entry.section in {"model", "prompt"}
        or entry.key in _MODEL_PROMPT_KEYS
    }
    if available_gpu_indices is None:
        available = {int(info.index) for info in gpu.list_gpus()}
    else:
        available = {int(index) for index in available_gpu_indices}
    selected: dict[str, Any] = {}
    skipped: list[str] = []
    for key in source:
        if key not in allowed or key not in coerced:
            continue
        if key in _ALWAYS_SKIPPED_KEYS:
            skipped.append(key)
            continue
        if key in _OPTIONAL_PATH_KEYS and not restore_paths:
            skipped.append(key)
            continue
        value = coerced[key]
        if key == "gpu_index":
            try:
                index = int(value)
            except (TypeError, ValueError):
                skipped.append(key)
                continue
            if index not in available:
                skipped.append(key)
                continue
            value = index
        elif key == "gpu_indices":
            raw_indices = value if isinstance(value, (list, tuple, set)) else [value]
            valid: list[int] = []
            for raw_index in raw_indices:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if index in available and index not in valid:
                    valid.append(index)
            if raw_indices and not valid:
                skipped.append(key)
                continue
            if len(valid) != len(raw_indices):
                skipped.append(key)
            value = valid
        selected[key] = value
    return selected, warnings, list(dict.fromkeys(skipped))


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
    raw_saved = _map_recovery_keys(extract_metadata_settings(document), registry)
    saved, _ = registry.coerce(raw_saved)
    if current_values is None:
        current = registry.defaults()
    elif isinstance(current_values, Mapping):
        current = dict(current_values)
    else:
        current = registry.values_to_dict(list(current_values))
    rows: list[dict[str, Any]] = []
    for entry in registry.entries():
        current_value = current.get(entry.key, entry.default)
        saved_value = saved.get(entry.key, entry.default) if entry.key in raw_saved else current_value
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
                info="Enter metadata.json or a single/batch run folder containing it.",
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
            restore_paths = gr.Checkbox(
                value=False,
                label="Also restore input/output paths",
                info="Restore input files and batch input/output folders; app, model, temp, and theme paths remain unchanged.",
            )
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
            resolved = resolve_metadata_path(source)
            document = load_metadata(resolved)
            applicable, warnings, skipped = _recovery_settings_details(document, registry)
            current = registry.values_to_dict(current_values)
            rows = build_recovery_diff_table(document, registry, current)
            differences = sum(bool(row["different"]) for row in rows)
            warning_text = " ".join(warnings[:5])
            message = f"Loaded {resolved}. {differences} stored setting(s) differ."
            if skipped:
                message += f" Skipped keys: {', '.join(skipped)}."
            if warning_text:
                message += f" Warnings: {warning_text}"
            ctx.app_log.log(message, scope="recover")
            recovered = {
                "path": str(resolved),
                "document": document,
                "settings": applicable,
                "warnings": warnings,
                "skipped": skipped,
            }
            return _styled_diff(rows), message, recovered, gr.update(value=str(resolved))
        except Exception as exc:
            ctx.app_log.error(f"Metadata recovery load failed: {exc}", scope="recover")
            return gr.skip(), f"<span class='vc-err'>{html.escape(str(exc))}</span>", {}, gr.skip()

    load.click(
        load_handler,
        inputs=[metadata_file, metadata_path, recent, *registry_components],
        outputs=[diff_table, recovery_status, recovered_state, metadata_path],
        show_progress="minimal", api_visibility="private",
    )

    def apply_handler(
        recovered: dict[str, Any],
        model_prompt_only: bool,
        include_paths: bool,
    ) -> tuple[Any, ...]:
        if not isinstance(recovered, dict):
            return (*[gr.skip() for _ in registry_components], "<span class='vc-warn'>Load metadata first.</span>")
        document = recovered.get("document")
        if not isinstance(document, Mapping):
            stored = recovered.get("settings")
            document = {"settings": stored} if isinstance(stored, Mapping) else None
        if not isinstance(document, Mapping):
            return (*[gr.skip() for _ in registry_components], "<span class='vc-warn'>Load metadata first.</span>")
        settings, warnings, skipped = _recovery_settings_details(
            document,
            registry,
            model_prompt_only=model_prompt_only,
            restore_paths=bool(include_paths),
        )
        values: list[Any] = []
        applied = 0
        for entry in registry.entries():
            allowed = (
                not model_prompt_only
                or entry.section in {"model", "prompt"}
                or entry.key in _MODEL_PROMPT_KEYS
            )
            if allowed and entry.key in settings:
                values.append(gr.update(value=settings[entry.key]))
                applied += 1
            else:
                values.append(gr.skip())
        scope = "model/prompt " if model_prompt_only else ""
        message = f"Applied {applied} recovered {scope}setting(s) to the UI."
        if skipped:
            message += f" Skipped keys: {', '.join(skipped)}."
        if warnings:
            message += " " + " ".join(str(value) for value in warnings[:5])
        ctx.app_log.log(message, scope="recover")
        return *values, message

    apply_all.click(
        lambda recovered, include_paths: apply_handler(recovered, False, include_paths),
        inputs=[recovered_state, restore_paths],
        outputs=[*registry_components, recovery_status],
        queue=False, show_progress="hidden", api_visibility="private",
    )
    apply_model_prompt.click(
        lambda recovered, include_paths: apply_handler(recovered, True, include_paths),
        inputs=[recovered_state, restore_paths],
        outputs=[*registry_components, recovery_status],
        queue=False, show_progress="hidden", api_visibility="private",
    )


__all__ = [
    "build",
    "build_recover_diff_table",
    "build_recovery_diff_table",
    "extract_metadata_settings",
    "present_recovery_settings",
    "recent_metadata_paths",
    "resolve_metadata_path",
]
