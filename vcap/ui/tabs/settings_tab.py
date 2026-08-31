"""Global paths and lightweight application preferences."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import gradio as gr

from vcap.core.app_settings import APP_SETTINGS_PATH, load_app_settings, save_app_settings
from vcap.core.paths import normalize_path
from vcap.ui.components import action_button
from vcap.ui.theme import THEME_CHANGE_JS

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


def build(ctx: "UiContext") -> None:
    """Render and register global application settings."""

    persisted = load_app_settings(APP_SETTINGS_PATH)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=360):
            gr.Markdown("### Storage")
            outputs_dir = gr.Textbox(
                value=str(ctx.outputs_dir),
                label="Outputs directory",
                info="Applies immediately to single runs and batch run metadata after saving.",
            )
            ctx.reg(
                "outputs_dir",
                outputs_dir,
                str(ctx.outputs_dir),
                section="global",
                description="Root directory for caption outputs and run metadata.",
                in_preset=False,
                kind="str",
            )
            temp_dir = gr.Textbox(
                value=str(ctx.temp_dir),
                label="Temporary directory",
                info="Takes effect after restart so Gradio, previews, and workers use the new path.",
            )
            ctx.reg(
                "temp_dir",
                temp_dir,
                str(ctx.temp_dir),
                section="global",
                description="Temporary media, preview, and compile-cache directory.",
                in_preset=False,
                kind="str",
            )
            models_dir = gr.Textbox(
                value=str(ctx.models_dir),
                label="Models directory",
                info="Takes effect after restart so model discovery uses the new path.",
            )
            ctx.reg(
                "models_dir",
                models_dir,
                str(ctx.models_dir),
                section="global",
                description="Directory containing downloaded model variants.",
                in_preset=False,
                kind="str",
            )

        with gr.Column(scale=4, min_width=340):
            gr.Markdown("### Experience")
            theme_mode = gr.Radio(
                choices=[
                    ("Dark (default)", "dark"),
                    ("Light", "light"),
                    ("System (follow OS)", "system"),
                ],
                value="dark",
                label="Theme",
                info="Stored in this browser; System follows live operating-system color changes.",
            )
            ctx.reg(
                "theme_mode",
                theme_mode,
                "dark",
                section="global",
                description="Browser color theme preference.",
                in_preset=False,
                in_metadata=False,
                choices=["dark", "light", "system"],
                kind="str",
            )
            save_processed = gr.Checkbox(
                value=bool(persisted.get("save_processed_files", False)),
                label="Save every processed file",
                info="Keep normalized clips, extracted media, and other intermediate artifacts.",
            )
            ctx.reg(
                "save_processed_files",
                save_processed,
                bool(persisted.get("save_processed_files", False)),
                section="global",
                description="Persist intermediate processed files instead of deleting them.",
                kind="bool",
            )
            scan_subfolders = gr.Checkbox(
                value=bool(persisted.get("scan_subfolders", False)),
                label="Scan subfolders by default",
                info="Default recursive-scan preference for folder tools.",
            )
            ctx.reg(
                "scan_subfolders",
                scan_subfolders,
                bool(persisted.get("scan_subfolders", False)),
                section="global",
                description="Default recursive folder scanning preference.",
                kind="bool",
            )
            gr.Markdown(
                "**Telemetry-free.** Captioning, presets, logs, and model checks stay on this machine. "
                "Gradio analytics are disabled by the application entry point.",
                elem_classes=["vc-card"],
            )

    gr.Markdown(
        "Outputs apply immediately after saving. Temporary and model directories take effect after restart.",
        elem_classes=["vc-help"],
    )
    with gr.Row(elem_classes=["vc-compact-row"]):
        save_global = action_button("💾 Save global settings", "fuchsia", size="lg", scale=2)
        save_status = gr.Markdown(
            "<span class='vc-help'>Global settings have not changed.</span>",
            elem_classes=["vc-status"],
            scale=5,
            min_width=320,
        )

    def save_globals(
        outputs_value: str,
        temp_value: str,
        models_value: str,
        save_files: bool,
        recursive: bool,
    ) -> tuple[Any, ...]:
        try:
            if not all(str(value or "").strip() for value in (outputs_value, temp_value, models_value)):
                raise ValueError("Outputs, temporary, and models directories cannot be empty")
            normalized_outputs = normalize_path(outputs_value)
            normalized_temp = normalize_path(temp_value)
            normalized_models = normalize_path(models_value)
            normalized_outputs.mkdir(parents=True, exist_ok=True)
            target = save_app_settings(
                {
                    "outputs_dir": normalized_outputs,
                    "temp_dir": normalized_temp,
                    "models_dir": normalized_models,
                    "save_processed_files": bool(save_files),
                    "scan_subfolders": bool(recursive),
                },
                APP_SETTINGS_PATH,
            )
            ctx.outputs_dir = normalized_outputs
            return (
                str(normalized_outputs),
                str(normalized_temp),
                str(normalized_models),
                f"<span class='vc-ok'>Saved global settings to {html.escape(str(target))}.</span>",
            )
        except Exception as exc:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                f"<span class='vc-err'>Could not save global settings: {html.escape(str(exc))}</span>",
            )

    save_global.click(
        save_globals,
        inputs=[outputs_dir, temp_dir, models_dir, save_processed, scan_subfolders],
        outputs=[outputs_dir, temp_dir, models_dir, save_status],
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )

    theme_mode.change(
        fn=None,
        inputs=theme_mode,
        outputs=[],
        js=THEME_CHANGE_JS,
        queue=False,
        show_progress="hidden",
        api_visibility="private",
    )
    ctx.states["theme_component"] = theme_mode
    ctx.states["scan_subfolders_component"] = scan_subfolders


__all__ = ["build"]
