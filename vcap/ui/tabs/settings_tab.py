"""Global paths and lightweight application preferences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from vcap.ui.theme import THEME_CHANGE_JS

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


def build(ctx: "UiContext") -> None:
    """Render and register global application settings."""

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=360):
            gr.Markdown("### Storage")
            outputs_dir = gr.Textbox(
                value=str(ctx.outputs_dir),
                label="Outputs directory",
                info="Single runs and batch run metadata are written below this directory.",
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
                info="Restart the app after changing this path so Gradio and workers use it.",
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
                info="Restart the app after changing this path so model discovery is refreshed.",
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
                choices=[("Dark", "dark"), ("Light", "light")],
                value="dark",
                label="Theme",
                info="Stored in this browser and restored on the next launch.",
            )
            ctx.reg(
                "theme_mode",
                theme_mode,
                "dark",
                section="global",
                description="Browser color theme preference.",
                in_preset=False,
                in_metadata=False,
                choices=["dark", "light"],
                kind="str",
            )
            save_processed = gr.Checkbox(
                value=False,
                label="Save every processed file",
                info="Keep normalized clips, extracted media, and other intermediate artifacts.",
            )
            ctx.reg(
                "save_processed_files",
                save_processed,
                False,
                section="global",
                description="Persist intermediate processed files instead of deleting them.",
                kind="bool",
            )
            scan_subfolders = gr.Checkbox(
                value=False,
                label="Scan subfolders by default",
                info="Default recursive-scan preference for folder tools.",
            )
            ctx.reg(
                "scan_subfolders",
                scan_subfolders,
                False,
                section="global",
                description="Default recursive folder scanning preference.",
                kind="bool",
            )
            gr.Markdown(
                "**Telemetry-free.** Captioning, presets, logs, and model checks stay on this machine. "
                "Gradio analytics are disabled by the application entry point.",
                elem_classes=["vc-card"],
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
