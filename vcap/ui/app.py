"""Gradio Blocks assembly and the shared UI dependency context."""

from __future__ import annotations

import html
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vcap import (
    APP_DIR,
    APP_NAME,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PRESETS_DEFAULT_DIR,
    PRESETS_DIR,
    TEMP_DIR,
    VERSION,
)

# Direct imports of this module (including smoke tests) still honor the Gradio
# cache contract. The CLI entry point sets the same value before importing UI.
os.environ.setdefault("GRADIO_TEMP_DIR", str(TEMP_DIR / "gradio"))

import gradio as gr

from vcap.core.gpu import list_gpus
from vcap.core.logs import AppLog, get_log
from vcap.core.presets import PresetStore
from vcap.core.registry import SettingsRegistry
from vcap.core.subprocess_runner import CancelToken
from vcap.pipeline.client import PipelineClient
from vcap.ui.components import PresetBarHandles, preset_bar, wire_preset_bar
from vcap.ui.tabs import (
    caption_tab,
    chat_tab,
    changelog_tab,
    dataset_tab,
    editor_tab,
    health_tab,
    recover_tab,
    settings_tab,
)


DEFAULT_PRESET_NAME = "Default - Qwen3-Omni Instruct video"


@dataclass
class UiContext:
    """Shared light-weight services, paths, registry, and per-app state handles."""

    settings_registry: SettingsRegistry
    preset_store: PresetStore
    pipeline_client: PipelineClient
    app_log: AppLog
    app_dir: Path = APP_DIR
    models_dir: Path = MODELS_DIR
    outputs_dir: Path = OUTPUTS_DIR
    temp_dir: Path = TEMP_DIR
    logs_dir: Path = LOGS_DIR
    presets_dir: Path = PRESETS_DIR
    presets_default_dir: Path = PRESETS_DEFAULT_DIR
    states: dict[str, Any] = field(default_factory=dict)
    preset_handles: PresetBarHandles | None = None
    caption_handles: caption_tab.CaptionTabHandles | None = None
    chat_handles: chat_tab.ChatTabHandles | None = None
    _active_cancel: CancelToken | None = field(default=None, repr=False)
    _runtime_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def registry(self) -> SettingsRegistry:
        """Concise compatibility alias used by tests and tab helpers."""

        return self.settings_registry

    @property
    def presets(self) -> PresetStore:
        return self.preset_store

    @property
    def pipeline(self) -> PipelineClient:
        return self.pipeline_client

    @property
    def log(self) -> AppLog:
        return self.app_log

    def reg(
        self,
        key: str,
        component: object,
        default: Any,
        *,
        section: str,
        description: str,
        **metadata: Any,
    ) -> object:
        """Register one user-facing setting and return its component unchanged."""

        if not str(description).strip():
            raise ValueError(f"UI setting {key!r} requires a description")
        return self.settings_registry.register(
            key,
            component,
            default,
            section=section,
            description=description,
            **metadata,
        )

    def activate_cancel(self, token: CancelToken) -> None:
        with self._runtime_lock:
            self._active_cancel = token

    def clear_active_cancel(self, token: CancelToken) -> None:
        with self._runtime_lock:
            if self._active_cancel is token:
                self._active_cancel = None

    def get_active_cancel(self) -> CancelToken | None:
        with self._runtime_lock:
            return self._active_cancel


def _gpu_summary() -> str:
    devices = list_gpus()
    if not devices:
        return "GPU telemetry unavailable"
    return " · ".join(
        f"GPU {item.index}: {html.escape(item.name)} ({item.total_gb:.1f} GB)"
        for item in devices
    )


def build_app() -> gr.Blocks:
    """Construct the complete Gradio application without launching a server."""

    context = UiContext(
        settings_registry=SettingsRegistry(),
        preset_store=PresetStore(
            PRESETS_DIR,
            PRESETS_DEFAULT_DIR,
            default_preset_name=DEFAULT_PRESET_NAME,
        ),
        pipeline_client=PipelineClient(subprocess_mode=True),
        app_log=get_log(),
    )

    with gr.Blocks(
        title=APP_NAME,
        fill_width=True,
        delete_cache=(86_400, 86_400),
        analytics_enabled=False,
    ) as demo:
        with gr.Row(elem_classes=["vc-header"]):
            with gr.Column(scale=7, min_width=420):
                gr.Markdown(
                    f"# {APP_NAME}\n"
                    "Local audiovisual captions, clips, subtitles, and dataset-ready metadata. "
                    "[Support SECourses on Patreon](https://www.patreon.com/SECourses)."
                )
            with gr.Column(scale=3, min_width=300):
                gr.HTML(
                    f"<div class='vc-header-meta'><strong>Version {html.escape(VERSION)}</strong><br>"
                    f"{_gpu_summary()}</div>"
                )

        preset_bar(context)
        with gr.Tabs(selected="caption", elem_id="vc-main-tabs") as main_tabs:
            context.states["main_tabs"] = main_tabs
            # Renders both "🎬 Caption" and its sibling "🎞️ Processing Pipeline".
            caption_tab.build(context)
            with gr.Tab("💬 Chat", id="chat"):
                chat_tab.build(context)
            with gr.Tab("✏️ Caption Editor", id="editor"):
                editor_tab.build(context)
            with gr.Tab("📦 Dataset & Export", id="dataset"):
                dataset_tab.build(context)
            with gr.Tab("⚙️ Global Settings", id="settings"):
                settings_tab.build(context)
            with gr.Tab("🔁 Recover Settings", id="recover") as recover_component:
                context.states["recover_tab_component"] = recover_component
                recover_tab.build(context)
            with gr.Tab("🩺 System & Models", id="health"):
                health_tab.build(context)
            with gr.Tab("📜 Changelog", id="changelog", render_children=False):
                changelog_tab.build(context)

        editor_tab.wire(context)
        recover_tab.wire(context)
        caption_tab.wire(context)
        chat_tab.wire(context)
        wire_preset_bar(context, demo)

        history_binding = context.states.get("run_history_binding")
        if isinstance(history_binding, dict):
            demo.load(
                history_binding["refresh_fn"],
                outputs=history_binding["refresh_outputs"],
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )

        theme_component = context.states["theme_component"]

        def restore_theme_mode(mode: str) -> str:
            return mode if mode in {"dark", "light", "system"} else "dark"

        demo.load(
            fn=restore_theme_mode,
            inputs=theme_component,
            outputs=theme_component,
            js=(
                "() => { let mode = localStorage.getItem('secourses_theme_mode'); "
                "if (!['dark','light','system'].includes(mode)) { mode = 'dark'; "
                "localStorage.setItem('secourses_theme_mode', 'dark'); } return [mode]; }"
            ),
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        context.states["scan_subfolders_component"].input(
            lambda enabled: bool(enabled),
            inputs=context.states["scan_subfolders_component"],
            outputs=context.caption_handles.media.recursive,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
        demo.load(
            lambda enabled: bool(enabled),
            inputs=context.states["scan_subfolders_component"],
            outputs=context.caption_handles.media.recursive,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )

    demo.unload(context.pipeline_client.shutdown)
    # Intentional public inspection handles for smoke tests and future T7 tabs.
    demo.vcap_context = context  # type: ignore[attr-defined]
    demo.settings_registry = context.settings_registry  # type: ignore[attr-defined]
    return demo


__all__ = ["DEFAULT_PRESET_NAME", "UiContext", "build_app"]
