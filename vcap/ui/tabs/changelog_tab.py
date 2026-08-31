"""Product release history and project links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v1.0.0",
        "2026-08-31",
        """
### First public release

- Caption video, audio, images, and text with TimeChat, AVoCaDO, and Qwen3-Omni model families.
- Run single files or resilient recursive batches through one subprocess pipeline with live progress, cancellation, and metadata.
- Prepare media with trimming, sampling, scene detection, model-limit splitting, trainer sub-splitting, and clip quality checks.
- Review and revise sidecar captions in the Caption Editor with autosave, filtering, approval flags, regeneration diffs, bulk edits, and approved-only export.
- Analyze trainer frame fitness and generate Kohya/Musubi dataset TOML for video and image datasets.
- Save universal presets, recover a complete UI from run metadata, inspect GPUs and attention backends, and download or verify model variants.
- Write UTF-8 TXT, JSON, SRT, VTT, JSONL, reasoning, logs, and versioned metadata with collision-safe run directories.
- Start with the Qwen3-Omni Instruct video preset on first launch and keep it first in the preset list.
- Review caption-only and text-prompt run outputs with a clear no-media preview state.
- Select any installed GPU for Transformers or GGUF workers; CUDA graphs are now the conservative compile default.
- Use tier-aware GGUF context and layer limits so Q4_K_M has a practical 24 GB configuration.
- Trim unmistakable capped AVoCaDO dataset-QA continuations while preserving ordinary caption text.
- Adapt automatic prompts per item in mixed-modality batches instead of sending video wording to audio or images.
- Cancel on the first click, abort in-flight GGUF streams, and report cancelled item counts in the terminal status.
- Keep model selection and narrow input tabs synchronized, and show each worker log line only once.
""".strip(),
    ),
]


def build(ctx: "UiContext") -> None:
    """Render newest-first release notes and SECourses project details."""

    del ctx
    gr.Markdown("## Release history")
    for index, (version, date, markdown) in enumerate(CHANGELOG_ENTRIES):
        with gr.Accordion(f"{version} · {date}", open=index == 0):
            gr.Markdown(markdown)

    gr.Markdown(
        """
### About SECourses Video Captioner Pro

Built by **SECourses** for local, dataset-focused audiovisual captioning and media preparation.

[Support SECourses on Patreon](https://www.patreon.com/SECourses) · [GitHub repository](https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro)
""".strip(),
        elem_classes=["vc-card"],
    )


__all__ = ["CHANGELOG_ENTRIES", "build"]
