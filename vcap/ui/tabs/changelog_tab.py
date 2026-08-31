"""Product release history and project links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v1.1.0",
        "2026-08-31",
        """
### Post-release fixes and workflow expansion

- Stop Transformers and GGUF generation on the correct EOS tokens and record whether each result ended at EOS, hit its length limit, or was cancelled.
- Apply model-family generation defaults consistently while preserving user sampling overrides and separate Thinking reasoning.
- Ignore prompt text safely for the prompt-free Qwen3-Omni Captioner and show a clear warning instead of failing.
- Add target-FPS, uniform, keyframe, and adaptive frame-sampling strategies with accurate trimmed-range timing.
- Use real SageAttention and xFormers integrations when compatible, with automatic SDPA fallback for unsupported calls or Flash Attention failures.
- Preserve recursive batch folder structure, exclude caption sidecars from folder input, and keep true filename collisions distinct.
- Improve audio-only quality checks, batch trim guidance, temporary clip cleanup, subtitle post-processing, and per-item error continuation.
- Show processed and remaining counts, item and job elapsed time, ETA, token rate, downloads, and model loading in both Gradio and the console without duplicate worker logs.
- Add optional previous-segment context for long AVoCaDO and Qwen3 Instruct/Thinking jobs.
- Add data-parallel folder batches with one isolated worker per selected GPU and round-robin item assignment.
- Record sampling, context, source-root, finish-reason, timing, and compact batch-summary data in run metadata.
- Add Dark, Light, and System themes, persistent global paths/preferences, tier-filtered model choices, and a cached compile readiness probe.
- Replace one-click Caption cancellation with a six-second arm-and-confirm flow and scope Caption and Editor keyboard shortcuts to the active tab.
- Protect shipped presets, keep user presets separate, auto-load the last-used preset, and make trigger/reasoning defaults explicit.
- Repair Caption Editor autosave, regeneration, Unicode folder scans, metadata-backed media previews, zero-limit filters, and caption-only export accounting.
- Expand Dataset & Export with synchronized trainer suggestions, crop/pad geometry, safe timestamped fitness plans, and deduplicated sub-split progress.
- Harden Recover Settings by excluding machine/theme directories, validating GPU choices, and requiring an opt-in before restoring source paths.
- Add human-readable model health status, local verification progress, and six Qwen3-Omni GGUF Q4/Q8 entries to the downloader menu.
- Add streamed interactive Chat with shared model reuse, multimodal history, cooperative stop, reasoning display, and JSON/Markdown transcript saves.
- Refresh the EOS-aware speed, latency, quality, and VRAM comparison with new AVoCaDO and Qwen3-Omni Thinking measurements.
""".strip(),
    ),
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
