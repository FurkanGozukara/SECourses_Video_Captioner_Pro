"""Product release history and project links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v1.3.1",
        "2026-09-02",
        """
### Block swap is visible and adjustable

- The Model panel now opens a `Block swap & offload plan` section by default. With `Automatic block swap` on, the `Decoder layers to block-swap` slider shows the swapped-layer count the loader is expected to choose for the selected variant, GPU, media budget, reserve, and slots, and the line under it reports resident layers, pinned RAM, the expected peak against free VRAM, tower staging, and any overflow warnings. The preview reruns whenever those inputs change.
- Uncheck `Automatic block swap` to set the swapped count yourself (0 keeps the whole decoder resident); the slider starts from the automatic value so small adjustments are easy. GGUF variants and the legacy expert offload keep the slider disabled with a note explaining why.
- When a model is resident, the preview uses the free VRAM measured before that model was placed (the figure the next load will see) and also reports the loaded plan. Presets and run metadata saved before this version that stored `gpu_layers` are translated on load; the Recover tab's model-only mode includes the new controls.
- GGUF loads now pass the `VRAM to keep free (GB)` control to `llama-server`'s fit target; previously the backend used its own 2 GB default regardless of the setting.
""".strip(),
    ),
    (
        "v1.3.0",
        "2026-09-01",
        """
### Decoder block swap instead of Windows shared-memory paging

- Replace the Accelerate hook offload with kohya/Musubi-style decoder block swap: the first N decoder layers stay resident, the rest live in pinned system RAM (`cudaHostRegister`, with pinned-chunk and pageable fallbacks) and stream through a ring of preallocated GPU slots on a dedicated CUDA copy stream that prefetches the next layer while the current one computes. Swapped captions are byte-identical to fully resident ones.
- Plan residency from measurements instead of tier constants: the loader reads per-layer byte sizes from the safetensors header, measures free VRAM, estimates the job's activation peak from the media it will process (frames, pixels, context, family), keeps a configurable reserve (default 2 GB) free, and logs the whole budget. `Decoder layers on GPU` accepts `auto`, `all`, or a resident count; `VRAM to keep free (GB)` and `Swap slots` are new controls.
- Cap the PyTorch allocator at the dedicated VRAM that was free at load so an overrun raises a recoverable out-of-memory error instead of silently paging (`VCAP_VRAM_HARD_CAP=0` disables it), slice the language-model head to the last token during generation (removes 1.6-2.5 GB of prefill logits), and record observed activation peaks per variant to tighten later plans.
- Show WDDM shared GPU memory in the Health meter and run metadata, and warn only when shared usage exceeds the pinned block-swap buffers, which Windows also reports as shared memory.
- Every VRAM tier now uses automatic block swap with a 2 GB reserve; INT8 and BF16 variants are offered on smaller tiers because they no longer need to fit entirely. GGUF variants start `llama-server` with `--fit on` and a target of the reserve plus 1,536 MiB of projector headroom, leaving `-ngl` to the fitter (an explicit `-ngl` makes llama.cpp abort fitting); Q8_0 Instruct went from 9.6 to 64.6 tok/s on a 32 GB GPU.
- Size residency from the media a job really contains (frames from clip durations, kinds from probing) rather than the preset's worst case, budget 768 MiB of allocator slack so the reserve holds against real free VRAM, trim the worker's working set after large loads, and retry ffprobe/ffmpeg spawns with backoff.
- Cap ConvRot prefill temporaries by chunking the `int_mm` projection rows (identical results, INT4/INT8 activation peaks roughly halved), stage the prefill-only audio/vision towers on CPU when that buys resident decoder layers, and feed observed reserved-memory peaks back into later plans as a per-variant ratio.
- Verified on an RTX 5090: swapped captions are byte-identical to resident ones for TimeChat and Qwen3-Omni (including all 48 layers swapped), every family runs on the automatic plan without touching shared GPU memory, and 63.4 GB Qwen3 BF16 checkpoints run within a 32 GB budget with 2.3 GB left free. See docs/BLOCK_SWAP.md.
""".strip(),
    ),
    (
        "v1.2.0",
        "2026-09-01",
        """
### Verified downloads and workflow polish

- Repair fresh-install model downloads by using the uploaded folders at the `MonsterMMORPG/Wan_GGUF` repository root, probing the alternate prefixed layout, and verifying every file by size and SHA-256. A live clean download of TimeChat INT4 sustained up to 178 MB/s, passed verification, loaded, and completed a caption.
- Add `MonsterMMORPG/Wan_GGUF` as a fallback mirror for Qwen3-Omni GGUF files, normalize mirrored files into the expected flat layout, and preserve resume and cancellation support.
- Check free disk space before Hugging Face and GGUF downloads with 5% headroom, and report clear `VCAP_STATUS` errors before transfer begins.
- Keep the task preset, rendered prompts, and description synchronized when model variants change, including family changes and the no-input state.
- Keep scanned source paths authoritative for batch jobs and mirror nested Unicode source folders into the selected batch output folder.
- Clamp SRT and WebVTT cues to each clip's real time window while preserving a minimum cue duration.
- Add per-segment ETA to single-file jobs, remove the repeated `ETA:` prefix, and deduplicate consecutive `run_log.txt` lines.
- Add optional desktop completion notifications and a chime, optional automatic opening of the output folder after single-file jobs, and a batch `Limit items` dry-run control.
- Preview word-replacement chips immediately while typing.
- Remove the unsupported `expandable_segments` allocator flag on Windows and strip it automatically from child-worker environments.
- Restore compile mode and recursive batch scanning through Recover Settings without unknown-key warnings.
- Complete end-to-end Chrome verification on an RTX 5090 across every model family and quantization path, including all six Qwen3 GGUF Q4/Q8 variants (121 GB downloaded) and 63.4 GB Qwen3 BF16 checkpoints under automatic CPU offload.
""".strip(),
    ),
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
