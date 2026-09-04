"""Product release history and project links."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from vcap.ui.app import UiContext


CHANGELOG_ENTRIES: list[tuple[str, str, str]] = [
    (
        "v1.6.0",
        "2026-09-04",
        """
### Dataset clip video and audio captions

- Added a split dataset layout that writes clean video captions to `video_caption/`, Whisper and/or Qwen3-Omni sound captions to `audio_caption/`, and an optional merged `<clip>.txt` beside every source or produced segment clip.
- Folder batches can save all media-facing artifacts beside their source files while keeping run metadata and work files in the numbered batch run directory; existing captions can be reused without loading the main caption model.
- Added configurable audio and merge templates, no-speech handling, 30-second sound-caption windowing, JSON metadata fields, Caption Editor part previews and merge-aware regeneration, coverage scanning, and three shipped dataset presets.

### Fixed in v1.6.0

- Task presets now stay selected while moving through empty upload and folder contexts.
- Changing model families can no longer leave a foreign-family task label in the dropdown.
- Returning to Upload files refreshes its preview and ignores late folder-scan results.
- Caption Editor recovers source media for mirrored batch captions and can open a batch run directory directly.
- Regenerating a scene caption processes only that scene's recorded time window.
- Editor regeneration reuses the shared resident model instead of starting a duplicate worker.
- A completed or cancelled run's final status is no longer overwritten by its cancel note.
- Expired cancel confirmations clear themselves instead of leaving a stale waiting message.
- Run history columns fit common desktop widths and keep a compact preview column last.
- The Clips result tab explains how to enable clip saving when a run produced none.
- Batch ETA ignores unsupported and zero-time items until a real item completes.
- Dataset sub-split suggestions use millisecond precision without invalid Number values.
- Autosaved caption edits immediately refresh character, word, token, and queue counts.
- Editor regeneration status hides third-party worker noise while retaining it in the live log.
- Unreadable folder items show a concise skipped-file message while preserving ffprobe diagnostics in logs.
- Idle model release and worker shutdown are reported in the console and live log.
- Application logs persist daily for 14 days, with a separate diagnostic file for crashed workers.
- Shipped Whisper and dataset presets contain only portable preset settings.
- GGUF startup shows a timed friendly status while raw llama-server output stays in the live log.
- Model precision, backend, and checkpoint details update in the dropdown's first event chain.
- Double-Escape cancellation uses an eight-second confirm window and re-arms after expiry.
- Silent videos are skipped cleanly by Whisper and corrupt media reports a concise ffmpeg error.
- Chat discards superseded load progress so final token statistics arrive with the answer.
- A preset-applied model remains resident after its run unless the user selects another variant while it is busy.
""".strip(),
    ),
    (
        "v1.5.0",
        "2026-09-04",
        """
### Whisper speech transcription, and every feature re-verified in Chrome

- New **🎙️ Transcribe** tab: faster-whisper (CTranslate2) transcription for single files, file paths, and recursive folder batches, with the complete faster-whisper parameter set (model, language, translate, beam/best-of/patience, temperature, penalties, thresholds, prompts, hotwords, word timestamps with normalized cues or word highlighting, chunking, Silero VAD), SRT/WebVTT/TXT/LRC/TSV/JSON outputs, trim range, live streamed segments with ETA and realtime speed, confirmed cancellation, Retry failed, Results ZIP, Open in Caption Editor, `metadata.json` + `run_log.txt` per run, run-history and Recover Settings integration, and F9/Esc hotkeys.
- Whisper models download automatically on first use with resumable progress in the terminal and the interface (`models/whisper/<alias>`); 17 aliases (large-v1/v2/v3, large-v3-turbo, distil, medium, small, base, tiny) are listed with their download size and downloaded state, and can be downloaded, verified, or deleted from the Transcribe tab, System & Models, and `Models_Downloader.py`. The defaults mirror the proven SECourses Whisper configuration (large-v1, float16, beam 5, repetition penalty 1.2, word timestamps + normalized subtitles); the reference app produces identical output for identical parameters.
- New **7. Speech transcript (Whisper)** stage in the Processing Pipeline: every caption run can transcribe the same input first, write `<name>_transcript.srt/.txt` next to the caption, and inject the clip-local speech into the prompt through the new `{{TRANSCRIPT}}` template variable; the run log records what was injected and `metadata.json` carries a transcript summary.
- Three shipped presets: **Transcribe - Whisper best quality (large-v1)**, **Transcribe - Whisper large-v3 turbo (fast)**, and **Caption + Whisper transcript (Qwen3-Omni Instruct)**. Whisper runs in its own subprocess that never imports PyTorch, so CUDA 12 CTranslate2 and CUDA 13 PyTorch coexist; the requirements gained `faster-whisper`, `ctranslate2`, `onnxruntime`, `nvidia-cublas-cu12` (and `nvidia-cudnn-cu12` on Linux).
- Fixed from the full real-user Chrome pass: Caption Editor **Regenerate selected** on a scene segment now re-captions only that clip window (40 s instead of 100 s, single caption replaced) and **Export approved** cuts the approved segment from the source instead of copying the whole video; segment rows show `…_segments/clip_0003.txt · 00:09.5–00:11.8`; the first scanned item is selected with its preview; find & replace defaults to whole words; typing an unknown value into the Task / prompt preset boxes no longer raises; Chat reuses the resident model instead of unloading and reloading the same variant; **Write plan JSON** works with the default folders; the progress bar starts at `Preparing…` instead of a full `ready` bar; Results ZIP renders on its own row; regeneration runs show up correctly in Run history; Enter sends a chat message (Shift+Enter adds a line); Save-as clears after a preset delete; Unload reports the real VRAM change; the System & Models timer no longer walks model folders every 3 seconds; Caption-tab downloads and scene previews can be cancelled; `output_formats`, `system_prompt`, and `trim_end_s` are typed in the registry; the literal text `None` can no longer become a system prompt.
- Selecting a long video that a browser cannot play (VP9/Opus MP4, MKV, HEVC…) no longer re-encodes it for the preview: the input block shows a first-frame poster immediately and the probe, resolved input, and Start are never delayed.
- Verified in Chrome on an RTX 5090: every tab, single/path/folder inputs with Unicode paths, cancel/confirm, batch skip and retry, presets, editor tools, dataset tools, settings, recover, health, Qwen3-Omni INT4 and GGUF Q4, TimeChat INT4, Whisper large-v1 and large-v3-turbo (5-, 10-, and 28-minute inputs), automatic model download, and the caption + transcript pipeline. See docs/QA_VERIFICATION_v1.5.0.md.
""".strip(),
    ),
    (
        "v1.4.1",
        "2026-09-04",
        """
### Stock Origin theme and a leaner interface

- The interface now uses the stock Gradio 6 **Origin** theme exactly as shipped, the same theme as the SECourses IndexTTS app: no design token is overridden, and the page width follows Gradio's own responsive limits instead of a custom cap.
- The application stylesheet shrank to what a theme cannot express: the multi-hue action buttons (now the IndexTTS recipe: 44 px tall, one weight, one glow, vertically centred in their row), the app's own markup (file tiles, progress and VRAM meters, find/replace chips, status words), and the header rule, preset-strip alignment, and confirmation bar.
- A **🌗 Light / dark theme** button sits at the top right of the header, as in IndexTTS: it flips the theme instantly in the browser and keeps the Global Settings radio in sync.
- Removed the bordered card boxes around Input, Result, Live log, progress, and Chat media (each is now a heading over native Gradio blocks), the sticky tab bar, the custom scrollbar, the per-row alignment classes, and the per-button size arguments; media previews use the components' own `height`.
""".strip(),
    ),
    (
        "v1.4.0",
        "2026-09-02",
        """
### Every parameter exposed, confirmed cancellation, leaner decode loop, user-requested features

- Every backend value a user can reasonably tune is now a Gradio control with a description, saved in presets and run metadata: seed, maximum caption characters, context carry words and wrapper prompt, fade threshold, re-encode codec/CRF/preset/audio bitrate (now honored by trimming too), quality-analysis frame count with black-luma and silence-RMS thresholds, adaptive sampling sensitivity, total pixel cap, OOM retries and degrade factor, pinned RAM budget and planner slack, subtitle minimum cue length and line wrapping, summary token limit, caption join separator, a repetition guard (`no_repeat_ngram_size` plus repeated-sentence removal), and the complete llama.cpp option set (frames, JPEG quality, threads, batch sizes, flash attention, cache reuse, tier-context bypass, min-p, repeat window, presence/frequency penalties, fit headroom, startup and idle timeouts, extra arguments). Controls that llama.cpp cannot use are disabled with a hint when a GGUF variant is selected.
- Every silent clamp now logs a warning (frame cap, per-frame pixel ceiling, maximum-new-tokens cap, GGUF context tier and frame budget, encoder fallback); `Maximum frames = 0` captions the audio track only on Qwen3-Omni; every extraction sample rate is resampled correctly for the model; the Generation slider follows the selected family's token cap; the 32 GB tier and the default preset select the fully resident INT4 Qwen3-Omni build.
- Cancel asks for confirmation with explicit **Yes, cancel** / **Keep running** buttons (mouse and `Esc`) and stops the worker; deleting model files asks the same way.
- New in the Caption tab: Unload model, Open in Caption Editor, Copy caption, Retry failed, Results ZIP, a Run history panel (open folder, open in editor, recover settings), ZIP upload for folder batches, batch media-kind and file-name filters, a personal prompt library, sampled-frame preview, long-video summary and chapters stage, and TimeChat flatten variants (motion+camera, audiovisual, speech SRT, chapters).
- New elsewhere: chat prompt presets, seed and repetition penalty; editor thumbnail gallery, dataset statistics, trainer token-limit flags, regenerate-all-in-filter, ZIP export; System & Models model deletion with on-disk size, update check, logs folder button; Global Settings logs directory and FFmpeg path.
- Decode loop: per-token console/UI work is throttled (bit-identical output), GGUF streaming reads whole HTTP chunks and starts `llama-server` with `--no-webui -np 1`, and ConvRot fuses gate/up projections and caches Hadamard matrices. Before/after tables and the static-cache/CUDA-graph investigation are in docs/QUANT_PERF.md.
- Cloud installers follow the proven Upscaler Pro v8 flow (Python 3.12 via deadsnakes or uv, `uv pip install`), build llama.cpp with CUDA on Linux for the installed GPU architectures only, and never auto-start the app. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set everywhere so GPU indices match `nvidia-smi`.
- Fixed from the live Chrome verification on an RTX 5090: the input mode now follows the selected input tab and inputs are resolved at Start; model and preset dropdowns reject unknown values and keep their labels; word replacements preserve capitalization; chat shows final token statistics for short replies; the first Start after a model change is no longer swallowed; editor regeneration receives every registered setting and always offers a compatible prompt preset; Retry failed covers unreadable items; status lines include unsupported and cancelled counts; Recover Settings opens on the newest run. See docs/QA_VERIFICATION_v1.4.0.md.
""".strip(),
    ),
    (
        "v1.3.2",
        "2026-09-02",
        """
### Switching models releases the previous model completely

- Selecting another model variant now releases the resident model immediately, or as soon as a running caption or chat job finishes.
- A full release now restores compiled forwards and clears Dynamo/Inductor state, removes the block-swap manager and pinned RAM, drops ConvRot device caches and cuBLAS workspaces, stops the GGUF `llama-server` process, and waits for the driver to return VRAM.
- Release verification checks weak-reference liveness and records freed VRAM plus before/after host memory in the model log.
- The worker `unload` command accepts `unless_variant`; its `unloaded` event reports `resident`, `released`, `skipped`, and the full release `report`.
""".strip(),
    ),
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
    )


__all__ = ["CHANGELOG_ENTRIES", "build"]
