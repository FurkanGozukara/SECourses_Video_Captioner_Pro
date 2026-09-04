# SECourses Video Captioner Pro

Local, dataset-focused audiovisual captioning and media preparation for NVIDIA GPUs. SECourses Video Captioner Pro provides one Gradio interface for single files, recursive batches, caption review, trainer-oriented clip preparation, and resumable model management.

[Support SECourses on Patreon](https://www.patreon.com/SECourses) · [Project repository](https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro)

## Features

- Use TimeChat for audiovisual video, AVoCaDO for visual or audiovisual video, Qwen3-Omni Instruct or Thinking for video, audio, images, and text, and the prompt-free Qwen3-Omni Captioner for one audio file up to 30 seconds.
- Run single inputs or recursive mixed-media folders through the same pipeline, mirror batch subfolders, skip completed files, continue after per-item failures, and write versioned metadata plus compact batch summaries.
- Transcribe video or audio with faster-whisper in the dedicated Transcribe tab, stream timestamped segments, export SRT/VTT/TXT/LRC/TSV/JSON, or run Whisper before captioning and inject clip-local speech through `{{TRANSCRIPT}}`.
- Turn folders of pre-cut clips into ready-to-train datasets with separate `video_caption/` and `audio_caption/` parts plus an optional merged caption beside each clip; reuse existing video captions without loading the main model.
- Trim media, sample video by target FPS, uniform timestamps, keyframes, or adaptive visual change, detect scenes, enforce model duration limits, sub-split with overlap, and optionally reject short, black, static, blurry, or silent clips.
- Carry the last 60 words from one generated segment into the next for long AVoCaDO and Qwen3 Instruct/Thinking jobs, while keeping TimeChat and Captioner prompt behavior model-native.
- Stop generation at the model EOS token and record `finish_reason`, token counts, prefill/decode timing, tokens per second, processing time, and peak VRAM in run results and metadata.
- Produce UTF-8 TXT, structured JSON, SRT, VTT, JSONL, optional Thinking reasoning, `run_log.txt`, and collision-safe single or batch output directories.
- Review source-backed or caption-only items in the Caption Editor with autosave, filters, approval flags, regeneration diffs, find/replace, bulk edits, and approved-only export with explicit no-media counts.
- Use Dataset & Export for trainer frame-fit suggestions, crop/pad bucket previews, timestamped fitness plans, an overlap-aware video sub-split tool, and Kohya/Musubi TOML generation.
- Chat through the shared resident worker with streamed Qwen3-Omni multimodal multi-turn history (video, audio, or images can be attached to any turn) or single-turn TimeChat/AVoCaDO video Q&A, watch the answer stream into the conversation (Thinking models stream their reasoning first as a collapsible thought block), track tokens, speed, and context usage per reply, then save the conversation as JSON and Markdown.
- Use protected shipped presets that also cover Chat, apply any preset the moment it is selected, save writable user presets, auto-load the last-used preset, restore settings from `metadata.json` in Recover Settings, and persist global paths and preferences in `app_settings.json`.
- Select one GPU or multiple data-parallel batch GPUs, apply tier-aware attention plans with automatic decoder block swap that keeps 2 GB of dedicated VRAM free instead of paging into Windows shared GPU memory, recover from supported OOM cases, inspect live VRAM/RAM/shared-memory and model health, and choose CUDA graphs or full `torch.compile` with fallbacks.
- Download and verify BF16, INT8 ConvRot, INT4 ConvRot W4A8, and all six Qwen3-Omni GGUF Q4/Q8 variants with resumable progress; GGUF runs through a private `llama-server` that fits itself to device memory, while 63.4 GB Qwen3 BF16 checkpoints run through pinned-RAM block swap on smaller GPUs.
- Tune every backend value from the interface: seed, repetition guard (`no_repeat_ngram_size` and repeated-sentence removal), maximum caption characters and join separator, subtitle cue minimum and line wrapping, context carry words and wrapper prompt, re-encode codec/CRF/preset/audio bitrate (also used by trimming), rejection thresholds (black luma, silence RMS, analysis frames), adaptive sampling sensitivity, total pixel cap, OOM retries and degrade factor, planner slack and pinned RAM budget, the summary stage token limit, and the full llama.cpp option set (frames, JPEG quality, threads, batch sizes, flash attention, cache reuse, tier-context bypass, min-p, repeat window, presence/frequency penalties, fit headroom, startup/idle timeouts, extra arguments). Every clamp the backend still applies is logged.
- Work faster after a run: Copy caption, Retry failed items, Results ZIP, Open in Caption Editor, Unload model, and a Run history panel that opens, edits, or recovers the settings of any earlier run. Folder batches accept a ZIP upload (extracted below `outputs/uploaded_batches`) and media-kind/file-name filters; a personal prompt library stores named system/user prompt pairs; System & Models can delete model files (with confirmation and on-disk size) and check for updates; Global Settings adds the logs directory and an FFmpeg path.

## Dataset clip captions (video + audio)

Open **Processing Pipeline > 8. Audio captions & dataset clip layout** to create a video-caption part and an audio-caption part for every input clip. A typical next-to-source batch produces:

```text
dataset/
  myVideo.mp4
  myVideo.txt
  video_caption/myVideo.txt
  audio_caption/myVideo.txt
```

The merged `myVideo.txt` is rendered from a configurable template and can be disabled when separate parts are preferred. Whisper supplies speech as plain text, one segment per line, or clip-local timestamped lines. Qwen3-Omni Captioner supplies a prompt-free sound description; audio longer than its 30-second limit is extracted at 16 kHz mono, captioned in windows of at most 30 seconds, and joined in order. Scene, fixed, and trainer splits use the same layout inside each segment directory, so saved clips are immediately paired with merged captions.

Use **Dataset clips - video + audio captions (Qwen3-Omni + Whisper)** for generated video plus speech captions, **Dataset clips - add Whisper audio captions to existing captions** to append audio information without loading the main caption model, or **Dataset clips - video + sound captions (Qwen3-Omni + Captioner)** for Whisper speech plus sound descriptions. Existing mode first preserves a clean caption in `video_caption/`, which keeps repeated overwrite runs idempotent. See [Dataset clip captions](docs/DATASET_CLIPS.md) for paths, templates, skip rules, and batch examples.

## Requirements

- Windows or Linux on x86-64.
- 64-bit Python 3.12.
- An NVIDIA RTX 3000-series GPU or newer. Supported presets span 6 GB through 80 GB VRAM; available models and speed vary sharply by tier.
- An NVIDIA driver and CUDA 13 environment compatible with the supplied PyTorch 2.13.0+cu130 wheels. The Windows installer also expects cuDNN 9.17 or newer.
- Git plus `ffmpeg` and `ffprobe` available on `PATH`.
- Windows: Visual Studio Community/Build Tools with the MSVC C++ workload for full TorchInductor compilation. The app can fall back to Triton-only, CUDA graphs, or eager execution.
- Linux GGUF users: the CUDA toolkit with `nvcc`, CMake, and a C++ toolchain are required. The pinned llama.cpp release has no prebuilt Ubuntu CUDA archive, so the cloud installers build it automatically.

Model downloads range from about 6.5 GB to 63.4 GB per variant. Leave additional disk space for the virtual environment, resumable partial files, outputs, and temporary media.

## Speech transcription (Whisper)

The **🎙️ Transcribe** tab accepts uploaded video/audio, a local file path, or a recursive folder batch. It streams segments while the worker runs, then presents plain text, SRT, segment confidence, JSON, and downloadable output files. Batch transcripts can mirror the source tree below `outputs/batch_transcripts` or be saved beside each source. Model download/verification, cancellation, retry, ZIP export, run metadata, live logs, and Caption Editor handoff use the same interaction patterns as caption runs.

The Processing Pipeline tab also has **7. Speech transcript (Whisper)**. Enabling it transcribes each video or audio item once before caption generation, writes the selected transcript sidecars beside the caption, and makes the overlapping speech available as `{{TRANSCRIPT}}` for every clip prompt. With prompt injection enabled, a wrapper is appended automatically when the selected prompt does not contain that variable.

The shipped **Whisper Quality (large-v1)** preset (read-only; its values are also the application's startup defaults, so the Transcribe tab is ready without loading anything) reproduces the proven large-v1 defaults: float16, beam/best-of 5, temperature 0, repetition penalty 1.2, 30-second chunks, word timestamps with normalization, VAD off, and all six outputs. The turbo preset changes only the model alias. Models download automatically on first use and interrupted downloads are resumable.

| Alias | Hugging Face repository | Download size | Note |
|---|---|---:|---|
| `large-v1` | `Systran/faster-whisper-large-v1` | 3.09 GB | Best-quality large-v1 default |
| `large-v3` | `Systran/faster-whisper-large-v3` | 3.09 GB | Latest full multilingual model |
| `large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | 1.62 GB | Fast multilingual large-v3 variant |
| `large-v2` | `Systran/faster-whisper-large-v2` | 3.09 GB | Previous full multilingual model |
| `distil-large-v3.5` | `distil-whisper/distil-large-v3.5-ct2` | 1.52 GB | Distilled multilingual large-v3.5 |
| `distil-large-v3` | `Systran/faster-distil-whisper-large-v3` | 1.52 GB | Distilled multilingual large-v3 |
| `distil-large-v2` | `Systran/faster-distil-whisper-large-v2` | 1.52 GB | Distilled multilingual large-v2 |
| `medium` | `Systran/faster-whisper-medium` | 1.53 GB | Balanced multilingual model |
| `medium.en` | `Systran/faster-whisper-medium.en` | 1.53 GB | English-only medium model |
| `distil-medium.en` | `Systran/faster-distil-whisper-medium.en` | 792.1 MB | Fast distilled English medium model |
| `small` | `Systran/faster-whisper-small` | 486.2 MB | Compact multilingual model |
| `small.en` | `Systran/faster-whisper-small.en` | 486.1 MB | Compact English-only model |
| `distil-small.en` | `Systran/faster-distil-whisper-small.en` | 335.5 MB | Fast distilled English small model |
| `base` | `Systran/faster-whisper-base` | 147.9 MB | Small multilingual model |
| `base.en` | `Systran/faster-whisper-base.en` | 147.8 MB | Small English-only model |
| `tiny` | `Systran/faster-whisper-tiny` | 78.2 MB | Smallest multilingual model |
| `tiny.en` | `Systran/faster-whisper-tiny.en` | 78.1 MB | Smallest English-only model |

Whisper runs in a separate CTranslate2 process and discovers the CUDA 12 cuBLAS/cuDNN libraries installed in the active environment. `Device: auto` tries CUDA and falls back to CPU/int8 when the runtime is unavailable; forcing `cuda` surfaces the runtime error instead. This CUDA runtime is independent of the PyTorch CUDA 13 runtime used by caption models.

## Installation

The release distribution places the installer scripts and `video_caption_requirements.txt` one directory above this repository folder.

### Windows

From the extracted distribution folder, run:

```text
Windows_Install_and_Update.bat
Windows_Run_Video_Captioner_Pro.bat
```

The installer clones or updates `SECourses_Video_Captioner_Pro`, creates its Python 3.12 virtual environment, and installs the distribution requirements with `uv`.

### RunPod or SimplePod

Upload the complete distribution to `/workspace`, then run:

```bash
cd /workspace
export HF_HOME="/workspace"
chmod +x RunPod_Install_SECourses_Video_Captioner_Pro.sh
./RunPod_Install_SECourses_Video_Captioner_Pro.sh
```

Start it later with:

```bash
export HF_HOME="/workspace"
cd /workspace
cd SECourses_Video_Captioner_Pro && git pull && source venv/bin/activate && unset LD_LIBRARY_PATH && python secourses_app.py --share
```

### Massed Compute or local Linux

Keep the complete distribution together, open a terminal in that directory, and run:

```bash
chmod +x Massed_Compute_Install.sh
./Massed_Compute_Install.sh
```

Then launch from the repository:

```bash
cd SECourses_Video_Captioner_Pro && git pull && source venv/bin/activate && unset LD_LIBRARY_PATH && python secourses_app.py --share
```

Both shell installers use `apt-get` for Git, FFmpeg, CMake, build-essential, and libcurl development files when needed. RunPod/SimplePod provisions a uv-managed Python 3.12; Massed Compute uses stable Python 3.12.13 or newer from deadsnakes when available and falls back to uv-managed 3.12.13. Both create `SECourses_Video_Captioner_Pro/venv`, install `video_caption_requirements.txt`, and attempt the pinned llama.cpp CUDA build without auto-starting the app. Set `HF_HOME` in the terminal as shown above or in the distribution instruction file.

## First Run and Models

`Windows_Run_Video_Captioner_Pro.bat` opens the local app at `http://127.0.0.1:7860`, or the next free port if 7860 is taken; pass `--server-name` and `--server-port` to pin a specific address and port. The terminal prints the URL actually used. A missing model is downloaded and validated when it is first used for captioning; interrupted downloads retain their resumable `.part` state.

Selecting a different model variant unloads the resident model right away, or as soon as the running job finishes. The release covers VRAM, pinned RAM, compiled graphs, and the GGUF `llama-server` process before the new model loads.

To choose models in advance or resume downloads manually, run this from the distribution folder:

```text
Windows_Download_Models_and_or_Resume.bat
```

On Linux/cloud, the equivalent menu is:

```bash
SECourses_Video_Captioner_Pro/venv/bin/python Models_Downloader.py
```

The downloader menu includes Qwen3-Omni Instruct, Thinking, and Captioner in both GGUF Q4_K_M and Q8_0 forms. On Linux the cloud installer automatically builds the pinned CUDA `llama-server`; if that best-effort step fails, use **System & Models -> Install / repair llama.cpp** to run the same installer again. Build output is retained in `llamacpp/downloads/b10621/build.log`; see [docs/GGUF_BACKEND.md](docs/GGUF_BACKEND.md) for the automatic layout and manual fallback.

The default first-launch preset is Qwen3-Omni Instruct video. Select the VRAM tier that matches the physical GPU; the app then applies the associated precision, media budget, and attention plan. Decoder placement is automatic on every tier: at load time the app measures free VRAM, estimates the job's activation peak from the media it will process, keeps 2 GB free, and block-swaps the remaining decoder layers from pinned RAM through a prefetching ring of GPU slots (see [docs/BLOCK_SWAP.md](docs/BLOCK_SWAP.md)). The `Block swap & offload plan` section of the Model panel shows what that resolves to: with `Automatic block swap` on, the `Decoder layers to block-swap` slider displays the swapped-layer count the loader is expected to choose for the selected variant, GPU, media budget, and reserve, and the line under it reports the expected peak against free VRAM. Uncheck `Automatic block swap` to set the swapped count yourself (0 keeps the whole decoder on the GPU); `VRAM to keep free (GB)` and `Swap slots` tune the automatic plan.

## Presets and Persistence

- `presets_default/` contains shipped read-only presets; the UI refuses to overwrite or delete them.
- `presets/` contains user presets and the last-used marker; saving or loading a preset marks it as last used.
- **A fresh start never restores the last-used preset.** Every launch applies the shipped `Default - Qwen3-Omni Instruct video` preset, so the app always opens in the same known state. Press **⟲ Load Last Values** in the preset bar to apply the preset you used last; that is the only thing that reads the marker.
- Selecting a preset in the universal preset dropdown applies it immediately; Load re-applies the selected preset after manual edits, and Reset restores application defaults. Task / prompt presets in the Caption tab apply on selection as well.
- Every preset also stores the Chat tab's system prompt and generation settings plus the context window. Four shipped presets select chat-capable models: `Chat assistant (Qwen3-Omni Instruct)` and `Chat with reasoning (Qwen3-Omni Thinking)` use the INT8 ConvRot Transformers path with native video, while the `… fast (… GGUF Q8)` variants run through the private `llama-server` for several times the decode speed with video sent as sampled frames plus audio. Qwen3-Omni Instruct and Thinking hold multi-turn conversations over video, audio, images, and text; TimeChat and AVoCaDO answer one question about exactly one video per Send; the Qwen3-Omni Captioner has no chat mode.

## Context Length

`Context length (tokens)` in the Caption tab's Generation section sets the total window for prompt, media, and reply. It is model-specific like the other generation controls (every shipped family allows 32,768) and is saved with universal presets. The window bounds the caption auto-split ceiling, trims the oldest chat turns to fit, and caps the reply so it never runs past the window.

Memory follows the backend: `llama-server` reserves the KV cache for the whole window at load (Qwen3-Omni needs about 96 KB per token, so 32k costs about 3 GB and 16k about 1.5 GB; TimeChat and AVoCaDO need about 56 KB per token), the VRAM tier clamps GGUF windows to 8k/16k/32k, and llama.cpp's fitter may shrink it further. Changing the window restarts the GGUF server; the Transformers backends grow the cache during generation and only fold the window into the block-swap activation budget. The control's help text shows each model's cost, the Caption tab's budget line shows the KV estimate at the current window, and the Speed line (Caption) and Tokens line (Chat) report `Context: used / window` after every reply. The GGUF chat presets request 24,576 tokens to leave more room for resident decoder layers.
- `app_settings.json` stores the outputs, temporary, and models directories plus the save-processed-files and recursive-scan preferences; environment variables still take precedence for application directories.
- Recover Settings reads `metadata.json` and restores compatible controls, while source paths require an explicit opt-in and unavailable GPU indices are skipped.

## Interface

The interface uses the stock Gradio 6 **Origin** theme exactly as shipped, the same theme as the SECourses IndexTTS app, so every colour, radius, shadow, and font comes from Gradio and the page width follows Gradio's own responsive limits. The application stylesheet covers only what a theme cannot express: the multi-hue action buttons (the same recipe as IndexTTS), the app's own markup (file tiles, progress and VRAM meters, find/replace chips, status words), and the header rule, preset-strip alignment, and confirmation bar. Fonts are the ones Gradio bundles, so a page load makes no request to Google Fonts and the app renders identically without internet access.

Dark is the default. The **🌗 Light / dark theme** button at the top right of the header flips between the two instantly, and Global Settings offers Dark, Light, and System; the choice is stored in the browser, applies immediately, and System follows live operating-system color changes. The two controls always agree. Theme choice is intentionally excluded from presets and run metadata.

**Open / Close All** in the preset bar expands or collapses every section of the tab you are looking at, including sections nested inside others; it runs entirely in the browser and never touches the server. **⟲ Load Last Values**, immediately to its left, applies the preset this machine used last.

The top-level order is Caption, Processing Pipeline, Transcribe, Chat, Caption Editor, Dataset & Export, Global Settings, Recover Settings, System & Models, and Changelog. Transcribe mirrors Caption's two-column workflow: inputs, streamed result, actions, progress, item tracker, live log, and resource meter stay on the left; model, language, decoding, timestamp/output, and VAD settings stay in numbered accordions on the right.

## Keyboard Shortcuts

Shortcuts are scoped to the active tab.

| Tab | Shortcut | Action |
|---|---|---|
| Any tab | `F4` | Open or close every section of the visible tab (same as **Open / Close All**). |
| Caption, Processing Pipeline | `F9` | Start captioning. |
| Caption, Processing Pipeline | `Esc` | Arm cancellation for six seconds; press again to confirm while a caption job is active. |
| Transcribe | `F9` | Start speech transcription. |
| Transcribe | `Esc` | Open the confirmation bar for an active transcription. |
| Caption Editor | `←` / `→` | Previous / next item when focus is outside a text field. |
| Caption Editor | `Ctrl+S` | Save the current caption, including while editing its textbox. |
| Caption Editor | `Ctrl+Enter` | Approve the current item. |
| Caption Editor | `Ctrl+Delete` | Reject the current item. |

## Console and UI Progress

Captioning, Whisper transcription, preprocessing, model downloads, and model loading report through both the terminal and Gradio. Running status includes processed/total counts, elapsed item and job time, remaining items, ETA when available, and generation or realtime speed; console status is rate-limited while the UI continues to refresh.

## Model Variants

Sizes are decimal GB from the produced/downloaded folders. GGUF totals include the required 1.325 GB Q8_0 multimodal projector.

| Family | Modality support | Variants, size, and recommended VRAM tier |
|---|---|---|
| TimeChat Captioner GRPO 7B | Video with audio; silent video receives a synthetic silent track | `timechat_bf16` — BF16 — 17.880 GB — 24 GB<br>`timechat_int8` — INT8 ConvRot — 10.275 GB — 12 GB (16 GB fully resident)<br>`timechat_int4` — INT4 ConvRot W4A8 — 6.468 GB — 6 GB (10 GB fully resident) |
| AVoCaDO | Video; audiovisual video | `avocado_bf16` — BF16 — 17.864 GB — 24 GB<br>`avocado_int8` — INT8 ConvRot — 10.275 GB — 12 GB (16 GB fully resident)<br>`avocado_int4` — INT4 ConvRot W4A8 — 6.452 GB — 6 GB (10 GB fully resident) |
| Qwen3-Omni 30B-A3B Instruct | Video, video+audio, audio, image, text | `qwen3_omni_instruct_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_instruct_int8` — INT8 ConvRot — 33.041 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_instruct_int4` — INT4 ConvRot W4A8 — 17.790 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_instruct_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_instruct_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |
| Qwen3-Omni 30B-A3B Thinking | Video, video+audio, audio, image, text; separate reasoning output | `qwen3_omni_thinking_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_thinking_int8` — INT8 ConvRot — 33.031 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_thinking_int4` — INT4 ConvRot W4A8 — 17.780 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_thinking_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_thinking_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |
| Qwen3-Omni 30B-A3B Captioner | One audio file, prompt-free; up to 30 seconds | `qwen3_omni_captioner_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_captioner_int8` — INT8 ConvRot — 33.041 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_captioner_int4` — INT4 ConvRot W4A8 — 17.790 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_captioner_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_captioner_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |

Qwen3 BF16 checkpoints are 63.4 GB and do not fit wholly in 32 GB VRAM. They load through automatic block swap: the decoder layers that do not fit stay in pinned system RAM (about 1.2 GB per swapped layer, so plan for 40 GB or more of free RAM on a 32 GB GPU) and stream through the GPU each token, which keeps dedicated VRAM within budget at substantially lower throughput than resident variants. The tier column above lists the smallest tier that still keeps the whole model resident; smaller cards use block swap automatically.

## Model Credits and Licenses

- **TimeChat-Captioner-GRPO-7B:** [model](https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B), [project/code](https://github.com/yaolinli/TimeChat-Captioner), and [project page](https://timechat-captioner.github.io/). The code is BSD-3-Clause. The Hugging Face checkpoint card does not declare a license; the author states that the weights also require compliance with the Apache-2.0 Qwen2.5-Omni base-model license.
- **AVoCaDO:** [model](https://huggingface.co/AVoCaDO-Captioner/AVoCaDO), [code](https://github.com/AVoCaDO-Captioner/AVoCaDO), [paper](https://arxiv.org/abs/2510.10395), and [project page](https://avocado-captioner.github.io/). Model and project are Apache-2.0.
- **Qwen3-Omni:** [official project](https://github.com/QwenLM/Qwen3-Omni), [Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), [Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking), and [Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner). The open checkpoints declare Apache-2.0 through their license metadata. GGUF files are third-party conversions from [ggml-org](https://huggingface.co/ggml-org) and [mradermacher](https://huggingface.co/mradermacher); retain the upstream model terms.

## Screenshots

> Screenshots will be added with the first tagged public release.

## Benchmarks

Fresh measurements below are three-generation means on physical GPU 0, an RTX 5090. Checkpoint size is decimal GB; peak memory is GiB. Qwen3 Instruct uses the production 32 GB profile (Flash Attention 2 and a 256-token frame budget). See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full comparison, quality deltas, GGUF measurements, methodology, and VRAM recommendations.

| Variant | Checkpoint GB | Load s | Peak GiB | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| TimeChat BF16 | 17.880 | 9.74 | 31.52 | 212.28 | 34.44 |
| TimeChat INT8 ConvRot | 10.275 | 7.70 | 24.40 | 1,904.59 | 34.54 |
| TimeChat INT4 ConvRot W4A8 | 6.468 | 5.24 | 20.86 | 1,834.54 | 34.73 |
| AVoCaDO BF16 | 17.864 | 24.21 | 20.45 | 6,554.05 | 36.03 |
| AVoCaDO INT8 ConvRot | 10.275 | 6.77 | 35.96 | 103.52 | 31.22 |
| AVoCaDO INT4 ConvRot W4A8 | 6.452 | 16.15 | 10.45 | 3,949.54 | 29.11 |
| Qwen3-Omni Instruct INT8 ConvRot | 33.041 | 13.32 | 31.38 | 437.07 | 12.08 |
| Qwen3-Omni Instruct INT4 ConvRot W4A8 | 17.790 | 16.75 | 18.84 | 5,692.81 | 17.91 |
| Qwen3-Omni Thinking INT8 ConvRot | 33.031 | 34.85 | 33.07 | 90.41 | 3.90 |
| Qwen3-Omni Thinking INT4 ConvRot W4A8 | 17.780 | 21.17 | 18.87 | 4,451.80 | 6.97 |
| Qwen3-Omni Captioner INT8 ConvRot | 33.041 | 22.42 | 29.94 | 584.26 | 20.49 |
| Qwen3-Omni Captioner INT4 ConvRot W4A8 | 17.790 | 13.93 | 16.74 | 1,188.83 | 20.13 |

## Documentation

- [Master plan and architecture](docs/PLAN.md)
- [Prompt and task presets](docs/PROMPT_PRESETS.md)
- [Resumable model downloader](docs/DOWNLOADER.md)
- [Qwen3-Omni GGUF backend](docs/GGUF_BACKEND.md)
- [Quantization quality report](docs/QUANT_REPORT.md)
- [Consolidated benchmarks](docs/BENCHMARKS.md)
- [Whisper speech transcription backend](docs/WHISPER.md)
- [QA verification log v1.6.0](docs/QA_VERIFICATION_v1.6.0.md)
- [QA verification log v1.5.0](docs/QA_VERIFICATION_v1.5.0.md)

## Troubleshooting

### `torch.compile`, MSVC, or a slow first run

The first compiled run can spend substantial time building and tuning kernels. On Windows, install the MSVC x64 C++ workload if full TorchInductor is desired. The app probes `cl.exe`/Visual Studio and falls back through Triton-only Inductor, CUDA graphs, and eager mode; disable compilation to isolate a toolchain problem.

### FFmpeg is not found

Confirm both commands work in a new terminal:

```text
ffmpeg -version
ffprobe -version
```

Then rerun the installer or add the FFmpeg `bin` directory to `PATH` before starting the app.

### CUDA OOM or extreme slowdown

Choose the tier at or below the GPU's physical VRAM and close other GPU-heavy programs. On Windows, WDDM can page into shared RAM instead of raising an immediate OOM; this appears as a peak above physical VRAM and can make a run dramatically slower. Prefer INT4 for extra headroom, reduce frames/resolution, or let the automatic preset and OOM retry lower the media budget.

### Silent video

TimeChat requires an audio timeline, so the app supplies a duration-matched silent waveform when the source has no audio track. AVoCaDO and Qwen3-Omni fall back to visual-only processing and report a warning. This is separate from the optional quality rule that can reject mostly silent clips.

### Interrupted model download

Run `Windows_Download_Models_and_or_Resume.bat` again, or retry the model in the app. The downloader keeps range state, validates published sizes/digests, and does not treat a `.part` file as a ready model.

## Support

Development, tutorials, and release support are funded through [SECourses on Patreon](https://www.patreon.com/SECourses).
