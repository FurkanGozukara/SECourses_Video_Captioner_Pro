# SECourses Video Captioner Pro

Local, dataset-focused audiovisual captioning and media preparation for NVIDIA GPUs. SECourses Video Captioner Pro provides one Gradio interface for single files, recursive batches, caption review, trainer-oriented clip preparation, and resumable model management.

[Support SECourses on Patreon](https://www.patreon.com/SECourses) · [Project repository](https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro)

## Features

- Use TimeChat for audiovisual video, AVoCaDO for visual or audiovisual video, Qwen3-Omni Instruct or Thinking for video, audio, images, and text, and the prompt-free Qwen3-Omni Captioner for one audio file up to 30 seconds.
- Run single inputs or recursive mixed-media folders through the same pipeline, mirror batch subfolders, skip completed files, continue after per-item failures, and write versioned metadata plus compact batch summaries.
- Trim media, sample video by target FPS, uniform timestamps, keyframes, or adaptive visual change, detect scenes, enforce model duration limits, sub-split with overlap, and optionally reject short, black, static, blurry, or silent clips.
- Carry the last 60 words from one generated segment into the next for long AVoCaDO and Qwen3 Instruct/Thinking jobs, while keeping TimeChat and Captioner prompt behavior model-native.
- Stop generation at the model EOS token and record `finish_reason`, token counts, prefill/decode timing, tokens per second, processing time, and peak VRAM in run results and metadata.
- Produce UTF-8 TXT, structured JSON, SRT, VTT, JSONL, optional Thinking reasoning, `run_log.txt`, and collision-safe single or batch output directories.
- Review source-backed or caption-only items in the Caption Editor with autosave, filters, approval flags, regeneration diffs, find/replace, bulk edits, and approved-only export with explicit no-media counts.
- Use Dataset & Export for trainer frame-fit suggestions, crop/pad bucket previews, timestamped fitness plans, an overlap-aware video sub-split tool, and Kohya/Musubi TOML generation.
- Chat through the shared resident worker with streamed Qwen3-Omni multimodal multi-turn history or single-turn TimeChat/AVoCaDO video Q&A, then save the conversation as JSON and Markdown.
- Use protected shipped presets, save writable user presets, auto-load the last-used preset, restore settings from `metadata.json` in Recover Settings, and persist global paths and preferences in `app_settings.json`.
- Select one GPU or multiple data-parallel batch GPUs, apply tier-aware attention and offload plans, recover from supported OOM cases, inspect live VRAM/RAM and model health, and choose CUDA graphs or full `torch.compile` with fallbacks.
- Download and verify BF16, INT8 ConvRot, INT4 ConvRot W4A8, and six Qwen3-Omni GGUF variants with resumable progress; GGUF runs through a private `llama-server` backend.

## Requirements

- Windows or Linux on x86-64.
- 64-bit Python 3.12.
- An NVIDIA RTX 3000-series GPU or newer. Supported presets span 6 GB through 80 GB VRAM; available models and speed vary sharply by tier.
- An NVIDIA driver and CUDA 13 environment compatible with the supplied PyTorch 2.13.0+cu130 wheels. The Windows installer also expects cuDNN 9.17 or newer.
- Git plus `ffmpeg` and `ffprobe` available on `PATH`.
- Windows: Visual Studio Community/Build Tools with the MSVC C++ workload for full TorchInductor compilation. The app can fall back to Triton-only, CUDA graphs, or eager execution.
- Linux GGUF users: CMake and a C++ toolchain are needed to build the pinned llama.cpp runtime because that release has no prebuilt Ubuntu CUDA archive.

Model downloads range from about 6.5 GB to 63.4 GB per variant. Leave additional disk space for the virtual environment, resumable partial files, outputs, and temporary media.

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
cd /workspace/SECourses_Video_Captioner_Pro
source venv/bin/activate
unset LD_LIBRARY_PATH
python secourses_app.py --share
```

### Massed Compute or local Linux

Keep the complete distribution together, open a terminal in that directory, and run:

```bash
chmod +x Massed_Compute_Install.sh
./Massed_Compute_Install.sh
```

Then launch from the repository:

```bash
cd SECourses_Video_Captioner_Pro
source venv/bin/activate
unset LD_LIBRARY_PATH
python secourses_app.py --share
```

Both shell installers use `apt-get` for missing Git, FFmpeg, CMake, and a C++ build toolchain when root or `sudo` is available, install Python 3.12 through `uv`, and create `SECourses_Video_Captioner_Pro/venv`. They preserve an existing `HF_HOME`; otherwise RunPod uses `/workspace` and Massed Compute/local Linux uses a `huggingface_cache` folder beside the installer.

## First Run and Models

`Windows_Run_Video_Captioner_Pro.bat` opens the local app at `http://127.0.0.1:7860`. A missing model is downloaded and validated when it is first used for captioning; interrupted downloads retain their resumable `.part` state.

To choose models in advance or resume downloads manually, run this from the distribution folder:

```text
Windows_Download_Models_and_or_Resume.bat
```

On Linux/cloud, the equivalent menu is:

```bash
SECourses_Video_Captioner_Pro/venv/bin/python Models_Downloader.py
```

The downloader menu includes Qwen3-Omni Instruct, Thinking, and Captioner in both GGUF Q4_K_M and Q8_0 forms. On Linux the model files can be downloaded from this menu, but the pinned `llama-server` runtime must be built with CMake and a C++ compiler and selected with `VCAP_LLAMACPP_SERVER`; see [docs/GGUF_BACKEND.md](docs/GGUF_BACKEND.md).

The default first-launch preset is Qwen3-Omni Instruct video. Select the VRAM tier that matches the physical GPU; the app then applies the associated precision, media budget, attention, and CPU-offload plan.

## Presets and Persistence

- `presets_default/` contains shipped read-only presets; the UI refuses to overwrite or delete them.
- `presets/` contains user presets and the last-used marker; saving or loading a preset marks it as last used, and startup loads it automatically.
- `app_settings.json` stores the outputs, temporary, and models directories plus the save-processed-files and recursive-scan preferences; environment variables still take precedence for application directories.
- Recover Settings reads `metadata.json` and restores compatible controls, while source paths require an explicit opt-in and unavailable GPU indices are skipped.

## Theme

Dark is the default. Global Settings offers Dark, Light, and System; the choice is stored in the browser, applies immediately, and System follows live operating-system color changes. Theme choice is intentionally excluded from presets and run metadata.

## Keyboard Shortcuts

Shortcuts are scoped to the active tab.

| Tab | Shortcut | Action |
|---|---|---|
| Caption | `F9` | Start captioning. |
| Caption | `Esc` | Arm cancellation for six seconds; press again to confirm while a caption job is active. |
| Caption Editor | `←` / `→` | Previous / next item when focus is outside a text field. |
| Caption Editor | `Ctrl+S` | Save the current caption, including while editing its textbox. |
| Caption Editor | `Ctrl+Enter` | Approve the current item. |
| Caption Editor | `Ctrl+Delete` | Reject the current item. |

## Console and UI Progress

Captioning, preprocessing, model downloads, and model loading report through both the terminal and Gradio. Running status includes processed/total counts, elapsed item and job time, remaining items, ETA when available, and generation tokens per second; console status is rate-limited while the UI continues to refresh.

## Model Variants

Sizes are decimal GB from the produced/downloaded folders. GGUF totals include the required 1.325 GB Q8_0 multimodal projector.

| Family | Modality support | Variants, size, and recommended VRAM tier |
|---|---|---|
| TimeChat Captioner GRPO 7B | Video with audio; silent video receives a synthetic silent track | `timechat_bf16` — BF16 — 17.880 GB — 24 GB<br>`timechat_int8` — INT8 ConvRot — 10.275 GB — 12 GB (16 GB fully resident)<br>`timechat_int4` — INT4 ConvRot W4A8 — 6.468 GB — 6 GB (10 GB fully resident) |
| AVoCaDO | Video; audiovisual video | `avocado_bf16` — BF16 — 17.864 GB — 24 GB<br>`avocado_int8` — INT8 ConvRot — 10.275 GB — 12 GB (16 GB fully resident)<br>`avocado_int4` — INT4 ConvRot W4A8 — 6.452 GB — 6 GB (10 GB fully resident) |
| Qwen3-Omni 30B-A3B Instruct | Video, video+audio, audio, image, text | `qwen3_omni_instruct_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_instruct_int8` — INT8 ConvRot — 33.041 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_instruct_int4` — INT4 ConvRot W4A8 — 17.790 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_instruct_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_instruct_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |
| Qwen3-Omni 30B-A3B Thinking | Video, video+audio, audio, image, text; separate reasoning output | `qwen3_omni_thinking_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_thinking_int8` — INT8 ConvRot — 33.031 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_thinking_int4` — INT4 ConvRot W4A8 — 17.780 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_thinking_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_thinking_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |
| Qwen3-Omni 30B-A3B Captioner | One audio file, prompt-free; up to 30 seconds | `qwen3_omni_captioner_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_captioner_int8` — INT8 ConvRot — 33.041 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_captioner_int4` — INT4 ConvRot W4A8 — 17.790 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_captioner_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_captioner_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |

Qwen3 BF16: 63.4 GB — does not fit a single 32 GB GPU; skipped (text-only logits reference was measured with CPU offload).

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
