# SECourses Video Captioner Pro

Local, dataset-focused audiovisual captioning and media preparation for NVIDIA GPUs. SECourses Video Captioner Pro provides one Gradio interface for single files, recursive batches, caption review, trainer-oriented clip preparation, and resumable model management.

[Support SECourses on Patreon](https://www.patreon.com/SECourses) · [Project repository](https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro)

## Features

- Caption video, audio, images, and text with TimeChat, AVoCaDO, and Qwen3-Omni Instruct, Thinking, and Captioner models.
- Run single files or resilient recursive batches through the same subprocess pipeline with live progress, ETA, cancellation, per-item errors, and versioned metadata.
- Trim and sample media; detect scenes; split around model and trainer limits; sub-split with overlap; and reject clips that are too short, black, static, blurry, or silent.
- Produce UTF-8 TXT, structured JSON, SRT, VTT, JSONL, optional reasoning, run logs, and collision-safe output directories.
- Review captions beside their source media with autosave, filters, approval flags, regeneration diffs, find/replace, bulk edits, and approved-only export.
- Analyze trainer frame fitness, preview resolution buckets, and export Kohya/Musubi dataset TOML for video and image datasets.
- Save universal presets, restore the complete UI from `metadata.json`, and start from a broad library of model-native, training-caption, ASR, translation, lyrics, OCR, and audio-analysis prompts.
- Select any installed GPU for Transformers or GGUF workers, use tier-aware attention/offload defaults, recover automatically from some OOMs, and inspect live VRAM/RAM and model health.
- Keep models loaded between jobs and, when compilation is enabled, default to conservative CUDA graphs or opt into full `torch.compile` with an automatic fallback ladder.
- Download and verify BF16/ConvRot variants with resumable range downloads; run Qwen3-Omni GGUF through a private, cancellable `llama-server` backend.
- Handle mixed-modality batches with per-item prompt adaptation, silent-video fallbacks, first-click cancellation, deduplicated worker logs, and clear caption-only/text-only preview states.

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

Both shell installers use `apt-get` for missing Git/FFmpeg when root or `sudo` is available, install Python 3.12 through `uv`, and create `SECourses_Video_Captioner_Pro/venv`.

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

The default first-launch preset is Qwen3-Omni Instruct video. Select the VRAM tier that matches the physical GPU; the app then applies the associated precision, media budget, attention, and CPU-offload plan.

## Model Variants

Sizes are decimal GB from the produced/downloaded folders. GGUF totals include the required 1.325 GB Q8_0 multimodal projector.

| Family | Modality support | Variants, size, and recommended VRAM tier |
|---|---|---|
| TimeChat Captioner GRPO 7B | Video with audio; silent video receives a synthetic silent track | `timechat_bf16` — BF16 — 17.880 GB — 24 GB<br>`timechat_int8` — INT8 ConvRot — 10.275 GB — 12 GB (16 GB fully resident)<br>`timechat_int4` — INT4 ConvRot W4A8 — 6.468 GB — 6 GB (10 GB fully resident) |
| AVoCaDO | Video; audiovisual video | `avocado_bf16` — BF16 — 17.880 GB — 24 GB<br>`avocado_int8` — INT8 ConvRot — 10.275 GB — 12 GB (16 GB fully resident)<br>`avocado_int4` — INT4 ConvRot W4A8 — 6.468 GB — 6 GB (10 GB fully resident) |
| Qwen3-Omni 30B-A3B Instruct | Video, video+audio, audio, image, text | `qwen3_omni_instruct_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_instruct_int8` — INT8 ConvRot — 33.041 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_instruct_int4` — INT4 ConvRot W4A8 — 17.790 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_instruct_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_instruct_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |
| Qwen3-Omni 30B-A3B Thinking | Video, video+audio, audio, image, text; separate reasoning output | `qwen3_omni_thinking_bf16` — BF16 — 63.4 GB — 80 GB<br>`qwen3_omni_thinking_int8` — INT8 ConvRot — 33.041 GB — 32 GB (48 GB fully resident)<br>`qwen3_omni_thinking_int4` — INT4 ConvRot W4A8 — 17.790 GB — 8 GB experimental (24 GB fully resident)<br>`qwen3_omni_thinking_gguf_q4` — GGUF Q4_K_M — 19.882 GB — 24 GB<br>`qwen3_omni_thinking_gguf_q8` — GGUF Q8_0 — 33.810 GB — 32 GB partial offload (48 GB fully resident) |
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
| AVoCaDO INT8 ConvRot | 10.275 | 6.77 | 35.96 | 103.52 | 31.22 |
| Qwen3-Omni Instruct INT8 ConvRot | 33.041 | 13.32 | 31.38 | 437.07 | 12.08 |
| Qwen3-Omni Instruct INT4 ConvRot W4A8 | 17.790 | 16.75 | 18.84 | 5,692.81 | 17.91 |
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
