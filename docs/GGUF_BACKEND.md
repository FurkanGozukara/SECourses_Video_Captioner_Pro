# Qwen3-Omni GGUF backend

SECourses Video Captioner Pro can run Qwen3-Omni audio, image, and video
captioning through a private `llama-server` process. The backend uses the GPU
selected in application settings (or worker `--gpu`) and does not import Gradio or route these files through
`Models_Downloader.py` / `Wan_GGUF`.

## Runtime build

The automatic installer is pinned to **[llama.cpp
b10621](https://github.com/ggml-org/llama.cpp/releases/tag/b10621)**
(2026-08-25), the nightly referenced by the v0.3.0 release. Qwen3-Omni
multimodal support needs b8775 or newer.

On Windows x64 the installer downloads both release assets and verifies their
GitHub release SHA-256 digests:

| Asset | SHA-256 |
|---|---|
| `llama-b10621-bin-win-cuda-13.3-x64.zip` | `23549ccc00b6a18d74348e95d4789f7e96c9efb11cf6e3f1b185baef34d7449f` |
| `cudart-llama-bin-win-cuda-13.3-x64.zip` | `1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e` |

Files are installed under `llamacpp/b10621/`. Downloads use HTTP Range resume
and are retained under `llamacpp/downloads/b10621/`. Run a forced refresh with:

```powershell
venv\Scripts\python.exe -m vcap.models.llamacpp_install --force
```

The b10621 release has no Ubuntu x64 CUDA binary. On Linux, build the two tools
from the pinned tag and point the application at the server executable:

```bash
git clone --branch b10621 --depth 1 https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j --target llama-server llama-mtmd-cli
export VCAP_LLAMACPP_SERVER="$PWD/llama.cpp/build/bin/llama-server"
```

The sibling `llama-mtmd-cli` must also exist. A self-built runtime can use
`-hf <repo>:Q4_K_M` for an independent command-line check, while the app uses
its explicitly downloaded local `--model` and `--mmproj` files.

## Registered models

All sizes below are decimal GB and the exact byte counts are stored in the
registry. Each variant downloads directly from its source Hugging Face repo.

| Variant key | Source and files | Download |
|---|---|---:|
| `qwen3_omni_instruct_gguf_q4` | `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`: `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf` | 18.557 + 1.325 GB |
| `qwen3_omni_instruct_gguf_q8` | same repo, `Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf` + mmproj | 32.484 + 1.325 GB |
| `qwen3_omni_thinking_gguf_q4` | `ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF`: Q4_K_M + named Q8_0 mmproj | 18.557 + 1.325 GB |
| `qwen3_omni_thinking_gguf_q8` | same repo, Q8_0 + named Q8_0 mmproj | 32.484 + 1.325 GB |
| `qwen3_omni_captioner_gguf_q4` | `mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF`: `Qwen3-Omni-30B-A3B-Captioner.Q4_K_M.gguf` + `Qwen3-Omni-30B-A3B-Captioner.mmproj-Q8_0.gguf` | 18.557 + 1.325 GB |
| `qwen3_omni_captioner_gguf_q8` | same repo, `.Q8_0.gguf` + mmproj | 32.484 + 1.325 GB |

Q4_K_M is offered from the 24 GB VRAM tier. Q8_0 becomes selectable at the
32 GB tier, but its 33.81 GB of files cannot be fully GPU-resident on a nominal
32 GB card. The server's memory fitter keeps the configured reserve free and
moves the remaining weights to host memory, so performance varies with the
fitted placement.

GGUF remains an explicit speed-first choice rather than the automatic Qwen3
preset. The optimized ConvRot path now reaches 20.40-21.84 tok/s on the tested
Captioner variants, while the Transformers processor preserves native temporal
A/V token interleaving and a much larger frame sample. Automatically trading
that fidelity for the GGUF frame-plus-audio fallback would be surprising.

## VRAM-tier server plan

The backend keeps the tier-specific context limits but delegates weight
placement to llama.cpp b10621. Every tier leaves `-ngl` unset and requests `--fit on` with a
target margin of the configured VRAM reserve plus 1,536 MiB of multimodal-projector headroom
(the fitter does not size the mtmd image/audio buffers; measured on a 32 GB GPU a bare 2,048 MiB
target left only ~0.7 GiB free at generation peak). With the default 2 GB reserve that is a
3,584 MiB target margin. This replaces the old fixed 36-layer Q8 plan: Q4 and
Q8 now use the same fitting policy, while larger cards naturally retain more
weights in VRAM.

| Detected tier | Context | GPU request | Fitter target |
|---:|---:|---:|---:|
| 16 GB or less | 8,192 | unset (fitter decides) | 3,584 MiB free |
| 24 GB | 16,384 | unset (fitter decides) | 3,584 MiB free |
| 32 GB or more | 32,768 | unset (fitter decides) | 3,584 MiB free |

For example, the 24 GB command is:

```text
llama-server --model <model.gguf> --mmproj <mmproj.gguf> -c 16384 --jinja --fit on --fit-target 3584 --device CUDA0 --split-mode none --main-gpu 0 --port <free> --host 127.0.0.1
```

`--fit-target` is a per-device margin in MiB; a single value is broadcast to
all selected devices. The application derives it from the load plan's
`vram_reserve_gb` setting (2.0 GB by default). `--fit-ctx` is not passed: the
application sets `-c` from the table, while that option only sets the minimum
context size that the fitter may choose for an unset context.

`CUDA0` is the process-local device in an isolated worker and maps to the
physical GPU chosen in settings. In in-process mode the command uses that GPU's
CUDA index directly. Diagnostic environment overrides are:

| Variable | Effect |
|---|---|
| `VCAP_LLAMACPP_CONTEXT_SIZE` | Overrides the tier context (minimum 4,096). |
| `VCAP_LLAMACPP_GPU_LAYERS` | Sends an explicit `-ngl`. llama.cpp then refuses to fit (`n_gpu_layers already set by user ... abort`) and will page a too-large model through WDDM, so pair it with `VCAP_LLAMACPP_N_CPU_MOE` or `VCAP_LLAMACPP_FIT=0` deliberately. |
| `VCAP_LLAMACPP_FIT_TARGET_MIB` | Overrides the target free margin in MiB. |
| `VCAP_LLAMACPP_FIT=0` | Sends `--fit off`. |
| `VCAP_LLAMACPP_N_CPU_MOE=N` | Sends `--n-cpu-moe N`, keeping the MoE weights of the first N layers on CPU. |

At startup the backend reads the final `n_gpu_layers`, `n_cpu_moe` or tensor
override lines, and `n_ctx` from the server log. The JSON-safe result is
available through `block_swap_summary()` and is stored in the common load
report's `block_swap` field.

## Server and API

The multimodal projector remains GPU-offloaded (the llama.cpp default).
`--no-mmproj-offload` was not needed for Q4 on the 32 GB test card. No `-ctk`
or `-ctv` option is ever supplied: the default F16 KV cache avoids the current
[Qwen3-Omni audio failure reported with quantized
KV](https://github.com/ggml-org/llama.cpp/issues/27136).

Startup is gated by `GET /health`. Captions stream from
`POST /v1/chat/completions` as SSE. The [pinned server
documentation](https://github.com/ggml-org/llama.cpp/blob/b10621/tools/server/README.md)
defines the OpenAI-compatible `image_url`, `input_audio`, and `input_video`
content parts used here. Audio is always a base64 PCM WAV. Sampling controls
map to `temperature`, `top_p`, `top_k`, `repeat_penalty`, and `max_tokens`.
Thinking requests preserve raw reasoning and split `<think>...</think>` from
the final caption.

The server remains loaded until model-cache unload, variant switch, process
exit, or cancellation. Cancellation closes the streaming response and kills
the complete server process tree.

## Video path

llama.cpp b10621 documents `input_video` on the OpenAI chat endpoint. The app
nevertheless defaults to `frames_audio`: it uniformly extracts 8-16
chronological frames with `vcap.core.media.read_video_frames`, sends them as
images, and sends the matching decoded audio as one `input_audio` part. This
avoids [an unresolved llama.cpp issue](https://github.com/ggml-org/llama.cpp/issues/27587)
in which native videos longer than roughly 10-13 seconds can hang a request.

This fallback loses the processor's native timestamp-level A/V token
interleaving. Frames remain ordered, but the audio follows as a separate media
part. Native server video can be tested with
`VCAP_LLAMACPP_VIDEO_MODE=native` or smoke-tool `--video-mode native`.

## Smoke test

The first command downloads about 19.9 GB. Downloads are resumable and
SHA-256 verified against the repositories' LFS metadata.

```powershell
venv\Scripts\python.exe tools\smoke\caption_gguf.py `
  --variant qwen3_omni_instruct_gguf_q4 `
  --input F:\SECourses_Video_Captioner_Pro_TEMP\test_media\lightning_storm_20s.mp4
```

## Verified measurements

Measurements below are real runs on physical GPU 0, an NVIDIA GeForce RTX 5090
with 31.84 GiB VRAM and driver 610.88. Peak VRAM is sampled through
`vcap.core.gpu.resource_snapshot`, so it is process-external GPU usage rather
than a Torch allocator number. Times include server startup but exclude the
already-completed model download.

| Test | Load | Decode | Peak VRAM | Output |
|---|---:|---:|---:|---|
| Instruct Q4, 20.02 s lightning video, 16 frames + audio | 28.93 s | 241.87 tok/s | 31.81 GiB | 171 tokens; detailed nighttime street and heavy-rain description |
| Captioner Q4, extracted 20.02 s audio | 12.03 s | 306.20 tok/s | 31.80 GiB | 465 tokens; detailed rain and thunder field-recording description |

Both requests completed through `/v1/chat/completions` with default mmproj GPU
offload and F16 KV. The Instruct run processed 4,450 prompt tokens in 8.93 s;
the Captioner run processed 283 prompt tokens in 0.26 s. The high peak was
transient: Instruct was at 23.51 GiB immediately after generation. The 24 GB
decode throughput was not separately benchmarked on 24 GB hardware. These
measurements predate the explicit 2,048 MiB fit target and remain a throughput
baseline rather than a promise about the newly fitted placement.

The GGUF contains only the Qwen thinker plus multimodal projector. Speech
generation (`talker` / `code2wav`) is not present, which is immaterial for text
captioning.
