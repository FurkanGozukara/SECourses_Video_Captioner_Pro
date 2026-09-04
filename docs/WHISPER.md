# Whisper Backend

SECourses Video Captioner Pro runs speech transcription in a dedicated Python
process. The parent writes a request below `temp/whisper/`, starts
`python -m vcap.whisper.worker`, and consumes JSON-lines events through
`vcap.core.subprocess_runner.WorkerProcess`. The child never imports PyTorch.
It selects the requested physical GPU with `CUDA_VISIBLE_DEVICES`; CTranslate2
therefore always receives local `device_index=0`.

## Installed Runtime

The development environment was verified on 2026-09-04 with:

| Package | Installed version |
| --- | ---: |
| `faster-whisper` | 1.2.1 |
| `ctranslate2` | 4.8.2 |
| `onnxruntime` | 1.29.0 |
| `nvidia-cublas-cu12` | 12.9.2.10 |

The Windows CTranslate2 wheel contains `ctranslate2/cudnn64_9.dll`, so a
separate Windows `nvidia-cudnn-cu12` installation was not needed. Linux installs
`nvidia-cudnn-cu12` through the requirements marker.

## Model Catalogue

Models are downloaded as visible, resumable Hugging Face snapshots below
`models/whisper/<alias>/`. Sizes are decimal units and were obtained by summing
`config.json`, `preprocessor_config.json`, `model.bin`, `tokenizer.json`, and
`vocabulary.*` metadata from each repository.

| Alias | Hugging Face repository | Bytes | Size |
| --- | --- | ---: | ---: |
| `large-v1` | `Systran/faster-whisper-large-v1` | 3,089,578,414 | 3.09 GB |
| `large-v3` | `Systran/faster-whisper-large-v3` | 3,090,835,702 | 3.09 GB |
| `large-v3-turbo` | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | 1,621,665,983 | 1.62 GB |
| `large-v2` | `Systran/faster-whisper-large-v2` | 3,089,578,858 | 3.09 GB |
| `distil-large-v3.5` | `distil-whisper/distil-large-v3.5-ct2` | 1,516,479,656 | 1.52 GB |
| `distil-large-v3` | `Systran/faster-distil-whisper-large-v3` | 1,516,479,628 | 1.52 GB |
| `distil-large-v2` | `Systran/faster-distil-whisper-large-v2` | 1,516,108,953 | 1.52 GB |
| `medium` | `Systran/faster-whisper-medium` | 1,530,571,735 | 1.53 GB |
| `medium.en` | `Systran/faster-whisper-medium.en` | 1,530,457,748 | 1.53 GB |
| `distil-medium.en` | `Systran/faster-distil-whisper-medium.en` | 792,060,626 | 792.1 MB |
| `small` | `Systran/faster-whisper-small` | 486,212,372 | 486.2 MB |
| `small.en` | `Systran/faster-whisper-small.en` | 486,098,798 | 486.1 MB |
| `distil-small.en` | `Systran/faster-distil-whisper-small.en` | 335,542,354 | 335.5 MB |
| `base` | `Systran/faster-whisper-base` | 147,882,941 | 147.9 MB |
| `base.en` | `Systran/faster-whisper-base.en` | 147,769,510 | 147.8 MB |
| `tiny` | `Systran/faster-whisper-tiny` | 78,203,619 | 78.2 MB |
| `tiny.en` | `Systran/faster-whisper-tiny.en` | 78,090,594 | 78.1 MB |

`Models_Downloader.py --ensure whisper:<alias>` downloads or resumes a model;
`--verify whisper:<alias>` checks for `config.json`, `model.bin`, and the absence
of `.part` or `.incomplete` files. Downloads require the remaining bytes plus
five percent free disk headroom. A cooperative cancellation leaves Hugging Face
resume data and an `.incomplete` guard; the next successful resume removes stale
guards only after the required model files are present. The parent sends the
worker a cancel line first, then terminates its process tree after three seconds
if a native transfer has not yielded control.

Worker download checks on 2026-09-04 completed a fresh `tiny` download in
4.408 s. A second run was cancelled with 66,991,736 local bytes and readiness
false, then resumed to ready in 4.388 s. A fresh `large-v1` run was cancelled
first (readiness false), resumed the 3.09 GB transfer in 34.937 s, and passed a
final idempotent readiness run in 0.796 s.

## Transcription Parameters

`WhisperParams` contains the complete standard faster-whisper decoding surface:
language or automatic detection, translate-to-English, device and compute type,
beam search and sampling values, repetition controls, fallback thresholds,
prompt/prefix/hotwords, token suppression, word timestamps and punctuation,
chunk and language-detection settings, and batched inference. Numeric UI input
is coerced and clamped before it reaches the worker.

The shipped **Transcribe - Whisper best quality (large-v1)** preset uses
`large-v1`, English, float16, beam 5, best-of 5, patience 1.0, temperature 0,
repetition penalty 1.2, 30-second windows, previous-text conditioning, word
timestamps, and batch size 1. A
positive `max_new_tokens` is cleared when any prompt source is active; otherwise
it is capped at 432 tokens. Previous-text conditioning is automatically disabled
at 60 or more windows to limit long-form repetition drift.

Optional Silero VAD uses faster-whisper's bundled ONNX model with
`CPUExecutionProvider`, 16 kHz mono audio, and 512-sample windows. Removed
silence is mapped back into source timestamps for both segments and words.

## Outputs

All outputs are UTF-8:

| Format | Contents |
| --- | --- |
| SRT | SubRip cues with millisecond timestamps |
| VTT | WebVTT cues |
| TXT | One transcript segment per line |
| LRC | Bracketed segment times, or aligned word times in highlight mode |
| TSV | Integer millisecond `start`, `end`, and `text` columns |
| JSON | Pretty-printed `TranscriptResult.to_dict()` |

Word normalization rebuilds subtitle segments at sentence and clause boundaries,
with safeguards of 92 characters, 24 words, or 8 seconds. Without normalization,
word highlighting emits `<u>` cues and an additional plain
`_noword_timestamps.srt` companion. Trimmed jobs add the source trim offset back
to every segment and word timestamp.

## CUDA Runtime

PyTorch in this application uses CUDA 13 (`cu130`), while the current
CTranslate2 wheel is built against CUDA 12 user-space libraries. Importing torch
does not solve that ABI requirement, and the Whisper worker deliberately never
imports it. `enable_cuda_runtime_autodiscovery()` discovers pip-installed
`nvidia/cublas`, `nvidia/cudnn`, and `nvidia/cuda_runtime` directories plus CUDA
toolkit environment paths. It prepends the platform library path, registers DLL
directories on Windows, and preloads CUDA runtime, cuBLAS, cuBLASLt, and cuDNN in
dependency order. This is why `nvidia-cublas-cu12` must be installed beside the
application's CUDA 13 PyTorch stack.

Automatic device selection uses `ctranslate2.get_cuda_device_count()`. An
automatic CUDA load that fails because of cuBLAS, cuDNN, or memory falls back to
CPU `int8`; explicitly forcing CUDA reports the error instead.

## Linux Notes

Linux uses the same worker protocol and visible model layout. The runtime shim
prepends discovered directories to `LD_LIBRARY_PATH` and preloads `.so` files
with global symbol visibility. Install both `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12`, keep an NVIDIA driver compatible with CUDA 12 libraries,
and ensure FFmpeg can decode the source media. CUDA selection remains physical
GPU index based through `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES`.

## Verified Smoke Runs

On physical GPU 0, `large-v1` produced:

- JFK WAV: 11.000 s media, 8.464 s transcription, 11.508 s wall time. Transcript:
  `And so my fellow Americans, ask not what your country can do for you. Ask what YOU CAN DO FOR YOUR COUNTRY!`
  A final warm-cache rerun completed in 0.637 s transcription / 3.712 s wall time
  with the same text.
- Five-minute MP3: 314.329 s media, 16.591 s transcription, 19.181 s wall time.
  A normalized word-sequence comparison against the supplied `test2.srt`
  measured 0.735 similarity (907 hypothesis words versus 1,068 reference
  words). The subject sequence matched (Lindsey Graham, Iran, the Strait of
  Hormuz, enriched uranium, and the closing military options), with expected
  punctuation and word-choice differences.

The worker path was also verified on the JFK WAV: model load plus transcription
completed in 6.646 s and wrote SRT, TXT, and JSON outputs; a final warm-cache
worker run completed in 3.194 s with the same transcript.
