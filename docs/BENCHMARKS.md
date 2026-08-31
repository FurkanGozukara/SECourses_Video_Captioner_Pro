# Benchmarks

Updated: 2026-09-01

## Test system and method

- GPU: NVIDIA GeForce RTX 5090, physical GPU 0, 31.84 GiB dedicated VRAM, driver 610.88.
- Isolation: every GPU command used `CUDA_VISIBLE_DEVICES=0`; GPU 1 was not used.
- Transformers stack: Python 3.12.10, PyTorch 2.13.0+cu130, Transformers 5.16.1, BF16 activations, and Triton Windows 3.7.1.
- Checkpoint sizes are decimal GB. Transformers peak values are PyTorch allocator high-water marks in GiB. GGUF peaks are process-external GPU samples and include the private `llama-server` process.
- A row marked **fresh C6 3-run EOS mean** used `tools/bench/benchmark.py` through the application's real loader and caption wrapper. Load time is the one load before three generations, peak VRAM is the maximum, and rates/tokens/wall time are arithmetic means. Seeds were 1234, 1235, and 1236.
- C6 video rows used `lightning_storm_20s.mp4`, the application's 32 GB production profile, and a 2,048-token ceiling. Thinking used `enable_thinking=True` and the sampling defaults (`temperature=0.6`, `top_p=0.95`, `top_k=20`); all 12 C6 generations stopped at EOS before the ceiling. Earlier fresh video rows used at most 512 new tokens. Qwen3-Omni Instruct used Flash Attention 2 and `max_pixels=256*32*32` (5,415 prompt tokens), avoiding the WDDM paging failure of the legacy 401,408-pixel SDPA profile.
- Fresh Captioner speed rows used `demon_singer_audio_18_sec.mp3`, SDPA, and at most 512 new tokens. The C1a INT4 quality run used the same audio, a 2,048-token ceiling, and stopped at EOS after 737 tokens.

Windows WDDM can page allocations into shared system memory. A peak above 31.84 GiB therefore does not mean the card has more physical VRAM; it means that run paged and may be much slower.

**Qwen3 BF16: 63.4 GB — does not fit a single 32 GB GPU; skipped (text-only logits reference was measured with CPU offload).**

## Consolidated comparison

| Model family | Variant | Checkpoint GB | Load s | Peak VRAM GiB | Prefill tok/s | Decode tok/s | Quality vs BF16 | Measurement |
|---|---|---:|---:|---:|---:|---:|---|---|
| TimeChat | BF16 | 17.880 | 9.74 | 31.52 | 212.28 | 34.44 | KL 0; max \|Δlogit\| 0 (exact) | fresh 3-run mean |
| TimeChat | INT8 ConvRot | 10.275 | 7.70 | 24.40 | 1,904.59 | 34.54 | KL 0.0108539; max \|Δlogit\| 0.4375 | fresh 3-run mean |
| TimeChat | INT4 ConvRot W4A8 | 6.468 | 5.24 | 20.86 | 1,834.54 | 34.73 | KL 0.150814; max \|Δlogit\| 2.84375 | fresh 3-run mean |
| AVoCaDO | BF16 | 17.864 | 24.21 | 20.45 | 6,554.05 | 36.03 | KL 0; max \|Δlogit\| 0 (exact) | fresh C6 3-run EOS mean |
| AVoCaDO | INT8 ConvRot | 10.275 | 6.77 | 35.96 | 103.52 | 31.22 | KL 0.00212637; max \|Δlogit\| 0.40625 | fresh 3-run mean |
| AVoCaDO | INT4 ConvRot W4A8 | 6.452 | 16.15 | 10.45 | 3,949.54 | 29.11 | KL 0.147627; max \|Δlogit\| 2.54492 | fresh C6 3-run EOS mean |
| Qwen3-Omni Instruct | BF16 | 63.4 | — | — | — | — | Text-only logits reference used CPU offload; media skipped | skipped: does not fit 32 GB |
| Qwen3-Omni Instruct | INT8 ConvRot | 33.041 | 13.32 | 31.38 | 437.07 | 12.08 | KL 1.65173e-05; max \|Δlogit\| 0.377686 | fresh 3-run mean |
| Qwen3-Omni Instruct | INT4 ConvRot W4A8 | 17.790 | 16.75 | 18.84 | 5,692.81 | 17.91 | KL 0.0292324; max \|Δlogit\| 3.27734 | fresh 3-run mean |
| Qwen3-Omni Instruct | GGUF Q4_K_M + Q8_0 mmproj | 19.882 | 28.93 | 31.81 | 498.32* | 241.87 | No BF16 logit comparison; coherent 171-token storm caption | GGUF_BACKEND measured run |
| Qwen3-Omni Instruct | GGUF Q8_0 + Q8_0 mmproj | 33.810 | — | — | — | — | not measured (not downloaded) | not measured (not downloaded) |
| Qwen3-Omni Thinking | BF16 | 63.4 | — | — | — | — | Media skipped: same single-GPU size limit | skipped: does not fit 32 GB |
| Qwen3-Omni Thinking | INT8 ConvRot | 33.031 | 34.85 | 33.07 | 90.41 | 3.90 | No BF16 logit comparison; 3/3 coherent storm captions with reasoning split | fresh C6 3-run EOS mean |
| Qwen3-Omni Thinking | INT4 ConvRot W4A8 | 17.780 | 21.17 | 18.87 | 4,451.80 | 6.97 | No BF16 logit comparison; 3/3 coherent storm captions with reasoning split | fresh C6 3-run EOS mean |
| Qwen3-Omni Thinking | GGUF Q4_K_M + Q8_0 mmproj | 19.882 | — | — | — | — | not measured (not downloaded) | not measured (not downloaded) |
| Qwen3-Omni Thinking | GGUF Q8_0 + Q8_0 mmproj | 33.810 | — | — | — | — | not measured (not downloaded) | not measured (not downloaded) |
| Qwen3-Omni Captioner | BF16 | 63.4 | — | — | — | — | Text-only logits reference used CPU offload; media skipped | skipped: does not fit 32 GB |
| Qwen3-Omni Captioner | INT8 ConvRot | 33.041 | 22.42 | 29.94 | 584.26 | 20.49 | KL 0.00339146; max \|Δlogit\| 0.75 | fresh 3-run mean |
| Qwen3-Omni Captioner | INT4 ConvRot W4A8 | 17.790 | 13.93 | 16.74 | 1,188.83 | 20.13 | No BF16 logit comparison; coherent 737-token speech/music caption stopped at EOS | fresh 3-run mean; C1a quality run |
| Qwen3-Omni Captioner | GGUF Q4_K_M + Q8_0 mmproj | 19.882 | 12.03 | 31.80 | 1,088.46* | 306.20 | No BF16 logit comparison; coherent 465-token rain/thunder caption | GGUF_BACKEND measured run |
| Qwen3-Omni Captioner | GGUF Q8_0 + Q8_0 mmproj | 33.810 | — | — | — | — | not measured (not downloaded) | not measured (not downloaded) |

`*` GGUF prefill rates are derived from the reported prompt counts and prefill times: 4,450 / 8.93 s for Instruct and 283 / 0.26 s for Captioner. The GGUF tests used the backend's frame-plus-audio fallback and are not token-for-token comparisons with the Transformers processor. Q4/Q8 checkpoint sizes include the required 1.325 GB Q8_0 multimodal projector.

Quantitative quality values come from the text-only logits verifier in [QUANT_REPORT.md](QUANT_REPORT.md). Rows without a BF16 logit comparison report only a caption sanity check. Caption content remains model- and prompt-dependent, so speed rows should not be read as a quality ranking.

The local Q8_0 folders for Instruct and Captioner were absent, as were both Thinking GGUF folders. Per the no-download benchmark rule, those rows remain explicitly marked **not measured (not downloaded)**; no other fitting local row remains unmeasured.

### EOS-stop wall clock

This is the practical elapsed time for one 20 s clip, including media decoding and model preprocessing but excluding the one-time model load. Unlike the older capped measurements, every run below completed naturally.

| Model | Mean generated tokens | Seconds per 20 s clip | Finish reasons |
|---|---:|---:|---|
| AVoCaDO BF16 | 272.0 | 13.90 | eos / eos / eos |
| AVoCaDO INT4 | 249.0 | 15.48 | eos / eos / eos |
| Qwen3-Omni Thinking INT8 | 577.7 | 211.86 | eos / eos / eos |
| Qwen3-Omni Thinking INT4 | 525.7 | 80.61 | eos / eos / eos |

## Which variant should I pick?

These are the automatic production choices encoded in `vcap/models/vram_presets.py`. "Full" means the preset keeps the relevant weights resident; lower tiers use the CPU placement shown and will be slower.

| VRAM tier | TimeChat / AVoCaDO | Qwen3-Omni Instruct / Thinking / Captioner |
|---:|---|---|
| 6 GB | **INT4**; heavy CPU offload, 32 frames | Not offered |
| 8 GB | **INT4**; partial decoder offload, conservative video | **INT4** experimental; four GPU layers, experts on CPU |
| 10 GB | **INT4** fully resident at training resolution | **INT4** with expert offload and conservative context |
| 12 GB | **INT8** with a small CPU-offloaded decoder tail | **INT4** with expert offload and conservative context |
| 16 GB | **INT8** fully resident | **INT4** with a light CPU-offloaded decoder tail |
| 24 GB | **BF16** fully resident | **INT4** fully resident; GGUF Q4_K_M is an optional speed-first choice |
| 32 GB | **BF16** fully resident; it was also faster than INT4 in the EOS AVoCaDO run | Automatic **INT8** prioritizes precision, but prefer **INT4 for Thinking throughput/headroom** (80.61 vs 211.86 s/clip); GGUF Q4 is the speed-first alternative when downloaded |
| 48 GB | **BF16** fully resident | **INT8** fully resident; GGUF Q4/Q8 fully offloaded |
| 80 GB | **BF16** fully resident | **BF16** fully resident |

## Source reports

- [Quantization verification and quality](QUANT_REPORT.md)
- [Qwen3-Omni ConvRot performance](QUANT_PERF.md)
- [Qwen3-Omni GGUF backend](GGUF_BACKEND.md)

## v1.2.0 live matrix (2026-09-01)

These end-to-end application runs used physical GPU 0, an NVIDIA GeForce RTX 5090, and were driven through Google Chrome. Times include model loading unless the result states otherwise. Generation stopped at EOS except where noted. This newer availability matrix supersedes the earlier "not downloaded" annotations above.

| Variant | Input | Result |
|---|---|---|
| `timechat_bf16` | 20 s video | 1,119 tokens at 33.7 tok/s; peak 16.7 GiB |
| `timechat_int8` | 20 s video, scene split into 5 clips | About 26 tok/s; SRT, clips, and sidecars verified |
| `timechat_int4` | 20 s video after a fresh-install download | About 25 tok/s |
| `avocado_bf16` | 20 s video, whole-file mode | 249 tokens at 30 tok/s |
| `avocado_int8` | 20 s video, whole-file mode | 330 tokens at 22 tok/s |
| `avocado_int4` | Video trimmed to 5-15 s and split into 3 clips | About 25 tok/s; trim preserved in metadata and timestamps |
| `qwen3_omni_instruct_bf16` (63.4 GB) | Image | Loaded through Accelerate CPU offload in 97.5 s; peak 51.6 GiB with WDDM spill; 0.42 tok/s; stopped at the configured length cap |
| `qwen3_omni_instruct_int8` | Image | 234 tokens at 4.3 tok/s; peak 30.2 GiB |
| `qwen3_omni_instruct_int4` | Unicode mixed batch: 2 videos, image, and WAV in a subfolder; sidecar excluded | 4/4 completed in 100.7 s with per-modality prompt adaptation; rerun skipped all 4 in 0.09 s |
| `qwen3_omni_instruct_gguf_q4` | 20 s video split into 5 clips | 30.6 s total; about 270 tok/s decode |
| `qwen3_omni_instruct_gguf_q8` | 20 s video, whole-file mode | Server completed in 15.5 s; peak 28.0 GiB; 9.6 tok/s |
| `qwen3_omni_thinking_int8` | Image | 497 tokens at 4.6 tok/s; separate reasoning file verified |
| `qwen3_omni_thinking_int4` | Image | 725 tokens at 12.4 tok/s; 2.4 KB reasoning file |
| `qwen3_omni_thinking_gguf_q4` | 20 s video | 784 tokens at 163 tok/s; reasoning verified |
| `qwen3_omni_thinking_gguf_q8` | 20 s video | 1,281 tokens at 26.5 tok/s; reasoning verified |
| `qwen3_omni_captioner_bf16` (63.4 GB) | 15 s MP3 | EOS after more than 450 tokens at about 0.5 tok/s; 1,347.7 s total including the CPU-offload load; audio tower functional under CPU offload |
| `qwen3_omni_captioner_int8` | 15 s MP3 | 492 tokens at 4.5 tok/s |
| `qwen3_omni_captioner_int4` | 15 s MP3 with prefix, suffix, replacement, and all 5 output formats | TXT, JSON, JSONL, SRT, and VTT verified with post-processing applied |
| `qwen3_omni_captioner_gguf_q4` | 15 s MP3 | 447 tokens at 300 tok/s |
| `qwen3_omni_captioner_gguf_q8` | 15 s MP3 | 481 tokens at 41.5 tok/s |
| `qwen3_omni_thinking_bf16` | Not run | Skipped because it uses the same proven 63.4 GB loader/offload path as Instruct BF16, while its reasoning path was covered by INT4, INT8, and both GGUF variants |
