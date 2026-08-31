# Benchmarks

Updated: 2026-08-31

## Test system and method

- GPU: NVIDIA GeForce RTX 5090, physical GPU 0, 31.84 GiB dedicated VRAM, driver 610.88.
- Isolation: every GPU command used `CUDA_VISIBLE_DEVICES=0`; GPU 1 was not used.
- Transformers stack: Python 3.12, PyTorch 2.13.0+cu130, Transformers 5.16.1, BF16 activations, and Triton Windows 3.7.1.
- Checkpoint sizes are decimal GB. Transformers peak values are PyTorch allocator high-water marks in GiB. GGUF peaks are process-external GPU samples and include the private `llama-server` process.
- A row marked **fresh 3-run mean** used `tools/bench/benchmark.py`. Load time is the one load before the three generations, peak VRAM is the maximum, and prefill/decode rates are arithmetic means across the three generations.
- Fresh video rows used `lightning_storm_20s.mp4` and at most 512 new tokens. Qwen3-Omni Instruct used the application's 32 GB production profile: Flash Attention 2 and `max_pixels=256*32*32` (5,415 prompt tokens). This avoids the known WDDM paging failure of the legacy 401,408-pixel SDPA profile.
- Fresh Captioner rows used `demon_singer_audio_18_sec.mp3`, SDPA, and at most 512 new tokens.

Windows WDDM can page allocations into shared system memory. A peak above 31.84 GiB therefore does not mean the card has more physical VRAM; it means that run paged and may be much slower.

**Qwen3 BF16: 63.4 GB — does not fit a single 32 GB GPU; skipped (text-only logits reference was measured with CPU offload).**

## Consolidated comparison

| Model family | Variant | Checkpoint GB | Load s | Peak VRAM GiB | Prefill tok/s | Decode tok/s | Quality vs BF16 | Measurement |
|---|---|---:|---:|---:|---:|---:|---|---|
| TimeChat | BF16 | 17.880 | 9.74 | 31.52 | 212.28 | 34.44 | KL 0; max \|Δlogit\| 0 (exact) | fresh 3-run mean |
| TimeChat | INT8 ConvRot | 10.275 | 7.70 | 24.40 | 1,904.59 | 34.54 | KL 0.0108539; max \|Δlogit\| 0.4375 | fresh 3-run mean |
| TimeChat | INT4 ConvRot W4A8 | 6.468 | 5.24 | 20.86 | 1,834.54 | 34.73 | KL 0.150814; max \|Δlogit\| 2.84375 | fresh 3-run mean |
| AVoCaDO | BF16 | 17.880 | 6.02 | 43.08 | 58.03 | 30.03 | KL 0; max \|Δlogit\| 0 (exact) | QUANT_REPORT measured run |
| AVoCaDO | INT8 ConvRot | 10.275 | 6.77 | 35.96 | 103.52 | 31.22 | KL 0.00212637; max \|Δlogit\| 0.40625 | fresh 3-run mean |
| AVoCaDO | INT4 ConvRot W4A8 | 6.468 | 6.44 | 32.41 | 152.73 | 29.06 | KL 0.147627; max \|Δlogit\| 2.54492 | QUANT_REPORT measured run |
| Qwen3-Omni Instruct | BF16 | 63.4 | — | — | — | — | Text-only logits reference used CPU offload; media skipped | skipped: does not fit 32 GB |
| Qwen3-Omni Instruct | INT8 ConvRot | 33.041 | 13.32 | 31.38 | 437.07 | 12.08 | KL 1.65173e-05; max \|Δlogit\| 0.377686 | fresh 3-run mean |
| Qwen3-Omni Instruct | INT4 ConvRot W4A8 | 17.790 | 16.75 | 18.84 | 5,692.81 | 17.91 | KL 0.0292324; max \|Δlogit\| 3.27734 | fresh 3-run mean |
| Qwen3-Omni Instruct | GGUF Q4_K_M + Q8_0 mmproj | 19.882 | 28.93 | 31.81 | 498.32* | 241.87 | No BF16 logit comparison; coherent 171-token storm caption | GGUF_BACKEND measured run |
| Qwen3-Omni Instruct | GGUF Q8_0 + Q8_0 mmproj | 33.810 | — | — | — | — | Not measured | not measured |
| Qwen3-Omni Thinking | BF16 | 63.4 | — | — | — | — | Not measured; same single-GPU size limit | skipped: does not fit 32 GB |
| Qwen3-Omni Thinking | INT8 ConvRot | 33.041 | — | — | — | — | Not measured | not measured |
| Qwen3-Omni Thinking | INT4 ConvRot W4A8 | 17.790 | — | — | — | — | Not measured | not measured |
| Qwen3-Omni Thinking | GGUF Q4_K_M + Q8_0 mmproj | 19.882 | — | — | — | — | Not measured | not measured |
| Qwen3-Omni Thinking | GGUF Q8_0 + Q8_0 mmproj | 33.810 | — | — | — | — | Not measured | not measured |
| Qwen3-Omni Captioner | BF16 | 63.4 | — | — | — | — | Text-only logits reference used CPU offload; media skipped | skipped: does not fit 32 GB |
| Qwen3-Omni Captioner | INT8 ConvRot | 33.041 | 22.42 | 29.94 | 584.26 | 20.49 | KL 0.00339146; max \|Δlogit\| 0.75 | fresh 3-run mean |
| Qwen3-Omni Captioner | INT4 ConvRot W4A8 | 17.790 | 13.93 | 16.74 | 1,188.83 | 20.13 | Not measured | fresh 3-run mean |
| Qwen3-Omni Captioner | GGUF Q4_K_M + Q8_0 mmproj | 19.882 | 12.03 | 31.80 | 1,088.46* | 306.20 | No BF16 logit comparison; coherent 465-token rain/thunder caption | GGUF_BACKEND measured run |
| Qwen3-Omni Captioner | GGUF Q8_0 + Q8_0 mmproj | 33.810 | — | — | — | — | Not measured | not measured |

`*` GGUF prefill rates are derived from the reported prompt counts and prefill times: 4,450 / 8.93 s for Instruct and 283 / 0.26 s for Captioner. The GGUF tests used the backend's frame-plus-audio fallback and are not token-for-token comparisons with the Transformers processor. Q4/Q8 checkpoint sizes include the required 1.325 GB Q8_0 multimodal projector.

Quality columns come from the text-only logits verifier in [QUANT_REPORT.md](QUANT_REPORT.md); the fresh rows replace only the load, peak, prefill, and decode measurements. Caption content remains model- and prompt-dependent, so speed rows should not be read as a quality ranking.

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
| 32 GB | **BF16** fully resident | **INT8** with a small decoder tail offloaded; INT4 for more headroom; GGUF Q4 fully offloaded or Q8 partially offloaded |
| 48 GB | **BF16** fully resident | **INT8** fully resident; GGUF Q4/Q8 fully offloaded |
| 80 GB | **BF16** fully resident | **BF16** fully resident |

## Source reports

- [Quantization verification and quality](QUANT_REPORT.md)
- [Qwen3-Omni ConvRot performance](QUANT_PERF.md)
- [Qwen3-Omni GGUF backend](GGUF_BACKEND.md)
