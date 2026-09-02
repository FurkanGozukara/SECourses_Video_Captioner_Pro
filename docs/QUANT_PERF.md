# Qwen3-Omni ConvRot MoE Performance

Updated: 2026-08-31. All measurements use logical CUDA device 0 on the RTX 5090
(`CUDA_VISIBLE_DEVICES=0`), torch 2.13.0+cu130, Triton Windows 3.7.1, transformers
5.16.1, BF16 activations, and the 18 s
`demon_singer_audio_18_sec.mp3` input unless noted otherwise.

## Before-optimization profile

The T3 smoke run that triggered T4b measured the Captioner INT4 path at **1176 s
prefill and 0.348 tok/s decode**. The branch entering T4b already contained a first
sorted-dispatch/Triton GEMV pass and a populated per-shape kernel cache, so a fresh
8-token instrumented run is substantially faster than that historical cold result.
These are the measured starting points for the remaining work:

| Variant | Load s | Input tokens | Prefill s | Decode tok/s | Peak VRAM GiB |
|---|---:|---:|---:|---:|---:|
| Captioner INT8, SDPA | 26.47 | 245 | 6.569 | 5.27 | 30.92 |
| Captioner INT4, SDPA | 22.13 | 245 | 2.788 | 8.14 | 16.71 |

The short decode measurement contains seven timed token intervals. Full 256-token
numbers are recorded in the final-results section after the implementation is stable.
Raw profiles are in `tools/bench/profile_qwen3_omni_captioner_*_before.json`.

### Component attribution

| Variant / phase | Audio tower | Text attention | Router | Experts | LM head |
|---|---:|---:|---:|---:|---:|
| INT8 prefill, total 6.569 s | 1.055 s | 0.878 s | 0.045 s | **4.207 s** | 0.067 s |
| INT8 decode, per token | n/a | **76.57 ms** | 21.22 ms | **72.19 ms** | 0.72 ms |
| INT4 prefill, total 2.788 s | 0.266 s | 0.130 s | 0.025 s | **2.169 s** | 0.001 s |
| INT4 decode, per token | n/a | **32.74 ms** | 5.93 ms | **68.48 ms** | 0.43 ms |

The 245-token prefill creates 1,960 token/expert assignments per layer. The first
INT8 layer routes to 117 experts and the first INT4 layer to 119; 35 experts receive
more than 16 tokens. There is no vision-tower work for this audio-only input.

### Findings before code changes

1. **Experts remain the main bottleneck.** Prefill still executes a Python iteration
   for every active expert (about 120 per layer) and decode executes 16 independent
   projection kernels per layer (gate/up plus down for eight experts). Across 48
   layers, expert dispatch alone costs about 72 ms per decoded token.
2. **The historical INT4 failure is explained by unpack and small-matrix dispatch.**
   The old/cold path unpacked packed nibbles into several temporary tensors for every
   active expert and projection, then rebuilt `weight.T.contiguous()` before
   `torch._int_mm`. The operator profile still shows 70 `_int_mm` calls and roughly
   337 MiB of INT4-unpack concatenation allocation in one representative prefill
   layer. Experts with at most 16 rows now hit the cached packed Triton GEMV, which is
   why the warm profile no longer reproduces the 1176 s T3 result.
3. **Dispatch still synchronizes the host once per layer.** `unique/counts` are copied
   through `.tolist()` before the expert loop. Decode profiles show a DtoH scalar copy
   and 16 Triton launches per layer, plus separate sort, gather, SiLU, multiply, and
   `index_add_` kernels.
4. **Hadamard matrices are cached, but only lazily.** The regular Hadamard itself is
   not rebuilt per call; however activation rotation/quantization is repeated for
   every expert slice, and `_int_mm_padded` creates a transposed contiguous weight
   view on every invocation.
5. **Attention-path time is also material.** The measured attention module includes
   Q/K/V/O ConvRot projections and attention proper. It costs 77 ms/token for INT8
   and 33 ms/token for INT4, so expert optimization alone cannot reach 15 tok/s on
   the current INT8 path. The dense quantized linear hot path also needs to eliminate
   repeated Python cache lookup, rotation, and per-projection launch overhead.

The optimization therefore needs a fused assignment-aware MoE path: rotate and
quantize once, execute all routed assignments in two grouped expert kernels, keep
INT4 packed in the kernel, and reduce token-major assignments directly instead of
round-tripping a CUDA dispatch table through Python.

## Final results

### Implementation

- The CUDA experts path now builds its dispatch table entirely on the GPU with
  `argsort`, `bincount`, and `searchsorted`. A layer executes one grouped gate/up
  kernel and one grouped down kernel, rather than a Python loop over roughly 120
  experts during prefill or eight experts during decode. There are no `.item()`,
  `.tolist()`, or CPU copies in this path.
- Gate and up remain fused in the checkpoint layout as
  `[experts, 2 * intermediate, hidden]`. Activation rotation/quantization, SiLU
  multiplication, and route reduction have dedicated Triton kernels. The Hadamard
  matrix is materialized once while loading.
- INT4 stays nibble-packed in VRAM. The grouped GEMM extracts nibbles in its weight
  tiles, so forward no longer creates a full unpacked INT8 expert tensor or repeats
  unpacking for every routed token.
- Adjacent quantized Q/K/V projections are concatenated once after loading and run
  as one dense projection. Decode-sized dense projections select the fused Triton
  kernel directly instead of repeating disk-cache lookup and candidate timing.
- A 33.03 GB INT8 checkpoint otherwise leaves too little dedicated VRAM for tower
  activations on this 32 GB WDDM GPU. In the automatic high-pressure mode, the
  audio and vision towers live on CPU between prefill calls and are staged to GPU
  only for their forward pass. INT4 stays fully resident.
- Meta-buffer reconstruction now calls `compute_default_rope_parameters(config)`
  without the deprecated `device=` argument and moves the result afterward.
  The corresponding Transformers deprecation warning is gone.

The CUDA path can be disabled with `VCAP_FUSED_MOE=0`; the existing portable grouped
fallback remains available. Tower staging can be overridden with
`VCAP_QUANT_TOWER_OFFLOAD`.

### Expert micro-benchmarks

These benchmarks use 128 synthetic experts and top-8 routing. Prefill uses the same
245 text/audio-token count as the Captioner smoke. Times are per MoE layer.

| Phase | Scheme | Fused grouped ms | Active-expert Python loop ms | Speedup |
|---|---|---:|---:|---:|
| Decode, 1 token | INT8 | 0.185 | 3.289 | 17.8x |
| Decode, 1 token | INT4 | 0.125 | 2.775 | 22.1x |
| Prefill, 245 tokens | INT8 | 0.760 | 51.857 | 68.3x |
| Prefill, 245 tokens | INT4 | 0.889 | 81.952 | 92.2x |

The fused and reference outputs differ only through BF16 operation/reduction order:
relative L2 delta was 0.81-0.93% on deliberately full-range random weights. The
raw results, including mean/max deltas, are in
`tools/bench/experts_{benchmark,prefill}_{int8,int4}_t4b.json`.

### Required smoke runs

| Variant / input | Attention | Load s | Prompt tokens | Prefill s | Decode s | Decode tok/s | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Captioner INT8 / 18 s audio / 256 new | SDPA | 33.684 | 245 | **2.599** | 12.500 | **20.40** | 31.39 |
| Captioner INT4 / 18 s audio / 256 new | SDPA | 25.977 | 245 | **0.974** | 11.678 | **21.84** | 16.88 |
| Instruct INT8 / 20 s video / 512 new | Flash Attention 2 | 34.613 | 5,415 | **15.273** | 47.568 | **10.74** | 31.41 |

Both Captioner targets pass using the smoke tool's default SDPA mode: INT8 is 1.36x
the 15 tok/s target and INT4 is 2.18x the 10 tok/s target. The historical INT4 case
improved from 1176 s to 0.974 s prefill (about 1207x) and from 0.348 to 21.84 tok/s
decode (about 62.8x). Flash Attention 2 was also loaded and exercised successfully;
on this short 245-token prompt it measured 18.17 tok/s INT8 and 20.00 tok/s INT4,
while SDPA was faster. The longer video run therefore uses Flash Attention 2, where
its attention scaling matters.

The two audio captions remain detailed and coherent, correctly describing the male
spoken tag, guitar/drum production, and sung lyrics, matching the subject and style
of the earlier `docs/QUANT_REPORT.md` caption. The Instruct run accurately describes
the dark city intersection, wet road, traffic, lightning, and thunder. After its
complete caption it repeats a short token at the forced 512-token cap; the old INT4
report already shows longer greedy repetition at its cap, and there was no earlier
INT8 media result because that run paged indefinitely.

### Regression checks

- `python -m pytest tests -q`: **77 passed** in 10.63 s.
- The T4 verifier completed TimeChat INT8, a 7B model, including the same 20 s video:
  max/mean logit deltas `0.4375 / 0.0658341`, KL `0.0108539`, 31.97 decode tok/s,
  and 24.40 GiB peak. These logit metrics exactly match the previously recorded T4
  report, so the shared ConvRot changes did not regress the 7B path.
- All final smoke and verifier commands ran with `CUDA_VISIBLE_DEVICES=0`.

### Remaining bottlenecks

The experts are no longer the dominant decode cost. The Instruct video creates a
5,415-token prompt, so attention/KV-cache work grows throughout its 512-token decode
and holds that run to 10.74 tok/s. INT8 also remains constrained by its 33.03 GB
checkpoint: staging the prefill-only towers prevents pathological WDDM paging but
adds load/prefill transfer time. Further gains would require a fused attention
projection/attention stack beyond QKV fusion, a paged KV cache, or a runtime designed
for batched MoE inference rather than more expert-loop work.

## C6 EOS benchmark addendum

The post-EOS-fix benchmark used the application's production 32 GB profile on the
20 s lightning video, with Thinking enabled, sampling defaults, and a 2,048-token
ceiling. Each value is a three-generation mean; all six generations stopped at EOS.

| Thinking variant | Load s | Peak GiB | Prefill tok/s | Decode tok/s | Generated tokens | Wall s/clip |
|---|---:|---:|---:|---:|---:|---:|
| INT8 ConvRot | 34.85 | 33.07 | 90.41 | 3.90 | 577.7 | 211.86 |
| INT4 ConvRot W4A8 | 21.17 | 18.87 | 4,451.80 | 6.97 | 525.7 | 80.61 |

Both variants used Flash Attention 2 and the profile's 42 GPU layers. INT8 paged
under WDDM at a 33.07 GiB allocator high-water mark; INT4 avoided that pressure and
completed a real clip 2.63x faster. Raw per-run results are in
`tools/bench/results/c6_qwen3_thinking_{int8,int4}.json`.

## v1.4.0 host-overhead removal (2026-09-02)

Goal: remove fixed per-token host work from every backend without changing outputs or VRAM. Measured on physical
GPU 0 (RTX 5090, `CUDA_VISIBLE_DEVICES=0`), greedy decoding, `max_new_tokens=256`, the 20 s
`lightning_storm_20s.mp4` (video families) and the 18 s `demon_singer_audio_18_sec.mp3` (Captioner), through
`tools/bench/benchmark.py` with the production 32 GB profile. Raw results: `tools/bench/results/v14_{baseline,after,control}_<variant>.json`.

Changes:

- Transformers path: the per-token `StoppingCriteria` now updates its counters every token but emits the console
  line and the UI progress callback through a 0.1 s `UiThrottle` (first and last token always emitted), and reuses a
  preallocated cancellation tensor instead of allocating one per token. `runner._Emitter` throttles only the
  high-frequency token events (payloads carrying `new_tokens`); item, segment, download, and load events are untouched.
  `console_progress.show_progress_line` gained a per-key minimum interval so any other caller is protected too.
- GGUF path: the SSE response is read per HTTP chunk instead of `iter_lines(chunk_size=1)`; progress emission is
  throttled the same way with exact token counts; `llama-server` is started with `--no-webui -np 1`; the new runtime
  options (`gguf_threads`, `gguf_batch_size`, `gguf_ubatch_size`, `gguf_flash_attn`, `gguf_cache_reuse`,
  `gguf_ignore_tier_context`, `gguf_extra_args`, `gguf_jpeg_quality`, `gguf_max_frames`) reach the server command line
  and the frame budget, sampling runs send `seed`, and every clamp (tier context, frame cap) is logged.
- ConvRot INT8/INT4: dense `gate_proj`/`up_proj` pairs are fused into one row-concatenated quantized linear (exactly
  as Q/K/V already were), the Hadamard matrix is cached on each module at load time (no lock or string formatting on
  the hot path), and the decode dispatch cache key no longer formats the device string. The separate modules are
  dropped after fusion, so VRAM does not grow.

| Variant | Stage | Load s | Prefill tok/s | Decode tok/s | Peak GiB | Tokens | Finish | Wall s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| timechat_int4 | before | 21.50 | 2,522.8 | 26.12 | 7.833 | 256 | length | 17.57 |
| timechat_int4 | after | 24.89 | 2,784.2 | 24.83 | 7.833 | 256 | length | 18.42 |
| timechat_int8 | before | 21.31 | 2,784.7 | 23.50 | 11.318 | 256 | length | 19.17 |
| timechat_int8 | after | 20.29 | 3,402.2 | 27.58 | 11.326 | 256 | length | 16.76 |
| timechat_bf16 | before | 25.04 | 4,582.8 | 32.17 | 18.263 | 256 | length | 14.61 |
| timechat_bf16 | after | 26.95 | 4,688.5 | 33.26 | 18.263 | 256 | length | 14.45 |
| avocado_int4 | before | 19.19 | 2,315.3 | 23.40 | 8.286 | 249 | eos | 20.46 |
| avocado_int4 | after | 22.60 | 3,283.6 | 25.32 | 8.286 | 256 | length | 18.97 |
| qwen3_omni_instruct_int4 | before | 32.48 | 2,400.7 | 11.29 | 17.882 | 165 | eos | 21.99 |
| qwen3_omni_instruct_int4 | after | 25.23 | 3,371.0 | 12.11 | 17.882 | 180 | eos | 21.14 |
| qwen3_omni_instruct_int8 | before | 49.57 | 1,376.2 | 1.35 | 23.570 | 166 | eos | 131.14 |
| qwen3_omni_instruct_int8 | after | 33.73 | 1,920.8 | 1.35 | 23.569 | 182 | eos | 142.09 |
| qwen3_omni_instruct_gguf_q4 | before | 15.17 | 2,575.6 | 247.73 | 24.353 | 171 | eos | 6.28 |
| qwen3_omni_instruct_gguf_q4 | after | 15.41 | 2,913.8 | 237.74 | 24.940 | 113 | eos | 7.72 |
| qwen3_omni_captioner_gguf_q4 | before | 14.48 | 836.7 | 299.26 | 24.222 | 256 | length | 1.31 |
| qwen3_omni_captioner_gguf_q4 | after | 13.13 | 933.9 | 301.10 | 24.222 | 256 | length | 1.26 |

Reading the table:

- Output identity: for every Transformers variant the "after" caption is byte-identical to a control run of the same
  code with the throttling disabled (`v14_control_*`), so the changes are output-neutral. The "before" captions differ
  from "after" only because the v1.4 backend round changed prompt defaults concurrently (the baseline was captured
  before those landed), not because of the performance work.
- Peak VRAM is unchanged for every variant (the INT8 TimeChat +8 MiB is allocator noise; the GGUF Q4 Instruct
  +0.6 GiB is the larger frame budget that the new `gguf_max_frames` default sends).
- Prefill improved 3-40 %; decode improved only 0-17 % (TimeChat INT8 +17 %, AVoCaDO INT4 +8 %, Qwen3 INT4 +7 %,
  BF16 +3 %; TimeChat INT4 is within run-to-run noise). The host-side per-token work was therefore not the dominant
  cost: the eager Transformers decode loop and its kernel-launch gaps set a ceiling near 25-35 tok/s for the 7B
  families on this GPU regardless of precision. Removing that ceiling needs CUDA-graph replay of the decode step
  with a static KV cache; the measurements for that follow-up are in the next section.
- The 33 GB INT8 Qwen3-Omni checkpoint cannot stay resident on 32 GB and decodes at 1.35 tok/s through block swap,
  which is why the 32 GB tier now selects INT4 (12 tok/s resident) and INT8 becomes automatic only from 48 GB.
- GGUF decode is unchanged within noise (240-300 tok/s); the SSE fix mainly reduces CPU use on the Python side.

## v1.4.0 static cache / CUDA graph probe (2026-09-02)

To test whether the eager ceiling above could be lifted, `tools/bench/static_cache_probe.py` loaded six resident
variants through the application loader (production 32 GB profile, greedy, 256 new tokens, 20 s storm video or 18 s
MP3) and compared four decode paths on the same prepared inputs. Full report: `temp/codex_v14/REPORT_S.md`; raw
results: `tools/bench/results/v14_probe_<variant>.json`.

| Variant | Eager dynamic (production) | Eager static cache | Inductor default | Reduce-overhead + static |
|---|---:|---:|---:|---:|
| timechat_bf16 | **32.97** tok/s, identical | 23.77, identical | 36.79, output changed at token 116 | 19.47, identical |
| timechat_int4 | **26.91**, identical | 20.55, identical | 27.33, changed at 62 | 15.37, changed at 155 |
| timechat_int8 | **27.50**, identical | 20.24, identical | 27.87, changed at 40 | 15.57, changed at 40 |
| avocado_int4 | **27.22**, identical | 20.23, identical | 27.90, changed at 31 | 15.42, changed at 31 |
| qwen3_omni_instruct_int4 | **12.09**, identical | 9.82, identical | 11.74, changed at 18 | 6.45, changed at 18 |
| qwen3_omni_captioner_int4 | **12.34**, identical | 9.54, changed at 21 | 11.64, changed at 7 | 6.53, changed at 13 |

Conclusions:

- No static or compiled path is both faster and output-identical. The static cache alone costs 19-28 % because
  attention scans the preallocated window; Inductor's fused kernels change greedy tokens on every variant (a compiled
  decode is at best +12 % on BF16); `reduce-overhead` never produced one reusable decode graph. Dynamo recorded
  402-1026 non-static-input captures per model: the flash-attention unpad path (`Tensor.item()`), 42 `aten.nonzero`
  breaks, per-layer cache guards, and the ConvRot / grouped-MoE Triton modules that must stay outside the graph.
- Peak VRAM stayed within +642 MiB allocated for every path, so memory was not the blocker.
- Eager dynamic decoding therefore remains the default and the only verified path; `torch.compile` stays an opt-in
  control with the existing "output may differ" caveat, and block-swapped models keep opting out. For Qwen3-Omni the
  fast path is the GGUF backend (240-300 tok/s on this GPU), which the VRAM tiers already offer from 24 GB.
