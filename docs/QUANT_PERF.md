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
