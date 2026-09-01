# Decoder Block Swap

Block swap keeps a model inside dedicated VRAM without relying on Windows WDDM shared-memory
paging. The first `R` decoder layers stay on the selected GPU. The remaining `S = N - R`
layers live in host memory and are copied through a ring of `K` fixed GPU slots during each
forward pass.

Each swapped layer is packed into a flat host byte buffer with 256-byte-aligned tensor offsets.
The loader first tries exact-size CUDA host registration. If that is unavailable, it uses
power-of-two pinned allocations with limited rounding waste, then falls back to pageable RAM.
Parameters and buffers are rebound to typed views of a GPU slot only while their layer runs;
between forwards they are CPU-visible views of the host pack.

A dedicated CUDA copy stream and preallocated events overlap transfers with decoder compute.
The root forward hook starts the first `K` copies. At each swapped layer, a compute-stream event
protects the slot most recently used by the preceding layer, the copy stream schedules the next
lookahead layer, and the compute stream waits for the current layer's ready event. There are no
per-token weight allocations and no host-side synchronization in these hooks.

## Settings

- `gpu_layers: auto` chooses the resident decoder-layer count from current free VRAM, the job's
  activation estimate, and the reserve. This is the recommended setting.
- `gpu_layers: all` forces every decoder layer to remain resident. If the estimate does not fit,
  Windows may page allocations into shared GPU memory.
- `gpu_layers: N` keeps exactly the first `N` decoder layers resident, capped at the model's layer
  count, and swaps the rest.
- `vram_reserve_gb` is dedicated VRAM deliberately left free at the expected peak. The default is
  `2.0` GB.
- `swap_slots` controls staging depth. `2` provides one-layer lookahead; `3` provides two-layer
  lookahead at the cost of one additional layer-sized GPU allocation.
- `pin_cpu` enables registered or pinned host packs. Disabling it uses pageable memory and usually
  lowers transfer throughput.

Explicit Accelerate settings (`offload_experts` or `max_memory`) select the legacy offload path
and disable block swap.

## Budget Math

Let `F` be currently free VRAM, `A` the activation estimate, `Q` the requested reserve, `B` the
largest decoder-layer size, `M` the non-decoder weight size, and `N` the number of decoder layers.
The available weight budget is:

```text
weights_budget = F - Q - A
```

For `auto`, all layers remain resident when `M + N*B <= weights_budget`. Otherwise the staging
slots are included before choosing the resident count:

```text
R = clamp(floor((weights_budget - M - K*B) / B), 0, N - 1)
S = N - R
resident_weight_bytes = M + R*B + K*B
expected_peak = resident_weight_bytes + A
pinned_host_bytes = S*B
```

The activation estimate includes runtime workspace slack, KV cache, media encoder peaks,
last-token logits, and MoE prefill intermediates. A forced numeric count is respected even when
the estimate exceeds `F - Q`; `all` is likewise reported as forced residency when it does not fit.

Three refinements sit on top of that formula:

- **Allocator slack.** The CUDA caching allocator keeps cached segments that never appear in
  `max_memory_allocated`; `expandable_segments` is unsupported on Windows builds of PyTorch and
  the measured gap scaled with the transient prefill volume (0.6-0.85x the activation peak), so
  `512 MiB + 0.5 x activation estimate` is budgeted next to the activations and counted in the
  expected peak. Windows workers additionally run with
  `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6`, which makes the allocator release
  unused cached blocks before the per-process VRAM cap is reached.
- **Tower staging.** The audio/vision encoders only run during prefill. When swapping is needed,
  the planner also evaluates keeping both towers on CPU between prefills (they are moved to the
  GPU for their forward and back afterwards, roughly 0.1-0.2 s per clip) and picks that layout
  when it buys resident decoder layers; the prefill phase is then budgeted as
  `towers + min(activation, 1 GiB)`. The plan line ends with `towers staged on CPU between
  prefills (x GiB)` when active.
- **Observed ratio.** After every generation the worker records `max(reserved, allocated) -
  resident weights` against the estimate that was planned for that job. The ratio (clamped to
  0.75-4.0, blended towards the latest observation) scales the next fresh estimate for that
  variant, so a family whose activations run hotter than the formula converges after one job
  without pinning small jobs to the largest one ever seen.

ConvRot prefill also changed: the `torch._int_mm` projection path used to materialise int32,
fp32, and bf16 copies of the whole `[tokens x N]` output (about 1.4 GB for one 7B MLP projection
at 7,600 tokens, which doubled INT4/INT8 activations relative to BF16 and fragmented the
allocator). It now processes rows in chunks sized to about 256 MiB of temporaries; per-row
activation scales make the result bit-identical.

## Tuning

Start with `gpu_layers=auto`, a `2.0` GB reserve, and two slots. Increase the reserve when Health
shows WDDM shared memory, generation approaches an out-of-memory recovery, or the job has larger
media/context requirements than the estimate. A larger reserve usually means fewer resident
layers and slower decoding.

If dedicated VRAM remains comfortably free, lower the reserve or set a conservative numeric
resident count. A third slot can improve overlap when compute is long enough to hide another
transfer, but it also consumes `B` more bytes of VRAM. Prefer pinned RAM; use `pin_cpu=false` only
when host pinning is unavailable or system-RAM pressure matters more than throughput.

The transfer floor per generated token is approximately:

```text
token transfer time = total swapped layer bytes / effective PCIe bandwidth
```

Measure the effective bandwidth on the target system rather than using the link's advertised
maximum. On the development machine (RTX 5090, Windows 11, WDDM driver 610.88) host-to-device
copies reached about **13.4 GiB/s** for registered, pinned, and pageable buffers alike, far below
the nominal PCIe 5.0 figure. At that rate one decoder layer costs roughly:

| Layer | Bytes | Transfer per token |
|---|---:|---:|
| Qwen3-Omni INT4 | 299 MiB | 22 ms |
| Qwen3-Omni INT8 | 596 MiB | 44 ms |
| Qwen3-Omni BF16 | 1189 MiB | 86 ms |
| TimeChat / AVoCaDO INT4 | 111 MiB | 8 ms |
| TimeChat / AVoCaDO INT8 | 222 MiB | 17 ms |
| TimeChat / AVoCaDO BF16 | 445 MiB | 33 ms |

Multiply by the number of swapped layers: decode with 20 swapped TimeChat INT4 layers measured
5.0 tok/s against 28.6 tok/s fully resident, exactly the transfer floor. Prefill hides most of
the transfer behind compute because every layer processes thousands of tokens per copy.

## Shared GPU memory readings

Windows counts page-locked host memory as "shared GPU memory", so a block-swapped model shows
its pinned layer packs in Task Manager and in the Health tab's shared-memory row even though no
device allocation was paged out. The runner therefore subtracts the pinned bytes (and a small
driver baseline) before warning; a warning means device memory really is being paged. The
worker also caps the PyTorch allocator at the dedicated VRAM measured free at load time
(`VCAP_VRAM_HARD_CAP=0` disables the cap), so an overrun raises an out-of-memory error that the
normal recovery handles instead of silently paging.

The loader additionally slices the language-model head input to the final position during
generation (`last_token_logits`), which removes the full-prompt logits buffer that previously
added 1.6-2.5 GB to every prefill.

## Limitations

- Swapped models skip `torch.compile` and CUDA graph capture because parameter and buffer bindings
  change between layer calls.
- Decoder KV cache and activations remain on the GPU; block swap moves weights only.
- The mechanism is forward-only and assumes decoder layers execute in their normal sequential
  order. Training and backward passes are not supported.
- A single-file safetensors checkpoint is required for streamed per-layer placement. Sharded
  checkpoints use the existing resident loader.

## Measured results (RTX 5090 32 GB, Windows 11, 2026-09-01)

All runs used `temp/codex_BS/verify_run.py` through the application's real loader and caption
wrapper with greedy decoding, the 20 s `lightning_storm_20s.mp4` (5,415 Qwen3 / 7,561 Qwen2.5
prompt tokens), an 18 s MP3, or a 960x540 image. "Free after" is `cudaMemGetInfo` free memory
right after generation; "Shared" is the WDDM per-process counter, which equals the pinned bytes
plus a ~0.08 GiB driver baseline when nothing is paged.

### Exactness: swapped output equals resident output

| Variant | Input | Resident / total | Slots | Gen peak GiB (swapped vs resident) | Decode tok/s (swapped vs resident) | Captions |
|---|---|---|---:|---|---|---|
| TimeChat INT4 | video | 8 / 28 | 2 | 7.38 vs 9.33 | 5.0 vs 28.6 | identical |
| Qwen3-Omni Instruct INT4 | video | 24 / 48 | 2 | 11.45 vs 17.88 | 1.6 vs 12.7 | identical |
| Qwen3-Omni Instruct INT4 | image | 0 / 48 | 3 | 3.59 vs 16.74 | 0.9 vs 13.2 | identical |
| Qwen3-Omni Thinking INT4 | image | 30 / 48 | 2 | 12.07 vs 16.74 | 2.1 vs 13.0 | identical |
| Qwen3-Omni Captioner INT4 | audio | 30 / 48 | 2 | 12.04 vs 16.74 | 2.1 vs 13.2 | identical |
| AVoCaDO INT8 | video | 10 / 28 | 2 | 10.46 vs 13.94 | 3.0 vs 28.2 | identical |

The decode rates sit exactly on the transfer floor (swapped bytes / 13.4 GiB/s): 20 x 111 MiB,
24 x 299 MiB, and 48 x 299 MiB per token respectively.

### Automatic plans (`gpu_layers=auto`, 2 GB reserve, ~30.3 GiB free at load)

| Variant | Input | Plan | Pinned GiB | GPU weights GiB | Gen peak GiB | Free after GiB | Decode tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| TimeChat BF16 | video | resident 28/28 | 0 | 16.69 | 18.26 | 10.55 | 33.9 |
| TimeChat INT8 | video | resident 28/28 | 0 | 9.57 | 12.82 | 14.55 | 28.4 |
| AVoCaDO BF16 | video | resident 28/28 | 0 | 16.69 | 18.79 | 9.57 | 35.2 |
| AVoCaDO INT8 | video | resident 28/28 | 0 | 9.57 | 13.94 | 12.61 | 28.8 |
| AVoCaDO INT4 | video | resident 28/28 | 0 | 6.02 | 10.45 | 16.25 | 28.4 |
| Qwen3-Omni Instruct INT8 | video | 35/48 resident, 13 swapped | 7.57 | 24.37 | 25.68 | 3.42 | 1.5 |
| Qwen3-Omni Thinking INT8 | image | 35/48 resident, 13 swapped | 7.57 | 24.37 | 24.54 | 4.54 | 1.5 |
| Qwen3-Omni Captioner INT8 | audio | 38/48 resident, 10 swapped | 5.82 | 26.12 | 26.26 | 2.78 | 1.9 |
| Qwen3-Omni Captioner BF16 (63.4 GB) | audio | 18/48 resident, 30 swapped | 34.82 | 26.6 | 26.74 | 2.87 | 0.36 |
| Qwen3-Omni Instruct BF16 (63.4 GB) | video | 17/48 resident, 31 swapped | 35.98 | 25.44 | 26.92 | 2.27 | 0.35 |
| Qwen3-Omni Thinking BF16 (63.4 GB) | image | 17/48 resident, 31 swapped | 35.98 | 25.44 | 25.63 | 3.96 | 0.35 |
| Qwen3-Omni Instruct INT4, reserve 20 GB (12 GB card simulation) | video | 11/48 resident, 37 swapped | 10.81 | 6.34 | 7.65 | 22.21 | 1.1 |
| TimeChat INT4, reserve 24 GB (8 GB card simulation) | video | 0/28 resident, 28 swapped | 3.04 | 3.19 | 6.50 | 21.41 | 3.8 |

No run paged: the shared-memory counter never exceeded the pinned bytes plus the driver
baseline, and the allocator cap was never hit. Compare with the previous release, where INT8
Thinking peaked at 33.07 GiB on this 31.84 GiB card and BF16 Instruct "CPU offload" peaked at
51.6 GiB, both through WDDM paging.

Two effects of the last-token logits hook are visible in the resident rows: TimeChat BF16 now
peaks at 18.26 GiB on the same clip that previously recorded 31.52 GiB, and Qwen3-Omni INT4
video peaks at 17.88 GiB with 16.57 GiB of weights, i.e. about 1.3 GiB of real activations.

### GGUF through `llama-server --fit`

| Variant | Input | Server flags | Prefill | Decode tok/s | External GPU peak GiB |
|---|---|---|---:|---:|---:|
| Instruct Q8_0 (33.8 GB) | video | `-ngl 999 --fit on` (fitter aborted: "already set by user") | 378 s | 0.41 | 31.79 (paged) |
| Instruct Q8_0 (33.8 GB) | video | `--fit on --fit-target 2048`, no `-ngl` | 12.7 s | 64.6 | 31.16 |
| Thinking Q8_0 (33.8 GB) | image | `--fit on --fit-target 3584`, no `-ngl` | 3.4 s | 57.0 | 29.1 |
| Captioner Q4_K_M (19.9 GB) | audio | `--fit on --fit-target 3584`, no `-ngl` | 0.3 s | 277 | 24.6 |

The previous fixed `-ngl 36` Q8 plan measured 9.6 tok/s; letting the fitter place the MoE
weights is about 6x faster and keeps the configured margin free once the 1,536 MiB projector
headroom is added to the target.

### Tier preset verification (every family, 6-32 GB)

Each tier preset was run through the real runner (`run_job`) on the RTX 5090 with the tier's own
precision, frame, pixel, and fps settings on the 20 s video (18 s MP3 for the Captioner). Smaller
cards were emulated by raising the reserve so that `free - reserve` equals what a card of that size
has left after the CUDA context and desktop use (`reserve = 2 GB + (31.84 - tier)`); "spare VRAM"
is the lowest free memory that card would have seen (start free minus the allocator's peak
*reserved* bytes), i.e. it already includes fragmentation. A pass means at least ~2 GB stayed free.

| Family | Tier GB | Scheme | Preset media | Plan chosen | Peak alloc GiB | Spare VRAM on that card GiB | tok/s (64 tok, incl. prefill) | Verdict |
|---|---:|---|---|---|---:|---:|---:|---|
| TimeChat | 6 | INT4 | 1.0 fps / 32 fr / 100k px | 0/28 resident, 28 swapped, towers staged | 3.27 | 0.88 | 4.02 | fits, < 2 GB spare |
| TimeChat | 8 | INT4 | 1.0 fps / 48 fr / 200k px | 0/28 resident, 28 swapped, towers staged | 3.54 | 2.52 | 4.06 | pass |
| TimeChat | 10 | INT4 | 2.0 fps / 80 fr / 297k px | 0/28 resident, 28 swapped, towers staged | 4.06 | 3.85 | 4.03 | pass |
| TimeChat | 12 | INT8 | 2.0 fps / 80 fr / 297k px | 8/28 resident, 20 swapped, towers staged | 6.21 | 3.65 | 2.88 | pass |
| TimeChat | 16 | INT8 | 2.0 fps / 128 fr / 297k px | 25/28 resident, 3 swapped, towers staged | 9.91 | 3.83 | 13.5 | pass |
| TimeChat | 24 | BF16 | 2.0 fps / 120 fr / 297k px | 28/28 resident | 18.26 | 2.83 | 48.28 | pass |
| TimeChat | 32 | BF16 | 2.0 fps / 160 fr / 297k px | 28/28 resident | 18.26 | 10.67 | 48.15 | pass |
| AVoCaDO | 6 | INT4 | 1.0 fps / 32 fr / 100k px | 0/28 resident, 28 swapped, towers staged | 3.27 | 0.89 | 4.02 | fits, < 2 GB spare |
| AVoCaDO | 8 | INT4 | 1.0 fps / 48 fr / 200k px | 0/28 resident, 28 swapped, towers staged | 3.54 | 2.52 | 4.02 | pass |
| AVoCaDO | 10 | INT4 | 2.0 fps / 80 fr / 401k px | 0/28 resident, 28 swapped, towers staged | 4.52 | 3.26 | 4.06 | pass |
| AVoCaDO | 12 | INT8 | 2.0 fps / 80 fr / 401k px | 5/28 resident, 23 swapped, towers staged | 5.83 | 3.65 | 2.53 | pass |
| AVoCaDO | 16 | INT8 | 2.0 fps / 128 fr / 401k px | 21/28 resident, 7 swapped, towers staged | 9.31 | 4.1 | 7.22 | pass |
| AVoCaDO | 24 | BF16 | 2.0 fps / 128 fr / 401k px | 28/28 resident | 18.79 | 1.86 | 48.05 | pass |
| AVoCaDO | 32 | BF16 | 2.0 fps / 256 fr / 401k px | 28/28 resident | 18.79 | 9.7 | 42.94 | pass |
| Qwen3-Omni Instruct | 6 | not offered | - | - | - | - | - |
| Qwen3-Omni Instruct | 8 | INT4 | 1.0 fps / 32 fr / 131k px | 0/48 resident, 48 swapped, towers staged | 3.03 | 3.33 | 0.92 | pass |
| Qwen3-Omni Instruct | 10 | INT4 | 1.0 fps / 48 fr / 131k px | 3/48 resident, 45 swapped, towers staged | 3.91 | 4.45 | 0.97 | pass |
| Qwen3-Omni Instruct | 12 | INT4 | 1.0 fps / 48 fr / 131k px | 10/48 resident, 38 swapped, towers staged | 5.96 | 4.35 | 1.14 | pass |
| Qwen3-Omni Instruct | 16 | INT4 | 1.0 fps / 64 fr / 196k px | 24/48 resident, 24 swapped, towers staged | 10.16 | 4.06 | 1.76 | pass |
| Qwen3-Omni Instruct | 24 | INT4 | 2.0 fps / 96 fr / 262k px | 48/48 resident | 17.88 | 4.07 | 20.2 | pass |
| Qwen3-Omni Instruct | 32 | INT8 | 2.0 fps / 96 fr / 262k px | 37/48 resident, 11 swapped, towers staged | 25.9 | 4.07 | 1.9 | pass |
| Qwen3-Omni Thinking | 6 | not offered | - | - | - | - | - |
| Qwen3-Omni Thinking | 8 | INT4 | 1.0 fps / 32 fr / 131k px | 0/48 resident, 48 swapped, towers staged | 3.03 | 3.33 | 0.92 | pass |
| Qwen3-Omni Thinking | 10 | INT4 | 1.0 fps / 48 fr / 131k px | 3/48 resident, 45 swapped, towers staged | 3.91 | 4.45 | 0.98 | pass |
| Qwen3-Omni Thinking | 12 | INT4 | 1.0 fps / 48 fr / 131k px | 10/48 resident, 38 swapped, towers staged | 5.96 | 4.35 | 1.14 | pass |
| Qwen3-Omni Thinking | 16 | INT4 | 1.0 fps / 64 fr / 196k px | 24/48 resident, 24 swapped, towers staged | 10.16 | 4.06 | 1.74 | pass |
| Qwen3-Omni Thinking | 24 | INT4 | 2.0 fps / 96 fr / 262k px | 48/48 resident | 17.88 | 4.07 | 20.25 | pass |
| Qwen3-Omni Thinking | 32 | INT8 | 2.0 fps / 96 fr / 262k px | 37/48 resident, 11 swapped, towers staged | 25.9 | 4.07 | 1.89 | pass |
| Qwen3-Omni Captioner | 6 | not offered | - | - | - | - | - |
| Qwen3-Omni Captioner | 8 | INT4 | 1.0 fps / 32 fr / 131k px | 0/48 resident, 48 swapped | 3.27 | 2.98 | 0.92 | pass |
| Qwen3-Omni Captioner | 10 | INT4 | 1.0 fps / 48 fr / 131k px | 4/48 resident, 44 swapped | 4.44 | 3.78 | 1.0 | pass |
| Qwen3-Omni Captioner | 12 | INT4 | 1.0 fps / 48 fr / 131k px | 11/48 resident, 37 swapped | 6.48 | 3.69 | 1.18 | pass |
| Qwen3-Omni Captioner | 16 | INT4 | 1.0 fps / 64 fr / 196k px | 25/48 resident, 23 swapped | 10.57 | 3.53 | 1.82 | pass |
| Qwen3-Omni Captioner | 24 | INT4 | 2.0 fps / 96 fr / 262k px | 48/48 resident | 16.71 | 5.27 | 14.8 | pass |
| Qwen3-Omni Captioner | 32 | INT8 | 2.0 fps / 96 fr / 262k px | 38/48 resident, 10 swapped | 26.26 | 3.21 | 2.02 | pass |

The 6 GB tier cannot reach 2 GB of headroom: with every decoder layer swapped and the towers
staged, the 7B model still needs ~3.3 GiB (embeddings, `lm_head`, slots, activations) of the
~4.4 GiB such a card has available, so it runs without paging but with under 1 GB spare. Qwen3-Omni
is not offered below 8 GB. Several plans are conservative by 1-2 GB (for example Qwen3 INT4 at
10-16 GB); the per-variant observed ratio reclaims part of that on later jobs, and lowering
`VRAM to keep free (GB)` trades the margin for resident layers.

### Honest cost model for the MoE models

Layer-granular swap moves every expert of a swapped layer for each token although only 8 of
128 experts are used, so a swapped Qwen3-Omni decodes at 13.4 GiB/s divided by the swapped
bytes: INT8 with 13 swapped layers is ~1.5 tok/s and BF16 with 31 swapped layers is ~0.35 tok/s.
That is slower than the paging it replaces in the INT8 case (4-5 tok/s), because the WDDM pager
only faulted in the expert pages that were touched. The trade is deterministic memory use and
no shared-memory growth. For throughput on a 32 GB card prefer INT4 (fully resident, 12-13
tok/s) or GGUF Q4/Q8 (57-280 tok/s); the natural next step for the MoE families is
expert-granular streaming, which would move only the routed experts (about 36 MiB per INT8
layer per token) and bring swapped INT8 close to resident speed.

### Host memory notes

- Pinned packs are real RAM: 31 swapped BF16 layers pin 36 GiB. Windows also charges WDDM
  device allocations to the process commit, so a 63 GB load shows ~67 GB of private bytes even
  though only the pinned part is physical host memory that cannot be paged.
- Right after a 63 GB load the worker briefly holds ~20 GB of mapped checkpoint pages on top
  of the pinned packs. The loader trims its working set after installing the swap, and
  ffprobe/ffmpeg child processes retry with backoff, because two runs with 37-38 GiB pinned
  saw the first child process fail to start on this 93 GB machine.
- Repeated load/unload cycles plateau (about 10 GB of process-level caches remain after the
  first unload on this system); the pinned packs themselves are released with the model.
