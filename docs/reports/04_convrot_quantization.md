# ConvRot INT8/INT4 for LLM/VLM transformers — full report

## 1. What "ConvRot" is

ConvRot = group-wise *regular* Hadamard rotation applied along the input (K) axis, immediately before symmetric integer quantization. QuaRot-family idea with one twist.

Canonical definition — `C:/Users/Furkan/Videos/convrot_int8/comfy-kitchen/comfy_kitchen/tensor/int8_utils.py`:
```python
def _build_hadamard(size, device="cpu", dtype=torch.float32):
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot)."""
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], dtype=dtype, device=device)
    h = h4; current_size = 4
    while current_size < size:
        h = torch.kron(h, h4); current_size *= 4
    return h / (size ** 0.5)

def _rotate_weight(weight, h, group_size):      # offline: W_rot = W @ H^T
    out_f, in_f = weight.shape
    weight_grouped = weight.reshape(out_f, in_f // group_size, group_size)
    return torch.matmul(weight_grouped, h.T.to(weight)).reshape(out_f, in_f)

def _rotate_activation(x, h, group_size):       # online: x_rot = x @ H
    n_groups = x.shape[-1] // group_size
    return torch.matmul(x.reshape(-1, n_groups, group_size), h.to(x)).reshape(x.shape)
```
Key points:
- Regular (not Sylvester) Hadamard. Base is the 4×4 whose rows/cols sum to 2. Size must be a power of 4 (4, 16, 64, 256, 1024). Avoid non-power-of-4 group sizes.
- Block-diagonal per group of 256 columns; `H` is 256×256; online activation rotation is cheap.
- `x_rot @ W_rot^T = x H H^T W^T = x W^T`. Exact, no calibration needed.
- Default group size 256; `in_features % 256 == 0` required.

### INT8 ConvRot = W8A8, dynamic per-token activations
- Weights: rotated, per-output-channel symmetric int8, `scale = rowabsmax/127`, fp32 `[out_features, 1]`.
- Activations: quantized at runtime, per-row (per-token), dynamically. No calibration, no `input_scale`.
- Format is called `int8_tensorwise` but with ConvRot the scale is per-row. `TensorWiseINT8Layout.quantize` in `comfy-kitchen/comfy_kitchen/tensor/int8.py` hard-raises if convrot without per-channel.

### INT4 ConvRot (`convrot_w4a4`) = W4A4
From `comfy-kitchen/comfy_kitchen/backends/eager/convrot_w4a4.py`: rotate group 256 → symmetric int4 in [-7, +7] → pack two nibbles per int8. Activations rotated online and quantized to int4 per-row. Eager quantizer uses per-ROW scales despite `quant_group_size=64`.

### Format boundary (4 incompatible things called "ConvRot")
| Name | What | Where |
|---|---|---|
| `int8_tensorwise` + `convrot:true` | signed int8 W8A8, group 256 | stock ComfyUI ≥0.27, comfy-kitchen ≥0.2.12 |
| `convrot_w4a4` | signed int4 W4A4, group 256 | stock ComfyUI `TensorCoreConvRotW4A4Layout` |
| `convrot_nvfp4` | NVFP4 W4A4, Blackwell only | official paper repo `ConvRot-main` (arXiv 2512.03673) |
| `int8_w8a8` | same bytes as #1, different marker | retired ComfyUI-INT8-Fast |

## 2. Canonical converter + tools

| Tool | Generic HF nn.Linear? | LLM proven? | Verdict |
|---|---|---|---|
| `ltx25_convrot_hq/quantize_hq.py` | Yes — pure name/shape driven, streaming | no, but only 3 regexes model-specific | Best starting point. 346 lines, no framework deps |
| `convert_to_quant` (ctq) | Yes | Yes — has `qwen35` filter | Good, but see bug |
| `comfy-quants` | config-driven | diffusion only | Best documentation |
| `comfy-kitchen` | tensor API | n/a | Kernel source of truth |
| `ai-toolkit/toolkit/util/convrot_quant.py` | Yes — walks nn.Linear live | model-agnostic | Best inference reference |
| `ggufy` | safetensors→GGUF, diffusion arch list | no | see §5 |

### ctq bug
`convert_to_quant/constants.py`:
```python
QWEN35_AVOID_KEY_NAMES = [
    ".layers.0.", ".layers.63.", "lm_head", "embed_tokens", "in_proj_a", "in_proj_b",
    "visual.pos_embed", "visual.patch_embed", "merger", "mtp.fc", "visual.blocks.0.",
]
```
Published recipe `qwen-2.5-vl-int8-convrot-comfyui/convert.sh` uses `--scaling_mode tensor --convrot` — but in `convert_to_quant/converters/learned_rounding.py:102`:
```python
if self.convrot and not (self.target_format == "int8" and self.scaling_mode == "row"):
    verbose("  - WARNING: ConvRot is currently only supported for INT8 row-wise quantization. It will be ignored.")
    self.convrot = False
```
With `--scaling_mode tensor`, `--convrot` is silently dropped. The published HF model `ansteinhuynh/Qwen2.5-VL-7B-Instruct-int8-convrot-comfyui` is likely plain int8 with a misleading name. **If using ctq, pass `--scaling_mode row`.**

### MoE
`G:/ComfyUI_V92/ComfyUI/comfy/ops.py:1447` `MoEExperts`: E experts in one 3D tensor (`{prefix}.weight` leading dim E, `{prefix}.weight_scale`, `{prefix}.comfy_quant` json `{"format": ..., "num_experts": E}`). For transformers, quantize each expert `[out, in]` slice independently.

## 3. Inference kernel and GPU support

Bottoms out in `torch._int_mm` (cuBLASLt IMMA). Portable eager version — `comfy-kitchen/comfy_kitchen/backends/eager/quantization.py:971`:
```python
def int8_linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16,
                convrot=False, convrot_groupsize=256, input_act=None):
    weight = weight.to(device=x.device).contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_8, x_scale = quantize_int8_rowwise(x_2d)          # dynamic per-token
    result = _int8_matmul_accumulate(x_8, weight.T.contiguous())   # -> int32
    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    weight_scale = weight_scale.reshape(1, -1)
    scaled_parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk = result[i:end_i].float()
        chunk_scales = x_scale[i:end_i].to(chunk) * weight_scale
        scaled_parts.append((chunk * chunk_scales).to(out_dtype))
    result = torch.cat(scaled_parts, dim=0)
    if bias is not None:
        result = result + bias.to(result).reshape(1, -1)
    return result.reshape(*orig_shape[:-1], weight.shape[0])
```
`_int8_matmul_accumulate` pads M to multiple of 32 (min 32), K to 8, N to 8 (32 on Turing).

Tightest QuantLinear forward — `ai-toolkit/toolkit/util/convrot_quant.py:1454`:
```python
if _int8_gemm_supported(x.device):
    aq, a_s = _int8_act_quant_padded(rotate(x, rot).reshape(-1, in_f), self.act_qmax)
    i32 = torch._int_mm(aq, self._qdata(module).t())
    out = _int8_epilogue(i32[:m], a_s[:m], self._scales(module), module.bias, x.dtype)
    return out.reshape(*x.shape[:-1], out_f)
```
Fallbacks: fused Triton GEMV for decode-size batches (`m <= FUSED_GEMV_MAX_M`), `_int_mm`, MPS int8pack (W8A16), dequant + `F.linear`.

**Decode-batch problem:** at batch=1 token generation M=1; `_int_mm` pads M to 32 — INT8 W8A8 likely slower than bf16 during decode, wins on prefill. ai-toolkit uses a fused Triton GEMV. Budget for this.

| GPU | SM | INT8 ConvRot | INT4 ConvRot (W4A4) |
|---|---|---|---|
| RTX 3090 | 8.6 | full speed | native m16n8k64 s4 MMA |
| RTX 4090 | 8.9 | yes | CUTLASS Sm89 |
| RTX 5090 | 12.0 | yes (`_int_mm` fine) | no native int4 — unpacks to int8 |
| Turing | 7.5 | yes (N align 32) | CUTLASS Sm75 |
Gate: `comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py:306` `_cuda_device_supports_native_int4_mma` returns `major == 8` only.

### Speed & quality (LTX-2.5 22B, `ltx25_convrot_benchmark/README.md`)
| Format | s/step | vs BF16 | Peak VRAM | Video L2 |
|---|---:|---:|---:|---:|
| BF16 | 1.405 | 100% | 24.37 GiB | 0% |
| INT8 ConvRot | 0.299 | 470% | 20.78 GiB | 2.480% |
| HQ INT8 | 0.309 | 455% | 21.25 GiB | 2.469% |
| Runtime FP8 | 0.328 | 428% | 20.34 GiB | 10.774% |
INT8 ConvRot ~4× closer to BF16 than runtime FP8.

## 4. INT4 specifics
- Packing: two nibbles per int8, low nibble = even column (`comfy-kitchen/.../eager/svdquant.py:69`):
```python
lo = values[..., 0::2].to(torch.int32) & 0x0F
hi = values[..., 1::2].to(torch.int32) & 0x0F
return (lo | (hi << 4)).to(torch.int8)
```
Stored `[out_features, in_features // 2]`, dtype int8. Range [-7,+7], `scale = rowabsmax/7`, fp32 per row. ConvRot group 256.
**Recommendation for 30B MoE LLM: do not use W4A4.** Use **W4A8**: ai-toolkit `convrotint4` (`ConvRotIntNQuantizer`, `get_convrot_quantizer("convrotint4")`): int4 bitpacked weights, int8 activations, unpacked on the fly through `torch._int_mm`. Same memory, far better quality, runs on every int8 tensor-core arch. ComfyUI also has `asym_w4a8_int8` (`QUANT_ALGOS["asym_w4a8_int8"]`) with per-group fp8 scales + per-channel fp32 + optional Lloyd-Max codebook.

## 5. GGUF
No tool here converts LLM safetensors to GGUF. `ggufy` handles diffusion archs only. For Qwen2.5-VL / Qwen3-Omni GGUF use llama.cpp `convert_hf_to_gguf.py` (separate track, mutually exclusive with ConvRot).

Bit-faithful recipe — `comfy-quants-main/docs/formats/int8_tensorwise.md`:
```text
if convrot and in_features % group_size == 0:        # group_size = 256
    H = regular_hadamard(gs, dtype=W.dtype)          # built AT the weight dtype
    w = (W.view(out, in//gs, gs) @ H.T).reshape(out, in)
scale      = clamp(w.abs().amax(dim=-1, keepdim=True).float() / 127, min=1e-30)  # fp32 [out, 1]
scale_math = scale.to(w.dtype); scale_math[scale_math == 0] = tiny(w.dtype)
q          = round(w / scale_math).clamp(-128, 127).to(int8)                     # DIVISION, at w.dtype
```

## 6. Recipe

### Safetensors layout per quantized Linear
```
<layer>.weight         int8    [out_features, in_features]
<layer>.weight_scale   float32 [out_features, 1]
<layer>.comfy_quant    uint8   JSON blob
<layer>.bias           bf16    [out_features]   (unquantized)
```
`.comfy_quant` payload (`quantize_hq.py:151`):
```python
def config_tensor(group_size):
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}
    encoded = json.dumps(config).encode("utf-8").ljust(72, b" ")
    return torch.tensor(list(encoded), dtype=torch.uint8)
```
INT4: `{"format": "convrot_w4a4", "convrot_groupsize": 256}`, weight int8 `[out, in//2]`, scale fp32 `[out]`.
Optional header `__metadata__["_quantization_metadata"]`: `{"format_version": "1.0", "layers": {"model.layers.0.mlp.up_proj": {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}}}` — for our own transformers loader the header JSON is more convenient.

Size: Qwen2.5-VL-7B bf16 ≈ 16.6 GB → ~9.0–9.5 GB INT8. Qwen3-Omni-30B-A3B bf16 ≈ 61 GB → ~32–34 GB INT8, ~18–20 GB W4A8.

### (a) Merge sharded HF bf16 → one file
`C:/Users/Furkan/Videos/convrot_int8/qwen-2.5-vl-int8-convrot-comfyui/merge_qwen_shards.py` (loads every shard into RAM — fine for 7B, not for 30B). For 30B stream it: build header from `safe_open(...).get_slice(k).get_shape()` per shard and write tensors one at a time (`quantize_hq.py:196` `make_header` shows the pattern).

### (b) INT8 ConvRot — adapt `quantize_hq.py`
Change: replace `EXCLUDE_SEG`/`ATTENTION_PARTS`/`is_main_video_ff` with LLM name rules; drop `model.diffusion_model.` prefix assumptions; relax `classify`'s digit requirement.
Quantize core — MSE-optimal per-row clipping, 3-stage search, best-of-{16,64,256} group size:
```python
def search_scales(rotated):
    absmax = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    best_alpha = torch.ones_like(absmax); best_mse = torch.full_like(absmax, float("inf"))
    def search(deltas, centered):
        nonlocal best_alpha, best_mse
        center = best_alpha if centered else torch.zeros_like(best_alpha)
        for delta in deltas:
            alpha = (center + delta).clamp(0.5, 1.0)
            scale = (absmax * alpha / 127.0).clamp_min(1e-30)
            quantized = (rotated / scale).round().clamp(-127, 127)
            mse = (quantized * scale - rotated).square().mean(dim=1, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_alpha = torch.where(better, alpha, best_alpha)
    search(torch.linspace(0.60, 1.00, 41, device=rotated.device), centered=False)
    search(torch.linspace(-0.012, 0.012, 25, device=rotated.device), centered=True)
    search(torch.linspace(-0.0010, 0.0010, 21, device=rotated.device), centered=True)
    scale = (absmax * best_alpha / 127.0).clamp_min(1e-30)
    return (rotated / scale).round().clamp(-127, 127).to(torch.int8), scale, best_mse
```
Fallback ctq: `ctq -i model-bf16.safetensors -o model-int8-convrot.safetensors --int8 --scaling_mode row --convrot --convrot_group_size 256 --qwen35 --comfy_quant --save-quant-metadata --low-memory --simple` (needs `tqdm`).

### (c) INT4 → W4A8 (`convrotint4` in ai-toolkit) or `asym_w4a8_int8`.

### Layers that must stay BF16
Non-negotiable: `embed_tokens`, `lm_head`, all norms, all biases, anything with `in_features % 256 != 0`.
Strongly recommended: first and last decoder block; `visual.patch_embed`, `visual.pos_embed`, `visual.blocks.0.`, `merger` / multimodal projector; audio encoder patch embed / conv front-end; MoE router / gate (`gate`, `router`); talker head.
Quantize: `q_proj/k_proj/v_proj/o_proj`, `gate_proj/up_proj/down_proj`, MoE expert weights, vision/audio transformer block MLPs+attn (blocks ≥1).
```python
EXCLUDE_SEG = re.compile(
    r"embed_tokens|lm_head|norm|bias|rotary|pos_embed|patch_embed|"
    r"merger|multi_modal_projector|mm_input_projection|"
    r"gate_logits|router|routing|logit|\.gate$|"
    r"visual\.blocks\.0\.|audio_tower\.layers\.0\.|talker"
)
```
Sensitivity-driven selection: quantize everything then restore individual projections to BF16 by measured output error (`ltx25_convrot_hq/probe_transformer.py`, `restore_compact_checkpoint.py`, `mix_best_checkpoint.py`). For 30B MoE, check expert `down_proj` first.

### Standalone QuantLinear
```python
class ConvRotInt8Linear(nn.Module):
    def __init__(self, in_f, out_f, group_size=256, bias=True):
        super().__init__()
        self.in_features, self.out_features, self.group_size = in_f, out_f, group_size
        self.register_buffer("weight", torch.empty(out_f, in_f, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_f, 1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_f)) if bias else None
    def forward(self, x):
        return int8_linear(x, self.weight, self.weight_scale, self.bias, out_dtype=x.dtype,
                           convrot=True, convrot_groupsize=self.group_size)
```
Walk the model swapping `nn.Linear` by name, load with `assign=True`. Cache the Hadamard per (group_size, device, dtype).

## Key file reference
| Purpose | Path |
|---|---|
| Hadamard + rotate | `C:/Users/Furkan/Videos/convrot_int8/comfy-kitchen/comfy_kitchen/tensor/int8_utils.py` |
| INT8 eager linear | `.../comfy-kitchen/comfy_kitchen/backends/eager/quantization.py` |
| INT8 layout | `.../comfy-kitchen/comfy_kitchen/tensor/int8.py` |
| INT4 W4A4 eager | `.../comfy-kitchen/comfy_kitchen/backends/eager/convrot_w4a4.py` |
| INT4 pack/unpack | `.../comfy-kitchen/comfy_kitchen/backends/eager/svdquant.py` (L69–98) |
| GPU gating | `.../comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py` (L296–345) |
| Best converter template | `C:/Users/Furkan/Videos/convrot_int8/ltx25_convrot_hq/quantize_hq.py` |
| Best QuantLinear reference | `C:/Users/Furkan/Videos/convrot_int8/ai-toolkit-main/ai-toolkit-main/toolkit/util/convrot_quant.py` (L1297–1478) |
| Shard merger | `C:/Users/Furkan/Videos/convrot_int8/qwen-2.5-vl-int8-convrot-comfyui/merge_qwen_shards.py` |
| Qwen skip lists | `C:/Users/Furkan/Videos/convrot_int8/convert_to_quant-main/convert_to_quant-main/convert_to_quant/constants.py` |
| ctq convrot bug | `.../convert_to_quant/converters/learned_rounding.py:102, 871` |
| Format spec | `C:/Users/Furkan/Videos/convrot_int8/comfy-quants-main/comfy-quants-main/docs/formats/int8_tensorwise.md` |
| ComfyUI loader | `G:/ComfyUI_V92/ComfyUI/comfy/ops.py` (L1111–1220 Linear, L1447+ MoE); `comfy/quant_ops.py` (L235–281) |
| Benchmarks | `C:/Users/Furkan/Videos/convrot_int8/ltx25_convrot_benchmark/README.md` |

## Action items
1. Verify any published "convrot" Qwen model before trusting it (check `"convrot"` in `.comfy_quant`).
2. INT4 W4A4 is wrong here — use W4A8.
3. Benchmark decode before committing to INT8; add fused GEMV (or fall back to dequant+bf16 matmul for M small) if slower.
