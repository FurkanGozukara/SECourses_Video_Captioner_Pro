# ComfyUI VRAM management techniques — report for a standalone transformers VLM/LLM app

ComfyUI source root: `G:\ComfyUI_V92\ComfyUI` (v0.34.0). Venv torch 2.13.0+cu130, transformers 5.15.1, `comfy_kitchen` 0.2.31, `comfy_aimdo` 0.4.15, sageattention 2.2.0 (sm75triton.sm86.sm89.sm120a), flash_attn 2.8.3.post1 (sm86.sm89.sm120a), xformers 0.0.35, triton-windows 3.7.1, gguf 0.19.0, torchao 0.18.0.

## 1. Model management — `comfy\model_management.py` (2127 lines)
- `VRAMState` enum L45–51; mode wiring L561–585; `unet_offload_device()` L1080; `unet_inital_load_device()` L1086–1105; `text_encoder_device()` L1193; `text_encoder_initial_device()` L1204–1219 (offload unless ≤1 GB or `mem_l > mem_o*0.5 and size*1.2 < mem_l`); `maximum_vram_for_weights()` L1107 = `total*0.88 - minimum_inference_memory()`.
- **Free memory**: `get_free_memory(dev, torch_free_too)` L1753–1797: `mem_free_total = cuda.mem_get_info()[0] + (reserved - active)`. `EXTRA_RESERVED_VRAM` L851–859: 400 MB, **600 MB on Windows**, +100 MB on >15 GB cards; `--reserve-vram G` overrides. `minimum_inference_memory()` L864 = `0.8 GB + reserved`.
- **Budget** (`load_models_gpu()` L913–1013, L994–1006):
```python
lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required),
                           min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
lowvram_model_memory -= loaded_memory
if lowvram_model_memory == 0: lowvram_model_memory = 0.1
if vram_set_state == VRAMState.NO_VRAM: lowvram_model_memory = 0.1
```
`MIN_WEIGHT_MEMORY_RATIO` L458 = 0.4, 0.0 on NVIDIA.
- **Partial loading** (`comfy\model_patcher.py`): `_load_list()` L945–980 enumerates leaf modules with params computing `module_mem` and `module_offload_mem`; `ModelPatcher.load()` L982–1111: `potential_offload = max(offload_buffer, module_offload_mem + sum(next NUM_STREAMS module costs))` (L1000) reserves a streaming buffer; if fits → `.to(device_to)`, else stays CPU with `m.comfy_cast_weights = True` (L1049–1051) and params pinned (L1085–1089). Whole leaf modules, largest-benefit-first. Forward: `comfy\ops.py` `CastWeightBiasOp` L482–486 + `forward()` (Linear L570–575) → `forward_comfy_cast_weights` → `cast_bias_weight(...)` (L337–438) copies CPU weight to GPU on a side stream into a reusable cast buffer; `uncast_bias_weight` (L441–459). Weights never written back. `partially_unload` L1171–1254; `partially_load` L1256–1282; `LowVramPatch` L171–202 applies LoRA on the fly; `archive_model_dtypes()` L1049–1054.
- **DynamicVRAM** (`ModelPatcherDynamic` L1749–2177): CUDA VMM demand paging via `comfy_aimdo` (`vbar.alloc`, `vbar_fault`, `vbar_free_memory`); not replicable in pure PyTorch.
- **soft_empty_cache** L2050–2066: `cuda.synchronize(); cuda.empty_cache(); cuda.ipc_collect()`; in `free_memory` L906–910 only when `mem_free_torch > mem_free_total * 0.25`.
- **OOM**: `OOM_EXCEPTION` L380–382, `is_oom(e)` L389–395 (also `torch.AcceleratorError` error_code==2 or "out of memory" in string), `discard_cuda_async_error()` L1602–1610. Retry sites: `attention.py` L437–450 (split attention doubles steps up to 64), `sd.py` L1253/L1390 (VAE tiled fallback — set `do_tile=True` and fall back *outside* the except block because the exception holds tensor refs), `cleanup_models_gc()` L1029–1046.
- **Pinned memory**: budget L1585–1593: Windows `MAX_PINNED_MEMORY = ram*0.40`; Linux `max(ram*0.40, min(ram*0.90, ram-4GB, ram+swap-16GB))`. `pin_memory(tensor)` L1612–1648 uses `torch.cuda.cudart().cudaHostRegister/Unregister` on existing `data_ptr()` (no copy, works on mmapped safetensors). Eviction ladder L648–743; `comfy\pinned_memory.py` `HostBuffer` slabs + `_steal_pin()` L19–48.
- **Async streaming**: `NUM_STREAMS` L1344–1357 (default 2 NVIDIA); `get_offload_stream(device)` L1463–1498 round-robins streams, returns None under `torch.compiler.is_compiling()`; `cast_to(..., stream=, r=)` L1531–1557; `get_cast_buffer` L1383–1413 per-stream int8 scratch; `cast_to_gathered` L1506–1528 packs weight+bias into one transfer (can DMA from file via `read_tensor_file_slice_into`).
- **Size estimation**: `module_size(module)` L635–641; `model_base.py memory_required()` L413–435; `load_safetensors()` `comfy\utils.py` L96–156 mmaps; `disable_weight_init.Linear.__init__` (`ops.py` L521–542) skips `torch.empty` (Windows doesn't overcommit).

## 2. Attention backends — `comfy\ldm\modules\attention.py`
Detection L20–54 (`SAGE_ATTENTION_IS_AVAILABLE`, `SAGE_ATTENTION_SUPPORTS_MASK = "attn_mask" in inspect.signature(sageattn).parameters` L29, `FLASH_ATTENTION_IS_AVAILABLE`); selection L853–899:
```python
optimized_attention = attention_basic
if model_management.sage_attention_enabled():      optimized_attention = attention_sage
elif model_management.flash_attention_enabled():   optimized_attention = attention_flash
elif model_management.xformers_enabled():          optimized_attention = attention_xformers
elif model_management.pytorch_attention_enabled(): optimized_attention = attention_pytorch
else: optimized_attention = attention_split if args.use_split_cross_attention else attention_sub_quad
```
`optimized_attention_for_device(device, mask, small_input)` L902–915. SDPA priority `comfy\ops.py` L64–99: `sdpa_kernel([FLASH, CUDNN, EFFICIENT, MATH], set_priority=True)` bypassed for `q.nelement() < 1024*128`; repeats KV when GQA unsupported (L83–94). Fallbacks: sage try/except → `attention_pytorch` (L674–685), pre-empts when mask present but unsupported (L647–648); flash raises on mask, catches all → SDPA (L827–845); `BROKEN_XFORMERS` L468–474; `SDP_BATCH_LIMIT = 2**15` L538–542. Capability gates in `comfy_kitchen`: flash `_MINIMUM_CAPABILITY = (8, 0)`, sage `(7, 5)`. `supports_fp8_compute` L1959–1981 (major≥9 or 8.9+; sm86 3090 → False), `supports_nvfp4_compute` L1983 (major≥10), `supports_mxfp8_compute` L1993.

## 3. Quantization at inference
`comfy\quant_ops.py` (300 lines) shim over `comfy_kitchen`; `QuantizedTensor` subclass routes `aten.linear` via `__torch_dispatch__` (design doc `QUANTIZATION.md`). Layouts L114–219; `QUANT_ALGOS` L235–281: `float8_e4m3fn`, `float8_e5m2`, `nvfp4`, `mxfp8`, `int8_tensorwise`, `convrot_w4a4`, `asym_w4a8_int8`. CUDA backend enabled only on cu130+ (L37–65); Triton backend by `--enable-triton-backend`.
`comfy\ops.py`: `mixed_precision_ops()` L1290–1649 → `MixedPrecisionOps.Linear` L1297–1445 (`forward` L1360–1425), `MoEExperts` L1447–1562, quantized `Embedding` L1564–1647; `fp8_linear()` L852–890; `linear_input_act()` L958–991 folds GELU/SwiGLU into INT8 activation quantizer; `QuantLinearFunc` L994–1090; `pick_operations()` L1651–1688. Loader `_load_quantized_module()` L1111–1247: JSON blob uint8 under `{prefix}comfy_quant`; scales `{prefix}weight_scale`, `weight_scale_2`, `weight_s_rel`, `weight_s_channel`, `weight_codebook`; ConvRot `convrot: bool` + `convrot_groupsize` (L1184–1188, L1196–1203); writer `_quantized_weight_state_dict()` L1250–1287.
Kernels (venv `comfy_kitchen`): `torch._int_mm` in `backends\eager\quantization.py` L722–800 (`_int8_matmul_accumulate` pads M→32, K→8, N→8/32 Turing), `torch._scaled_mm` in `scaled_mm_v2.py` L140, CUDA kernels `backends\cuda\__init__.py`.
`custom_nodes\ComfyUI-QuantOps`: `unified_ops.py` L42–701; `quant_layouts\int8_layout.py` `BlockWiseINT8Layout` L96–341, Triton `int8_gemm` L456–616, fallback L411–453; `fp8_variants.py`; `tensorwise_int8_layout.py` L13–53 per-channel patch (`kernels\int8_kernels.py` L1461+); `bnb4bit_ops.py` NF4/FP4 without bitsandbytes (`dequantize_bnb_4bit()` L154–210).
3090 vs 5090: sm86 fp8/nvfp4 dequantize (weight-only compression); INT8 `_int_mm` and Triton int8/fp8 run on sm86. sm120a native fp8/mxfp8/nvfp4.

## 4. GGUF — `custom_nodes\ComfyUI-GGUF`
`gguf>=0.13.0` (0.19.0 installed). `dequant.py`: `dequantize_tensor()` L15–28; `dequantize()` L30–44; per-type kernels L61–285; registry L287–301 supports BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS (others → numpy fallback). `ops.py`: `GGMLTensor` L44–91, `GGMLLayer` L93–225 (`get_weight()` L166–191), `GGMLOps` L227–271 — dequant inside forward every call. `loader.py`: `gguf_sd_loader()` L70, `gguf_mmproj_loader()` L271, `llama_permute()` L230.
```python
def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype  = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)

def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks   = dequantize_functions[qtype]
    rows     = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks   = rows.reshape((n_blocks, type_size))
    return dequantize_blocks(blocks, block_size, type_size, dtype).reshape(oshape)

def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    return (d.view(torch.float16).to(dtype) * x.view(torch.int8))
```

## 5. torch.compile
`comfy_extras\nodes_torch_compile.py` — backends `["inductor", "cudagraphs"]`; `model.clone(disable_dynamic=True)` — compile forces model off DynamicVRAM; `skip_torch_compile_dict()`. `comfy_api\torch_helpers\torch_compile.py` compiles selected submodules. `run_every_op()` L34–38 skips interrupt check while compiling; `get_offload_stream()` returns None while compiling. GGUF `ops.py` L21–42 version-gated `torch.compiler.disable`. CUDA graphs: `comfy\model_prefetch.py` L62–152 per-block `torch.cuda.CUDAGraph` for fully-resident decode steps; `--disable-cuda-graphs`.

## 6. LLM / VLM code — `comfy\text_encoders\llama.py` (1355 lines)
Configs: `Qwen25_7BVLI_Config` L323–345 (Qwen2.5-VL 7B), `Qwen3_8BConfig` L249, `Qwen3VL_*` L276–289. `Llama2_` L762–954: `get_dynamic_vram__units()` L793 returns `list(self.layers)` (declares swap units); `forward()` L824–954 builds causal mask, `optimized_attention_for_device(..., small_input=True)` L857, CUDA-graph static buffers L859–883, `make_prefetch_queue(list(self.layers), ...)` L898 + `prefetch_queue_pop(...)` L924 (prefetch N+1 while computing N). `Attention.forward` L540–624: `FixedKV` + `comfy_kitchen.flash_attention_decode` L574–580, ring/append KV L599–615, `enable_gqa=True` L622. `init_kv_cache()` L800–813 pre-allocates full cache. `BaseGenerate.generate()` L1017–1068; `logits()` L1002–1012 streams `lm_head` via `CastBiasWeightContext`. Qwen2.5-VL: `Qwen25_7BVLI` L1250–1301 with MRoPE; vision `qwen_vl.py` (`process_qwen2vl_images()` L9–89 smart-resize, `qwen2vl_mrope_position_ids()` L91, `Qwen2VLVisionTransformer` L289). Qwen3-VL `qwen3vl.py`. MoE: `gpt_oss.py` L160–250 with `ops.MoEExperts` — whole expert bank streamed, no per-expert offload (the gap to fill for Qwen3-Omni-30B-A3B).

## 7. Multi-GPU
`main.py` L46–106: Windows pins `CUDA_VISIBLE_DEVICES=0` unless specified; `--default-device N`; `--cuda-device`. `model_management.py`: `get_all_torch_devices()` L215–245, `get_gpu_device_options()` L247–258, `cuda_device_context(device)` L292–314, `set_torch_device()` L1825. `comfy\multigpu.py`: `MultiGPUThreadPool` L16–76 (one persistent thread per extra GPU), `create_multigpu_deepclones()` L125–197 (data parallel), `load_balance_devices()` L199–233. No tensor/pipeline parallelism.

## 8. Excerpts
**(a) Manual block swap + prefetch** (`custom_nodes\ComfyUI-WanVideoWrapper\wanvideo\modules\model.py` L3240–3298):
```python
            for b, block in enumerate(self.blocks):
                if self.prefetch_blocks > 0:
                    for prefetch_offset in range(1, self.prefetch_blocks + 1):
                        prefetch_idx = b + prefetch_offset
                        if prefetch_idx < len(self.blocks) and self.blocks_to_swap > 0 and prefetch_idx >= swap_start_idx:
                            with torch.cuda.stream(cuda_stream):
                                self.blocks[prefetch_idx].to(self.main_device, non_blocking=self.use_non_blocking)
                                events[prefetch_idx].record(cuda_stream)
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    if self.prefetch_blocks > 0 and not events[b].query():
                        events[b].synchronize()
                    block.to(self.main_device)
                x, ... = block(x, **kwargs)
                if b >= swap_start_idx and self.blocks_to_swap > 0:
                    block.to(self.offload_device, non_blocking=self.use_non_blocking)
```
**(b) Free memory** (`model_management.py` L1786–1797):
```python
            stats         = torch.cuda.memory_stats(dev)
            mem_active    = stats['active_bytes.all.current']
            mem_reserved  = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch
```

## 9. Recommended standalone `VRAMManager` design for transformers VLM/LLMs
**Do not use `device_map="auto"`** for the swap path (static placement, blocking `.to()` hooks, no prefetch, no pinned reuse). Load with `device_map={"": "cpu"}` / `low_cpu_mem_usage=True`, then install own hooks.

1. **Budget**: `free_vram = mem_get_info()[0] + (reserved - active)`; `reserved = 600MB (Windows) [+100MB if >15GB]`; `min_inference = 0.8GB + reserved`; `weight_budget = max(0, free_vram + already_resident - min_inference - kv_cache_bytes - vision_peak)`. `kv_cache_bytes = 2 * layers * kv_heads * head_dim * max_len * batch * dtype_size` (allocate cache first); `vision_peak` measured once per resolution bucket with `torch.cuda.max_memory_allocated()`.
2. **Placement pass**: enumerate leaf modules with params; sort by streaming cost; keep resident while `resident + module + stream_buffer < weight_budget`, `stream_buffer = sum of next NUM_STREAMS module sizes`. Always resident: embeddings, norms, rotary, MoE routers, vision tower (if many images). Offload first: `lm_head`, decoder blocks from the last backwards, MoE experts.
3. **Manual block-swap hooks**: coarse — `register_forward_pre_hook` on each decoder layer that waits on `events[i]`, issues `blocks[i+1..i+k].to(cuda, non_blocking=True)` on a dedicated stream; `register_forward_hook` does `block.to("cpu", non_blocking=True)`; `k=2`. Fine — for `lm_head` and expert banks stream the weight into a preallocated int8 scratch buffer.
4. **Pinned buffers**: `torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.nbytes, 1)` in place (guard cpu/contiguous/not pinned); cap 40% RAM on Windows; size-bucketed LRU; call `discard_cuda_async_error()` equivalent on failure. Pin only the offloaded tail.
5. **Expert offload for Qwen3-Omni-30B-A3B**: contiguous pinned 3-D tensor per layer `[E, out, in]`; after router `topk_idx`, copy the ~8 needed experts on the offload stream during attention compute; LRU of K hottest experts resident (16–32 typical, cuts transfers 50–70%); speculative prefetch of historically popular experts; int8/fp8 experts halve traffic (5090 fp8 `_scaled_mm`; 3090 `torch._int_mm`).
6. **OOM policy**: catch `torch.cuda.OutOfMemoryError` and `torch.AcceleratorError` (error_code==2 or "out of memory"); set flag, exit except block first, then `synchronize(); empty_cache(); ipc_collect()`, shrink `weight_budget` by 15%, re-run placement, retry. Only `empty_cache` when `mem_free_torch > 0.25 * mem_free_total`.
7. **Attention selection**: for causal LLM decode: `flash_attn_func`/`flash_attn_with_kvcache` → SDPA with priority → math. Skip sage for the LLM (INT8, tuned for non-causal DiT, no arbitrary masks); fine for ViT. try/except around fast kernel with SDPA fallback and log once.
8. **Keep HF transformers modeling + processor; replace only accelerate hooks.** Load `from_pretrained(..., dtype=..., low_cpu_mem_usage=True, device_map=None, attn_implementation="flash_attention_2"|"sdpa")`, then placement + hooks. Use `StaticCache`. torch.compile only the resident core with swapping disabled; CUDA graphs for fully-resident decode.

Flags: `Windows_Run_VRAM_Optimized.bat` line 83 `--normalvram` does not exist; `--disable-pinned-memory` recommended in `Windows_Run_GPU_Lowest_VRAM.bat` due to OOM bug — budget pinned memory conservatively.
