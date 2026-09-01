"""ConvRot single-file checkpoint metadata, kernels, modules, and streaming loader."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_HADAMARD_LOCK = threading.Lock()
_KERNEL_CACHE_PATH = Path(__file__).with_name(".kernel_cache.json")
_KERNEL_CACHE_LOCK = threading.Lock()
_KERNEL_CACHE_MEMORY: dict | None = None
_TRITON_KERNEL = None
_TRITON_MOE_KERNEL = None
_TRITON_QUANT_KERNEL = None
_TRITON_SILU_MUL_KERNEL = None
_TRITON_ROUTE_REDUCE_KERNEL = None
_TRITON_FAILED = False
_DECODE_DISPATCH_CACHE: dict[tuple[str, int], torch.Tensor] = {}


@dataclass(frozen=True)
class QuantLayerMeta:
    name: str
    format: str
    weight_shape: tuple[int, int]
    scale_shape: tuple[int, ...]
    convrot_groupsize: int
    bits: int


@dataclass(frozen=True)
class QuantMeta:
    path: Path
    format_version: str
    scheme: str
    group_size: int
    scale_layout: str
    layers: Mapping[str, QuantLayerMeta]


@dataclass(frozen=True)
class LoadReport:
    quantized_layers: int
    bf16_layers: int
    bytes_loaded: int
    seconds: float


def _checkpoint_path(path: str | os.PathLike[str]) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_dir():
        value = value / "model.safetensors"
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def read_quant_metadata(path: str | os.PathLike[str]) -> QuantMeta | None:
    checkpoint = _checkpoint_path(path)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        encoded = (handle.metadata() or {}).get("_quantization_metadata")
    if not encoded:
        return None
    raw = json.loads(encoded)
    layers: dict[str, QuantLayerMeta] = {}
    default_group = int(raw.get("group_size", 256))
    for name, item in raw.get("layers", {}).items():
        shape = item.get("weight_shape")
        if not shape:
            raise ValueError(f"Quantized layer {name} has no weight_shape")
        layers[name] = QuantLayerMeta(
            name=name,
            format=str(item.get("format", "")),
            weight_shape=(int(shape[0]), int(shape[1])),
            scale_shape=tuple(int(value) for value in item.get("scale_shape", (shape[0], 1))),
            convrot_groupsize=int(item.get("convrot_groupsize", default_group)),
            bits=int(item.get("bits", 8)),
        )
    return QuantMeta(
        path=checkpoint,
        format_version=str(raw.get("format_version", "")),
        scheme=str(raw.get("scheme", "")),
        group_size=default_group,
        scale_layout=str(raw.get("scale_layout", "row")),
        layers=layers,
    )


def iter_safetensors_tensors(path: str | os.PathLike[str]) -> Iterator[tuple[str, torch.Tensor]]:
    checkpoint = _checkpoint_path(path)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            yield name, handle.get_tensor(name)


def estimate_checkpoint_vram_gb(path: str | os.PathLike[str]) -> float:
    checkpoint = _checkpoint_path(path)
    total = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_slice(name)
            total += math.prod(tensor.get_shape()) * _DTYPE_BYTES[tensor.get_dtype()]
    return total / 1024**3


def _regular_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (size, str(device), dtype)
    with _HADAMARD_LOCK:
        cached = _HADAMARD_CACHE.get(key)
        if cached is not None:
            return cached
        value = size
        while value > 1 and value % 4 == 0:
            value //= 4
        if size < 4 or value != 1:
            raise ValueError(f"Regular Hadamard size must be a power of four, got {size}")
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            device=device,
            dtype=torch.float32,
        )
        matrix = h4
        while matrix.shape[0] < size:
            matrix = torch.kron(matrix, h4)
        matrix = (matrix / math.sqrt(size)).to(dtype=dtype)
        _HADAMARD_CACHE[key] = matrix
        return matrix


def _rotate_activation(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if x.shape[-1] % group_size:
        raise ValueError(f"Input width {x.shape[-1]} is not divisible by ConvRot group {group_size}")
    matrix = _regular_hadamard(group_size, x.device, x.dtype)
    shape = x.shape
    return x.reshape(-1, shape[-1] // group_size, group_size).matmul(matrix).reshape(shape)


def _unpack_int4(weight: torch.Tensor) -> torch.Tensor:
    packed = weight.to(torch.int16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack((low, high), dim=-1).reshape(*weight.shape[:-1], weight.shape[-1] * 2).to(torch.int8)


def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.is_cuda and _triton_available():
        return _triton_quantize_activation(x)
    scale = (x.float().abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    quantized = (x / scale.to(x.dtype)).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _int_mm_padded(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    original_m = a.shape[0]
    original_n = weight.shape[0]
    m = _round_up(max(original_m, 32), 32) if a.is_cuda else original_m
    k = _round_up(a.shape[1], 8)
    n_alignment = 32 if a.is_cuda and torch.cuda.get_device_capability(a.device) == (7, 5) else 8
    n = _round_up(original_n, n_alignment)
    if (m, k) != a.shape:
        padded = torch.zeros((m, k), device=a.device, dtype=torch.int8)
        padded[:original_m, : a.shape[1]] = a
        a = padded
    transposed = weight.T.contiguous()
    if transposed.shape != (k, n):
        padded_weight = torch.zeros((k, n), device=weight.device, dtype=torch.int8)
        padded_weight[: transposed.shape[0], : transposed.shape[1]] = transposed
        transposed = padded_weight
    operator = getattr(torch, "int8_mm", torch._int_mm)
    return operator(a, transposed)[:original_m, :original_n]


def _load_kernel_cache() -> dict:
    global _KERNEL_CACHE_MEMORY
    with _KERNEL_CACHE_LOCK:
        if _KERNEL_CACHE_MEMORY is not None:
            return _KERNEL_CACHE_MEMORY
        try:
            value = json.loads(_KERNEL_CACHE_PATH.read_text(encoding="utf-8"))
            if value.get("format_version") != "1.0":
                value = {"format_version": "1.0", "entries": {}}
        except Exception:
            value = {"format_version": "1.0", "entries": {}}
        _KERNEL_CACHE_MEMORY = value
        return value


def _save_kernel_cache(cache: dict) -> None:
    with _KERNEL_CACHE_LOCK:
        try:
            partial = _KERNEL_CACHE_PATH.with_name(_KERNEL_CACHE_PATH.name + ".partial")
            partial.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
            os.replace(partial, _KERNEL_CACHE_PATH)
        except OSError:
            pass


def _triton_available() -> bool:
    global _TRITON_FAILED
    if _TRITON_FAILED:
        return False
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401

        return True
    except Exception:
        _TRITON_FAILED = True
        return False


def _get_triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        M,
        K,
        N,
        weight_stride,
        PACKED: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N
        offsets_m = tl.arange(0, 16)
        mask_m = offsets_m < M

        maximum = tl.zeros((16,), tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offsets_k = k0 + tl.arange(0, BLOCK_K)
            values = tl.load(
                x_ptr + offsets_m[:, None] * K + offsets_k[None, :],
                mask=mask_m[:, None] & (offsets_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            maximum = tl.maximum(maximum, tl.max(tl.abs(values), axis=1))
        activation_scale = tl.maximum(maximum / 127.0, 1.0e-30)
        activation_scale_math = activation_scale.to(x_ptr.dtype.element_ty).to(tl.float32)

        accumulator = tl.zeros((16, BLOCK_N), tl.int32)
        for k0 in range(0, K, BLOCK_K):
            offsets_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < K
            values = tl.load(
                x_ptr + offsets_m[:, None] * K + offsets_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            activation = (values / activation_scale_math[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
            activation = libdevice.rint(activation)
            activation = tl.minimum(tl.maximum(activation, -127.0), 127.0).to(tl.int8)
            if PACKED:
                byte_offsets = offsets_k // 2
                packed = tl.load(
                    weight_ptr + offsets_n[:, None] * weight_stride + byte_offsets[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0,
                ).to(tl.int32)
                shift = (offsets_k & 1) * 4
                nibble = (packed >> shift[None, :]) & 15
                weight = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.int8)
            else:
                weight = tl.load(
                    weight_ptr + offsets_n[:, None] * weight_stride + offsets_k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :],
                    other=0,
                ).to(tl.int8)
            accumulator = tl.dot(activation, tl.trans(weight), accumulator, out_dtype=tl.int32)

        weight_scale = tl.load(scale_ptr + offsets_n, mask=mask_n, other=0.0)
        output = accumulator.to(tl.float32) * (activation_scale[:, None] * weight_scale[None, :])
        if HAS_BIAS:
            output += tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0)[None, :]
        tl.store(
            output_ptr + offsets_m[:, None] * N + offsets_n[None, :],
            output,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    _TRITON_KERNEL = kernel
    return kernel


def _triton_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    packed: bool,
) -> torch.Tensor:
    import triton

    if x.shape[0] > 16:
        raise ValueError("Fused GEMV supports at most 16 rows")
    output = torch.empty((x.shape[0], scale.shape[0]), device=x.device, dtype=x.dtype)
    block_n = 16 if packed else 32
    kernel = _get_triton_kernel()
    kernel[(triton.cdiv(output.shape[1], block_n),)](
        x.contiguous(),
        weight,
        scale.reshape(-1),
        bias if bias is not None else scale,
        output,
        x.shape[0],
        x.shape[1],
        output.shape[1],
        weight.shape[-1],
        PACKED=packed,
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=256,
        num_warps=4,
        num_stages=2,
    )
    return output


def _get_triton_quant_kernel():
    global _TRITON_QUANT_KERNEL
    if _TRITON_QUANT_KERNEL is not None:
        return _TRITON_QUANT_KERNEL
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def kernel(x_ptr, output_ptr, scale_ptr, M, K, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        maximum = 0.0
        for k0 in range(0, K, BLOCK_K):
            offsets = k0 + tl.arange(0, BLOCK_K)
            values = tl.load(x_ptr + row * K + offsets, mask=offsets < K, other=0.0).to(tl.float32)
            maximum = tl.maximum(maximum, tl.max(tl.abs(values), axis=0))
        scale = tl.maximum(maximum / 127.0, 1.0e-30)
        scale_math = scale.to(x_ptr.dtype.element_ty).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offsets = k0 + tl.arange(0, BLOCK_K)
            mask = offsets < K
            values = tl.load(x_ptr + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
            quantized = (values / scale_math).to(x_ptr.dtype.element_ty).to(tl.float32)
            quantized = libdevice.rint(quantized)
            quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0).to(tl.int8)
            tl.store(output_ptr + row * K + offsets, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)

    _TRITON_QUANT_KERNEL = kernel
    return kernel


def _triton_quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or not x.is_contiguous():
        x = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    _get_triton_quant_kernel()[(x.shape[0],)](
        x,
        output,
        scale,
        x.shape[0],
        x.shape[1],
        BLOCK_K=256,
        num_warps=4,
        num_stages=1,
    )
    return output, scale


def _get_triton_silu_mul_kernel():
    global _TRITON_SILU_MUL_KERNEL
    if _TRITON_SILU_MUL_KERNEL is not None:
        return _TRITON_SILU_MUL_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(gate_up_ptr, output_ptr, rows, width: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = rows * width
        mask = offsets < total
        row = offsets // width
        column = offsets - row * width
        gate = tl.load(gate_up_ptr + row * (2 * width) + column, mask=mask).to(tl.float32)
        up = tl.load(gate_up_ptr + row * (2 * width) + width + column, mask=mask).to(tl.float32)
        value = gate * tl.sigmoid(gate) * up
        tl.store(output_ptr + offsets, value, mask=mask)

    _TRITON_SILU_MUL_KERNEL = kernel
    return kernel


def _triton_silu_mul(gate_up: torch.Tensor, width: int) -> torch.Tensor:
    import triton

    output = torch.empty((gate_up.shape[0], width), device=gate_up.device, dtype=gate_up.dtype)
    _get_triton_silu_mul_kernel()[(triton.cdiv(output.numel(), 256),)](
        gate_up,
        output,
        gate_up.shape[0],
        width=width,
        BLOCK=256,
        num_warps=4,
    )
    return output


def _get_triton_route_reduce_kernel():
    global _TRITON_ROUTE_REDUCE_KERNEL
    if _TRITON_ROUTE_REDUCE_KERNEL is not None:
        return _TRITON_ROUTE_REDUCE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(values_ptr, routing_ptr, output_ptr, H: tl.constexpr, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
        token = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < H
        accumulator = tl.zeros((BLOCK,), tl.float32)
        for index in range(TOP_K):
            value = tl.load(values_ptr + (token * TOP_K + index) * H + offsets, mask=mask).to(tl.float32)
            routing = tl.load(routing_ptr + token * TOP_K + index).to(tl.float32)
            accumulator += value * routing
        tl.store(output_ptr + token * H + offsets, accumulator, mask=mask)

    _TRITON_ROUTE_REDUCE_KERNEL = kernel
    return kernel


def _triton_route_reduce(
    values: torch.Tensor,
    routing: torch.Tensor,
    token_count: int,
    top_k: int,
    hidden_dim: int,
) -> torch.Tensor:
    import triton

    output = torch.empty((token_count, hidden_dim), device=values.device, dtype=values.dtype)
    _get_triton_route_reduce_kernel()[(token_count, triton.cdiv(hidden_dim, 256))](
        values,
        routing,
        output,
        H=hidden_dim,
        TOP_K=top_k,
        BLOCK=256,
        num_warps=4,
    )
    return output


def _get_triton_moe_kernel():
    """Return the assignment-aware grouped INT8/packed-INT4 MoE GEMM kernel."""

    global _TRITON_MOE_KERNEL
    if _TRITON_MOE_KERNEL is not None:
        return _TRITON_MOE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        x_ptr,
        x_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        dispatch_ptr,
        block_expert_ptr,
        output_ptr,
        assignment_count,
        dispatch_size,
        top_k,
        K: tl.constexpr,
        N: tl.constexpr,
        weight_stride_e: tl.constexpr,
        weight_stride_n: tl.constexpr,
        PACKED: tl.constexpr,
        SOURCE_IS_TOKEN: tl.constexpr,
        ROWS_PER_BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_m = tl.program_id(0)
        block_n = tl.program_id(1)
        offsets_m = tl.arange(0, BLOCK_M)
        dispatch_offsets = block_m * ROWS_PER_BLOCK + offsets_m
        dispatch_mask = (offsets_m < ROWS_PER_BLOCK) & (dispatch_offsets < dispatch_size)
        assignments = tl.load(dispatch_ptr + dispatch_offsets, mask=dispatch_mask, other=assignment_count)
        expert = tl.load(block_expert_ptr + block_m)
        valid_rows = dispatch_mask & (assignments < assignment_count)
        safe_expert = tl.where(expert >= 0, expert, 0)
        source_rows = assignments // top_k if SOURCE_IS_TOKEN else assignments
        offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offsets_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < K
            activation = tl.load(
                x_ptr + source_rows[:, None] * K + offsets_k[None, :],
                mask=valid_rows[:, None] & mask_k[None, :],
                other=0,
            ).to(tl.int8)
            if PACKED:
                packed = tl.load(
                    weight_ptr
                    + safe_expert * weight_stride_e
                    + offsets_n[:, None] * weight_stride_n
                    + (offsets_k[None, :] // 2),
                    mask=(expert >= 0) & mask_n[:, None] & mask_k[None, :],
                    other=0,
                ).to(tl.int32)
                shift = (offsets_k & 1) * 4
                nibble = (packed >> shift[None, :]) & 15
                weight = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.int8)
            else:
                weight = tl.load(
                    weight_ptr
                    + safe_expert * weight_stride_e
                    + offsets_n[:, None] * weight_stride_n
                    + offsets_k[None, :],
                    mask=(expert >= 0) & mask_n[:, None] & mask_k[None, :],
                    other=0,
                ).to(tl.int8)
            accumulator = tl.dot(activation, tl.trans(weight), accumulator, out_dtype=tl.int32)

        activation_scale = tl.load(x_scale_ptr + source_rows, mask=valid_rows, other=0.0)
        weight_scale = tl.load(
            weight_scale_ptr + safe_expert * N + offsets_n,
            mask=(expert >= 0) & mask_n,
            other=0.0,
        )
        output = accumulator.to(tl.float32) * activation_scale[:, None] * weight_scale[None, :]
        tl.store(
            output_ptr + assignments[:, None] * N + offsets_n[None, :],
            output,
            mask=valid_rows[:, None] & mask_n[None, :],
        )

    _TRITON_MOE_KERNEL = kernel
    return kernel


def _grouped_dispatch(
    expert_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build a padded expert-sorted dispatch table without a device-to-host sync."""

    assignment_count = expert_ids.numel()
    if assignment_count <= 16:
        key = (str(expert_ids.device), assignment_count)
        dispatch = _DECODE_DISPATCH_CACHE.get(key)
        if dispatch is None:
            dispatch = torch.arange(assignment_count, device=expert_ids.device, dtype=torch.int32)
            _DECODE_DISPATCH_CACHE[key] = dispatch
        return dispatch, expert_ids, 1

    order = torch.argsort(expert_ids)
    sorted_experts = expert_ids[order]
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    padded_counts = ((counts + block_m - 1) // block_m) * block_m
    raw_offsets = torch.cumsum(counts, dim=0) - counts
    padded_ends = torch.cumsum(padded_counts, dim=0)
    padded_offsets = padded_ends - padded_counts
    destinations = torch.arange(assignment_count, device=expert_ids.device)
    destinations = destinations + (padded_offsets - raw_offsets)[sorted_experts]

    capacity = _round_up(assignment_count + num_experts * (block_m - 1), block_m)
    dispatch = torch.full(
        (capacity,), assignment_count, device=expert_ids.device, dtype=torch.int32
    )
    dispatch[destinations] = order.to(torch.int32)
    block_positions = torch.arange(0, capacity, block_m, device=expert_ids.device)
    block_experts = torch.searchsorted(padded_ends, block_positions, right=True).to(torch.int32)
    block_experts.masked_fill_(block_experts >= num_experts, -1)
    return dispatch, block_experts, block_m


def _triton_grouped_moe_mm(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    dispatch: torch.Tensor,
    block_experts: torch.Tensor,
    assignment_count: int,
    top_k: int,
    *,
    packed: bool,
    source_is_token: bool,
    rows_per_block: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    import triton

    n = int(weight_scale.shape[1])
    k = int(activation.shape[1])
    output = torch.zeros((assignment_count, n), device=activation.device, dtype=output_dtype)
    block_n = 16 if packed else 32
    kernel = _get_triton_moe_kernel()
    kernel[(block_experts.numel(), triton.cdiv(n, block_n))](
        activation,
        activation_scale,
        weight,
        weight_scale,
        dispatch,
        block_experts,
        output,
        assignment_count,
        dispatch.numel(),
        top_k,
        K=k,
        N=n,
        weight_stride_e=weight.stride(0),
        weight_stride_n=weight.stride(1),
        PACKED=packed,
        SOURCE_IS_TOKEN=source_is_token,
        ROWS_PER_BLOCK=rows_per_block,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=256,
        num_warps=4,
        num_stages=2,
    )
    return output


def _kernel_cache_key(
    scheme: str, x: torch.Tensor, out_features: int, in_features: int
) -> tuple[str, str]:
    gpu = torch.cuda.get_device_name(x.device) if x.is_cuda else "cpu"
    capability = torch.cuda.get_device_capability(x.device) if x.is_cuda else (0, 0)
    gpu_key = f"{gpu}|sm{capability[0]}{capability[1]}"
    shape_key = f"{scheme}|{str(x.dtype).split('.')[-1]}|m{min(x.shape[0], 16)}|k{in_features}|n{out_features}"
    return gpu_key, shape_key


def _run_linear_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    scheme: str,
    kernel: str,
) -> torch.Tensor:
    packed = scheme == "int4_convrot_w4a8"
    if kernel == "triton":
        return _triton_gemv(x, weight, scale, bias, packed)
    int8_weight = _unpack_int4(weight) if packed else weight
    if kernel == "bf16":
        dequantized = int8_weight.to(x.dtype) * scale.to(x.dtype)
        return F.linear(x, dequantized, bias)
    if kernel != "int_mm":
        raise ValueError(kernel)
    rows = int(x.shape[0])
    chunk = _int_mm_row_chunk(rows, int(int8_weight.shape[0]), int(x.shape[1]))
    if rows <= chunk:
        return _int_mm_linear_rows(x, int8_weight, scale, bias)
    # Prefill: bound the int32/fp32 temporaries of the [rows, N] product. Activation
    # scales are per row, so chunking reproduces the unchunked result exactly.
    output = torch.empty((rows, int(int8_weight.shape[0])), device=x.device, dtype=x.dtype)
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        output[start:stop] = _int_mm_linear_rows(x[start:stop], int8_weight, scale, bias)
    return output


_INT_MM_CHUNK_BUDGET_BYTES = 256 * 2**20


def _int_mm_row_chunk(rows: int, out_features: int, in_features: int) -> int:
    """Rows per int_mm call so the int32 + fp32 + output temporaries stay near 256 MiB."""

    per_row = 10 * max(1, out_features) + max(1, in_features)
    chunk = _INT_MM_CHUNK_BUDGET_BYTES // per_row
    chunk = max(256, (chunk // 32) * 32)
    return min(max(rows, 1), chunk)


def _int_mm_linear_rows(
    x: torch.Tensor,
    int8_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    activation, activation_scale = _quantize_activation(x)
    accumulated = _int_mm_padded(activation, int8_weight)
    output = accumulated.float() * (activation_scale * scale.reshape(1, -1).float())
    if bias is not None:
        output = output + bias.float().reshape(1, -1)
    return output.to(x.dtype)


def _benchmark_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    scheme: str,
) -> str:
    if not x.is_cuda:
        return "bf16"
    gpu_key, shape_key = _kernel_cache_key(scheme, x, scale.shape[0], x.shape[1])
    cache = _load_kernel_cache()
    entry = cache.setdefault("entries", {}).setdefault(gpu_key, {}).get(shape_key)
    candidates = ["int_mm", "bf16"]
    if x.shape[0] <= 16 and _triton_available():
        candidates.append("triton")
    if entry and entry.get("selected") in candidates:
        return str(entry["selected"])

    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    with torch.inference_mode():
        for candidate in candidates:
            try:
                for _ in range(2):
                    _run_linear_kernel(x, weight, scale, bias, scheme, candidate)
                torch.cuda.synchronize(x.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):
                    _run_linear_kernel(x, weight, scale, bias, scheme, candidate)
                end.record()
                end.synchronize()
                timings[candidate] = start.elapsed_time(end) / 5.0
            except Exception as exc:
                failures[candidate] = f"{type(exc).__name__}: {exc}"[:500]
                torch.cuda.synchronize(x.device)
                torch.cuda.empty_cache()
    if not timings:
        raise RuntimeError(f"All ConvRot kernels failed: {failures}")
    selected = min(timings, key=timings.get)
    cache["entries"][gpu_key][shape_key] = {
        "selected": selected,
        "timings_ms": timings,
        "failures": failures,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    _save_kernel_cache(cache)
    return selected


def _quantized_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    scheme: str,
    group_size: int,
) -> torch.Tensor:
    shape = x.shape
    rotated = _rotate_activation(x, group_size).reshape(-1, shape[-1])
    kernel = "int_mm"
    if rotated.shape[0] <= 16:
        # Decode is latency-bound. The fused kernel consumes row-major INT8 or
        # packed INT4 weights directly and avoids both the cache lookup and the
        # dequant/unpack allocations considered by the offline microbenchmark.
        kernel = "triton" if rotated.is_cuda and _triton_available() else _benchmark_kernel(
            rotated, weight, scale, bias, scheme
        )
    output = _run_linear_kernel(rotated, weight, scale, bias, scheme, kernel)
    return output.reshape(*shape[:-1], scale.shape[0])


class _ConvRotLinearBase(nn.Module):
    scheme: str

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 256,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.register_parameter(
            "bias",
            nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _quantized_linear(
            x, self.weight, self.weight_scale, self.bias, self.scheme, self.group_size
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


class ConvRotInt8Linear(_ConvRotLinearBase):
    scheme = "int8_convrot"

    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        device = kwargs.get("device")
        super().__init__(in_features, out_features, **kwargs)
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device)
        )


class ConvRotInt4W4A8Linear(_ConvRotLinearBase):
    scheme = "int4_convrot_w4a8"

    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        device = kwargs.get("device")
        super().__init__(in_features, out_features, **kwargs)
        self.register_buffer(
            "weight", torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.empty(out_features, 1, dtype=torch.float32, device=device)
        )


class _FusedQKVProjection(nn.Module):
    """One ConvRot projection shared by the Q/K/V proxy modules."""

    def __init__(self, projections: tuple[_ConvRotLinearBase, ...]) -> None:
        super().__init__()
        first = projections[0]
        self.scheme = first.scheme
        self.group_size = first.group_size
        self.in_features = first.in_features
        self.output_sizes = tuple(item.out_features for item in projections)
        self.register_buffer("weight", torch.cat([item.weight for item in projections], dim=0))
        self.register_buffer(
            "weight_scale", torch.cat([item.weight_scale for item in projections], dim=0)
        )
        biases = [item.bias for item in projections]
        self.register_parameter(
            "bias",
            nn.Parameter(torch.cat(biases), requires_grad=False) if all(item is not None for item in biases) else None,
        )
        self._cached_input: weakref.ReferenceType[torch.Tensor] | None = None
        self._cached_outputs: tuple[torch.Tensor, ...] | None = None

    def project(self, index: int, x: torch.Tensor) -> torch.Tensor:
        cached_input = self._cached_input() if self._cached_input is not None else None
        if cached_input is not x or self._cached_outputs is None:
            combined = _quantized_linear(
                x,
                self.weight,
                self.weight_scale,
                self.bias,
                self.scheme,
                self.group_size,
            )
            self._cached_input = weakref.ref(x)
            self._cached_outputs = combined.split(self.output_sizes, dim=-1)
        output = self._cached_outputs[index]
        if index == len(self.output_sizes) - 1:
            self._cached_input = None
            self._cached_outputs = None
        return output


class _QKVProjectionProxy(nn.Module):
    def __init__(self, shared: _FusedQKVProjection, index: int, *, owner: bool) -> None:
        super().__init__()
        self.in_features = shared.in_features
        self.out_features = shared.output_sizes[index]
        self.index = int(index)
        if owner:
            self.shared = shared
            self.__dict__["_shared_ref"] = None
        else:
            self.__dict__["_shared_ref"] = weakref.ref(shared)

    def _shared(self) -> _FusedQKVProjection:
        owned = self._modules.get("shared")
        if owned is not None:
            return owned
        value = self.__dict__["_shared_ref"]()
        if value is None:
            raise RuntimeError("The fused QKV projection owner was released")
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared().project(self.index, x)


def _fuse_quantized_qkv(model: nn.Module) -> int:
    """Fuse adjacent Q/K/V ConvRot modules after their checkpoint tensors are loaded."""

    fused = 0
    for module in list(model.modules()):
        projections = tuple(getattr(module, name, None) for name in ("q_proj", "k_proj", "v_proj"))
        if not all(isinstance(item, _ConvRotLinearBase) for item in projections):
            continue
        first = projections[0]
        if any(
            item.scheme != first.scheme
            or item.group_size != first.group_size
            or item.in_features != first.in_features
            or (item.bias is None) != (first.bias is None)
            for item in projections[1:]
        ):
            continue
        shared = _FusedQKVProjection(projections)
        module.q_proj = _QKVProjectionProxy(shared, 0, owner=True)
        module.k_proj = _QKVProjectionProxy(shared, 1, owner=False)
        module.v_proj = _QKVProjectionProxy(shared, 2, owner=False)
        fused += 1
    return fused


class _OnDemandCudaModule(nn.Module):
    """Keep a prefill-only tower on CPU and stage it for the duration of its forward."""

    def __init__(self, module: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.module = module.to("cpu")
        self.target_device = torch.device(device)
        self._stage_lock = threading.RLock()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("module"), name)

    def forward(self, *args, **kwargs):
        with self._stage_lock:
            self.module.to(self.target_device)
            try:
                return self.module(*args, **kwargs)
            finally:
                # Parameter copies back to CPU serialize after the tower's work,
                # so returned CUDA features are complete before weights leave VRAM.
                self.module.to("cpu")
                torch.cuda.empty_cache()


def _offload_prefill_towers_if_needed(
    model: nn.Module,
    checkpoint: Path,
    target_device: torch.device,
    *,
    force: bool = False,
) -> tuple[str, ...]:
    if target_device.type != "cuda":
        return ()
    setting = os.environ.get("VCAP_QUANT_TOWER_OFFLOAD", "auto").strip().lower()
    if not force and setting in {"0", "false", "off"}:
        return ()
    _, total = torch.cuda.mem_get_info(target_device)
    pressure = checkpoint.stat().st_size / max(total, 1)
    if not force and setting == "auto" and pressure < 0.88:
        return ()
    names = []
    # Transformers derives the root model's input device from its first
    # parameter. Keep that contract on CUDA even when both prefill-only towers
    # live on CPU between calls.
    if "_vcap_device_anchor" not in model._parameters:
        model.register_parameter(
            "_vcap_device_anchor",
            nn.Parameter(torch.empty(0, device=target_device), requires_grad=False),
        )
    for name in ("audio_tower", "visual"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module) and not isinstance(module, _OnDemandCudaModule):
            setattr(model, name, _OnDemandCudaModule(module, target_device))
            names.append(name)
    return tuple(names)


class _ConvRotExpertsBase(nn.Module):
    scheme: str

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        group_size: int = 256,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.group_size = int(group_size)
        self.act_fn = act_fn
        packed_divisor = 2 if self.scheme == "int4_convrot_w4a8" else 1
        self.register_buffer(
            "gate_up_weight",
            torch.empty(
                num_experts,
                2 * intermediate_dim,
                hidden_dim // packed_divisor,
                dtype=torch.int8,
                device=device,
            ),
        )
        self.register_buffer(
            "gate_up_weight_scale",
            torch.empty(num_experts, 2 * intermediate_dim, 1, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "down_weight",
            torch.empty(
                num_experts,
                hidden_dim,
                intermediate_dim // packed_divisor,
                dtype=torch.int8,
                device=device,
            ),
        )
        self.register_buffer(
            "down_weight_scale",
            torch.empty(num_experts, hidden_dim, 1, dtype=torch.float32, device=device),
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        assignment_count = token_count * top_k
        expert_ids = top_k_index.reshape(-1)
        dispatch, block_experts, rows_per_block = _grouped_dispatch(
            expert_ids, self.num_experts, 16
        )
        packed = self.scheme == "int4_convrot_w4a8"

        rotated = _rotate_activation(hidden_states, self.group_size)
        activation, activation_scale = _quantize_activation(rotated)
        gate_up = _triton_grouped_moe_mm(
            activation,
            activation_scale,
            self.gate_up_weight,
            self.gate_up_weight_scale,
            dispatch,
            block_experts,
            assignment_count,
            top_k,
            packed=packed,
            source_is_token=True,
            rows_per_block=rows_per_block,
            output_dtype=hidden_states.dtype,
        )
        activation_name = type(self.act_fn).__name__.lower()
        if (
            self.act_fn is F.silu
            or getattr(self.act_fn, "__name__", "") in {"silu", "silu_forward"}
            or "silu" in activation_name
        ):
            intermediate = _triton_silu_mul(gate_up, self.intermediate_dim)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = self.act_fn(gate) * up
        rotated_intermediate = _rotate_activation(intermediate, self.group_size)
        intermediate_q, intermediate_scale = _quantize_activation(rotated_intermediate)
        current = _triton_grouped_moe_mm(
            intermediate_q,
            intermediate_scale,
            self.down_weight,
            self.down_weight_scale,
            dispatch,
            block_experts,
            assignment_count,
            top_k,
            packed=packed,
            source_is_token=False,
            rows_per_block=rows_per_block,
            output_dtype=hidden_states.dtype,
        )
        return _triton_route_reduce(
            current,
            top_k_weights,
            token_count,
            top_k,
            self.hidden_dim,
        )

    def _fallback_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        token_count, hidden_dim = hidden_states.shape
        experts = top_k_index.reshape(-1)
        assignment_tokens = (
            torch.arange(token_count, device=hidden_states.device)[:, None]
            .expand_as(top_k_index)
            .reshape(-1)
        )
        assignment_weights = top_k_weights.reshape(-1)
        order = torch.argsort(experts)
        sorted_experts = experts[order]
        sorted_tokens = assignment_tokens[order]
        sorted_weights = assignment_weights[order]
        unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        output = torch.zeros_like(hidden_states)

        # Copy the compact dispatch table once. Calling .item() for every id
        # and offset serializes the CUDA stream three times per active expert
        # (384 synchronizations/layer on a full-prefill MoE batch).
        dispatch = torch.stack((unique, counts), dim=1).tolist()
        begin = 0
        for expert, count in dispatch:
            expert = int(expert)
            end = begin + int(count)
            if expert >= self.num_experts:
                begin = end
                continue
            token_index = sorted_tokens[begin:end]
            current = hidden_states[token_index]
            gate_up = _quantized_linear(
                current,
                self.gate_up_weight[expert],
                self.gate_up_weight_scale[expert],
                None,
                self.scheme,
                self.group_size,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = _quantized_linear(
                current,
                self.down_weight[expert],
                self.down_weight_scale[expert],
                None,
                self.scheme,
                self.group_size,
            )
            current = current * sorted_weights[begin:end, None]
            output.index_add_(0, token_index, current.to(output.dtype))
            begin = end
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        _, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden size {self.hidden_dim}, got {hidden_dim}")
        fused_enabled = os.environ.get("VCAP_FUSED_MOE", "1").strip().lower() not in {
            "0",
            "false",
            "off",
        }
        if hidden_states.is_cuda and fused_enabled and _triton_available():
            return self._fused_forward(hidden_states, top_k_index, top_k_weights)
        return self._fallback_forward(hidden_states, top_k_index, top_k_weights)


class ConvRotInt8Experts(_ConvRotExpertsBase):
    scheme = "int8_convrot"


class ConvRotInt4Experts(_ConvRotExpertsBase):
    scheme = "int4_convrot_w4a8"


def fuse_bf16_experts_from_per_expert(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fuse ``{expert}.{gate,up,down}_proj.weight`` tensors like Transformers 5.x."""
    pattern = re.compile(r"^(?:.*\.experts\.)?(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    grouped: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        match = pattern.match(name)
        if match:
            grouped.setdefault(int(match.group(1)), {})[match.group(2)] = tensor
    if not grouped:
        raise ValueError("No per-expert projection tensors were found")
    expected = list(range(max(grouped) + 1))
    if sorted(grouped) != expected or any(set(grouped[index]) != {"gate_proj", "up_proj", "down_proj"} for index in expected):
        raise ValueError("Expert tensors are incomplete")
    gate_up = torch.stack(
        [torch.cat((grouped[index]["gate_proj"], grouped[index]["up_proj"]), dim=0) for index in expected]
    )
    down = torch.stack([grouped[index]["down_proj"] for index in expected])
    return {"gate_up_proj": gate_up, "down_proj": down}


def _strip(name: str, prefix: str) -> str:
    return name[len(prefix) :] if prefix and name.startswith(prefix) else name


def _parent_and_leaf(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = name.rpartition(".")
    modules = dict(model.named_modules())
    if parent_name not in modules:
        raise KeyError(f"Module path not found: {parent_name}")
    return modules[parent_name], leaf


def _replace_child(model: nn.Module, name: str, value: nn.Module) -> None:
    parent, leaf = _parent_and_leaf(model, name)
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _assign_tensor(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parent, leaf = _parent_and_leaf(model, name)
    if leaf in parent._parameters:
        old = parent._parameters[leaf]
        requires_grad = bool(old.requires_grad) if old is not None else False
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=requires_grad)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        raise KeyError(f"Tensor target not found: {name}")


_EXPERT_TENSOR = re.compile(
    r"^((?:.*\.)?experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight(_scale)?$"
)
_DECODER_LAYER = re.compile(r"^model\.layers\.(\d+)(?:\.|$)")


def _placement_device(
    name: str,
    default: torch.device,
    layer_device: Callable[[int], torch.device | str] | None,
) -> torch.device:
    if layer_device is None:
        return default
    match = _DECODER_LAYER.match(name)
    if match is None:
        return default
    return torch.device(layer_device(int(match.group(1))))


def _materialize_quant_experts(module: _ConvRotExpertsBase, device: torch.device) -> None:
    for name, tensor in list(module._buffers.items()):
        if tensor is not None and tensor.device.type == "meta":
            module._buffers[name] = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)


def _assign_quant_expert(
    module: _ConvRotExpertsBase,
    expert: int,
    projection: str,
    is_scale: bool,
    value: torch.Tensor,
) -> None:
    intermediate = module.intermediate_dim
    if projection == "down_proj":
        target = module.down_weight_scale if is_scale else module.down_weight
        target[expert].copy_(value)
        return
    target = module.gate_up_weight_scale if is_scale else module.gate_up_weight
    start = 0 if projection == "gate_proj" else intermediate
    target[expert, start : start + intermediate].copy_(value)


def _materialize_bf16_expert_parameter(
    module: nn.Module, projection: str, device: torch.device, dtype: torch.dtype
) -> nn.Parameter:
    parameter_name = "down_proj" if projection == "down_proj" else "gate_up_proj"
    parameter = module._parameters[parameter_name]
    if parameter.device.type == "meta":
        parameter = nn.Parameter(
            torch.empty(parameter.shape, device=device, dtype=dtype), requires_grad=False
        )
        module._parameters[parameter_name] = parameter
    return parameter


def _assign_bf16_expert(
    module: nn.Module,
    expert: int,
    projection: str,
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    parameter = _materialize_bf16_expert_parameter(module, projection, device, dtype)
    if projection == "down_proj":
        parameter.data[expert].copy_(value)
        return
    intermediate = parameter.shape[1] // 2
    start = 0 if projection == "gate_proj" else intermediate
    parameter.data[expert, start : start + intermediate].copy_(value)


def _move_non_meta_buffers(
    model: nn.Module,
    device: torch.device,
    layer_device: Callable[[int], torch.device | str] | None = None,
) -> None:
    for module_name, module in model.named_modules():
        module_device = _placement_device(module_name, device, layer_device)
        for name, value in list(module._buffers.items()):
            if (
                value is not None
                and value.device.type != "meta"
                and value.device != module_device
            ):
                module._buffers[name] = value.to(module_device)


def _materialize_meta_buffers(
    model: nn.Module,
    device: torch.device,
    layer_device: Callable[[int], torch.device | str] | None = None,
) -> None:
    """Rebuild deterministic non-persistent buffers lost under ``torch.device('meta')``."""
    for module_name, module in model.named_modules():
        module_device = _placement_device(module_name, device, layer_device)
        for name, value in list(module._buffers.items()):
            if value is None or value.device.type != "meta":
                continue
            rebuilt = None
            if name == "positional_embedding" and hasattr(
                module, "compute_default_singular_positional_embedding"
            ):
                rebuilt = module.compute_default_singular_positional_embedding()
            elif name == "inv_freq" and hasattr(module, "compute_default_rope_parameters"):
                rebuilt, attention_scaling = module.compute_default_rope_parameters(
                    module.config
                )
                module.attention_scaling = attention_scaling
            elif name == "inv_freq" and hasattr(module, "dim") and hasattr(module, "theta"):
                rebuilt = 1.0 / (
                    module.theta
                    ** (
                        torch.arange(
                            0,
                            module.dim,
                            2,
                            dtype=torch.float32,
                            device=module_device,
                        )
                        / module.dim
                    )
                )
            elif name == "original_inv_freq" and "inv_freq" in module._buffers:
                inv_freq = module._buffers["inv_freq"]
                if inv_freq.device.type != "meta":
                    rebuilt = inv_freq.clone()
            if rebuilt is not None:
                module._buffers[name] = rebuilt.to(device=module_device, dtype=value.dtype)


def apply_quantized_checkpoint(
    model: nn.Module,
    safetensors_path: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    progress_cb: Callable[[int, int, str], None] | None = None,
    strip_prefix: str = "thinker.",
    layer_device: Callable[[int], torch.device | str] | None = None,
    tower_offload: bool | None = None,
) -> LoadReport:
    """Swap quantized modules and stream one safetensors file into a meta model."""
    started = time.perf_counter()
    checkpoint = _checkpoint_path(safetensors_path)
    target_device = torch.device(device)
    meta = read_quant_metadata(checkpoint)
    quantized = meta is not None
    if quantized and meta.scale_layout != "row":
        raise NotImplementedError(f"Unsupported scale layout: {meta.scale_layout}")

    modules = dict(model.named_modules())
    quant_expert_parents: set[str] = set()
    if meta is not None:
        linear_class = ConvRotInt8Linear if meta.scheme == "int8_convrot" else ConvRotInt4W4A8Linear
        experts_class = ConvRotInt8Experts if meta.scheme == "int8_convrot" else ConvRotInt4Experts
        for checkpoint_layer, layer_meta in meta.layers.items():
            target_layer = _strip(checkpoint_layer, strip_prefix)
            expert_match = re.match(
                r"^((?:.*\.)?experts)\.\d+\.(?:gate_proj|up_proj|down_proj)$",
                target_layer,
            )
            if expert_match:
                quant_expert_parents.add(expert_match.group(1))
                continue
            original = modules.get(target_layer)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"Quant metadata target is not nn.Linear: {target_layer}")
            replacement = linear_class(
                layer_meta.weight_shape[1],
                layer_meta.weight_shape[0],
                group_size=layer_meta.convrot_groupsize,
                bias=original.bias is not None,
                device="meta",
            )
            _replace_child(model, target_layer, replacement)

        modules = dict(model.named_modules())
        for parent_name in sorted(quant_expert_parents):
            original = modules.get(parent_name)
            required = ("num_experts", "hidden_dim", "intermediate_dim", "act_fn")
            if original is None or any(not hasattr(original, item) for item in required):
                raise TypeError(f"Unrecognized Transformers experts module: {parent_name}")
            replacement = experts_class(
                original.num_experts,
                original.hidden_dim,
                original.intermediate_dim,
                original.act_fn,
                group_size=meta.group_size,
                device="meta",
            )
            _replace_child(model, parent_name, replacement)
        modules = dict(model.named_modules())
        for parent_name in quant_expert_parents:
            parent_device = _placement_device(parent_name, target_device, layer_device)
            _materialize_quant_experts(modules[parent_name], parent_device)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        total_bytes = sum(
            math.prod(handle.get_slice(name).get_shape())
            * _DTYPE_BYTES[handle.get_slice(name).get_dtype()]
            for name in names
        )
        loaded_bytes = 0
        bf16_linear_names: set[str] = set()
        modules = dict(model.named_modules())
        for name in names:
            value = handle.get_tensor(name)
            loaded_bytes += value.numel() * value.element_size()
            target_name = _strip(name, strip_prefix)
            tensor_device = _placement_device(target_name, target_device, layer_device)
            expert_match = _EXPERT_TENSOR.match(target_name)
            if expert_match:
                parent_name, expert_text, projection, scale_suffix = expert_match.groups()
                expert_module = modules.get(parent_name)
                if expert_module is None:
                    raise KeyError(f"Expert module not found: {parent_name}")
                if isinstance(expert_module, _ConvRotExpertsBase):
                    destination_dtype = torch.float32 if scale_suffix else torch.int8
                    moved = value.to(tensor_device, dtype=destination_dtype)
                    _assign_quant_expert(
                        expert_module, int(expert_text), projection, bool(scale_suffix), moved
                    )
                else:
                    if scale_suffix:
                        raise ValueError(f"BF16 checkpoint unexpectedly contains {name}")
                    moved = value.to(tensor_device, dtype=dtype)
                    _assign_bf16_expert(
                        expert_module,
                        int(expert_text),
                        projection,
                        moved,
                        tensor_device,
                        dtype,
                    )
                if progress_cb:
                    progress_cb(loaded_bytes, total_bytes, name)
                continue

            parent, leaf = _parent_and_leaf(model, target_name)
            destination = parent._parameters.get(leaf)
            if destination is None:
                destination = parent._buffers.get(leaf)
            if destination is None:
                raise KeyError(f"Checkpoint tensor has no target: {target_name}")
            destination_dtype = destination.dtype
            if destination_dtype.is_floating_point:
                destination_dtype = torch.float32 if target_name.endswith("weight_scale") else dtype
            moved = value.to(tensor_device, dtype=destination_dtype)
            _assign_tensor(model, target_name, moved)
            if target_name.endswith(".weight") and moved.is_floating_point() and moved.ndim == 2:
                bf16_linear_names.add(target_name[: -len(".weight")])
            if progress_cb:
                progress_cb(loaded_bytes, total_bytes, name)

    if meta is not None:
        _fuse_quantized_qkv(model)
        _regular_hadamard(meta.group_size, target_device, dtype)
    _materialize_meta_buffers(model, target_device, layer_device)
    _move_non_meta_buffers(model, target_device, layer_device)
    if tower_offload is True or (meta is not None and tower_offload is None):
        if tower_offload is True:
            _offload_prefill_towers_if_needed(
                model,
                checkpoint,
                target_device,
                force=True,
            )
        else:
            _offload_prefill_towers_if_needed(model, checkpoint, target_device)
    remaining_meta_parameters = [name for name, value in model.named_parameters() if value.device.type == "meta"]
    remaining_meta_buffers = [name for name, value in model.named_buffers() if value.device.type == "meta"]
    if remaining_meta_parameters or remaining_meta_buffers:
        preview = (remaining_meta_parameters + remaining_meta_buffers)[:8]
        raise RuntimeError(f"Checkpoint load left meta tensors: {preview}")
    model.eval()
    if target_device.type == "cuda":
        # QKV fusion briefly owns both the separate and concatenated buffers.
        # Return those staging blocks to CUDA before a near-capacity INT8 model
        # enters generation instead of letting WDDM page live expert weights.
        torch.cuda.empty_cache()
    return LoadReport(
        quantized_layers=len(meta.layers) if meta else 0,
        bf16_layers=len(bf16_linear_names),
        bytes_loaded=loaded_bytes,
        seconds=time.perf_counter() - started,
    )


__all__ = [
    "ConvRotInt4Experts",
    "ConvRotInt4W4A8Linear",
    "ConvRotInt8Experts",
    "ConvRotInt8Linear",
    "LoadReport",
    "QuantLayerMeta",
    "QuantMeta",
    "apply_quantized_checkpoint",
    "estimate_checkpoint_vram_gb",
    "fuse_bf16_experts_from_per_expert",
    "iter_safetensors_tensors",
    "read_quant_metadata",
]
