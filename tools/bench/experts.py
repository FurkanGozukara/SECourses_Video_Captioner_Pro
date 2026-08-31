"""Benchmark compact Qwen3-Omni expert dispatch against Python-loop baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vcap.models.quant.convrot import (  # noqa: E402
    ConvRotInt4Experts,
    ConvRotInt8Experts,
    _quantized_linear,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-Omni quantized expert dispatch for decode or prefill."
    )
    parser.add_argument("--scheme", choices=("int8", "int4"), default="int4")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/bench/experts_benchmark.json"),
        help="JSON result path",
    )
    return parser.parse_args(argv)


def run_projection(module, current: torch.Tensor, expert: int) -> torch.Tensor:
    gate_up = _quantized_linear(
        current,
        module.gate_up_weight[expert],
        module.gate_up_weight_scale[expert],
        None,
        module.scheme,
        module.group_size,
    )
    gate, up = gate_up.chunk(2, dim=-1)
    return _quantized_linear(
        F.silu(gate) * up,
        module.down_weight[expert],
        module.down_weight_scale[expert],
        None,
        module.scheme,
        module.group_size,
    )


def transformers_active_loop(module, hidden, top_index, top_weights):
    expert_mask = F.one_hot(top_index, num_classes=module.num_experts).permute(2, 1, 0)
    expert_hit = torch.where(expert_mask.sum(dim=(-1, -2)) > 0)[0]
    output = torch.zeros_like(hidden)
    for expert_tensor in expert_hit:
        nth_expert, batch_index = torch.where(expert_mask[expert_tensor])
        expert = int(expert_tensor.item())
        current = run_projection(module, hidden[batch_index], expert)
        current = current * top_weights[batch_index, nth_expert, None]
        output.index_add_(0, batch_index, current.to(output.dtype))
    return output


def naive_128_loop(module, hidden, top_index, top_weights):
    output = torch.zeros_like(hidden)
    for expert in range(module.num_experts):
        mask = top_index == expert
        if not bool(mask.any()):
            continue
        batch_index, nth_expert = torch.where(mask)
        current = run_projection(module, hidden[batch_index], expert)
        current = current * top_weights[batch_index, nth_expert, None]
        output.index_add_(0, batch_index, current.to(output.dtype))
    return output


def measure(function, iterations: int) -> float:
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def run(args: argparse.Namespace) -> int:
    forced_gpu = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
    if forced_gpu and os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != forced_gpu:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={forced_gpu} for this dev benchmark")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")
    torch.manual_seed(7)
    module_class = ConvRotInt8Experts if args.scheme == "int8" else ConvRotInt4Experts
    module = module_class(128, 2048, 768, F.silu, device="cuda:0").eval()
    with torch.no_grad():
        low = -127 if args.scheme == "int8" else -8
        high = 128 if args.scheme == "int8" else 8
        module.gate_up_weight.random_(low, high)
        module.down_weight.random_(low, high)
        module.gate_up_weight_scale.fill_(0.001)
        module.down_weight_scale.fill_(0.001)

    hidden = torch.randn(args.tokens, 2048, device="cuda:0", dtype=torch.bfloat16)
    top_index = torch.rand(args.tokens, 128, device="cuda:0").topk(8, dim=-1).indices
    top_weights = torch.softmax(torch.randn(args.tokens, 8, device="cuda:0"), dim=-1).to(torch.bfloat16)
    compact = lambda: module(hidden, top_index, top_weights)
    active = lambda: transformers_active_loop(module, hidden, top_index, top_weights)
    naive = lambda: naive_128_loop(module, hidden, top_index, top_weights)

    reference = compact()
    active_result = active()
    naive_result = naive()
    active_delta = (reference - active_result).float()
    naive_delta = (reference - naive_result).float()
    active_error = float(active_delta.abs().max())
    naive_error = float(naive_delta.abs().max())
    timings = {
        "compact_sorted_ms": measure(compact, args.iterations),
        "transformers_active_loop_ms": measure(active, args.iterations),
        "naive_128_loop_ms": measure(naive, args.iterations),
    }
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "scheme": module.scheme,
        "tokens": args.tokens,
        "experts": 128,
        "active_experts": 8,
        "iterations": args.iterations,
        "timings_ms": timings,
        "speedup_vs_transformers_active": timings["transformers_active_loop_ms"]
        / timings["compact_sorted_ms"],
        "speedup_vs_naive_128": timings["naive_128_loop_ms"]
        / timings["compact_sorted_ms"],
        "max_abs_delta_vs_active": active_error,
        "mean_abs_delta_vs_active": float(active_delta.abs().mean()),
        "relative_l2_delta_vs_active": float(
            torch.linalg.vector_norm(active_delta)
            / torch.linalg.vector_norm(active_result.float()).clamp_min(1e-12)
        ),
        "max_abs_delta_vs_naive": naive_error,
        "mean_abs_delta_vs_naive": float(naive_delta.abs().mean()),
        "relative_l2_delta_vs_naive": float(
            torch.linalg.vector_norm(naive_delta)
            / torch.linalg.vector_norm(naive_result.float()).clamp_min(1e-12)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
