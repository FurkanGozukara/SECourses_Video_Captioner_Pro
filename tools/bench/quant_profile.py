"""Profile Qwen3-Omni ConvRot towers and text-generation components on GPU 0."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


_DEV_GPU = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
if _DEV_GPU:
    existing_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing_gpu and existing_gpu != _DEV_GPU:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={_DEV_GPU} for this dev profile")
    os.environ["CUDA_VISIBLE_DEVICES"] = _DEV_GPU

ROOT = Path(__file__).resolve().parents[2]
QUANT_TOOLS = ROOT / "tools" / "quantize"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_TOOLS))

import torch  # noqa: E402

from _model_utils import (  # noqa: E402
    generate_with_metrics,
    generation_recipe,
    identify_model,
    move_inputs,
    multimodal_inputs,
)
from vcap.models.loader import load_model, unload_model  # noqa: E402
from vcap.models.offload import OffloadPlan  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operator-profile", action="store_true")
    return parser.parse_args(argv)


def _phase(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    value = args[0] if args and isinstance(args[0], torch.Tensor) else None
    if value is None:
        for key in ("hidden_states", "input_ids", "inputs_embeds"):
            candidate = kwargs.get(key)
            if isinstance(candidate, torch.Tensor):
                value = candidate
                break
    if value is None:
        return "prefill"
    if value.ndim >= 3:
        sequence = int(value.shape[-2])
    elif value.ndim == 2 and not value.is_floating_point():
        sequence = int(value.shape[-1])
    else:
        sequence = int(value.shape[0])
    return "prefill" if sequence > 1 else "decode"


class _CudaModuleTimers:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event, float]] = []
        self._pending: dict[int, list[tuple[str, str, torch.cuda.Event, float]]] = defaultdict(list)
        self.handles: list[Any] = []

    def add(self, module: torch.nn.Module, category: str, fixed_phase: str | None = None) -> None:
        def before(current, args, kwargs):
            phase = fixed_phase or _phase(args, kwargs)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._pending[id(current)].append((category, phase, event, time.perf_counter()))

        def after(current, args, kwargs, output):
            del args, kwargs, output
            category_value, phase, start, cpu_start = self._pending[id(current)].pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records.append((category_value, phase, start, end, time.perf_counter() - cpu_start))

        self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        self.handles.append(module.register_forward_hook(after, with_kwargs=True))

    def close(self) -> dict[str, dict[str, dict[str, float | int]]]:
        torch.cuda.synchronize()
        totals: dict[str, dict[str, dict[str, float | int]]] = defaultdict(
            lambda: defaultdict(lambda: {"calls": 0, "cuda_ms": 0.0, "cpu_ms": 0.0})
        )
        for category, phase, start, end, cpu_seconds in self.records:
            item = totals[phase][category]
            item["calls"] = int(item["calls"]) + 1
            item["cuda_ms"] = float(item["cuda_ms"]) + start.elapsed_time(end)
            item["cpu_ms"] = float(item["cpu_ms"]) + cpu_seconds * 1000.0
        for handle in self.handles:
            handle.remove()
        return {phase: dict(categories) for phase, categories in totals.items()}


def _profile_expert_call(module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> str:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as result:
        module(*values)
        torch.cuda.synchronize()
    return result.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)


def run(args: argparse.Namespace) -> int:
    if args.max_new_tokens < 2:
        raise ValueError("--max-new-tokens must be at least 2 to measure decode")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    loaded = load_model(
        args.variant,
        device="cuda:0",
        attention=args.attention,
        offload=OffloadPlan(),
    )
    model = loaded.model
    timers = _CudaModuleTimers()
    captured: dict[str, tuple[torch.nn.Module, tuple[torch.Tensor, ...]]] = {}
    count_tensors: dict[str, torch.Tensor] = {}
    try:
        timers.add(model, "model_forward")
        timers.add(model.audio_tower, "audio_tower", "prefill")
        timers.add(model.visual, "vision_tower", "prefill")
        timers.add(model.lm_head, "lm_head")
        for name, module in model.named_modules():
            if name.startswith("model.layers.") and name.endswith(".self_attn"):
                timers.add(module, "attention")
            elif name.startswith("model.layers.") and name.endswith(".mlp.gate"):
                timers.add(module, "router")
            elif name.startswith("model.layers.") and name.endswith(".mlp.experts"):
                timers.add(module, "experts")

                def capture(current, hook_args, hook_kwargs):
                    del hook_kwargs
                    phase = _phase(hook_args, {})
                    if phase not in captured:
                        values = tuple(value.detach() for value in hook_args[:3])
                        captured[phase] = (current, values)
                        count_tensors[phase] = torch.bincount(
                            values[1].reshape(-1), minlength=int(current.num_experts)
                        )

                timers.handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))

        identity = identify_model(loaded.model_dir)
        inputs, recipe = multimodal_inputs(loaded.processor, identity, args.input)
        recipe = replace(generation_recipe(identity), max_new_tokens=args.max_new_tokens)
        inputs = move_inputs(inputs, torch.device("cuda:0"))
        metrics = generate_with_metrics(model, loaded.processor, inputs, recipe)
        timings = timers.close()

        expert_counts = {}
        for phase, counts in count_tensors.items():
            counts_cpu = counts.cpu()
            active = counts_cpu[counts_cpu > 0]
            expert_counts[phase] = {
                "assignments": int(active.sum()),
                "active_experts": int(active.numel()),
                "min": int(active.min()) if active.numel() else 0,
                "median": float(active.float().median()) if active.numel() else 0.0,
                "max": int(active.max()) if active.numel() else 0,
                "experts_over_16": int((active > 16).sum()),
            }

        operator_tables = {}
        if args.operator_profile:
            for phase, (module, values) in captured.items():
                operator_tables[phase] = _profile_expert_call(module, values)

        report = {
            "variant": args.variant,
            "input": str(args.input.resolve()),
            "attention": loaded.load_report.attention,
            "load": asdict(loaded.load_report),
            "generation": metrics,
            "module_timings": timings,
            "expert_dispatch": expert_counts,
            "expert_operator_profiles": operator_tables,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "expert_operator_profiles"}, indent=2, default=str))
        for phase, table in operator_tables.items():
            print(f"\n=== EXPERT OPERATORS: {phase.upper()} ===\n{table}")
        return 0
    finally:
        if timers.handles:
            for handle in timers.handles:
                try:
                    handle.remove()
                except Exception:
                    pass
        unload_model(loaded)


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
