"""Benchmark one local model variant through the application's caption path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _gpu_layers(value: str) -> int | str:
    normalized = str(value).strip().casefold()
    if normalized in {"auto", "all"}:
        return normalized
    try:
        count = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected 'auto', 'all', or a non-negative integer") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("GPU layer count must be non-negative")
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="Registered local model variant key")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional checkpoint folder; its name is used as --variant when omitted",
    )
    parser.add_argument(
        "--input",
        "--video",
        dest="input",
        type=Path,
        required=True,
        help="Video, audio, or image input",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured generations (default: 3)")
    parser.add_argument("--max-new-tokens", type=int, help="Override the family generation default")
    parser.add_argument("--prompt-preset", help="Optional application prompt-preset id")
    parser.add_argument("--profile-tier", type=int, default=32, help="Production VRAM tier (default: 32)")
    parser.add_argument("--attention", help="Override the production profile attention backend")
    parser.add_argument(
        "--gpu-layers",
        type=_gpu_layers,
        help="Resident decoder layers: auto, all, or a non-negative count (default: profile plan)",
    )
    parser.add_argument(
        "--vram-reserve-gb",
        type=float,
        help="Dedicated VRAM to keep free (default: profile plan)",
    )
    parser.add_argument(
        "--swap-slots",
        type=int,
        help="GPU block-swap staging slots (default: profile plan)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA GPU index (default: 0)")
    parser.add_argument("--seed", type=int, default=1234, help="Base sampling seed (default: 1234)")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Force deterministic greedy decoding regardless of the family default",
    )
    parser.add_argument(
        "--unfused-gate-up-control",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Raw JSON output path (default: tools/bench/results/<variant>_<timestamp>.json)",
    )
    return parser.parse_args(argv)


def _resolve_variant(args: argparse.Namespace) -> tuple[str, Path | None]:
    model_dir = args.model_dir.expanduser().resolve(strict=False) if args.model_dir else None
    variant = str(args.variant or (model_dir.name if model_dir else "")).strip()
    if not variant:
        raise ValueError("Provide --variant or --model-dir")
    if model_dir is not None and not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    return variant, model_dir


def _progress(*args: object) -> None:
    if not args:
        return
    first = args[0]
    message = first.get("message") if isinstance(first, dict) else first
    if message and not str(message).startswith(("Generating:", "Loading tensor")):
        print(str(message), flush=True)


def _caption_preview(text: str, limit: int = 200) -> str:
    return " ".join(str(text).split())[:limit]


def _install_unfused_gate_up_control() -> None:
    """Keep legacy dense MLP projections separate for an A/B benchmark run."""

    from torch import nn
    from vcap.models.quant import convrot

    original = convrot._fuse_quantized_qkv

    def qkv_only(model: nn.Module) -> int:
        hidden: list[tuple[nn.Module, nn.Module, nn.Module]] = []
        for module in list(model.modules()):
            gate = module._modules.get("gate_proj")
            up = module._modules.get("up_proj")
            if isinstance(gate, convrot._ConvRotLinearBase) and isinstance(
                up, convrot._ConvRotLinearBase
            ):
                hidden.append((module, gate, up))
                module._modules["gate_proj"] = nn.Identity()
                module._modules["up_proj"] = nn.Identity()
        try:
            return original(model)
        finally:
            for module, gate, up in hidden:
                module._modules["gate_proj"] = gate
                module._modules["up_proj"] = up

    convrot._fuse_quantized_qkv = qkv_only


def summarize(
    variant: str,
    checkpoint_bytes: int,
    load_s: float,
    load_peak_vram_gib: float,
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the stable aggregate consumed by the Markdown report."""

    if not measurements:
        raise ValueError("At least one measurement is required")
    reasons = [str(item["finish_reason"]) for item in measurements]
    return {
        "variant": variant,
        "checkpoint_gb": checkpoint_bytes / 1_000_000_000,
        "load_s": float(load_s),
        "peak_vram_gib": max(
            float(load_peak_vram_gib),
            *(float(item["peak_vram_gib"]) for item in measurements),
        ),
        "prefill_tok_s_mean": statistics.mean(float(item["prefill_tok_s"]) for item in measurements),
        "decode_tok_s_mean": statistics.mean(float(item["decode_tok_s"]) for item in measurements),
        "generated_tokens_mean": statistics.mean(int(item["generated_tokens"]) for item in measurements),
        "wall_clock_s_mean": statistics.mean(float(item["wall_clock_s"]) for item in measurements),
        "finish_reason": reasons[0] if len(set(reasons)) == 1 else "mixed: " + ", ".join(reasons),
        "caption_preview": str(measurements[-1]["caption_preview"]),
        "runs": len(measurements),
    }


def markdown_table(summary: dict[str, Any]) -> str:
    """Render one complete benchmark row without dropping EOS diagnostics."""

    preview = str(summary["caption_preview"]).replace("|", "\\|")
    return "\n".join(
        [
            (
                "| Variant | Checkpoint GB | Load s | Peak VRAM GiB | Prefill tok/s | "
                "Decode tok/s | Generated tokens | Wall s/caption | Finish | Caption preview |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            (
                f"| {summary['variant']} | {summary['checkpoint_gb']:.3f} | "
                f"{summary['load_s']:.2f} | {summary['peak_vram_gib']:.2f} | "
                f"{summary['prefill_tok_s_mean']:.2f} | {summary['decode_tok_s_mean']:.2f} | "
                f"{summary['generated_tokens_mean']:.1f} | {summary['wall_clock_s_mean']:.2f} | "
                f"{summary['finish_reason']} | {preview} |"
            ),
        ]
    )


def _default_output(variant: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "tools" / "bench" / "results" / f"{variant}_{stamp}.json"


def run(args: argparse.Namespace) -> int:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    source = args.input.expanduser().resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(source)
    variant_key, model_dir = _resolve_variant(args)

    selected_gpu = str(int(args.gpu))
    forced_gpu = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
    if forced_gpu and forced_gpu != selected_gpu:
        raise RuntimeError(f"VCAP_DEV_FORCE_GPU={forced_gpu} rejects physical GPU {selected_gpu}")
    existing_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing_visible and existing_visible != selected_gpu:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={existing_visible} does not match requested physical GPU {selected_gpu}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu

    import torch
    import transformers

    from vcap.models import captioner_for_loaded
    from vcap.models.base import Callbacks, GenParams, MediaInput, PreprocessParams, PromptSpec
    from vcap.models.loader import load_model, unload_model
    from vcap.models.offload import BudgetHint, OffloadPlan
    from vcap.models.registry import MODEL_SPECS, get_variant, variant_to_family
    from vcap.models.vram_presets import preset_for

    if args.unfused_gate_up_control:
        _install_unfused_gate_up_control()

    registered_variant = get_variant(variant_key)
    family = variant_to_family(registered_variant.key)
    spec = MODEL_SPECS[family]
    profile = preset_for(family, args.profile_tier)
    profile_offload = profile.offload
    offload = OffloadPlan(
        gpu_layers=profile_offload.gpu_layers if args.gpu_layers is None else args.gpu_layers,
        offload_experts=profile_offload.offload_experts,
        max_memory=profile_offload.max_memory,
        pin_cpu=profile_offload.pin_cpu,
        vram_reserve_gb=(
            profile_offload.vram_reserve_gb
            if args.vram_reserve_gb is None
            else args.vram_reserve_gb
        ),
        swap_slots=profile_offload.swap_slots if args.swap_slots is None else args.swap_slots,
    )
    defaults = {item.name: item.default for item in spec.param_schema}
    max_new_tokens = int(args.max_new_tokens or defaults.get("max_new_tokens", 2048))
    generation = GenParams(
        temperature=float(defaults.get("temperature", 0.0)),
        top_p=float(defaults.get("top_p", 1.0)),
        top_k=int(defaults.get("top_k", 0)),
        repetition_penalty=float(defaults.get("repetition_penalty", 1.0)),
        max_new_tokens=max_new_tokens,
        do_sample=False if args.greedy else bool(defaults.get("do_sample", False)),
        enable_thinking=bool(defaults.get("enable_thinking", True)),
    )
    preprocessing = PreprocessParams(
        fps=float(profile.fps),
        max_frames=int(profile.max_frames),
        max_pixels=int(profile.max_pixels),
        min_pixels=spec.limits.min_pixels,
        use_audio_in_video="video_audio" in spec.capabilities,
    )
    attention = str(args.attention or profile.attention)
    output_path = (args.output or _default_output(registered_variant.key)).expanduser().resolve(strict=False)

    loaded = None
    measurements: list[dict[str, Any]] = []
    try:
        load_kwargs: dict[str, Any] = {
            "device": "cuda:0",
            "gpu_index": int(args.gpu),
            "attention": attention,
            "offload": offload,
            "progress_cb": _progress,
            "hf_dir": model_dir,
        }
        if "budget_hint" in inspect.signature(load_model).parameters:
            load_kwargs["budget_hint"] = BudgetHint(
                max_frames=preprocessing.max_frames,
                max_pixels=preprocessing.max_pixels,
                fps=preprocessing.fps,
                max_new_tokens=generation.max_new_tokens,
                context_tokens=spec.limits.context_tokens,
            )
        loaded = load_model(registered_variant.key, **load_kwargs)
        captioner = captioner_for_loaded(loaded)
        prompt = PromptSpec(preset_id=args.prompt_preset) if args.prompt_preset else None
        for index in range(args.runs):
            run_seed = int(args.seed) + index
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = captioner.caption(
                MediaInput(path=source),
                prompt,
                generation,
                preprocessing,
                Callbacks(progress=_progress),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_s = time.perf_counter() - started
            prefill_s = float(result.timing.prefill_s)
            prompt_tokens = int(result.usage.prompt_tokens)
            measurement = {
                "run": index + 1,
                "seed": run_seed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(result.usage.new_tokens),
                "finish_reason": str(result.usage.finish_reason),
                "prefill_s": prefill_s,
                "decode_s": float(result.timing.decode_s),
                "prefill_tok_s": prompt_tokens / max(prefill_s, 1e-9),
                "decode_tok_s": float(result.timing.tokens_per_s),
                "generation_s": float(result.timing.total_s),
                "wall_clock_s": wall_s,
                "peak_vram_gib": float(result.peak_vram_gb),
                "caption_chars": len(result.text),
                "caption_preview": _caption_preview(result.text),
                "caption": result.text,
                "reasoning_chars": len(result.reasoning),
                "warnings": list(result.warnings),
            }
            measurements.append(measurement)
            print(
                f"run {index + 1}/{args.runs}: finish={measurement['finish_reason']}, "
                f"{measurement['generated_tokens']} tokens, prefill "
                f"{measurement['prefill_tok_s']:.2f} tok/s, decode "
                f"{measurement['decode_tok_s']:.2f} tok/s, wall {wall_s:.2f}s, "
                f"peak {measurement['peak_vram_gib']:.2f} GiB",
                flush=True,
            )

        summary = summarize(
            registered_variant.key,
            int(loaded.load_report.checkpoint_bytes),
            float(loaded.load_report.seconds),
            float(loaded.load_report.peak_vram_gb),
            measurements,
        )
        report = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "summary": summary,
            "configuration": {
                "variant": registered_variant.key,
                "family": family,
                "input": str(source),
                "runs": int(args.runs),
                "unfused_gate_up_control": bool(args.unfused_gate_up_control),
                "profile_tier_gb": int(args.profile_tier),
                "attention_requested": attention,
                "attention_resolved": loaded.load_report.attention,
                "offload": asdict(offload),
                "generation": asdict(generation),
                "preprocessing": asdict(preprocessing),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "load": asdict(loaded.load_report),
            "measurements": measurements,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print("\n" + markdown_table(summary))
        print(f"raw JSON: {output_path}")
        return 0
    finally:
        if loaded is not None:
            unload_data = asdict(unload_model(loaded))
            print("unload: " + json.dumps(unload_data, default=str))


def main(argv: list[str] | None = None) -> int:
    try:
        from vcap.core.logs import setup_utf8_stdio

        setup_utf8_stdio()
    except ImportError:
        pass
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
