"""Reusable single-file model load, VRAM, prefill, and decode benchmark."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
QUANT_TOOLS = ROOT / "tools" / "quantize"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(QUANT_TOOLS))

from _common import ensure_utf8_stdio  # noqa: E402
from _model_utils import (  # noqa: E402
    generate_with_metrics,
    generation_recipe,
    identify_model,
    instantiate_meta,
    move_inputs,
    multimodal_inputs,
)
from vcap.models.quant.convrot import apply_quantized_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a Video Captioner Pro model variant.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Self-contained model folder")
    parser.add_argument("--video", type=Path, required=True, help="Video or audio input")
    parser.add_argument("--runs", type=int, default=3, help="Measured generation runs (default: 3)")
    parser.add_argument("--max-new-tokens", type=int, help="Optional short-run override")
    parser.add_argument("--output", type=Path, help="Write the Markdown row/table to this file")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    forced_gpu = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
    if forced_gpu and os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != forced_gpu:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={forced_gpu} for this dev benchmark")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    model_dir = args.model_dir.resolve()
    media_path = args.video.resolve()
    identity = identify_model(model_dir)
    processor = identity.processor_class.from_pretrained(model_dir)
    model = instantiate_meta(identity, model_dir)

    torch.cuda.reset_peak_memory_stats()
    last_progress = 0.0

    def progress(loaded: int, total: int, name: str) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        if loaded < total and now - last_progress < 0.15:
            return
        last_progress = now
        short = name if len(name) <= 64 else "..." + name[-61:]
        print(
            f"\rload: {loaded / 1e9:.2f}/{total / 1e9:.2f} GB | {short}".ljust(130),
            end="",
            flush=True,
        )

    load = apply_quantized_checkpoint(
        model,
        model_dir / "model.safetensors",
        device="cuda:0",
        dtype=torch.bfloat16,
        progress_cb=progress,
    )
    print()
    inputs, recipe = multimodal_inputs(processor, identity, media_path)
    if args.max_new_tokens is not None:
        recipe = replace(recipe, max_new_tokens=args.max_new_tokens)
    inputs = move_inputs(inputs, torch.device("cuda:0"))

    measurements = []
    for index in range(args.runs):
        result = generate_with_metrics(model, processor, inputs, recipe)
        measurements.append(result)
        print(
            f"run {index + 1}/{args.runs}: prefill {result['prefill_tok_s']:.2f} tok/s, "
            f"decode {result['decode_tok_s']:.2f} tok/s, peak {result['peak_vram_gb']:.2f} GiB"
        )

    size_gb = sum(path.stat().st_size for path in model_dir.iterdir() if path.is_file()) / 1e9
    row = {
        "variant": model_dir.name,
        "size_gb": size_gb,
        "load_s": load.seconds,
        "peak_vram_gb": max(item["peak_vram_gb"] for item in measurements),
        "prefill_tok_s": statistics.mean(item["prefill_tok_s"] for item in measurements),
        "decode_tok_s": statistics.mean(item["decode_tok_s"] for item in measurements),
        "caption_chars": len(measurements[-1]["caption"]),
    }
    markdown = "\n".join(
        [
            "| Variant | Size GB | Load s | Peak VRAM GB | Prefill tok/s | Decode tok/s | Caption chars |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| {row['variant']} | {row['size_gb']:.3f} | {row['load_s']:.2f} | "
                f"{row['peak_vram_gb']:.2f} | {row['prefill_tok_s']:.2f} | "
                f"{row['decode_tok_s']:.2f} | {row['caption_chars']} |"
            ),
        ]
    )
    print(markdown)
    print(json.dumps(row, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
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
