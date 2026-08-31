"""Accuracy, caption, speed, and VRAM verification for ConvRot checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import atomic_write_json, ensure_utf8_stdio, utc_now  # noqa: E402
from _model_utils import (  # noqa: E402
    generation_recipe,
    identify_model,
    instantiate_meta,
    last_token_logits,
    move_inputs,
    multimodal_inputs,
    text_prompt_inputs,
)
from vcap.models.quant.convrot import apply_quantized_checkpoint  # noqa: E402


DEFAULT_VIDEO = Path("F:/SECourses_Video_Captioner_Pro_TEMP/test_media/lightning_storm_20s.mp4")
DEFAULT_AUDIO = Path("F:/SECourses_Video_Captioner_Pro_TEMP/test_media/demon_singer_audio_18_sec.mp3")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one or more single-file model variants against a BF16 reference."
    )
    parser.add_argument("--bf16-dir", type=Path, required=True, help="Merged BF16 reference folder")
    parser.add_argument(
        "--variant-dir", type=Path, action="append", required=True, help="Variant folder; repeatable"
    )
    parser.add_argument("--media", type=Path, help="Video/audio test file (model-specific default)")
    parser.add_argument("--skip-media", action="store_true", help="Only compare text-prompt logits")
    parser.add_argument(
        "--max-new-tokens", type=int, help="Testing override; official model default is used when omitted"
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/QUANT_REPORT.md"), help="Aggregate Markdown report"
    )
    return parser.parse_args(argv)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_reference_logits(bf16_dir: Path, processor, identity) -> tuple[torch.Tensor, dict]:
    clear_memory()
    started = time.perf_counter()
    kwargs = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if identity.family == "qwen3_omni_moe":
        kwargs.update(
            device_map="auto",
            max_memory={0: "26GiB", "cpu": "60GiB"},
        )
    else:
        kwargs.update(device_map={"": "cuda:0"})
    model = identity.model_class.from_pretrained(bf16_dir, **kwargs).eval()
    load_seconds = time.perf_counter() - started
    _, inputs = text_prompt_inputs(processor, identity)
    inputs = move_inputs(inputs, torch.device("cuda:0"))
    logits = last_token_logits(model, inputs)
    metrics = {
        "load_seconds": load_seconds,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "device_map_auto": identity.family == "qwen3_omni_moe",
    }
    del model, inputs
    clear_memory()
    return logits, metrics


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (candidate - reference).abs()
    reference_log_probs = torch.log_softmax(reference.float(), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    kl = (reference_probs * (reference_log_probs - candidate_log_probs)).sum(dim=-1).mean()
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "kl_reference_to_variant": float(kl),
    }


def verify_variant(
    variant_dir: Path,
    bf16_dir: Path,
    processor,
    identity,
    reference_logits: torch.Tensor,
    media_path: Path,
    skip_media: bool,
    max_new_tokens: int | None,
) -> dict:
    clear_memory()
    torch.cuda.reset_peak_memory_stats()
    model = instantiate_meta(identity, variant_dir)
    last_progress = 0.0

    def progress(loaded: int, total: int, name: str) -> None:
        nonlocal last_progress
        now = time.perf_counter()
        if loaded < total and now - last_progress < 0.15:
            return
        last_progress = now
        elapsed = max(time.perf_counter() - load_started, 1e-6)
        rate = loaded / elapsed
        eta = (total - loaded) / rate if rate else 0.0
        short = name if len(name) < 64 else "..." + name[-61:]
        print(
            f"\rload {variant_dir.name}: {loaded / 1e9:.2f}/{total / 1e9:.2f} GB "
            f"ETA {eta:.0f}s | {short}".ljust(145),
            end="",
            flush=True,
        )

    load_started = time.perf_counter()
    load_report = apply_quantized_checkpoint(
        model,
        variant_dir / "model.safetensors",
        device="cuda:0",
        dtype=torch.bfloat16,
        progress_cb=progress,
    )
    print()
    peak_after_load = torch.cuda.max_memory_allocated() / 1024**3
    _, text_inputs = text_prompt_inputs(processor, identity)
    text_inputs = move_inputs(text_inputs, torch.device("cuda:0"))
    candidate_logits = last_token_logits(model, text_inputs)
    logit_metrics = compare_logits(reference_logits, candidate_logits)

    result = {
        "variant": variant_dir.name,
        "family": identity.family,
        "bf16_reference": str(bf16_dir),
        "created_at": utc_now(),
        "size_gb": sum(item.stat().st_size for item in variant_dir.iterdir() if item.is_file()) / 1e9,
        "load": asdict(load_report),
        "peak_load_vram_gb": peak_after_load,
        "logits": logit_metrics,
        "media": None,
    }
    del text_inputs, candidate_logits

    qwen3_bf16 = identity.family == "qwen3_omni_moe" and variant_dir.resolve() == bf16_dir.resolve()
    if skip_media:
        result["media"] = {"skipped": "--skip-media"}
    elif qwen3_bf16:
        result["media"] = {
            "skipped": "Qwen3-Omni BF16 exceeds the 32 GB GPU; reference logits use CPU offload only."
        }
    else:
        media_inputs, recipe = multimodal_inputs(processor, identity, media_path)
        if max_new_tokens is not None:
            recipe = replace(recipe, max_new_tokens=max_new_tokens)
        media_inputs = move_inputs(media_inputs, torch.device("cuda:0"))
        from _model_utils import generate_with_metrics

        result["media"] = generate_with_metrics(model, processor, media_inputs, recipe)
        result["media"]["path"] = str(media_path)
        result["media"]["recipe"] = asdict(recipe)
        del media_inputs
    result["peak_total_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    del model
    clear_memory()
    atomic_write_json(variant_dir / "verification_report.json", result)
    return result


def rebuild_markdown(report_path: Path, models_root: Path, reference_metrics: dict | None = None) -> None:
    reports = []
    if models_root.exists():
        for path in sorted(models_root.glob("*/verification_report.json")):
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    lines = [
        "# Quantization Verification Report",
        "",
        f"Updated: {utc_now()}",
        "",
        "GPU measurements use process-local CUDA device 0; `CUDA_VISIBLE_DEVICES` selects the physical GPU.",
        "",
        "| Variant | Size GB | Load s | Peak VRAM GB | Max abs logit diff | Mean abs diff | KL | Prefill tok/s | Decode tok/s | Caption chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        media = item.get("media") or {}
        lines.append(
            "| {variant} | {size:.3f} | {load:.2f} | {peak:.2f} | {mx:.6g} | {mean:.6g} | "
            "{kl:.6g} | {prefill} | {decode} | {chars} |".format(
                variant=item.get("variant", "?"),
                size=item.get("size_gb", 0.0),
                load=item.get("load", {}).get("seconds", 0.0),
                peak=item.get("peak_total_vram_gb", item.get("peak_load_vram_gb", 0.0)),
                mx=item.get("logits", {}).get("max_abs_delta", float("nan")),
                mean=item.get("logits", {}).get("mean_abs_delta", float("nan")),
                kl=item.get("logits", {}).get("kl_reference_to_variant", float("nan")),
                prefill=f"{media['prefill_tok_s']:.2f}" if "prefill_tok_s" in media else "skipped",
                decode=f"{media['decode_tok_s']:.2f}" if "decode_tok_s" in media else "skipped",
                chars=len(media.get("caption", "")) if "caption" in media else "skipped",
            )
        )
    for item in reports:
        media = item.get("media") or {}
        lines.extend(["", f"## {item.get('variant', '?')}", ""])
        if "caption" in media:
            lines.extend([media["caption"], ""])
        elif "skipped" in media:
            lines.extend([f"Media run skipped: {media['skipped']}", ""])
        lines.append(
            f"Logits: exact={item.get('logits', {}).get('exact', False)}, "
            f"max |delta|={item.get('logits', {}).get('max_abs_delta', 'n/a')}, "
            f"mean |delta|={item.get('logits', {}).get('mean_abs_delta', 'n/a')}."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = report_path.with_name(report_path.name + ".partial")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(partial, report_path)


def run(args: argparse.Namespace) -> int:
    forced_gpu = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
    if forced_gpu and os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != forced_gpu:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={forced_gpu} for this dev verification")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bf16_dir = args.bf16_dir.resolve()
    identity = identify_model(bf16_dir)
    processor = identity.processor_class.from_pretrained(bf16_dir)
    reference_logits, reference_metrics = load_reference_logits(bf16_dir, processor, identity)
    print(f"Reference logits ready; BF16 load {reference_metrics['load_seconds']:.2f}s")

    recipe = generation_recipe(identity)
    media = args.media
    if media is None:
        media = DEFAULT_AUDIO if recipe.modality == "audio" else DEFAULT_VIDEO
    media = media.resolve()
    if not args.skip_media and not media.exists():
        raise FileNotFoundError(media)

    failures = 0
    for variant_dir in args.variant_dir:
        try:
            result = verify_variant(
                variant_dir.resolve(),
                bf16_dir,
                processor,
                identity,
                reference_logits,
                media,
                args.skip_media,
                args.max_new_tokens,
            )
            print(json.dumps({key: value for key, value in result.items() if key != "media"}, indent=2))
            if result.get("media", {}).get("caption"):
                print(f"caption: {result['media']['caption']}")
        except Exception as exc:
            failures += 1
            print(f"ERROR verifying {variant_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
            clear_memory()
    models_root = bf16_dir.parent
    rebuild_markdown(args.report.resolve(), models_root, reference_metrics)
    return 1 if failures else 0


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
