"""Streaming INT8 and INT4-W4A8 ConvRot converter."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Pattern

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ProgressLine,
    RawFileCache,
    SafeOpenCache,
    TensorSpec,
    atomic_write_json,
    copy_model_support_files,
    copy_range,
    discover_tensor_records,
    ensure_utf8_stdio,
    infer_family,
    infer_source_repo,
    make_safetensors_header,
    read_safetensors_header,
    total_spec_bytes,
    utc_now,
    write_cpu_tensor,
)


SCHEME_NAMES = {
    "int8": "int8_convrot",
    "int4": "int4_convrot_w4a8",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a single-file BF16 HF checkpoint with regular-Hadamard ConvRot."
    )
    parser.add_argument("--in", dest="source", type=Path, required=True, help="BF16 model folder or file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Self-contained output model folder")
    parser.add_argument("--scheme", choices=sorted(SCHEME_NAMES), required=True)
    parser.add_argument("--group-size", type=int, default=256, help="ConvRot group size (default: 256)")
    parser.add_argument(
        "--exclude-regex", action="append", default=[], help="Additional exclusion regex; repeatable"
    )
    parser.add_argument(
        "--include-regex", action="append", default=[], help="Only quantize names matching a regex; repeatable"
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output checkpoint")
    return parser.parse_args(argv)


def regular_hadamard(size: int, device, dtype):
    import torch

    if size < 4:
        raise ValueError("ConvRot group size must be a power of four and at least 4")
    value = size
    while value > 1 and value % 4 == 0:
        value //= 4
    if value != 1:
        raise ValueError(f"ConvRot group size must be a power of four, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        device=device,
        dtype=torch.float32,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return (h / math.sqrt(size)).to(dtype=dtype)


def rotate_weight(weight, hadamard, group_size: int):
    out_features, in_features = weight.shape
    return (
        weight.reshape(out_features, in_features // group_size, group_size)
        .matmul(hadamard.T.to(weight))
        .reshape(out_features, in_features)
    )


def search_scales(rotated, qmax: int):
    """Three-stage per-row MSE clipping search ported from quantize_hq.py."""
    import torch

    rows = rotated.reshape(-1, rotated.shape[-1])
    absmax = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    best_alpha = torch.ones_like(absmax)
    best_mse = torch.full_like(absmax, float("inf"))
    max_candidate_elements = 96_000_000

    def search(deltas, centered: bool) -> None:
        nonlocal best_alpha, best_mse
        candidate_batch = max(1, min(len(deltas), max_candidate_elements // max(rows.numel(), 1)))
        for start in range(0, len(deltas), candidate_batch):
            delta = deltas[start : start + candidate_batch, None, None]
            center = best_alpha[None] if centered else torch.zeros_like(best_alpha)[None]
            alpha = (center + delta).clamp(0.5, 1.0)
            scales = (absmax[None] * alpha / float(qmax)).clamp_min(1e-30)
            reconstructed = (rows[None] / scales).round().clamp(-qmax, qmax) * scales
            mse = (reconstructed - rows[None]).square().mean(dim=2)
            chunk_mse, chunk_index = mse.min(dim=0)
            chunk_alpha = alpha[:, :, 0].gather(0, chunk_index[None]).squeeze(0)[:, None]
            better = chunk_mse[:, None] < best_mse
            best_mse = torch.where(better, chunk_mse[:, None], best_mse)
            best_alpha = torch.where(better, chunk_alpha, best_alpha)
            del reconstructed, mse, scales, alpha

    search(torch.linspace(0.60, 1.00, 41, device=rows.device), centered=False)
    search(torch.linspace(-0.012, 0.012, 25, device=rows.device), centered=True)
    search(torch.linspace(-0.0010, 0.0010, 21, device=rows.device), centered=True)
    scale = (absmax * best_alpha / float(qmax)).clamp_min(1e-30)
    quantized = (rows / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return quantized.reshape(rotated.shape), scale, best_mse


def quantize_grouped_for_study(rotated, qmax: int = 7, scale_group: int = 128):
    if rotated.shape[1] % scale_group:
        return None
    grouped = rotated.reshape(rotated.shape[0], -1, scale_group)
    quantized, scales, mse = search_scales(grouped.reshape(-1, scale_group), qmax)
    reconstructed = quantized.float() * scales
    squared_error = (reconstructed - grouped.reshape(-1, scale_group)).square().sum().item()
    return {
        "scale_group": scale_group,
        "squared_error": squared_error,
        "scale_count": scales.numel(),
    }


def pack_int4_signed(values):
    import torch

    if values.shape[-1] % 2:
        raise ValueError("INT4 packing requires an even input dimension")
    low = values[..., 0::2].to(torch.int32) & 0x0F
    high = values[..., 1::2].to(torch.int32) & 0x0F
    return (low | (high << 4)).to(torch.int8).contiguous()


def compile_regexes(values: list[str], label: str) -> list[Pattern[str]]:
    try:
        return [re.compile(value) for value in values]
    except re.error as exc:
        raise ValueError(f"Invalid {label} regex: {exc}") from exc


def default_exclusion(layer: str, shape: tuple[int, ...], group_size: int) -> str | None:
    lower = layer.lower()
    if len(shape) != 2:
        return "not_2d"
    if shape[1] % group_size:
        return "input_not_divisible"
    if any(token in lower for token in ("embed_tokens", "embedding", "embeddings", "lm_head")) or lower.endswith(
        ("_token", ".token")
    ):
        return "embedding_or_lm_head"
    if any(token in lower for token in ("norm", "rotary", "rope", "inv_freq", "position")):
        return "norm_or_position"
    if re.search(r"(?:^|\.)mlp\.gate$", lower) or any(
        token in lower for token in ("router", "routing", "gate_logits")
    ):
        return "moe_router"
    if any(token in lower for token in ("patch_embed", "pos_embed", "position_embedding")):
        return "patch_or_position_embed"
    if any(token in lower for token in ("merger", "projector", "mm_projector", "multi_modal_projector")):
        return "multimodal_merger_or_projector"
    if re.search(r"(?:^|\.)visual\.blocks\.0(?:\.|$)", lower):
        return "first_vision_block"
    if re.search(r"(?:^|\.)audio_tower\.(?:encoder\.)?layers\.0(?:\.|$)", lower):
        return "first_audio_layer"
    if "audio_tower" in lower and re.search(r"(?:^|\.)(?:conv\w*|conv2d\w*)(?:\.|$)", lower):
        return "audio_conv_frontend"
    if "audio_tower" in lower:
        tail = lower.split("audio_tower.", 1)[1]
        if tail in {"proj", "proj1", "proj2", "project", "out_proj"} or tail.startswith(
            ("proj.", "proj1.", "proj2.", "project.", "out_proj.")
        ):
            return "audio_projector"
    return None


def classify_layers(records, group_size: int, excludes, includes):
    quantized: dict[str, tuple[int, int]] = {}
    reasons: collections.Counter[str] = collections.Counter()
    layer_reasons: dict[str, str] = {}
    for tensor_name in sorted(records):
        if not tensor_name.endswith(".weight"):
            continue
        layer = tensor_name[: -len(".weight")]
        shape = records[tensor_name].shape
        reason = default_exclusion(layer, shape, group_size)
        if reason is None and includes and not any(regex.search(layer) for regex in includes):
            reason = "not_included"
        if reason is None and any(regex.search(layer) for regex in excludes):
            reason = "user_excluded"
        if reason is not None:
            reasons[reason] += 1
            layer_reasons[layer] = reason
            continue
        quantized[layer] = (shape[0], shape[1])
    return quantized, reasons, layer_reasons


def build_plan(records, layers, scheme: str):
    specs: dict[str, TensorSpec] = {}
    for tensor_name in sorted(records):
        record = records[tensor_name]
        layer = tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else None
        if layer in layers:
            out_features, in_features = layers[layer]
            weight_shape = (out_features, in_features) if scheme == "int8" else (out_features, in_features // 2)
            specs[tensor_name] = TensorSpec("I8", weight_shape)
            specs[f"{layer}.weight_scale"] = TensorSpec("F32", (out_features, 1))
        else:
            specs[tensor_name] = TensorSpec(record.dtype, record.shape)
    return specs


def quant_metadata(layers, scheme: str, group_size: int) -> dict:
    entries = {}
    for layer, shape in layers.items():
        if scheme == "int8":
            entries[layer] = {
                "format": "int8_tensorwise",
                "weight_shape": list(shape),
                "scale_shape": [shape[0], 1],
                "convrot": True,
                "convrot_groupsize": group_size,
            }
        else:
            entries[layer] = {
                "format": "int4_packed_w4a8",
                "bits": 4,
                "weight_shape": list(shape),
                "packed_shape": [shape[0], shape[1] // 2],
                "scale_shape": [shape[0], 1],
                "packing": "signed_twos_complement_low_nibble_even",
                "activation_bits": 8,
                "convrot": True,
                "convrot_groupsize": group_size,
            }
    return {
        "format_version": "1.0",
        "scheme": SCHEME_NAMES[scheme],
        "group_size": group_size,
        "scale_layout": "row",
        "layers": entries,
    }


def is_timechat_source(source_dir: Path) -> bool:
    candidates = {source_dir.name.lower(), source_dir.parent.name.lower()}
    info_path = source_dir / "vcap_model_info.json"
    if info_path.exists():
        try:
            candidates.add(json.loads(info_path.read_text(encoding="utf-8")).get("key", "").lower())
        except Exception:
            pass
    return any("timechat" in value for value in candidates)


def validate_existing(output: Path, expected_scheme: str, expected_layers: int) -> bool:
    if not output.exists():
        return False
    try:
        metadata, _ = read_safetensors_header(output)
        quant = json.loads(metadata["_quantization_metadata"])
        return (
            metadata.get("vcap_scheme") == expected_scheme
            and quant.get("scheme") == expected_scheme
            and len(quant.get("layers", {})) == expected_layers
        )
    except Exception:
        return False


def convert(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for conversion")
    forced_gpu = os.environ.get("VCAP_DEV_FORCE_GPU", "").strip()
    if forced_gpu and os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != forced_gpu:
        raise RuntimeError(
            f"Set CUDA_VISIBLE_DEVICES={forced_gpu} before running this dev converter"
        )

    started = time.perf_counter()
    source = args.source.resolve()
    source_dir = source if source.is_dir() else source.parent
    out_dir = args.out_dir.resolve()
    output = out_dir / "model.safetensors"
    if source_dir == out_dir:
        raise ValueError("Input and output folders must differ")

    group_size = args.group_size
    regular_hadamard(group_size, "cpu", torch.float32)
    excludes = compile_regexes(args.exclude_regex, "exclude")
    includes = compile_regexes(args.include_regex, "include")
    source_metadata, records = discover_tensor_records(source)
    if "_quantization_metadata" in source_metadata:
        raise ValueError("Input is already quantized")
    layers, exclusion_counts, layer_reasons = classify_layers(records, group_size, excludes, includes)
    if not layers:
        raise ValueError("No eligible 2-D Linear weights were found")

    scheme_name = SCHEME_NAMES[args.scheme]
    out_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        if validate_existing(output, scheme_name, len(layers)):
            print(f"Verified output already exists; skipping conversion: {output}")
            return 0
        raise FileExistsError(f"Existing output failed validation (use --force): {output}")

    specs = build_plan(records, layers, args.scheme)
    qmeta = quant_metadata(layers, args.scheme, group_size)
    source_repo = source_metadata.get("source_repo") or infer_source_repo(source_dir)
    header_metadata = {
        "format": "pt",
        "vcap_scheme": scheme_name,
        "source_repo": source_repo,
        "_quantization_metadata": json.dumps(qmeta, separators=(",", ":"), ensure_ascii=False),
    }
    header, order = make_safetensors_header(specs, header_metadata)
    total_output_data = total_spec_bytes(specs)
    source_data_bytes = sum(record.nbytes for record in records.values())
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    progress = ProgressLine(args.scheme, len(order), total_output_data)

    hadamard = regular_hadamard(group_size, torch.device("cuda:0"), torch.float32)
    layer_reports: list[dict[str, object]] = []
    layout_study: list[dict[str, object]] = []
    do_layout_study = args.scheme == "int4" and is_timechat_source(source_dir)
    output_item = 0
    completed_bytes = 0

    with partial.open("wb") as target, RawFileCache() as raw_files, SafeOpenCache() as tensors:
        target.write(header)
        for tensor_name in sorted(records):
            record = records[tensor_name]
            layer = tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else None
            if layer not in layers:
                copy_range(raw_files.get(record.path), target, record.start, record.nbytes)
                output_item += 1
                completed_bytes += record.nbytes
                progress.update(output_item, completed_bytes, tensor_name)
                continue

            before = time.perf_counter()
            original = tensors.get_tensor(record)
            weight = original.to(device="cuda:0", dtype=torch.float32, non_blocking=False)
            rotated = rotate_weight(weight, hadamard, group_size)
            qmax = 127 if args.scheme == "int8" else 7
            quantized, scales, row_mse = search_scales(rotated, qmax)
            squared_error = (row_mse.sum() * rotated.shape[1]).item()
            energy = weight.square().sum().clamp_min(1e-30).item()
            relative_error = math.sqrt(squared_error / energy) * 100.0

            if do_layout_study and len(layout_study) < 6 and weight.numel() <= 24_000_000:
                grouped = quantize_grouped_for_study(rotated, qmax=7, scale_group=128)
                if grouped is not None:
                    layout_study.append(
                        {
                            "layer": layer,
                            "shape": list(weight.shape),
                            "energy": energy,
                            "row_squared_error": squared_error,
                            "row_relative_l2_pct": relative_error,
                            "group128_squared_error": grouped["squared_error"],
                            "group128_relative_l2_pct": math.sqrt(grouped["squared_error"] / energy) * 100.0,
                            "row_scale_count": scales.numel(),
                            "group128_scale_count": grouped["scale_count"],
                        }
                    )

            stored_weight = quantized if args.scheme == "int8" else pack_int4_signed(quantized)
            stored_weight = stored_weight.cpu()
            stored_scales = scales.reshape(weight.shape[0], -1).cpu().float()
            write_cpu_tensor(target, stored_weight)
            output_item += 1
            completed_bytes += specs[tensor_name].nbytes
            progress.update(output_item, completed_bytes, tensor_name)
            scale_name = f"{layer}.weight_scale"
            write_cpu_tensor(target, stored_scales)
            output_item += 1
            completed_bytes += specs[scale_name].nbytes
            progress.update(output_item, completed_bytes, scale_name)

            layer_reports.append(
                {
                    "layer": layer,
                    "shape": list(weight.shape),
                    "relative_weight_error_pct": relative_error,
                    "squared_error": squared_error,
                    "seconds": time.perf_counter() - before,
                }
            )
            del original, weight, rotated, quantized, scales, row_mse, stored_weight, stored_scales
            torch.cuda.empty_cache()

        target.flush()
        os.fsync(target.fileno())

    if output_item != len(order) or completed_bytes != total_output_data:
        raise RuntimeError(
            f"Writer plan mismatch: {output_item}/{len(order)} tensors, "
            f"{completed_bytes}/{total_output_data} bytes"
        )
    os.replace(partial, output)
    progress.update(len(order), total_output_data, "complete", force=True)

    metadata_check, output_records = read_safetensors_header(output)
    parsed_qmeta = json.loads(metadata_check["_quantization_metadata"])
    if set(output_records) != set(specs) or parsed_qmeta["scheme"] != scheme_name:
        raise RuntimeError("Output header verification failed")

    copy_model_support_files(source_dir, out_dir)
    family = infer_family(out_dir / "config.json")
    errors = [float(item["relative_weight_error_pct"]) for item in layer_reports]
    layout_summary = None
    if layout_study:
        energy = sum(float(item["energy"]) for item in layout_study)
        row_error = sum(float(item["row_squared_error"]) for item in layout_study)
        group_error = sum(float(item["group128_squared_error"]) for item in layout_study)
        layout_summary = {
            "decision": "row",
            "reason": (
                "Row scales preserve one INT8 GEMM plus one output-channel epilogue; group scales "
                "need K-split accumulation or an additional lossy re-expression onto a row INT8 grid."
            ),
            "sample_layers": len(layout_study),
            "row_relative_l2_pct": math.sqrt(row_error / energy) * 100.0,
            "group128_relative_l2_pct": math.sqrt(group_error / energy) * 100.0,
            "row_to_group_mse_ratio": row_error / max(group_error, 1e-30),
            "layers": layout_study,
        }

    report = {
        "source": str(source),
        "output": str(output),
        "scheme": scheme_name,
        "format_version": "1.0",
        "group_size": group_size,
        "scale_layout": "row",
        "quantized_layers": len(layer_reports),
        "kept_tensors": len(records) - len(layer_reports),
        "output_tensor_count": len(output_records),
        "bytes_before": source_data_bytes,
        "bytes_after": output.stat().st_size,
        "data_bytes_after": total_output_data,
        "compression_ratio": source_data_bytes / total_output_data,
        "mean_relative_weight_error_pct": sum(errors) / len(errors),
        "max_relative_weight_error_pct": max(errors),
        "conversion_seconds": time.perf_counter() - started,
        "default_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "user_exclude_regex": args.exclude_regex,
        "user_include_regex": args.include_regex,
        "int4_scale_layout_study": layout_summary,
        "layers": layer_reports,
    }
    atomic_write_json(out_dir / "quantization_report.json", report)
    info = {
        "key": out_dir.name,
        "family": family,
        "scheme": scheme_name,
        "source_repo": source_repo,
        "total_bytes": output.stat().st_size,
        "tensor_count": len(output_records),
        "quantized_layer_count": len(layer_reports),
        "created_at": utc_now(),
    }
    atomic_write_json(out_dir / "vcap_model_info.json", info)
    print(
        f"DONE: {len(layer_reports):,} quantized layers, {len(records) - len(layer_reports):,} kept tensors; "
        f"{source_data_bytes / 1e9:.3f} -> {output.stat().st_size / 1e9:.3f} GB "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    try:
        return convert(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
