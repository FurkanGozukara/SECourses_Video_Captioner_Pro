"""Stream sharded Hugging Face safetensors into one thinker checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ProgressLine,
    RawFileCache,
    TensorSpec,
    atomic_write_json,
    copy_model_support_files,
    copy_range,
    discover_tensor_records,
    ensure_utf8_stdio,
    hash_range,
    infer_family,
    infer_source_repo,
    make_safetensors_header,
    read_safetensors_header,
    total_spec_bytes,
    utc_now,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream HF safetensors shards into one BF16 model.safetensors file."
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Source Hugging Face model folder")
    parser.add_argument("--out-dir", type=Path, required=True, help="Self-contained output model folder")
    parser.add_argument(
        "--drop-prefix",
        action="append",
        default=[],
        help="Drop keys beginning with this prefix; may be supplied more than once",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output checkpoint")
    return parser.parse_args(argv)


def validate_existing(
    output: Path, source_records: dict, kept_names: list[str], drop_prefixes: list[str]
) -> bool:
    if not output.exists():
        return False
    try:
        metadata, records = read_safetensors_header(output)
        if metadata.get("vcap_scheme") != "bf16":
            return False
        if set(records) != set(kept_names):
            return False
        encoded_drops = metadata.get("dropped_prefixes", "[]")
        if json.loads(encoded_drops) != drop_prefixes:
            return False
        if any(records[name].nbytes != source_records[name].nbytes for name in kept_names):
            return False
    except Exception:
        return False
    return True


def prove_transformers_load(out_dir: Path, family: str) -> dict[str, int]:
    import torch

    if family == "qwen2_5_omni":
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration as model_class
    elif family == "qwen3_omni_moe":
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration as model_class
    else:
        raise ValueError(f"No thinker load check is defined for family {family}")

    model, loading = model_class.from_pretrained(
        out_dir,
        dtype=torch.bfloat16,
        device_map={"": "meta"},
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    result = {
        "missing_keys": len(loading.get("missing_keys", ())),
        "unexpected_keys": len(loading.get("unexpected_keys", ())),
        "mismatched_keys": len(loading.get("mismatched_keys", ())),
    }
    del model
    if any(result.values()):
        raise RuntimeError(f"Transformers load check failed: {result}")
    return result


def merge(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir.resolve()
    output = out_dir / "model.safetensors"
    drop_prefixes = list(dict.fromkeys(args.drop_prefix))

    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    _, source_records = discover_tensor_records(model_dir)
    kept_names = sorted(
        name for name in source_records if not any(name.startswith(prefix) for prefix in drop_prefixes)
    )
    dropped_names = sorted(set(source_records) - set(kept_names))
    if not kept_names:
        raise ValueError("All tensors were dropped")
    non_bf16 = [name for name in kept_names if source_records[name].dtype != "BF16"]
    if non_bf16:
        raise ValueError(f"Expected a BF16 checkpoint; {len(non_bf16)} tensors are not BF16")

    out_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        if validate_existing(output, source_records, kept_names, drop_prefixes):
            print(f"Verified output already exists; skipping merge: {output}")
            return 0
        raise FileExistsError(f"Existing output failed validation (use --force): {output}")

    source_repo = infer_source_repo(model_dir)
    specs = {name: TensorSpec(source_records[name].dtype, source_records[name].shape) for name in kept_names}
    metadata = {
        "format": "pt",
        "vcap_scheme": "bf16",
        "source_repo": source_repo,
        "dropped_prefixes": drop_prefixes,
    }
    header, order = make_safetensors_header(specs, metadata)
    total_bytes = total_spec_bytes(specs)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    progress = ProgressLine("merge", len(order), total_bytes)

    completed = 0
    with partial.open("wb") as target, RawFileCache() as sources:
        target.write(header)
        for index, name in enumerate(order, 1):
            record = source_records[name]
            copy_range(sources.get(record.path), target, record.start, record.nbytes)
            completed += record.nbytes
            progress.update(index, completed, name, force=index == len(order))
        target.flush()
        os.fsync(target.fileno())
    os.replace(partial, output)

    _, output_records = read_safetensors_header(output)
    rng = random.Random(42)
    sample = rng.sample(kept_names, min(5, len(kept_names)))
    checks: list[dict[str, object]] = []
    for name in sample:
        source = source_records[name]
        merged = output_records[name]
        source_sha = hash_range(source.path, source.start, source.nbytes)
        output_sha = hash_range(output, merged.start, merged.nbytes)
        if source_sha != output_sha:
            raise RuntimeError(f"Byte verification failed for {name}")
        checks.append({"tensor": name, "bytes": source.nbytes, "sha256": source_sha})
        print(f"verified byte-identical: {name} ({source.nbytes / 1e6:.1f} MB)")

    copy_model_support_files(model_dir, out_dir)
    family = infer_family(out_dir / "config.json")
    loading = prove_transformers_load(out_dir, family)
    print(f"Transformers meta load: zero missing/unexpected/mismatched keys ({family})")

    info = {
        "key": out_dir.name,
        "family": family,
        "scheme": "bf16",
        "source_repo": source_repo,
        "total_bytes": output.stat().st_size,
        "tensor_count": len(kept_names),
        "created_at": utc_now(),
        "dropped_tensor_count": len(dropped_names),
        "dropped_prefixes": drop_prefixes,
        "byte_identity_checks": checks,
        "transformers_load_check": loading,
    }
    atomic_write_json(out_dir / "vcap_model_info.json", info)
    print(
        f"DONE: {len(kept_names):,} tensors, {output.stat().st_size / 1e9:.3f} GB, "
        f"{time.perf_counter() - started:.1f}s -> {output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    try:
        return merge(parse_args(argv))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
