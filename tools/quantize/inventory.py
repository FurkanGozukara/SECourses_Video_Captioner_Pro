"""Build MODEL_FILES.md and compare shared multimodal tower tensors."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ProgressLine,
    discover_tensor_records,
    ensure_utf8_stdio,
    hash_range,
    utc_now,
)


FOLDERS = [
    "timechat_bf16",
    "timechat_int8",
    "timechat_int4",
    "avocado_bf16",
    "avocado_int8",
    "avocado_int4",
    "qwen3_omni_instruct_bf16",
    "qwen3_omni_instruct_int8",
    "qwen3_omni_instruct_int4",
    "qwen3_omni_instruct_gguf_q4",
    "qwen3_omni_thinking_bf16",
    "qwen3_omni_thinking_int8",
    "qwen3_omni_thinking_int4",
    "qwen3_omni_captioner_bf16",
    "qwen3_omni_captioner_int8",
    "qwen3_omni_captioner_int4",
    "qwen3_omni_captioner_gguf_q4",
]
ORIGINALS = Path("F:/SECourses_Video_Captioner_Pro_TEMP/originals")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hash produced model folders and compare tower tensors.")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("docs/MODEL_FILES.md"))
    return parser.parse_args(argv)


def hash_file(path: Path, progress: ProgressLine, item: int, completed: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
            completed += len(chunk)
            progress.update(item, completed, f"{path.parent.name}/{path.name}")
    return digest.hexdigest(), completed


def tower_manifest(model_dir: Path, prefixes: tuple[str, ...]) -> dict:
    _, records = discover_tensor_records(model_dir)
    names = sorted(name for name in records if name.startswith(prefixes))
    tensor_hashes = {}
    total_bytes = 0
    manifest = hashlib.sha256()
    for index, name in enumerate(names, 1):
        record = records[name]
        digest = hash_range(record.path, record.start, record.nbytes)
        tensor_hashes[name] = digest
        total_bytes += record.nbytes
        manifest.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
        if index == len(names) or index % 100 == 0:
            print(f"\rtowers {model_dir.name}: {index}/{len(names)} tensors".ljust(100), end="", flush=True)
    print()
    return {
        "tensor_hashes": tensor_hashes,
        "tensor_count": len(names),
        "total_bytes": total_bytes,
        "manifest_sha256": manifest.hexdigest(),
    }


def comparison(name: str, manifests: dict[str, dict]) -> dict:
    labels = list(manifests)
    common = set(manifests[labels[0]]["tensor_hashes"])
    union = set(common)
    for label in labels[1:]:
        keys = set(manifests[label]["tensor_hashes"])
        common &= keys
        union |= keys
    mismatches = []
    for tensor_name in sorted(common):
        values = {manifests[label]["tensor_hashes"][tensor_name] for label in labels}
        if len(values) != 1:
            mismatches.append(tensor_name)
    return {
        "name": name,
        "labels": labels,
        "common": len(common),
        "union": len(union),
        "mismatches": mismatches,
        "all_identical": len(common) == len(union) and not mismatches,
        "manifests": manifests,
    }


def run(args: argparse.Namespace) -> int:
    models_root = args.models_root.resolve()
    missing = [name for name in FOLDERS if not (models_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing requested model folders: {', '.join(missing)}")

    files = [
        path
        for folder in FOLDERS
        for path in sorted((models_root / folder).iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and not path.name.endswith(".partial")
    ]
    total_bytes = sum(path.stat().st_size for path in files)
    progress = ProgressLine("sha256", len(files), total_bytes)
    completed = 0
    inventory: dict[str, list[dict]] = {folder: [] for folder in FOLDERS}
    for index, path in enumerate(files, 1):
        digest, completed = hash_file(path, progress, index, completed)
        inventory[path.parent.name].append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": digest}
        )
    progress.update(len(files), total_bytes, "complete", force=True)

    q25 = comparison(
        "Qwen2.5-Omni TimeChat vs AVoCaDO",
        {
            "TimeChat": tower_manifest(
                ORIGINALS / "TimeChat-Captioner-GRPO-7B",
                ("thinker.visual.", "thinker.audio_tower."),
            ),
            "AVoCaDO": tower_manifest(
                ORIGINALS / "AVoCaDO", ("thinker.visual.", "thinker.audio_tower.")
            ),
        },
    )
    qwen3 = comparison(
        "Qwen3-Omni Thinking vs Instruct vs Captioner",
        {
            "Thinking": tower_manifest(
                ORIGINALS / "Qwen3-Omni-30B-A3B-Thinking",
                ("thinker.visual.", "thinker.audio_tower."),
            ),
            "Instruct": tower_manifest(
                ORIGINALS / "Qwen3-Omni-30B-A3B-Instruct",
                ("thinker.visual.", "thinker.audio_tower."),
            ),
            "Captioner": tower_manifest(
                ORIGINALS / "Qwen3-Omni-30B-A3B-Captioner",
                ("thinker.visual.", "thinker.audio_tower."),
            ),
        },
    )

    lines = [
        "# Model Files",
        "",
        f"Generated: {utc_now()}",
        "",
        "Hugging Face root: `MonsterMMORPG/Wan_GGUF/Video_Captioner_Pro/`",
        "",
        "Sizes use decimal GB. SHA-256 values were calculated by streaming each file.",
    ]
    for folder in FOLDERS:
        entries = inventory[folder]
        folder_bytes = sum(item["bytes"] for item in entries)
        lines.extend(
            [
                "",
                f"## {folder}",
                "",
                f"Total: {folder_bytes / 1e9:.6f} GB ({folder_bytes:,} bytes)",
                "",
                "| File | Bytes | Size GB | SHA-256 | Hugging Face path |",
                "|---|---:|---:|---|---|",
            ]
        )
        for item in entries:
            hf_path = f"MonsterMMORPG/Wan_GGUF/Video_Captioner_Pro/{folder}/{item['name']}"
            lines.append(
                f"| `{item['name']}` | {item['bytes']:,} | {item['bytes'] / 1e9:.6f} | "
                f"`{item['sha256']}` | `{hf_path}` |"
            )

    lines.extend(["", "## Tower Sharing", ""])
    for item in (q25, qwen3):
        verdict = "IDENTICAL" if item["all_identical"] else "DIFFERENT"
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"Verdict: **{verdict}** across {item['common']:,} common tower tensors "
                f"({item['union']:,} tensors in the union).",
                "",
                "| Checkpoint | Tower tensors | Tower bytes | Manifest SHA-256 |",
                "|---|---:|---:|---|",
            ]
        )
        for label, manifest in item["manifests"].items():
            lines.append(
                f"| {label} | {manifest['tensor_count']:,} | {manifest['total_bytes']:,} | "
                f"`{manifest['manifest_sha256']}` |"
            )
        if item["mismatches"]:
            lines.extend(["", f"Mismatched tensors: {len(item['mismatches']):,}."])
            for tensor_name in item["mismatches"][:50]:
                lines.append(f"- `{tensor_name}`")
            if len(item["mismatches"]) > 50:
                lines.append(f"- ... and {len(item['mismatches']) - 50:,} more")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(partial, args.output)
    print(f"Wrote {args.output} for {len(FOLDERS)} folders and {len(files)} files")
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
