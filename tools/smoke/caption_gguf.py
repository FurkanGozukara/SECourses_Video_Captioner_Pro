"""Download, load, and caption one input through Qwen3-Omni GGUF."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="qwen3_omni_instruct_gguf_q4")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-preset", default=None)
    parser.add_argument("--max-new-tokens", "--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--video-mode", choices=("frames_audio", "native"), default="frames_audio")
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA GPU index")
    parser.add_argument("--force-llamacpp", action="store_true")
    return parser


def _progress(*args: object) -> None:
    if not args:
        return
    first = args[0]
    if isinstance(first, dict):
        message = first.get("message")
    else:
        message = first
    if message and not str(message).startswith("Generating:"):
        print(str(message), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max(0, int(args.gpu)))
    from vcap.core.logs import setup_utf8_stdio

    setup_utf8_stdio()
    source = args.input.expanduser().resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(source)
    os.environ["VCAP_LLAMACPP_VIDEO_MODE"] = args.video_mode

    from vcap import ensure_app_dirs
    from vcap.core.gpu import resource_snapshot
    from vcap.models.base import Callbacks, GenParams, MediaInput, PreprocessParams, PromptSpec
    from vcap.models.llamacpp_backend import ensure_gguf
    from vcap.models.llamacpp_install import ensure_llamacpp
    from vcap.models.loader import load_model, unload_model
    from vcap.models.registry import MODEL_SPECS, get_variant, variant_to_family

    ensure_app_dirs()
    variant = get_variant(args.variant)
    if variant.backend != "llamacpp":
        raise ValueError(f"Smoke tool requires a GGUF llama.cpp variant, got {args.variant}")
    family = variant_to_family(variant.key)
    spec = MODEL_SPECS[family]
    ensure_llamacpp(_progress, force=args.force_llamacpp)
    ensure_gguf(variant.key, _progress)
    before = resource_snapshot(args.gpu)
    loaded = load_model(
        variant.key,
        device="cuda:0",
        gpu_index=args.gpu,
        progress_cb=_progress,
    )
    try:
        backend = loaded.model
        prompt = None
        if args.prompt is not None or args.prompt_preset is not None:
            prompt = PromptSpec(preset_id=args.prompt_preset, user_prompt=args.prompt)
        sampled = family in {"qwen3_omni_thinking", "qwen3_omni_captioner"}
        temperature = args.temperature if args.temperature is not None else (0.6 if sampled else 0.0)
        generation = GenParams(
            temperature=float(temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=1.0,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            do_sample=bool(sampled and temperature > 0),
        )
        preprocessing = PreprocessParams(
            fps=max(0.01, float(args.fps)),
            max_frames=max(1, int(args.max_frames)),
            max_pixels=max(spec.limits.min_pixels, int(args.max_pixels)),
            min_pixels=spec.limits.min_pixels,
            use_audio_in_video="video_audio" in spec.capabilities,
        )
        result = backend.caption(
            MediaInput(path=source),
            prompt,
            generation,
            preprocessing,
            Callbacks(progress=_progress),
        )
        print("\n=== CAPTION ===")
        print(result.text)
        if result.reasoning:
            print("\n=== REASONING ===")
            print(result.reasoning)
        print("\n=== METRICS ===")
        print(
            json.dumps(
                {
                    "variant": variant.key,
                    "llama_server": str(backend.server_path),
                    "endpoint": f"{backend.base_url}/v1/chat/completions",
                    "video_mode": backend.video_mode,
                    "load": asdict(loaded.load_report),
                    "usage": asdict(result.usage),
                    "timing": asdict(result.timing),
                    "peak_vram_gb": result.peak_vram_gb,
                    "gpu_before": before,
                    "gpu_after": resource_snapshot(args.gpu),
                    "cancelled": result.cancelled,
                    "warnings": result.warnings,
                },
                indent=2,
                default=str,
            )
        )
    finally:
        unload = unload_model(loaded)
        print("\n=== UNLOAD ===")
        print(json.dumps(asdict(unload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
