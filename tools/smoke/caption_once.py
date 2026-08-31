"""Load one local model and caption exactly one media input."""

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
    parser.add_argument("--variant", default="timechat_bf16")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt-preset", default=None)
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--hf-dir", type=Path, default=None,
                        help="Original sharded HF folder used when a converted folder is unavailable.")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--compile-mode", default="cudagraphs")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA GPU index")
    return parser


def _wrapper(loaded: object) -> object:
    family = loaded.spec.family
    if family == "timechat":
        from vcap.models.timechat import TimeChatCaptioner

        return TimeChatCaptioner(loaded)
    if family == "avocado":
        from vcap.models.avocado import AvocadoCaptioner

        return AvocadoCaptioner(loaded)
    from vcap.models.qwen3_omni import captioner_for_loaded

    return captioner_for_loaded(loaded)


def main() -> int:
    args = _parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max(0, int(args.gpu)))
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    from vcap.models.base import GenParams, MediaInput, PreprocessParams, PromptSpec
    from vcap.models.loader import load_model, unload_model
    from vcap.models.offload import OffloadPlan
    from vcap.models.registry import MODEL_SPECS, variant_to_family

    family = variant_to_family(args.variant)
    spec = MODEL_SPECS[family]
    loaded = load_model(
        args.variant,
        device="cuda:0",
        gpu_index=args.gpu,
        attention=args.attention,
        offload=OffloadPlan(),
        hf_dir=args.hf_dir,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
    )
    try:
        wrapper = _wrapper(loaded)
        prompt = PromptSpec(preset_id=args.prompt_preset) if args.prompt_preset else None
        sampled = family in {"qwen3_omni_thinking", "qwen3_omni_captioner"}
        gen = GenParams(
            temperature=0.6 if sampled else 0.0,
            top_p=0.95 if sampled else 1.0,
            top_k=20 if sampled else 0,
            repetition_penalty=1.0,
            max_new_tokens=max(1, args.max_new_tokens),
            do_sample=sampled,
        )
        limits = spec.limits
        pre = PreprocessParams(
            fps=args.fps or limits.default_fps,
            max_frames=args.max_frames or int(limits.max_frames or 768),
            max_pixels=args.max_pixels or limits.default_max_pixels,
            min_pixels=limits.min_pixels,
            use_audio_in_video="video_audio" in spec.capabilities,
        )
        result = wrapper.caption(MediaInput(path=args.input), prompt, gen, pre)
        print("\n=== CAPTION ===")
        print(result.text)
        if result.reasoning:
            print("\n=== REASONING ===")
            print(result.reasoning)
        print("\n=== METRICS ===")
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "load": asdict(loaded.load_report),
                    "usage": asdict(result.usage),
                    "timing": asdict(result.timing),
                    "peak_vram_gb": result.peak_vram_gb,
                    "cancelled": result.cancelled,
                    "warnings": result.warnings,
                },
                indent=2,
                default=str,
            )
        )
    finally:
        report = unload_model(loaded)
        print("\n=== UNLOAD ===")
        print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
