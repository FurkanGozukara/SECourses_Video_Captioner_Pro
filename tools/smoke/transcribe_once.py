#!/usr/bin/env python3
"""Transcribe one media file through the standalone Whisper backend."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe one media file with faster-whisper")
    parser.add_argument("media", help="audio or video file")
    parser.add_argument("--model", default="large-v1", help="Whisper model alias")
    parser.add_argument("--gpu", type=int, default=0, help="physical CUDA GPU index")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max(0, args.gpu))
    repository = Path(__file__).resolve().parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    from vcap.whisper.engine import WhisperEngine
    from vcap.whisper.params import WhisperParams

    params = WhisperParams(model=args.model, device="cuda", gpu_index=max(0, args.gpu))
    engine = WhisperEngine(
        params,
        log=lambda message, level="info": print(f"[{level}] {message}", file=sys.stderr),
    )
    started = time.perf_counter()
    try:
        result = engine.transcribe(args.media)
    finally:
        engine.unload()
    wall_s = time.perf_counter() - started
    print(result.text)
    print(
        f"duration_s={result.duration_s:.3f} transcribe_s={result.elapsed_s:.3f} "
        f"wall_s={wall_s:.3f} device={result.device} compute_type={result.compute_type}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
