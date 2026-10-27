"""Exercise caption Conv3d and Whisper math; no audio file is transcribed.

Run with the project venv and --gpu set to the intended GPU UUID. Whisper's
tiny.en checkpoint is downloaded into the normal Hub cache on first use.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vcap.core.cuda_runtime import bootstrap_cuda_runtime, runtime_library_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--order", choices=("whisper-first", "torch-first"), default="whisper-first")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    bootstrap_cuda_runtime()
    report = {"python": sys.executable, "gpu_selector": args.gpu, "order": args.order,
              "packages": {name: metadata.version(name) for name in ("torch", "ctranslate2")},
              "runtime_directories": runtime_library_dirs()}

    def whisper():
        import ctranslate2
        import numpy as np
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download("Systran/faster-whisper-tiny.en", allow_patterns=["model.bin", "config.json", "vocabulary.*"])
        results = []
        for device, precision in (("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")):
            started = time.monotonic()
            model = ctranslate2.models.Whisper(checkpoint, device=device, compute_type=precision, intra_threads=4)
            features = ctranslate2.StorageView.from_array(np.zeros((1, 80, 3000), dtype=np.float32))
            encoded = model.encode(features)
            generated = model.generate(encoded, [[50257, 50362]], max_length=4, beam_size=1)
            assert len(generated) == 1
            results.append({"device": device, "compute_type": precision, "seconds": round(time.monotonic() - started, 3)})
            del model, encoded, generated
        report["whisper"] = results

    def caption():
        import torch
        report["gpu"] = str(torch.cuda.get_device_properties(0))
        report["torch_cuda"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version()
        with torch.inference_mode():
            conv = torch.nn.Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16)).to("cuda", dtype=torch.bfloat16)
            output = conv(torch.zeros((1, 3, 2, 32, 32), device="cuda", dtype=torch.bfloat16))
            torch.cuda.synchronize()
            assert output.isfinite().all()
            report["caption_conv3d"] = list(output.shape)

    for operation in ((whisper, caption) if args.order == "whisper-first" else (caption, whisper)):
        operation()
    if sys.platform == "linux":
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if any(name in line for name in ("libcudnn", "libcublas", "libcudart", "libctranslate"))})
    else:
        import psutil
        libraries = sorted({mapping.path for mapping in psutil.Process().memory_maps()
                            if any(name in mapping.path.lower() for name in ("cudnn", "cublas", "cudart", "ctranslate2.dll"))})
    report["libraries"] = libraries
    assert not any(any(name in path for name in ("cublas64_12", "cudart64_12", "libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12")) for path in libraries), libraries
    assert any("cublas" in path for path in libraries), libraries
    runtime_libraries = [path for path in libraries if Path(path).name.lower().startswith(
        ("cublas", "cudnn", "cudart", "ctranslate", "libcublas", "libcudnn", "libcudart", "libctranslate"))]
    assert all(Path(path).resolve().is_relative_to(Path(sys.prefix).resolve()) for path in runtime_libraries), libraries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
