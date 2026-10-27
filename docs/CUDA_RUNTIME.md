# Shared CUDA 13 runtime (v2.0)

Whisper uses the application's venv in its own subprocess. Its backend is
upstream faster-whisper and CTranslate2; our integration and the CUDA 13 build
script are maintained here. No separate backend fork is required.

## Failure and repair

PyTorch 2.13.0+cu130 requires `nvidia-cudnn-cu13==9.20.0.48` on Linux.
The former requirements also installed `nvidia-cudnn-cu12`. Both distributions
write the same eight `nvidia/cudnn/lib` paths. Newer cuDNN wheels additionally
leave sublibraries that are absent from the older set. A mixed set reproduced
the reported `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` in BF16 Conv3d exactly.
Restoring the matching set made that same operation pass.

The original RunPod container was not directly inspected. The conflicting
requirements, overlapping official wheel manifests, and the matching Linux
reproduction establish the defect and the repair; they do not identify every
library that was loaded in that user's container.

The stock CTranslate2 4.8.4 wheel requests CUDA 12 cuBLAS. A preliminary test
appeared to work alongside PyTorch CUDA 13 but mapped system CUDA 12 libraries.
Changing a search path cannot change the major library name requested by that
binary. The supplied wheels instead compile the upstream source with CUDA 13.0.

Normal app startup and all three installers repair older installations:

1. Remove the conflicting CUDA 12 cuDNN/cuBLAS packages from this venv.
2. Restore the exact cuDNN dependency declared by the installed PyTorch and
   verify its library hashes against the wheel RECORD. Unexpected orphan
   sublibraries are preserved outside the library search path.
3. Install the platform's hash-pinned CUDA 13 CTranslate2 wheel when needed.
4. Select the current venv's libraries before native imports. On Linux the app
   re-executes once if its loader environment needs adjustment; workers receive
   the selected paths before they start. Windows registers the venv DLL folder.
   Competing runtime folders are excluded while NVIDIA driver paths remain.

From the project directory, an explicit check/repair is also available:

```bash
python -m vcap.core.cuda_runtime --repair
```

Use the activated project venv. No manual `unset LD_LIBRARY_PATH` is needed in
the start command. The runtime setup does not change FFmpeg.

## Reproducible backend builds

Upstream source: [CTranslate2 v4.8.4](https://github.com/OpenNMT/CTranslate2/tree/5de572ef60b67a0276c5989ee73d0072808ff4c3),
including its pinned recursive submodules. The local package version is
`4.8.4+cu130`. There are no inference source patches.

Builds use NVCC 13.0.88, CUDA 13.0 headers/libraries, cuDNN 9.20 headers/import
libraries, C++17, and pybind11. CMake enables CUDA, cuDNN, dynamic cuBLAS loading,
Ruy CPU inference, and the compiler's OpenMP runtime. SM targets are 75, 80, 86,
89, 90, 100, 103, 120, and 121, with compute_121 PTX. Windows uses MSVC 2022;
Linux uses GCC 12 on Ubuntu 22.04 and auditwheel's manylinux_2_35 policy.

Install build tools in a development venv (`pybind11`, `setuptools`, `wheel`;
Linux also `auditwheel`, `patchelf`, and `pyelftools`). A CUDA 13.0 toolkit is
required, not merely a CUDA-capable driver. Then run the project venv's Python:

```text
tools/build_ctranslate2_cuda13.py --source <recursive-upstream-checkout> --toolkit <cuda-13.0-root> --cudnn-include <headers> --cudnn-library <libcudnn.so.9-or-cudnn.lib> --jobs 8
```

The builder checks the upstream commit and NVCC major/minor, bundles upstream
license notices, and produces a Python 3.12 wheel. On Linux auditwheel bundles
the backend and OpenMP libraries while excluding CUDA/cuDNN supplied by PyTorch.
On Windows the wheel includes the backend DLL and uses PyTorch's runtime DLLs.
End users install prebuilt wheels; they do not compile CTranslate2.

Artifacts are pinned to Hugging Face commit
`38577fbe9bb716faaa124c3916ca69cc25b1409a` in `requirements.txt`:

| Platform | Bytes | SHA-256 |
| --- | ---: | --- |
| Linux x86-64, glibc 2.35+, CPython 3.12 | 34,586,982 | `953e7ebac5d257f0bfcd124453bf39932d231749c73e980d0fabe6c59c4a4dcb` |
| Windows x86-64, CPython 3.12 | 13,990,926 | `26ffd79fc304434c52d9f8cea4ff5c9c5c8a2692c07fe337b79cd6e1cac5c4c0` |

## Verification

The CUDA 13 backend passed real tiny.en encoder/generation operations on an
RTX 5090 (Windows) and RTX A6000 (Linux/Massed Compute), with GPU FP16,
GPU INT8/FP16, and CPU INT8. BF16 vision Conv3d passed in the same process.
Fresh processes exercised Whisper first and PyTorch first. Mapped library
paths confirmed CUDA 13 cuBLAS, CUDA 13 runtime, and matching venv cuDNN without
CUDA 12 cuBLAS. These are synthetic feature tensors; no audio was transcribed
or evaluated.

The developer check is available with the project venv:

```text
tools/check_cuda_runtime.py --gpu <GPU-UUID> --order whisper-first --output <receipt.json>
```

Repeat with `--order torch-first` to check the other import order. The check
downloads tiny.en into the normal Hub cache on first use. Local receipts and
the exact failed/repaired convolution traces are retained under
`temp/cudnn_runtime_20261028/`. The isolated Linux lab is under
`/home/Ubuntu/secourses_captioner_validation/20261028/` on the Massed Compute
validation host. Other applications and system CUDA installations were not
modified.
