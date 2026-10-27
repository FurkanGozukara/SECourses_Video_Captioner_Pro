# v2.0 downloader verification (2026-10-28)

## CUDA 13 follow-up in the final v2.0 package

The final v2.0 also fixes the conflicting cuDNN runtime and replaces the stock
CUDA 12 CTranslate2 binary with CUDA 13 builds for both platforms. Full build
provenance, wheel hashes, the exact reproduced error, and the repair procedure
are recorded in [CUDA_RUNTIME.md](CUDA_RUNTIME.md).

Real Whisper tiny.en encoder/generation operations passed on Windows/RTX 5090
and Linux/Massed Compute/RTX A6000: GPU FP16, GPU INT8/FP16, and CPU INT8. Both
Whisper-first and PyTorch-first load orders passed, alongside BF16 vision
Conv3d. Process library maps confirmed the venv's CUDA 13 libraries. An isolated
Linux mixed-cuDNN installation reproduced the exact reported error; the repair
restored matching wheel hashes and the same convolution passed. Migration from
the stock backend fetched the published, hash-pinned Linux wheel automatically.
The published Windows wheel was also installed and hash-verified through pip.

The normal Windows BAT launched v2.0 and its real UI completed a Qwen3-Omni
Instruct INT4 image caption: **177 new tokens, EOS, done=1, failed=0**. The caption
correctly described the blue-circle test image. Output and the screenshot
`output/playwright/cuda13_v2_completed.png` were inspected. Earlier test outputs
were preserved. The **10 focused runtime/subprocess checks passed on both OSes**.

## Confirmed cause

The application launched the downloader with
`Path("venv/bin/python").resolve()`. On Linux that selects the underlying base
interpreter, which no longer discovers the venv's `pyvenv.cfg` and packages.
A real Linux/Massed Compute reproduction selected `/usr/bin/python3.12` and
failed with the reported `ModuleNotFoundError: No module named 'huggingface_hub'`.
The corrected child used the venv path, retained the correct `sys.prefix`, and
imported the installed package successfully.

`vcap/core/python_runtime.py` now selects the project venv or current Python
without resolving executable symlinks. The downloader bridge and Whisper
worker share that selector, including explicit Python overrides. Ordinary
source-file paths still resolve normally.

The canonical downloader now lives in the Git checkout. Updates therefore
update the automatic downloader instead of leaving it on an older ZIP copy.
Explicit overrides still take priority; distribution-level copies remain a
fallback. The standalone ZIP copy and Git copy share the existing cache and
models locations. HTTP metadata retries now retain `Retry-After` headers from
error responses whose boolean value is false. Logging also handles the current
Hub client's `httpx2` / `httpcore2` logger names.

## Checks on both operating systems

The **31 downloader/interpreter checks passed on Windows and real Linux**.
The Linux checks ran in an isolated project Python 3.12 venv on the Massed
Compute host, without changing its other applications, system packages, or GPUs.
Windows used huggingface-hub 1.33.0; Linux used huggingface-hub 2.3.0.

The checks include real child processes with dependencies installed only in a
temporary venv, paths containing spaces and Unicode, default and explicit
interpreter selection, bundled/legacy/override downloader discovery, and the
existing catalog and status-protocol tests. A real loopback HTTP server covers:

- Multiple range connections and reuse of already verified files.
- HTTP 429 and 503 recovery, expired links, truncated responses, and malformed
  `Content-Range` headers.
- Servers that ignore byte ranges and require a single-stream fallback.
- Corrupt bytes detected by SHA verification, successful retry, and refusal to
  publish a persistently corrupt file.
- Interruption with `.part` and `.part.json` preservation, followed by fetching
  only the missing bytes.
- Respect for `Retry-After` when the HTTP response evaluates to false.

## Real Hugging Face transfers

All **15 regular model configurations** were freshly downloaded and verified
against Hub metadata on Windows and Linux. This checked every catalog subfolder
without downloading every precision's weight file.

Two complete model downloads then ran through the application's real automatic
first-use bridge, including dependency imports, all auxiliary files, native Xet
transfer, full published SHA verification, and final ready status:

| Host | Model | Weight bytes | Result |
| --- | --- | ---: | --- |
| Windows | `timechat_int4` | 6,451,611,760 | Complete and verified |
| Linux / Massed Compute | `qwen3_omni_instruct_int4` | 17,780,288,064 | Complete and verified |

The new Git-managed downloader was additionally exercised by preserving and
moving each downloaded `config.json` aside, then invoking the automatic bridge
again. Both hosts fetched an identical replacement config and reused the
verified weights without rewriting them. The preserved original configs and
JSON receipts remain with the local validation evidence.

These are live Windows and Linux/Massed Compute downloader checks. The original
RunPod dependency error was reproduced on Linux; no separate fresh deployment
inside the reporting user's RunPod container is claimed.

## Application and suite

The normal `Windows_Run_Video_Captioner_Pro.bat` launched v2.0, and the real
browser UI displayed version 2.0. Restarting the owned local test app preserved
the earlier caption outputs. No task assets were lost.

The final full suite reported **664 passed, 9 skipped, 8 failed**. The eight failures
are the same UI/protocol failures already reproduced on the unchanged baseline
and listed in [v1.9.3 verification](QA_VERIFICATION_v1.9.3.md#broader-suite-and-existing-failures).
The prior [15-variant model-generation checks](QA_VERIFICATION_v1.9.3.md#launcher-and-real-checkpoints)
remain relevant; this follow-up changes runtime selection rather than model
loading algorithms. The additional real Qwen INT4 UI run is described above.

Local evidence is retained in `temp/downloader_reliability_20261028/`, including
the baseline failure, native network receipts, full-model checks, missing-config
recovery, and the Windows suite output. The UI screenshot is
`output/playwright/downloader_v2.png`.

## Release artifact

`tools/package_release.py` builds the existing nine-file installer distribution
from the Git-managed downloader and `installers/` sources. The distribution's
requirements wrapper selects the current Git-managed `requirements.txt`.
The builder preserves the previous ZIP, synchronizes loose distribution files,
verifies every member byte-for-byte, and checks ZIP CRCs.

`SECourses_Video_Captioner_Pro_v2_0.zip` is **45,050 bytes**, SHA-256:

`10f209ae0fa466840f80e9270638930254d3562329486e9850dc9c050725a269`

The final package includes the runtime repair in all installers and updated
Linux startup instructions. Both Linux FFmpeg blocks were checked byte-for-byte
against the initial v2.0 ZIP and are unchanged. The earlier 45,408-byte v2.0 ZIP
is preserved under `temp/release_builds/previous/`, identified by its original
SHA-256 `eff2069aa46c034b3e01a8a6726ee9d510f33d9e890d4202e282f8df4d34cdfd`.
