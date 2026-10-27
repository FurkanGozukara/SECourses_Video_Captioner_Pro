# QA verification - v1.9.3 (2026-10-27)

## Cause and repair

A fresh environment installed Transformers 5.21.0, released on October 26. Its
Qwen3-Omni and Qwen2.5-Omni thinkers now own the text decoder at
`model.language_model` and the towers at `model.audio_tower` / `model.visual`.
The published checkpoints retain names such as
`thinker.model.embed_tokens.weight`. Our single-file loader stripped `thinker.`
but did not apply Transformers' checkpoint conversion mapping, leaving names
that no longer existed in the constructed model. The reported
`KeyError: 'Module path not found: model.embed_tokens'` was reproduced against
the actual installed model's module lookup. It is a loader compatibility error;
changing prompts, allocating more VRAM, or downloading the same weights again
does not repair that lookup.

The layout resolver now maps checkpoint tensors and quantization metadata to
the instantiated model, preserves already converted names, and still supports
the previous layout. Decoder placement, block swap, explicit Accelerate maps,
on-demand tower wrappers, and the compilation target use the same layout.
Accelerate's automatically inferred maps already contain runtime names and
are left intact. Header-only memory planning also recognizes checkpoints
exported with the new nested decoder and tower names.

The installer requirements shipped beside the repository now request
`transformers>=5.21.0` and `av>=19.0.2`, permitting newer versions.

The fresh environment also exposed a separate Whisper compatibility error:
released faster-whisper 1.2.1 calls `av.open(metadata_errors="ignore")`, but
PyAV 19 removed that keyword. Both transcription and trim preparation now use
the application's FFmpeg decoder for that specific TypeError. Compatible
upstream decoders remain in use, and unrelated exceptions propagate.

Upstream references: [Transformers release source](https://github.com/huggingface/transformers/blob/dded62b5e641816d552098b1b8de578d17121740/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py),
[checkpoint conversion mapping](https://github.com/huggingface/transformers/blob/dded62b5e641816d552098b1b8de578d17121740/src/transformers/conversion_mapping.py),
[faster-whisper PyAV issue](https://github.com/SYSTRAN/faster-whisper/issues/1589).

## Environment and targeted checks

Python 3.12, PyTorch 2.13.0+cu130, Transformers 5.21.0, Accelerate 1.15.0,
safetensors 0.8.0, huggingface-hub 1.33.0, tokenizers 0.23.3, PyAV 19.0.2,
faster-whisper 1.2.1. Local GPU work used physical GPU 0, an RTX 5090, with its
UUID verified against CUDA enumeration. Other host workloads were present;
timings are functional-run evidence, not performance comparisons.

`tests/test_transformers_layout.py` loads legacy-named checkpoints into tiny
real Qwen3-Omni and Qwen2.5-Omni models from the installed library. BF16 state
and forward logits match the original models exactly. INT4/INT8 metadata,
packed experts, layer placement, swap/compile decoder selection, explicit
versus inferred device maps, and actual tower owners are covered. Both the
Windows unmapped checkpoint reader and the `safetensors.safe_open` reader used
on Linux are exercised. The rotary regression uses the current model layout.

Whisper checks verify decoded PCM sample values and trim timestamp offsets
without listening or ASR quality assessment. The final targeted run of layout,
offload-budget, loader-wiring, rotary, and decoder-compatibility tests passed
**37/37**. The synthetic loader-wiring fixture now mocks both CUDA and physical
VRAM probes; mixing mocked CUDA memory with live physical telemetry had made
that test depend on other GPU jobs.

## Launcher and real checkpoints

The supplied `Windows_Run_Video_Captioner_Pro.bat` launched the real UI. Port
7860 was already occupied by a model downloader, so the captioner was launched
on 7872; the existing service was preserved and no task data was lost.

In the UI, Qwen3-Omni Instruct INT4 and BF16 each described the test image's
blue circle on a white background and stopped at EOS. BF16 loaded the original
63.4 GB checkpoint and exercised 31 swapped layers, two staging slots, and
vision-tower staging. The BF16 caption was:

> The image displays a solid blue circle centered on a plain white background.

The remaining Transformers variants are checked sequentially through
`tools/smoke/caption_once.py`, loading, generating, and unloading each model.
Video models receive a two-second test pattern with a tone track; the audio
Captioner receives its extracted mono track. Smoke runs use a 24-token limit,
so a successful generation need not be a complete caption or final Thinking
answer. Live GPU testing is on Windows; no live RunPod installation is claimed.

All **15/15** Transformers variants completed loading, generation, and unloading.

| Family | BF16 | INT8 ConvRot | INT4 ConvRot |
| --- | --- | --- | --- |
| `timechat` | pass | pass | pass |
| `avocado` | pass | pass | pass |
| `qwen3_omni_instruct` | pass | pass | pass |
| `qwen3_omni_thinking` | pass | pass | pass |
| `qwen3_omni_captioner` | pass | pass | pass |

## Broader suite and existing failures

The full suite after the repair reported **632 passed, 9 skipped, 8 failed**.
The later expanded reader and header tests add six passing cases, exercised
in the targeted run above. Before the Whisper repair, nine failures were
reproduced against an untouched archive of the starting commit using the same
fresh environment. The additional failure was the PyAV decoder TypeError,
which is now fixed. These eight unrelated failures remain:

| Test | Existing failure |
| --- | --- |
| `test_dataset_clips::test_folder_light_scan_reports_caption_coverage` | Expected literal `<stem>.txt`; summary correctly contains escaped `&lt;stem&gt;.txt`. |
| `test_f3_ui_fixes::test_zip_upload_descends_wrapper_folder_and_can_enable_recursive_scan` | Expected `Extracted 2 files` in a caption-coverage summary. |
| `test_f3_ui_fixes::test_non_playable_video_uses_a_poster_without_assigning_gradio_video` | Expected older preview help wording. |
| `test_f4_fixes::test_model_change_returns_immediately_and_one_run_is_submitted` | `IndexError: tuple index out of range` in callback arguments. |
| `test_model_switch_unload::test_pipeline_client_release_model_through_the_worker_protocol` | Worker retention assertion fails after release. |
| `test_v16_fixes::test_d1_d21_cancel_note_is_separate_and_escape_rearms_after_expiry` | `IndexError: tuple index out of range` in callback arguments. |
| `test_v16_fixes::test_d13_idle_release_and_worker_stop_are_logged` | Expected worker-stop log text is absent. |
| `test_v17_fixes::test_at_most_one_deferred_listener_per_event` | Deferred-listener duplication assertion fails. |

Local receipts, complete logs, the untouched baseline, and the archive manifest
are retained in `temp/runpod_qwen3_diagnosis_20261027/`. The UI screenshot is
`output/playwright/qwen521_bf16.png`.

## Installer artifact

`SECourses_Video_Captioner_Pro_v1_9_3.zip` preserves the existing nine-file
installer distribution and contains the updated requirements. All members
were checked byte-for-byte against their source files and passed the ZIP CRC
check. Size: 45,397 bytes. SHA-256:

`33a7c693a3ca24576cd2ef2522b924a2a16fc68300f24b11bb91ee4f2e58819d`
