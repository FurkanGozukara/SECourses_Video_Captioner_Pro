# QA verification log - v1.9.0 (2026-09-06)

## Scope

Two requests from the Discord community:

1. An option to write only the `.txt` caption (no JSON or other sidecars),
   in the same folder as the source clips of a batch video + audio run.
2. An experimental way to load a custom checkpoint for the video caption
   model and for the audio caption model (for example a Q2/Q3 GGUF of
   Qwen3-Omni, an abliterated build, or Qwen2.5-VL), with an optional mmproj
   path, used whenever the application would otherwise load the regular model.

Implemented as **Save only .txt captions** (Processing Pipeline > 6.
Post-processing, mirrored in the Caption tab's Folder batch panel) and
**Override model loading (experimental)** (Caption > Model), see the README
sections *Plain .txt-only captions* and *Custom model overrides*.

## Chrome verification

Tested in Google Chrome with Gradio 6.26.0, Python 3.12.10, Windows 11, and an
RTX 5090. Durations in this log are smoke-test timings, not benchmarks.

| Check | Result |
| --- | --- |
| Folder batch panel | The new **Save only the .txt caption (skip JSON and other sidecar files)** checkbox renders directly under **Save outputs next to the source files** with its explanatory text. |
| Mirror -> main sync | Ticking the folder-panel copy set the real control in Processing Pipeline > 6 to on; its hint switched to "Each item or clip writes exactly one <name>.txt caption". |
| Preset load resets both | Loading the shipped **Dataset clips - video + audio captions (Qwen3-Omni + Whisper)** preset (stores the switch off) turned both controls off; ticking the main control turned the folder copy on again, and unticking it turned both off with the "Off:" hint. |
| txt-only batch, next to source | Two 5-second clips (`storm one.mp4`, `storm two.mp4`) with the Dataset preset, save next to source, txt-only on, Qwen3-Omni Instruct INT4 + Whisper large-v1. Result folder: only `storm one.txt` and `storm two.txt` beside the videos; no `.json`, no `video_caption/` or `audio_caption/`, no transcript sidecars. Each `.txt` holds the video caption followed by the Whisper text. `metadata.json`, `run_log.txt`, `summary.json`, and `captions_index.json` stayed in `outputs/batch_0147_qwen3`; metadata records `caption_txt_only: true`, `audio_captions: 2`, and only the merged path per item. |
| Override status line | Pointing **Video / main caption model override** at `models/custom_qwen2.5vl/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` showed "✓ ... (GGUF (llama.cpp), 4.7 GB) + mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf" (projector auto-detected beside the model). |
| Qwen2.5-VL GGUF override (cross-architecture) | Base variant Qwen3-Omni Instruct INT4 (Transformers). The run switched to llama.cpp, logged "Custom GGUF override", read `/props` (`modalities: vision`), warned "The loaded model has no audio encoder; the video's audio track was not sent", sent 11 frames, and finished in 16.5 s (`outputs/0148_qwen3`). Caption: "A car drives down the street." (8 tokens; the shipped Qwen3-Omni prompt presets are not tuned for this model). `model_info.override` records kind, path, mmproj, and size. |
| Qwen3-Omni Q4 GGUF override on a Transformers base | Same base variant with `models/qwen3_omni_instruct_gguf_q4/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf`. The cache logged the override switch, llama-server reported `modalities: vision, audio`, and the clip received a 105-token detailed night-street caption stopped by EOS (`outputs/0149_qwen3`, 2,916 prompt tokens). |
| Invalid path plus Whisper folder | With the video override set to `D:\nope\missing-model.gguf` and the audio override set to `models/whisper/large-v3-turbo`, the status line showed "✗ Video / main model: The override path does not exist: ..." and "✓ Audio caption model: large-v3-turbo (Whisper CTranslate2 model, 1.6 GB)". |
| Start refuses a bad override | Start Captioning stopped immediately with the progress title "Model override" and the message "Video / main model override: The override path does not exist: ..."; no run directory, worker, or download was created. |
| Transcribe tab note | With the Whisper folder override set, the Transcribe tab's Model section showed "⚠ Custom Whisper model folder in use: ...\models\whisper\large-v3-turbo" with the explanation that the dropdown is ignored until the field is cleared. |
| Clearing the fields | Emptying both override fields returned the status line to "No override set; the registered files of the selected variant are used." and hid the Transcribe note. |

## Automated verification

- New: `tests/test_v19_txt_only_and_overrides.py` (26 tests): PostSpec/JobSpec
  contract, path classification (GGUF file/folder/shards/projector selection,
  Transformers, CTranslate2, error messages), family compatibility rules,
  synthetic override variants, llama.cpp command/readiness/modalities and the
  audio-strip retry helper, model-cache keys, loader rejection of Whisper
  folders, Whisper params/engine override, sound-caption model choice,
  fail-fast validation, fake-session reload, txt-only single and split-layout
  runs, batch skip rules, and the UI registry/status helpers.
- `pytest tests -q`: **607 passed, 9 skipped, 9 failed** (135 s). The 9
  failures are pre-existing and reproduce on the untouched v1.8.1 tree (they
  cover the vp9 preview label text, ZIP wrapper scan, cancel-note timing,
  idle-release logging, the deferred-listener audit, the worker-protocol
  release test, the folder light-scan coverage text, and the model-change
  timing test); none touch the new code paths.
- Shipped presets: all 19 now store `caption_txt_only: false` and the four
  empty override paths, so `tests/test_whisper_presets.py` and
  `tests/test_prompts.py` keep their exact-key contracts.

## v1.9.1 follow-up (2026-09-07)

A second pass over v1.9.0 through `Windows_Run_Video_Captioner_Pro.bat` and
Google Chrome. Every v1.9.0 check above was repeated on a fresh instance and
passed: folder-panel mirror and Post-processing switch in
both directions, a txt-only batch of two clips beside their sources (only the
two `.txt` files; `metadata.json` records `caption_txt_only: true`), a regular
run still writing `.txt` + `.json`, override status lines for a valid GGUF, a
missing path, and a Whisper folder, Start refusing the missing path, a real
Qwen2.5-VL GGUF override run (`model_info.override` recorded, vision-only
modalities, audio skipped with a warning), a Chat turn on the resident
override server, a Transcribe run through the large-v3-turbo folder
(`model_path` in the run metadata), preset load resetting the override fields
and the txt-only switch, and Unload model releasing the llama-server.
Three cosmetic gaps were found and fixed in v1.9.1:

| Fix | Verified in Chrome |
| --- | --- |
| Chat model line ignored the override and still promised audio for a vision-only model. | With the Qwen2.5-VL GGUF set, the line reads "… · override Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf (GGUF (llama.cpp)) — multi-turn chat with text; video, audio, and image support follows the custom model's mmproj …". With `D:\nope\missing-model.gguf` it turns red ("the custom model override cannot be used: The override path does not exist …") and Send returns that message immediately with no model load. |
| Whisper log named the dropdown alias after loading the override folder. | Transcribe with the folder override logs "Whisper model large-v3-turbo (custom folder …\models\whisper\large-v3-turbo, replaces large-v1) loaded in 1.1s". |
| llama.cpp still-frames note said "plus separate audio" although audio was not sent. | The override run logs "Video used 11 chronological still frames with no audio input." directly after "The loaded model has no audio encoder; the video's audio track was not sent." |

Automated: three tests added to `tests/test_v19_txt_only_and_overrides.py`
(frames note, Whisper label and client fallback, chat note/mode with GGUF,
text-only GGUF, missing path, wrong family, and Transformers overrides).
`pytest tests -q`: **611 passed, 9 skipped, 8 failed** (110 s); the 8
failures are the same pre-existing ones and reproduce with identical
assertions on the untouched v1.8.1 tree.

## v1.9.2 follow-up (2026-09-12)

A user reported every Qwen3-Omni load failing on a fresh install with
`RuntimeError: Checkpoint load left meta tensors: ['visual.rotary_pos_emb.inv_freq',
'visual.rotary_pos_emb.original_inv_freq']` (runner → loader →
`convrot.apply_quantized_checkpoint`). Root cause: transformers 5.17.0 (the
current release; `transformers` was unpinned in the requirements) rewrote
`Qwen3OmniMoeVisionRotaryEmbedding` and `Qwen2_5OmniVisionRotaryEmbedding` to be
config driven (`compute_axial_rope_parameters`, an extra non-persistent
`original_inv_freq` buffer, no `dim`/`theta` attributes). The streaming loader
builds the model on the meta device and `_materialize_meta_buffers` only knew
the older recipes, so both vision buffers stayed on meta after the single-file
checkpoint was streamed. On GPUs that stage the towers on the CPU between
prefills the same gap surfaced earlier as `Cannot copy out of meta tensor; no
data!`.

Fix (`vcap/models/quant/convrot.py`): `_rope_init_function` resolves the recipe
the way the transformers constructor does (`compute_<rope_type>_rope_parameters`,
then `ROPE_INIT_FUNCTIONS` for scaled rope types, then
`compute_default_rope_parameters`, the 4.x `rope_init_fn`, any
`compute_*_rope_parameters`, and finally the `dim`/`theta` formula);
`original_inv_freq` is cloned from the rebuilt `inv_freq` regardless of buffer
order; the leftover-meta error names the owning module classes and the
transformers version. The installer requirements now request
`transformers>=5.17.0`.

Automated: `tests/test_rotary_meta_buffers.py` (7 tests: the 5.17 axial vision
class, the legacy `dim`/`theta` class, default and registry-scaled text rotary,
end-to-end `apply_quantized_checkpoint`, the diagnostic error, and a tiny real
Qwen3-Omni thinker built with the installed transformers). Verified on
transformers 5.16.1 and on a fresh venv (Python 3.12.10, torch 2.13.0+cu130,
transformers 5.17.0, accelerate 1.15.0). `pytest tests -q` on the fresh venv:
**618 passed, 9 skipped, 8 failed** (134 s); the 8 failures are the same
pre-existing UI/pipeline tests listed under v1.9.0 and fail identically with the
original `convrot.py` and with transformers 5.16.1.

Model matrix (`tools/smoke/caption_once.py --attention auto --max-new-tokens 48`,
RTX 5090 32 GB, a synthetic 6 s `testsrc2` clip with a tone track, tone WAV for
the Captioner family), every variant through the application's caption path with
load → caption → unload:

| Variant | Load | Result |
| --- | --- | --- |
| avocado_int4 / int8 / bf16 | 19.4 s / 20.6 s / 23.7 s | video caption OK |
| timechat_int4 / int8 / bf16 | 17.9 s / 19.1 s / 16.4 s | timestamped JSON caption OK |
| qwen3_omni_captioner_int4 / int8 / bf16 | 13.1 s / 21.7 s / 47.9 s | audio caption OK (counts the beeps) |
| qwen3_omni_instruct_int4 / int8 | 18.7 s / 37.2 s | colour-bar description OK |
| qwen3_omni_thinking_int4 / int8 | 25.4 s / 37.1 s | colour-bar description OK |

All 13 ran with FlashAttention 2 and released their VRAM on unload. The
`qwen3_omni_instruct_bf16` and `qwen3_omni_thinking_bf16` runs were stopped for
time: the 60 GB BF16 checkpoints page on this 94 GiB machine (the Captioner BF16
generated at 0.27 tok/s) and they share the architecture and size of the
Captioner BF16 that passed.
