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

Tested a fresh instance of the app (`secourses_app.py --server-port 62155`) in
installed Google Chrome with Gradio 6.26.0, Python 3.12.10, Windows 11. GPU 0
(RTX 5090, 31.8 GB) had about 12 GB free because other processes were resident,
so every model below ran partially from host memory; timings are therefore
slow and are not performance numbers.

| Check | Result |
| --- | --- |
| Folder batch panel | The new **Save only the .txt caption (skip JSON and other sidecar files)** checkbox renders directly under **Save outputs next to the source files** with its explanatory text. |
| Mirror -> main sync | Ticking the folder-panel copy set the real control in Processing Pipeline > 6 to on; its hint switched to "Each item or clip writes exactly one <name>.txt caption". |
| Preset load resets both | Loading the shipped **Dataset clips - video + audio captions (Qwen3-Omni + Whisper)** preset (stores the switch off) turned both controls off; ticking the main control turned the folder copy on again, and unticking it turned both off with the "Off:" hint. |
| txt-only batch, next to source | Two 5-second clips (`storm one.mp4`, `storm two.mp4`) with the Dataset preset, save next to source, txt-only on, Qwen3-Omni Instruct INT4 + Whisper large-v1. Result folder: only `storm one.txt` and `storm two.txt` beside the videos; no `.json`, no `video_caption/` or `audio_caption/`, no transcript sidecars. Each `.txt` holds the video caption followed by the Whisper text. `metadata.json`, `run_log.txt`, `summary.json`, and `captions_index.json` stayed in `outputs/batch_0147_qwen3`; metadata records `caption_txt_only: true`, `audio_captions: 2`, and only the merged path per item. |
| Override status line | Pointing **Video / main caption model override** at `models/custom_qwen2.5vl/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` showed "✓ ... (GGUF (llama.cpp), 4.7 GB) + mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf" (projector auto-detected beside the model). |
| Qwen2.5-VL GGUF override (cross-architecture) | Base variant Qwen3-Omni Instruct INT4 (Transformers). The run switched to llama.cpp, logged "Custom GGUF override", read `/props` (`modalities: vision`), warned "The loaded model has no audio encoder; the video's audio track was not sent", sent 11 frames, and finished in 16.5 s (`outputs/0148_qwen3`). Caption: "A car drives down the street." (8 tokens; the shipped Qwen3-Omni prompt presets are not tuned for this model). `model_info.override` records kind, path, mmproj, and size. |
| Qwen3-Omni Q4 GGUF override on a Transformers base | Same base variant with `models/qwen3_omni_instruct_gguf_q4/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf`. The cache logged the override switch, llama-server reported `modalities: vision, audio`, and the clip received a 105-token detailed night-street caption stopped by EOS (`outputs/0149_qwen3`, 2,916 prompt tokens, 0.58 tok/s with most weights fitted to host memory on the crowded GPU). |
| Invalid path plus Whisper folder | With the video override set to `D:\nope\missing-model.gguf` and the audio override set to `models/whisper/large-v3-turbo`, the status line showed "✗ Video / main model: The override path does not exist: ..." and "✓ Audio caption model: large-v3-turbo (Whisper CTranslate2 model, 1.6 GB)". |
| Start refuses a bad override | Start Captioning stopped immediately with the progress title "Model override" and the message "Video / main model override: The override path does not exist: ..."; no run directory, worker, or download was created. |
| Transcribe tab note | With the Whisper folder override set, the Transcribe tab's Model section showed "⚠ Custom Whisper model folder in use: ...\models\whisper\large-v3-turbo" with the explanation that the dropdown is ignored until the field is cleared. |
| Clearing the fields | Emptying both override fields returned the status line to "No override set; the registered files of the selected variant are used." and hid the Transcribe note. |

Screenshots were reviewed live during the session; browser diagnostics are not
retained for this version.

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
