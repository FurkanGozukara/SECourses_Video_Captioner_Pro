# Full application QA audit — 2026-09-05

Status: in progress. This log distinguishes real Chrome checks, automated
regressions, source review, and checks that require an unavailable environment.
No claim of universal correctness is made from a passing sample.

Baseline: commit `68fa49e`, app 1.7.0, Windows, Python 3.12.10,
Gradio 6.26.0, PyTorch 2.13.0+cu130, RTX 5090 32 GB and RTX 3090 24 GB.
Chrome opens the actual app at http://127.0.0.1:7860. Models are reused from
the existing distribution; QA media and outputs live in this checkout.

## Coverage inventory

Every row is pending until evidence is recorded below. Backend parameter
permutations are covered by targeted tests as well as representative UI jobs.

| Area | Features and states to exercise |
|---|---|
| Startup and layout | Fresh startup, default preset, local fonts, dark/light/system, reload, all ten tabs, narrow viewport, F4 and Open/Close All |
| Universal presets | Every shipped preset; mouse and keyboard select; Load; Save; overwrite; protected defaults; delete/cancel; Reset; Load Last Values; restart persistence |
| Inputs | Upload single/multiple/ZIP; local paths; folders; recursive toggle; mixed media; Unicode; filters; corrupt/missing/empty sources; preview and trim |
| Models | All 21 variants, readiness/download/verification, family constraints, attention choices, VRAM tiers, single/multiple GPUs, resident reuse/unload, block swap/offload, OOM recovery, compile |
| Prompts and decoding | Every task preset, family/modality filtering, variables, personal library CRUD, reset, sampling, seed, token/context caps, penalties, EOS, Thinking reasoning |
| Caption pipeline | Whole/fixed/scene/trainer segmentation; duration caps; overlap; trim; FPS/uniform/keyframe/adaptive sampling; resize; encoding; quality rejection; context carry; summary |
| Outputs and jobs | TXT/JSON/SRT/VTT/JSONL/reasoning; metadata/logs; collision safety; mirrored/next-to-source batch; skip/overwrite/failure continuation; progress; cancellation/retry; copy/open/ZIP/editor handoff; history |
| Whisper | Upload/path/folder; real transcription; language/translation; decoding/VAD/timestamps; all six outputs; model download/verify; cancellation/retry/ZIP/editor handoff |
| Transcript integration | Pre-caption Whisper, local transcript injection, custom wrapper, saved sidecars |
| Dataset clip captions | Generated/existing video captions; Whisper/Captioner sound sources; long audio windows; merge template; no-merge; overwrite/skip/idempotence; segment layout; editor parts |
| Chat | Qwen text/image/audio/video, multi-turn attachments, Thinking streams, GGUF and Transformers, TimeChat/AVoCaDO video Q&A; validation; stop confirmation; clear; JSON/Markdown save; context pruning |
| Caption Editor | Folder/metadata load; table/gallery; paging/filter/search/sort; preview; autosave/manual save; approve/reject/navigation/hotkeys; parts; regeneration/keep/revert/all; regex replace; bulk tools; export/ZIP; statistics |
| Dataset & Export | All trainer targets, custom FPS/frames, every bucket/resize policy, fitness plans, image/video/auto TOML, preset defaults, validation, overlap/copy/reencode sub-split |
| Global Settings | Paths, FFmpeg, all preferences, save/reload, folder actions, theme synchronization, shortcuts reference |
| Recover Settings | Metadata file/path/recent refresh; diff; apply all/model+prompt; path opt-in; malformed input; missing GPU |
| System & Models | Environment/GPU/RAM/disk, worker ping/unload, model inventory, verify/download/cancel, delete confirmation, llama.cpp install/runtime, update check |
| Distribution/docs | Windows launcher/install scripts, requirements, advertised behavior consistency; Linux-specific paths assessed separately |

## Evidence and findings

- Initial Chrome screenshot reviewed: desktop layout renders correctly, default
  model is Qwen3 Instruct INT4 on the detected 32 GB tier, existing model ready.
- Test environment initially lacked pytest; installed it in this checkout's venv.
- Baseline suite: 575 passed, 8 skipped, one failure from a test assuming the
  checkout contains a `models/` directory. Fixed the test to honor MODELS_DIR.
  Per the user's clarification, automated checks are not feature signoff:
  every feature result in this report must have a Chrome interaction.
- User hardware constraint: only RTX 5090 / GPU 0. Multi-GPU execution is
  excluded from live signoff.
- Real Chrome run `outputs/0002_qwen3`: Qwen3 Instruct INT4, quoted Windows
  video path, Whole, 160 tokens, prefix/suffix, TXT/JSON/SRT/VTT/JSONL,
  42-character subtitle wrapping. Completed in 19.49 s, 12.21 tok/s,
  context 5,431 / 32,768. Files inspected; Copy matched the visible caption;
  Results ZIP created; source-backed editor handoff worked.
- Earlier empty-media run `0001_qwen3` exposed unintended text fallback
  with a video preset. Fixed: empty media requires an intentional text task;
  folder batches always require matching files. Restarted Chrome check now
  shows Input required without loading a model.
- Editor typing with physical keys autosaved, but paste/value replacement
  did not update the review state or disk, and subsequent review actions
  restored old text. Changed edit tracking to value change with an equality
  guard for programmatic selection updates. Chrome regression pending.
- Unicode user preset `QA_full_audit_ü` saved. Restart still applied the
  shipped Default preset, preserving the user preset for Load Last Values.
- Editor paste regression passed in Chrome after restart; bulk actions kept the
  new text, and TXT/JSON agreed. Table and gallery page 2 showed items 26–27
  of a 27-item queue. Approved export: 1 media pair; excluding caption-only
  items reported no-media 1; including them exported 2 items. Statistics
  displayed character/token ranges, duplicate check, trigger coverage and words.
- Found additional deferred-event conflicts involving browser-only handlers
  (model/task prompt markers and editor row selection); they now use multiple
  rather than deferred execution. Regression invariant includes JS-only events.
- Whisper uploaded JFK WAV: real CUDA large-v1, 6.6 s total, 12.8× realtime,
  correct speech, two confidence rows, all six outputs, copy and ZIP verified.
- Whisper VAD + batched + word highlighting + 2–9 s trim failed with
  `AssertionError: non-negative timestamp expected`. The adapter supplied
  sample offsets while the installed public batched API expects seconds.
  Fixed and Chrome Retry failed passed in `0005_whisper`: 4.1 s total,
  17.4× realtime. A 165-second standard/VAD/highlighting run also passed.
- A 1,650-second transcription completed in 71.1 s, but clicking Cancel
  exposed a disappearing confirmation bar: every progress yield hid it.
  Fixed stream updates to preserve an armed/cancelled button and confirmation.
  Chrome cancellation regression pending after restart.
- F9 after choosing a dropdown was ignored by the global dropdown guard.
  F4/F9 now pass that guard. Escape/arrows remain reserved for dropdown input.
  Browser regression pending after restart.
- Whisper old Results ZIP remained visible after subsequent failed/successful
  runs. Added clearing on Start, F9, and Retry. Browser regression pending.
- Cancellation regression passed in Chrome (`0008_whisper`): confirmation
  remained visible during segment progress and Yes stopped the 1,650-second job
  after 29.3 s. GPU 0 memory returned to idle.
- Recursive Whisper batch (`batch_0009_whisper`) transcribed nested speech and
  Unicode video, reporting the intentionally corrupt MP4 as failed while
  continuing. `batch_0010_whisper` skipped both existing transcripts; the old
  ZIP disappeared on Start. `batch_0011_whisper` respected a limit of two and
  wrote speech sidecars next to its source. Explicit overwrite rerun
  `batch_0012_whisper` replaced those transcripts successfully.
- Whisper-to-editor handoff loaded the source folder and playable audio.
  Autosave-off plus Ctrl+S saved an edited transcript to its real TXT file.
- F4 works with dropdown focus. A second F9 cause was discovered: Whisper
  shortcut buttons used visible=False, removing them from the rendered DOM.
  Changed to visible="hidden", matching the working Caption/Editor buttons.
- After restart, F9 with Device dropdown focus started real RTX 5090 Whisper
  inference (`0013_whisper`); Escape opened confirmation, Keep running resumed,
  and a second Escape + Yes cancelled successfully (18.5 s total).
- Dataset fitness exercised all five targets (Wan, Hunyuan, LTX, MiniMax,
  Custom), all three resolution buckets and four resize policies. The 1-second
  portrait clip was rejected for LTX's 121-frame minimum, and accepted with
  Custom 20 frames at 30 FPS. Portrait crop and letterbox geometry displayed
  real crop/padding; area and keep-AR previews were inspected.
- LTX TOML generated through Chrome and matched 121 frames, 25 FPS,
  832×480, cache path and dataset path. Invalid frame text produced feedback.
- Real sub-splits produced five clips across two videos in both copy mode
  (precise fallback verified in split_manifest.json) with 0.5 s overlap, and
  precise mode with 1 s overlap. Source and actual frame counts match manifests.
- Chrome reproduced an image-export bug in a mixed image/video folder:
  "No image dataset folders found" despite a PNG being present. Fixed the
  dataset-kind filter to accept mixed folders for either explicit media kind.
  Browser regression pending restart.
- Mixed-folder image export regression passed after restart in Chrome.
  `outputs/qa_image.toml` contains image_directory, num_repeats=3, and
  bucket_no_upscale=true. MiniMax export matched 124 frames and 24 FPS.
  All ten split outputs across both modes have exact expected/actual frame
  counts according to their manifests.
- README corrected two advertising inconsistencies: last-used presets require
  Load Last Values, and caption cancellation confirmation lasts eight seconds.
