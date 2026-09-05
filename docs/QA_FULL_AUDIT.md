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
- Qwen3 Instruct INT4 real Chat: empty Send validated; Enter sent a text turn,
  answered 12×7 correctly (84), and an image attached to the next turn was
  recognized as a TV color-bar test pattern. The model remembered ORCHID from
  the first turn. Image answer: 72 tokens, 12.42 tok/s, EOS, context 573/32768.
- Global Settings light-theme screenshot reviewed; top theme toggle restored
  Dark and synchronized the radio. Keyboard reference rendered correctly.
- Browser automation detached during a chat file-chooser wait (infrastructure
  timeout, not an app failure). Continuing in a fresh Chrome tab; prior chat
  was not saved before the interruption.
- Further Chat checks: uploaded JFK audio produced the correct transcript;
  a quoted Unicode video path on the next turn produced an accurate nighttime
  storm description. Saved JSON/Markdown in `0014_chat_qwen3` retain messages,
  attachments, parameters, EOS/timing/context statistics (video: 57 tokens,
  11.90 tok/s, 5521/32768 context).
- Found two real Chrome bugs: Copy last answer wrote `[object Object]` because
  Gradio 6 sends normalized text blocks; the Chatbot toolbar Clear erased only
  the visual messages, and prior turns returned on the next Send. Fixed text
  extraction and wired both Clear controls to the same state reset, including
  the token counters. Browser regression pending restart.
- Real Chat stop passed: double-click confirmation stopped a story at 806
  generated tokens / 68.4 s, finish=cancelled, retaining the resident model.
  Cancelled conversation saved in `0015_chat_qwen3`.
- Chat fixes passed after restart with Qwen3 Instruct GGUF Q4 on GPU 0:
  Copy last answer returned SAVED exactly; toolbar Clear reset the displayed
  history and token/context metrics; the next reply was NO HISTORY with only
  the new pair visible and 28 prompt tokens. First GGUF conversation saved in
  `0016_chat_qwen3` (72.78 tok/s, EOS).
- GGUF Q4 accepted image+audio together and transcribed the speech, but wrongly
  described the image as also containing the speaker. An isolated image run
  correctly identified the TV test pattern (with extra speculative wording).
  This is recorded as a model-output quality failure, not a successful accuracy
  check. The subsequent GGUF video answer accurately described the storm.
- Qwen3 Instruct GGUF Q8 passed real video+audio chat: 53 tokens, 49.71 tok/s,
  EOS; 32-frame cap and chronological-frame/audio warning displayed.
- Thinking GGUF Q4 showed a collapsible thought and separate Reasoning panel;
  calculated 3/10 correctly, reported a 768-token length stop, and saved 1,525
  reasoning characters separately from the answer (`0019_chat_qwen3`). Turning
  thinking off yielded just the fraction; Copy returned the answer's Markdown.
- Thinking GGUF Q8 passed storm-video reasoning: 71 tokens, 45.28 tok/s, EOS,
  final answer "Thunderstorm with lightning and rain." Same-family precision
  changes retained the conversation; cross-family selection cleared history.
  Found stale counters on cross-family changes; fixed, pending restart check.
- System UI copied a complete environment report, detected all 21 downloaded
  caption variants and CUDA llama.cpp b10621, reported the local Git commits
  ahead of origin, showed loaded Thinking Q8 / fit, and verified TimeChat INT4
  (16 files / 6.0 GiB, sizes and SHA). Runtime repair reused the external
  executable successfully. System unload completed before Whisper tests.
- Fresh Whisper base download (not present initially) was cancelled mid-progress
  through Chrome and resumed to verified completion. Its RTX 5090 INT8/FP16
  transcription passed (`0022_whisper`). BF16 + automatic language detection
  + initial prompt/hotwords/repeated prompt + beam/best-of 3/patience 1.2 and
  96-token window ran successfully (`0023_whisper`, detected en at 0.9453).
  INT8/BF16 and FP32 also executed successfully. Accuracy of the small base
  model varied: substitution and incomplete speech were observed, unlike the
  accurate large-v1 default. These runs establish compatibility, not equal quality.
- Base Whisper precision sweep also passed INT8 and FP16 with timestamped
  filenames (`0026`–`0028`). Reducing repetition penalty from the large-v1
  preset's 1.2 to 1.0 restored the complete JFK sentence in `0028_whisper`.
  This is a parameter/model accuracy interaction, not lost UI output.
- TimeChat BF16 (`0029_timechat`) ran fixed 10-second clips with 0.5-second
  overlap, uniform eight-frame sampling, and saved clips. Three clips completed
  in 50.4 s; about 32.4 tok/s, 384 total tokens. The deliberately low 128-token
  limit truncated each model-native JSON answer before Wan paragraph conversion.
  Uniform thumbnails showed eight correct times from 0.00 to 19.99 seconds.
  Chrome played saved clip 2 (9.976633 s) to its end without media errors.
- The Clips tab incorrectly retained its empty-state message above the working
  gallery. Fixed the final update to clear the message's value as well as its
  visibility; browser regression pending restart.
- TimeChat BF16 Chat answered night (8 tokens, 31.16 tok/s) and storm weather
  (18 tokens, 30.80 tok/s). Second Send replaced the first exchange as advertised.
  Both conversations saved. Switching to AVoCaDO cleared messages and all three
  counters, passing the cross-family counter regression after restart.
- AVoCaDO BF16 Chat produced an accurate storm/camera description but ignored
  the requested one-sentence brevity and reached 128 tokens. Automatic block
  swap was active (1,024 layer transfers), about 3.66 tok/s. Conversation saved.
- Corrected README's selectable CUDA-graphs claim: Chrome exposes Inductor
  default and max-autotune without explicit graph replay. Legacy fallback is
  still present internally.
- Corrected three Distil-Whisper large catalogue/README labels from multilingual
  to English recognition, matching the upstream model cards:
  [v2](https://huggingface.co/distil-whisper/distil-large-v2),
  [v3](https://huggingface.co/distil-whisper/distil-large-v3),
  [v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5).
  Catalogue display regression pending restart.
