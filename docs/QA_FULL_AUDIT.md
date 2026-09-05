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
- TimeChat INT8 scene mode (`0033_timechat`) detected and captioned five scenes
  at 0–5.105, 5.105–9.510, 9.510–11.812, 11.812–16.750, 16.750–20.020 s.
  Completed in 124.2 s, final decode 26.22 tok/s, EOS. The 1,024-token budget
  permitted native structured output conversion to motion paragraphs.
  Saved five clips; the corrected empty-state message disappeared in Chrome.
- TimeChat INT4 + explicit SDPA + Inductor default ran in Chrome
  (`0034_timechat`): trim 0–5 s, keyframe preview, normalize, Wan trainer mode.
  Completed 2 segments in 142.3 s, final decode 21.22 tok/s, 512-token caps;
  no runtime compilation fallback. Keyframe preview showed 0.00 and 4.14 s.
  Quality failure: first caption invented a close-up person wearing a hoodie.
  Trainer splitting also produced an overlap tail of only 0.5 s; inspect trim
  and frame-quantization behavior before final signoff.
- Editor regex `item0[12]` narrowed the 27-item queue to two. Approving item01
  advanced to item02; Reject set item02 rejected. Rejected filter showed only
  item02. Invalid regex `[` produced an explicit unterminated-character-set
  message while retaining the existing queue.
- All three corrected Distil-Whisper large descriptions displayed English
  speech recognition in Chrome after restart.
- AVoCaDO INT8 + explicit Sage attention + adaptive sampling + video audio off
  completed `0035_avocado` in 35.6 s, 176 tokens, 31.39 tok/s, EOS. Preview
  selected 5.00, 7.84, 12.15, 13.55 s within the selected 5–15 s range.
- Confirmed a real input-trim bug: fast copying a requested 5–15 s range kept
  keyframe pre-roll, yielding an 11.04 s clip labelled 5–16.04 s. This also
  let codec padding create a redundant trainer overlap tail. Input trims now
  re-encode precisely, and planning caps duration to the requested window;
  later clip splitting retains the selected fast/precise behavior.
- Chrome regression `0036_avocado`: AVoCaDO INT4, xFormers, audiovisual adaptive
  sampling; 215 tokens, 33.30 tok/s, EOS, 34.2 s total. Saved requested 5–15 s
  clip has 300 frames / 10.01 s (within one source frame), and a coherent storm
  audio/visual caption. `0037_avocado` repeated the five-second Wan trainer
  case: one clip labelled 0–5.00 s, no redundant overlap tail; 6.5 s total.
- User preset QA_trim_regression persisted across restart and immediately
  restored the saved settings, including the 5–15 s trim. Local media paths
  were entered again explicitly for the regression.
- Captioner INT4 ran the Audio SFX shipped preset on JFK speech in
  `0038_qwen3`: 256-token limit, 10.92 tok/s, 48.2 s total. Captioner Chat
  rejected Hello with the model-specific no-chat-mode explanation.
- Captioner GGUF Q4 (`0039_qwen3`) ignored a supplied instruction exactly as
  its prompt-free contract requires and logged the ignored prompt. 512-token
  cap, 265.19 tok/s, 20.6 s total. Personal prompt QA_prompt_ü passed Save,
  changed-text/Load restoration (both system and user text), and Delete.
- Captioner GGUF Q8 (`0040_qwen3`) completed a 20-second thunderstorm WAV at
  32.79 tok/s, EOS, 42.9 s total. It recognized thunder/rain, but invented
  recording characteristics and called the known mono WAV stereo. This is an
  explicit model-output quality failure despite successful inference.
- Editor flags survived app restart. Page 2 showed items 26–27 and Prev page
  returned to 1–25. Changing Per page to 10 and Gallery produced ten cards,
  Page 1/3, showing 1–10 of 27.
- Found shared input-tab state mismatch: changing main tabs restored the
  visible Upload panel although backend mode and preview still used File path.
  The shared select handler now synchronizes the Tabs selected property with
  its mode state. Chrome regressions after restart passed for both Caption
  (storm WAV) and Transcribe (French FLAC) across main-tab navigation.
- Real French audiobook fixture from Hugging Face's public audio examples:
  [monte_cristo.flac](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/blob/main/monte_cristo.flac),
  16.75 s / 16 kHz mono. Downloaded only into ignored QA fixtures.
  large-v1 French transcription (`0041_whisper`, 7.0 s) and English translation
  (`0042`, `0043`) executed on CUDA. Translation grammar improved at repetition
  penalty 1.0, but semantic errors remained (e.g. reins rendered as queens).
  Full large-v3 automatically downloaded 3.09 GB on Start and translated the
  sample in `0044_whisper` (40.0 s including download); semantic errors and an
  invented ending remained. Translation execution passes; accuracy does not.
- Found Global Settings' recursive default affected Caption only. Transcribe
  and Caption Editor ignored it. Fixed both initial load and immediate changes
  to reach all three controls. Chrome restart showed both checked; switching
  the global preference off unchecked both immediately.
- Saved a new global output directory and Save every processed file. Real
  Captioner INT8 run `qa_global_runs/0001_qwen3` wrote there and retained its
  1–10 s trimmed WAV in `storm_processed/trimmed.wav` (288,078 bytes).
  Completed in 121.1 s, 160 tokens, 2.16 tok/s with automatic block swap.
  Its audio caption again incorrectly described mono content as stereo.
  Restored original output directory and the two preferences afterwards.
- Global empty-directory validation rejected an empty output path and retained
  the prior app_settings.json. Empty-string browser fill was unreliable in this
  interaction; keyboard Ctrl+A/Backspace visibly cleared it before submission.
- Restart-only path test pending: saved temporary directory
  `temp/qa_runtime_temp` and FFmpeg path `C:/ffmpeg_latest`; will verify a real
  run and restore them. Next launch also omits VCAP_MODELS_DIR so model discovery
  must use the persisted Global Settings models directory.

### Native BF16 loading and interrupted-job recovery

- Qwen3 Captioner BF16 with explicit Flash Attention 2 and a 128-token cap
  crashed three times during the first checkpoint tensor read (`0045`–`0047`).
  Worker exit code was 3221225477 / 0xC0000005. Enabling faulthandler in the
  worker captured the native stack at `torch.storage.__getitem__`, called by
  safetensors `get_tensor`; this preceded GPU inference.
- Fixed Windows checkpoint loading to use safetensors 0.8's `pread` backend,
  avoiding PyTorch's writable map of the complete 63.4 GB checkpoint. Kept
  Linux's mmap behavior and documented the minimum dependency and upgrade
  command. Upstream provides this reader specifically for memory pressure and
  slow mmap scenarios ([safetensors PR 760](https://github.com/safetensors/safetensors/pull/760)).
- The native crash also left the item marked Running and disabled Retry failed.
  Interrupted jobs now finalize queued/running rows, preserve run and output
  destinations, and offer retry for failed paths. Completed rows and artifacts
  remain available. Cancellation uses the same finalization with Cancelled status.
  Crash logs now briefly wait for process termination so their header contains
  the actual exit code instead of None.
- Chrome regression `0047_qwen3` reproduced the real native crash with the new
  failure handler: Failed row, correct native exit code, enabled Retry failed.
  Clicking Retry failed after the loader change reran the same file, BF16
  model, and Flash Attention 2 setting without restarting the app or browser.
- Retry `0048_qwen3` completed: 1 done, 0 failed, 410.8 seconds total,
  128 tokens at 0.36 tok/s. Model loading took 51.2 seconds and reported
  25.44 GiB peak VRAM, 17 resident layers, 31 swapped layers / 35.98 GiB pinned
  RAM, and two staging slots. Generation transferred 4,605.5 GiB through block
  swap. This proves BF16 execution with RAM offload on GPU 0, not full residency.
  Saved TXT/JSON/metadata exist. The output again falsely calls the mono WAV
  stereo and truncates at the deliberately small token cap; these are explicit
  output-quality limitations.
- The successful run also validates persisted custom temporary/FFmpeg paths
  and models discovery with no VCAP_MODELS_DIR environment override. Restoring
  the temporary directory and automatic FFmpeg discovery remains pending.
- Additional Chrome checks: Changelog expanded the initial release and collapsed
  v1.7.0 correctly. Editor Min/Max chars=31 selected items 10–27 (18 matches).
  Over token limit selected all 27 eight-token captions at limit 7, showed warning
  flags, and selected zero at limit 8. A temporary narrow viewport reported no
  horizontal document overflow; screenshot capture was unsuitable for visual
  signoff, so responsive layout coverage remains incomplete. Viewport reset.
