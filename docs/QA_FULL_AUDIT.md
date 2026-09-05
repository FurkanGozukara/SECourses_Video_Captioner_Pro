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
| Distribution/docs | Use the user's fresh installation as baseline; flag setup failures encountered during app use and check advertised behavior. Fresh-install and launcher reruns excluded by the user's follow-up instruction. |

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
- Restart-only path test: saved temporary directory `temp/qa_runtime_temp`
  and FFmpeg path `C:/ffmpeg_latest`. Run `0048_qwen3` subsequently verified
  them with no VCAP_MODELS_DIR override. Restored `temp` and automatic FFmpeg
  discovery through Chrome, saved, restarted, and verified System environment.

### Native BF16 loading and interrupted-job recovery

- Qwen3 Captioner BF16 with explicit Flash Attention 2 and a 128-token cap
  crashed three times during the first checkpoint tensor read (`0045`–`0047`).
  Worker exit code was 3221225477 / 0xC0000005. Enabling faulthandler in the
  worker captured the native stack at `torch.storage.__getitem__`, called by
  safetensors `get_tensor`; this preceded GPU inference.
- Initially changed Windows checkpoint loading to safetensors 0.8's `pread` backend,
  avoiding PyTorch's writable map of the complete 63.4 GB checkpoint. Kept
  Linux's mmap behavior and documented the minimum dependency and upgrade
  command. Upstream provides this reader specifically for memory pressure and
  slow mmap scenarios ([safetensors PR 760](https://github.com/safetensors/safetensors/pull/760)).
  The model-switch stress check below exposed a remaining header-map reservation;
  the final reader now uses unmapped file I/O for headers and tensors.
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
  and models discovery with no VCAP_MODELS_DIR environment override. The
  temporary directory and automatic FFmpeg discovery were subsequently restored.
- Additional Chrome checks: Changelog expanded the initial release and collapsed
  v1.7.0 correctly. Editor Min/Max chars=31 selected items 10–27 (18 matches).
  Over token limit selected all 27 eight-token captions at limit 7, showed warning
  flags, and selected zero at limit 8. A temporary narrow viewport reported no
  horizontal document overflow; screenshot capture was unsuitable for visual
  signoff, so responsive layout coverage remains incomplete. Viewport reset.

### Checkpoint streaming, unload memory, and remaining Qwen variants

- Instruct BF16 with explicit eager attention completed `0049_qwen3` on the
  test-pattern image: 15 tokens, EOS, correct short caption, 97.95 s total,
  53.5 s loading, 0.36 tok/s, 17 resident and 31 swapped decoder layers.
- Switching to Instruct INT8 released 25.78 GiB VRAM, but private host memory
  remained 42.76 GiB. The following INT8 load (`0050_qwen3`) failed with Windows
  error 1455 at checkpoint open and retained about 23 GiB of GPU allocation.
  Terminated that QA worker after the failed job to release its resources.
- Upstream safetensors 0.8 source creates a temporary copy-on-write mapping of
  the complete file even with `backend="pread"`, to parse the header. Replaced
  the Windows path with a bounded reader that validates the header, dtype,
  shapes, offsets, complete coverage, and file bounds, then reads each tensor
  directly into owned Torch memory. Linux continues to use safetensors mmap.
  See [upstream open implementation](https://github.com/safetensors/safetensors/blob/v0.8.0/bindings/python/src/lib.rs).
- Chrome Retry failed completed `0051_qwen3` with the same Instruct INT8 model,
  12 manually swapped layers, three slots, and pinning disabled: 47.79 s total,
  15 tokens, EOS, 1.29 tok/s, correct test-pattern caption. Explicit unload freed
  26.28 GiB VRAM, but host memory still remained at 13.40 GiB; investigation
  continues. A removed block-swap manager now drops its packs and layer refs.
- Thinking INT8 with automatic block swap completed `0052_qwen3`: 102.61 s,
  128 tokens at 1.95 tok/s. The limit interrupted its reasoning, leaving an
  empty final caption and a populated Reasoning tab. This is an incomplete
  output, not a correct caption. Unload freed 26.88 GiB VRAM, but host private
  memory remained 13.18 GiB. Temporary diagnostics found zero surviving Python
  host-owner objects; checking remaining tensor storage/native memory next.
- Fixed the plan preview/log to say pageable RAM when pinning is disabled,
  report zero pinned bytes, and omit the inapplicable pin-budget warning.
  Chrome displayed 10.5 GiB pageable RAM with pinning disabled.
- Added cleanup of completed traceback frames after failed model loads, so
  exception logging cannot keep partially loaded GPU/RAM tensors alive.
  Further real-browser load-failure verification remains pending.
- The user requested QA instances be terminated after use. QA workers are now
  terminated after their completed checks; the QA app will also be stopped
  when the complete audit finishes. GPU 1 remains outside the test scope.
- Thinking INT8 with thinking disabled and the short-image task completed
  `0053_qwen3`: correct caption, 37 tokens, EOS, 1.67 tok/s, 59.39 s total.
  Keep model loaded off invoked unload automatically. Diagnostics then found
  zero live large CPU tensor storages, yet private host memory remained 12.29 GiB.
  Removed the temporary diagnostics after capturing this evidence.
- Subprocess cleanup now stops the empty worker after explicit/model-selection
  unload, idle expiry, and jobs/chat turns with Keep model loaded disabled.
  This releases native allocations that outlive Python tensor cleanup. Models
  remain reusable between jobs when Keep model loaded is on.
- Thinking BF16 completed `0054_qwen3` with the new unmapped reader: 109.83 s
  total, 49.2 s load, 25.44 GiB peak VRAM, 19 tokens at 0.35 tok/s, EOS, correct
  test-pattern caption. Automatic placement kept 17 layers resident and 31
  swapped (35.98 GiB pinned). Keep model loaded off stopped worker PID 17672
  and wrapper 79548 after the run; both disappeared and GPU 0 returned to
  2,159 MiB. This releases the 39.59 GiB of private host memory still reported
  immediately after the in-worker unload.
- Thinking INT4 completed `0055_qwen3`: 31.60 s, 22 tokens, 4.71 tok/s, EOS.
  This completes real Chrome inference coverage of all 21 caption variants.
  Used KV cache off, no-repeat n-gram size 3, thinking off, and short-image task.
  Regex replacement changed "test pattern" to "calibration chart"; whole-word
  `color` did not affect `colorful`/`colors`. Prefix/suffix joined with ` | `
  correctly. TXT, JSON, and Dataset JSONL contain the processed caption.
  SRT/VTT were selected but not produced for this image input; their timestamp
  wrapping remains covered by the earlier video test, not this image run.
  Explicit Unload freed 17.08 GiB VRAM and stopped the worker automatically.
- Run `0056_qwen3` verified an empty join separator and Maximum caption
  characters 80: the result had 77 characters and ended at a word boundary.
  It completed in 27.55 s, 18 tokens, 7.58 tok/s with KV cache enabled.
  Idle unload set to 0.1 minutes failed to trigger after the job. Source review
  found browser reload called permanent client shutdown, killing its idle
  monitor. Browser disconnect now releases session resources while keeping the
  monitor alive; full shutdown remains separate.
- Chrome regression after a real page reload completed `0057_qwen3` in 34.57 s,
  33 tokens, EOS, correct image caption. With idle unload at 0.1 minutes, the
  job ended at 20:19:40, unload completed at 20:19:47, and the worker stopped
  by 20:19:49. The idle log confirmed release; the app had no worker child
  process afterward. Explicit unload, post-job automatic unload, and idle
  unload after reload have now passed with actual model jobs.
- Load Last Values after a fresh page restored the last-used universal preset
  (Audio SFX captions), matching the documented behavior; it restores that
  preset rather than unsaved individual controls.
- Per the user's follow-up, fresh-install and launcher reruns are excluded.
  The initial missing pytest package was a development dependency installed
  for the early baseline suite; no missing required runtime package has been
  found during the real Chrome checks so far.

### Mixed caption batch and Whisper integration

- `batch_0058_qwen3` used Instruct INT4, recursive folder scan, glob list
  `*.mp4;*.wav;*.png`, and video/audio/image kinds with Text excluded. Chrome
  counted four inputs: a corrupt MP4, nested PNG/WAV, and `storm ü.mp4`.
  Output folder `qa_caption_batch_ü` preserved nested paths and Unicode names.
  Result: 3 done, 0 failed, 1 unsupported in 87.47 s. The corrupt MP4 was
  identified as unreadable before model processing and did not stop the batch.
- Enabled stage 7 using Whisper base, English, repetition penalty 1,
  SubRip/TXT/LRC sidecars, suffix `_qa_speech`, and custom injection wrapper
  `[QA_SPEECH]\n{{TRANSCRIPT}}`. Image transcription was skipped. Speech sidecars
  contained the full JFK sentence. The storm produced zero speech segments and
  empty sidecars; the log explicitly injected the no-speech result for its clip.
  Caption and transcript filenames remained distinct. The worker exited when
  the completed batch unloaded its model.
- Found a prompt-selection mismatch: metadata recorded `image_short_caption`
  while the user prompt was the selected joint-media template. The image used
  that template and truncated at 256 tokens; the audio/video received the
  short-image preset's per-modality fallbacks. This makes recorded settings
  misleading and can change the requested task across a mixed batch.
- Prompt context callbacks now read the latest validated selection from session
  state instead of a dropdown value captured by an earlier event. Validation
  also synchronizes the dropdown choices and value. Chrome regression switched
  Thinking INT4 to Instruct INT4, immediately selected joint-media description,
  navigated through Pipeline and Transcribe, then changed to a filtered folder
  batch. The joint preset and its prompt remained selected throughout.
- Saved regression `batch_0059_qwen3` completed one nested PNG in 30.53 s with
  `*.png` filtering and Limit items 1. Metadata now contains
  `prompt_preset_id: qwen3_joint_describe` and the matching native prompt.
  Used a deliberate 16-token cap to check settings execution; its partial
  caption is not a quality benchmark. Automatic worker shutdown also passed.

### Caption folder skipping, ZIP feedback, and immediate task changes

- `batch_0060_qwen3` skipped the existing filtered PNG in 0.01 s. The log
  confirmed no caption model was loaded. `batch_0061_qwen3` skipped all three
  completed image/audio/video outputs and reported the corrupt MP4 as unsupported
  in 0.11 s, again without loading a caption model.
- Chrome ZIP upload selected the deepest single media folder, preserved the
  UTF-8 filename `renk_ü.png`, and extracted below `outputs/uploaded_batches`.
  A harmless test archive included parent traversal, a Windows absolute path,
  and macOS metadata entries; all three were skipped and neither marker escaped
  the extraction directory. An earlier malformed fixture accidentally contained
  `?` in its filename due to PowerShell pipe encoding and correctly showed an
  extraction error; it was replaced by the intended UTF-8 fixture.
- Found two feedback bugs: literal `<stem>` was interpreted as an HTML tag,
  hiding the rest of the scan summary, and an automatic rescan immediately
  replaced the ZIP extraction report. Escaped the placeholder and gave ZIP
  results their own status component. After restart, Chrome showed the full
  coverage/overwrite summary and persistent extraction/skipped-entry details.
  Clearing the ZIP also cleared its extraction report.
  Simplified the glob help to one example so Gradio no longer treats paired
  wildcard asterisks as emphasis; Chrome now visibly shows `*.mp4`.
- `batch_0062_qwen3` saved PNG captions beside the extracted source, with run
  metadata/logs in the numbered batch directory; completed in 38.17 s. It
  exposed a second prompt race: immediate Start captured the selected short-image
  task ID with the preceding joint-media prompt text, before the render callback
  finished. This is separate from the model/input-context race fixed earlier.
- Start now captures automatic-prompt provenance alongside the visible fields
  and renders automatic fields from the selected task and variables. Manually
  edited fields and personally loaded prompts are preserved. Chrome repeated
  task selection followed immediately by a 64-token limit and Start:
  `batch_0063_qwen3` saved matching `image_short_caption` ID/text, generated a
  correct 17-token TV test-pattern caption with EOS, and completed in 27.97 s
  at 12.47 tok/s. The worker exited after unloading.
- `batch_0064_qwen3` used a manually edited prompt, sampling enabled,
  temperature 0.6, top-p 0.9, top-k 20, and seed 1234. The image completed with
  the requested `QA_MANUAL` prefix (41 tokens, EOS); a video with valid stream
  headers but deliberately truncated frame data failed during PyAV decoding.
  Chrome showed 1 done / 1 failed and enabled Retry; metadata preserved the
  manual prompt and all sampling values. Total 30.67 s.
- Repaired only the QA video at the same path, changed to Whole / 8 frames and
  a compatible custom task, then clicked Retry. `batch_0065_qwen3` processed
  only that video and produced a correct storm-street caption, 26 tokens with
  EOS, 36.39 s total. The prior image caption's SHA-256 and modification time
  were unchanged. The worker exited after both the partial failure and retry.
- Added an identical image as a third file and set Limit items to 1 with
  overwrite off. `batch_0066_qwen3` skipped the two existing outputs and
  captioned the pending image, confirming skipped files do not consume the
  limit. Repeating the earlier manual prompt and seed 1234 after a new worker
  load produced identical 41-token text and the same SHA-256. Total 37.79 s;
  Chrome showed 1 done / 2 skipped and the worker exited.

### Scene preview controls

- An eight-second QA fixture contains asymmetric fade-out/fade-in around a
  two-second black interval. With Content threshold 100, minimum scene 0.2 s,
  no merging/cap, and Detect fades enabled, Chrome displayed boundaries at
  3.00, 4.08, and 5.04 s for both Fade threshold 12 and 100. Source inspection
  confirmed the preview callback omitted the Fade threshold input; actual
  caption jobs already passed the setting correctly.
- Wired the missing preview input. After restart, Chrome showed the fade
  boundary at 4.40 s for threshold 100, returning to 4.08 s for threshold 12.
  The other two content boundaries were unchanged, confirming the selected
  fade setting now reaches the detector.
- The dedicated Threshold/fades detector at threshold 12 produced two ranges,
  0–4.08 and 4.08–8.00 s. Merge below 5 s combined them into one 8 s range;
  Maximum scene 2 s then produced four contiguous 2 s ranges. Setting model
  limit 1 s displayed `will auto-split` on all four rows.
- Adaptive threshold 3 with downscale 2 and fades/merging off produced three
  ranges bounded by 3.00 and 5.04 s. Raising minimum scene length to 10 s
  suppressed those cuts and displayed one 8 s range with `below minimum`
  and `will auto-split` warnings. All checks were Chrome preview actions.

### Automatic quality rejection

- `batch_0067_qwen3` enabled all six rules: minimum duration 2 s, maximum
  black ratio 0.98 at luma 16, minimum motion 0.01, minimum sharpness 100,
  required audio, maximum silence ratio 0.95 at RMS 0.001, and 8 analysis
  frames. Four QA clips were rejected in 0.90 s with no caption model load.
  Black/silent media measured black ratio 1, motion 0, sharpness 0, silence 1
  and fired all four corresponding reasons. Blurred moving media measured
  sharpness 2.10 and failed only sharpness; a clip without audio failed the
  audio requirement; a 1 s clip failed minimum duration. Chrome showed all
  four as skipped and retained each reason in the log and metadata.
- Added a 4 s moving test pattern with sound. Under the same settings,
  `batch_0068_qwen3` completed 1 clip and rejected the other 4 in 70.90 s.
  The accepted clip measured motion 7.14, sharpness 667.77, negligible black
  pixels, and silence ratio 0. The deliberately short 64-token generation
  limit produced partial text; this run validates quality filtering and batch
  continuation, not complete caption quality. The worker exited afterward.
- A typed Quality frames value of 3 was rejected by Gradio's minimum-4 input
  validation before a new job/model load. Corrected to 4 and set black luma
  255 plus silence RMS 0.1, with overwrite enabled for the accepted control.
  `batch_0069_qwen3` then measured black/silence ratios 1 and rejected it for
  both reasons in 0.33 s. Metadata recorded all three changed parameters;
  Chrome displayed 0 done / 1 skipped without a caption model load.

### Caption cancellation and resume

- `batch_0070_qwen3` requested a long description of two QA images. The
  confirmation remained visible through streaming updates, expired safely
  without stopping the job, and its Keep running action dismissed it while
  generation continued. Yes, cancel stopped the second image after the first
  had completed: 1 done / 1 cancelled in 124.58 s. The completed caption
  remained available; the cancelled partial generation was not saved as a
  finished caption. The worker exited and released its model allocation.
- Added a third QA image and disabled overwrite. `batch_0071_qwen3` skipped
  the existing caption, then Yes, cancel stopped the active image and marked
  the queued image `Cancelled before processing`: 1 skipped / 2 cancelled in
  33.40 s. Retry failed correctly remained disabled because neither was a
  failure. Chrome showed the cancelled terminal state and Results ZIP.
- Restarting with a short prompt and the same overwrite-off folder settings
  resumed the unfinished files in `batch_0072_qwen3`: 2 done / 1 skipped in
  36.04 s, both generations ending at EOS. The original caption's SHA-256
  and modification time stayed unchanged through cancellation and resume.
  All three jobs used only GPU 0 and Keep model loaded off.

### Split transcripts, context carry, summary, and HEVC

- `0073_qwen3` processed the 11 s speech-video fixture as two precise fixed
  clips, with no overlap, libx265 / CRF 25 / veryfast, saved clips,
  normalization at 2 FPS, 48 kHz normalized audio, and a 524,288 total pixel
  cap. Both clips and the separate summary ended at EOS; the job completed
  in 60.17 s and the worker exited. A typed context-carry size of 8 was
  rejected before job creation; the corrected supported minimum was 10.
- Whisper `base.en` was absent initially, downloaded automatically (147.8 MB),
  loaded on isolated GPU 0, and transcribed the complete JFK sentence with
  22 words. All six sidecars were written with `_qa_all_speech` suffix.
  An explicit `{{TRANSCRIPT}}` token worked with automatic prompt appending
  disabled: the two clips recorded different local speech windows (9 and
  14 words, sharing the boundary word), both marked injected. The captions
  quoted portions of the supplied speech, not every transcribed word.
- Context carry used 10 words and custom `QA_PREVIOUS: {{CONTEXT}}` wording;
  the run log confirmed application to clip 2. The custom summary prompt
  produced `QA_SUMMARY` plus two timestamped chapter lines (143 tokens),
  saved in `_summary.txt`, the output JSON, and metadata. The caption table
  settled at 259 total tokens, including the two captions and summary.
- The pixel-cap log reported reduction from 262,144 to 131,072 pixels per
  sampled frame. Normalized artifacts were HEVC at 672×384 / exactly 2 FPS
  with AAC 48 kHz; persistent splits were HEVC at source geometry/cadence.
  Chrome played the second HEVC clip to its 5.4945 s end without an error.
- The split verifier warned that clip 2 contained 161 rather than 165 frames.
  Artifact inspection traced this to the fixture: its source video contains
  326 frames and ends at 10.8775 s, while audio/container duration is 11 s.
  The two outputs preserve all 326 source frames (165 + 161).
- Normalization correctly honored libx265, exposing an outdated H.264-only
  help string. Updated it to describe the selected codec, FPS, and dimensions;
  the corrected text was verified in Chrome after restart. Input previews use
  Gradio's conservative codec check and therefore show an HEVC first frame,
  although this Chrome can play HEVC in the clip gallery. Corrected the input
  note to describe that fallback without claiming HEVC is unplayable; verified
  the new note on a recovered HEVC folder after another restart.

### Settings recovery

- Loaded `0073_qwen3` through Recent run: 43 stored values differed after a
  restart. Model + prompt only restored the Custom prompt while preserving
  pipeline defaults (128 frames, normalization off, 16 kHz). Apply to UI
  restored 8 frames, fixed 5.5 s clips, libx265, normalization, 48 kHz, and
  10-word context carry. Without path opt-in, the input remained empty.
- Full recovery exposed an invalid-dropdown bug: caption metadata included
  the transient personal-prompt selection despite its registry exclusion.
  Recovery coerced its empty value to an invalid choice, breaking subsequent
  metadata Load/upload and other registry-wide actions. Caption runs now
  honor `in_metadata`; recovery also ignores excluded controls from older
  files. Extended the input/output path opt-in to all four Whisper path fields.
- After restart, applied the same old metadata, uploaded a modified copy,
  and loaded/applied it successfully. The fixture contained an obsolete
  personal-prompt name, unavailable GPU 9999, and distinct Caption/Whisper
  and machine paths. Chrome skipped the obsolete selection, warned and
  defaulted the unavailable GPU to GPU 0, and removed 9999 from the GPU list.
  GPU 1 remained unselected. Whisper paths stayed empty without opt-in and
  restored correctly when opted in. Outputs, temp, models, logs, and FFmpeg
  machine paths remained unchanged, as verified in Global Settings.
- New `batch_0074_qwen3` metadata excludes personal-library controls, chat
  controls, logs/FFmpeg paths, and theme. Cleared the upload, refreshed Recent
  run, loaded `batch_0077_qwen3` using the run-folder path, and applied it
  without an error (zero settings differences). Recovery also restored this
  layout after the next application restart.

### Existing video captions and Whisper audio parts

- `batch_0074_qwen3`: three clips with existing captions, Whisper-only audio
  source, timestamped transcript style, custom filename/audio/merge templates,
  and placeholder policy. Completed in 3.74 s with three audio parts and
  merged captions, without loading the main caption model. All six transcript
  sidecars were written even with stage 7 disabled, as documented for stage 8.
  Whisper base.en without VAD hallucinated a period on the silent clip;
  this is not counted as correct transcription of silence.
- `batch_0075_qwen3` reran with overwrite off: three skipped in 0.09 s,
  with no caption-model load. `batch_0076_qwen3` enabled Silero VAD, selected
  one-segment-per-line rendering, and overwrote the layout in 3.66 s. Silence
  produced zero words and exactly `QA_NO_SPEECH` in its audio part. The
  template's empty sound token collapsed without residual template syntax.
- A Windows UTF-8 BOM in the existing silent caption leaked into the merged
  text. Changed existing-caption reading to UTF-8 with optional BOM removal;
  the same Chrome overwrite run produced clean video and merged parts.
  `batch_0077_qwen3` repeated the merge in 3.45 s; all 12 TXT artifacts were
  byte-identical, proving that existing clean video parts prevent duplicate
  audio appending.
- `batch_0078_qwen3` used a fresh output directory, plain-paragraph speech,
  merged files off, skip-empty policy, and an additional clip without a
  video caption. Completed four items in 4.68 s. There were three video
  parts and three audio parts, no main caption TXT files, no silent audio
  part, and no fabricated video part for the orphan clip. The orphan's
  speech was saved separately with a diagnostic message.
- `batch_0079_qwen3` with overwrite off skipped the three existing audio
  parts and processed the silent file again. This matches the documented
  existing-caption skip rule, which requires an audio part; a skipped empty
  audio part therefore does not suppress future transcription attempts.
- Corrected messages that claimed a merged caption was written in parts-only
  mode, and that audio had been saved when both parts were absent.
  `batch_0080_qwen3` verified both cases after restart: two silent items,
  0 audio captions / 2 no-speech in 1.98 s, with accurate per-item messages.
  These jobs released their workers after completion.

### Sound windows, progress, and generated caption parts

- `batch_0081_qwen3` used existing video captions and auto-selected Captioner
  INT4 for a 35 s storm recording, an audio-free video, and digital silence.
  The long recording split into 30 s and 5 s windows; no-audio returned empty.
  The silent fixture decoded to exactly 64,000 zero samples, but was sent to
  Captioner and produced 512 tokens of invented synthesizer/music description.
  The run took 145.88 s. The sound phase also lacked live progress and usage
  accounting, leaving the UI at a loading/phase message and zero final tokens.
- Added a PCM16 exact-zero check after extraction and before model loading.
  Quiet nonzero samples remain eligible. Sound windows now use the common
  generation path and forward window number, token progress, speed, and
  context to the UI. Metadata records each window's bounds, usage, timing,
  finish reason, and peak memory; item totals include sound generation.
  Updated an existing mock-extraction fixture to write valid PCM WAV data.
  Feature verification remained through Chrome, without running a backend suite.
- `batch_0082_qwen3` overwrote the same silent/no-audio outputs in 0.36 s:
  0 audio captions / 2 no-speech, no Captioner model load, the old invented
  silent audio part removed, and merged files equal to the clean video parts.
- `batch_0083_qwen3` used the 35 s recording with an intentional 128-token
  limit per window. Chrome displayed window 2/2 with streaming tokens,
  12.8 tok/s, and context usage, then 256 total tokens. Metadata recorded
  0–30 and 30–35 s windows, 128 tokens each, and `length` finish reasons.
  The 46.31 s run validates windowing and progress; both texts are partial.
  An item timing of 0.0 s exposed a second accounting omission, fixed by
  including sound-phase time and peak memory in the item result.
- `0084_qwen3` generated two precise 5.5 s HEVC clips with Instruct INT4,
  transcribed speech using base.en/VAD in memory, then switched to explicitly
  selected Captioner GGUF Q4. The Instruct model unloaded before GGUF loading.
  Video generations used 18 + 33 tokens; sound used 470 + 426, all ending at
  EOS. Chrome displayed 947 tokens and 54.9 s item time, consistent with
  metadata (54.85 s item / 57.50 s job, 25.53 GiB peak). Separate video/audio
  parts and merged captions exist for both clips and the combined item.
  Empty stage-7 format selection produced no transcript sidecars.
- Sound descriptions are not reliable ground truth: the storm output included
  unsupported metallic-friction claims, and GGUF speech descriptions inferred
  acoustic spaces, historical context, and recording provenance beyond the
  observable speech. Successful execution and EOS are not accuracy guarantees.
- `0084` exposed a transcript-boundary defect: stage-8 audio parts copied an
  entire overlapping Whisper sentence into both clips despite available word
  timestamps. The prompt-injection text already selected local words correctly.
  Updated caption-part rendering to select overlapping words and shift their
  timestamps to the clip origin, retaining segment fallback when word timing
  is unavailable.
- `0085_qwen3` repeated the split workflow with Whisper-only audio parts and
  timestamped rendering: 40.36 s, two video generations ending at EOS, no
  sidecars, and correct local transcript files. Clip 1 contains the opening
  9 words over 0–5.5 s; clip 2 contains 14 overlapping words over 0–4.8 s.
  The boundary word appears in both because its word interval crosses the cut.
  All regression workers exited after their jobs.
