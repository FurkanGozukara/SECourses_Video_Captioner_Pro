# QA verification log - v1.6.0 (2026-09-04)

Every feature of the application was exercised again through the real interface in Google Chrome on Windows 11, first on the
v1.5.0 build (defect hunt) and then on the v1.6.0 build (dataset clip captions + fixes), started each time from
`Windows_Run_Video_Captioner_Pro.bat --server-port 7860`. Physical GPU 0 (RTX 5090, 32 GB) was the only GPU used.
Implementation was delegated to codex (`gpt-5.6-sol`) tasks N1 (dataset clip captions), B (defects found in the v1.5.0
pass), and N2 (follow-ups found while verifying N1 with real models); the orchestrator verified every result in Chrome.

## Environment

| Item | Value |
|---|---|
| OS / Python / torch / transformers / gradio | Windows 11 Pro 10.0.26200 / 3.12.10 / 2.13.0+cu130 / 5.16.1 / 6.26.0 |
| Whisper runtime | faster-whisper 1.2.1, ctranslate2 4.8.2, onnxruntime 1.29.0 |
| GPU | GPU 0, RTX 5090 (GPU 1 untouched) |
| Test media | `temp/qa_media` (20 s storm video, 18 s MP3, PNG, 11 s JFK WAV), `temp/qa_batch_ünicode` (Unicode names, nested folders, a corrupt MP4, silent videos), `temp/qa_dataset_clips` (four pre-cut 8 s clips with narration cut from the Voyager launch film, one silent clip, one Unicode-named clip in a Unicode subfolder), `F:\…\test_media\vöyager ünicode 日本語 テスト.mp4` (78 s) |
| Automated suite | `pytest tests -q`: **561 passed, 8 skipped** (v1.5.0 baseline: 493 passed, 8 skipped) |

## v1.5.0 baseline pass (defects found → fixed in v1.6.0)

| # | Finding on v1.5.0 | Fix (verified on v1.6.0) |
|---|---|---|
| D1/D21 | After **Yes, cancel** (or an expired Esc arm) the **Status:** line kept `Cooperative cancellation requested.` / `Waiting for cancel confirmation.` although the run had ended | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element) |
| D2 | Run history table columns cramped at 1450 px (`Ite ms`, hidden Done/When/Preview) | Run history uses fixed column widths and a 90-character preview; no wrapped `Ite ms` header |
| D3 | Clips result tab showed an empty gallery placeholder when clips were not saved | Clips tab shows `No clips were saved for this run. Enable "Save produced clips" …` when nothing was saved |
| D4 | Batch ETA showed `ETA 0s` after an unsupported item | ETA is computed from items that actually ran (`test_d4_eta_waits_for_a_positive_duration`) |
| D5 | Sub-split **Target seconds** flagged invalid (5.0625 vs step 0.1) | Target seconds uses step 0.001 / precision 3 (`test_d5_dataset_target_seconds_uses_three_decimal_step`) |
| D6 | Caption Editor had no media for folder-batch results (mirror folder → `no media`, run dir → 0 items) | Scanning `outputs/qa_batch_v16` shows image/audio previews for every item; scanning `outputs/batch_0106_qwen3` lists 4 items with media; batch runs now write `captions_index.json` |
| D7 | Editor character/word/token line and queue columns did not refresh after an autosaved edit | Autosave refreshes the stats line and queue cells (`test_d7_autosave_refreshes_stats_and_queue_cells`) |
| D8 | Task / prompt preset silently replaced by `wan22_t2v_dense` when switching to Folder batch (metadata proved the batch ran with the Wan prompt) | Live: Upload → Folder batch → Upload keeps `Model-native · Qwen3-Omni — dense audiovisual video`; `test_d8_prompt_preset_survives_upload_folder_upload` |
| D9 | Editor regeneration status showed raw worker log lines (`INFO:root:Successfully loaded: 'mslk.dll'`) | Regeneration status filters raw worker lines (`test_d9_editor_regeneration_filters_raw_worker_status`) |
| D10 | Editor **Regenerate selected** on a scene segment captioned the whole 78 s source (clip window dropped for batch-kind jobs) | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)0 |
| D11 | Editor regeneration reloaded the resident model in a fresh worker (26 s) | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)1 |
| D12 | Folder light scan showed the raw ffprobe error for a corrupt file | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)2 |
| D13 | Idle unload stopped the worker silently | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)3 |
| D14 | `logs/` stayed empty; Open logs folder opened an empty directory | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)4 |
| D15 | Three shipped Whisper presets contained the developer's absolute paths and every non-preset key | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)5 |
| D16 | Selecting the Upload files input tab did not refresh the preview/probe/modality (stale folder/path media shown) | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)6 |
| D17 | Raw llama-server stderr shown in the Status line while the GGUF server started | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)7 |
| D18 | Precision/Backend/Checkpoint line lagged the model dropdown by seconds | Cancelled run ends with `Status: Cancelled: 1 cancelled, 0 done, 0 skipped, 0 failed in 14.1s` (cancel note lives in its own element)8 |
| D20 | Foreign-family preset label (`Qwen3-Omni — describe image`) left in the Task preset box after a family change with stale modality; run used a silent fallback | Run history uses fixed column widths and a 90-character preview; no wrapped `Ite ms` header0 |
| D22 | Whisper folder batch failed silent videos with `tuple index out of range` and showed `[Errno …]` for a corrupt file | Run history uses fixed column widths and a 90-character preview; no wrapped `Ite ms` header2 |
| D23 | Chat status/Tokens lines lagged behind the finished answer after a model load | Run history uses fixed column widths and a 90-character preview; no wrapped `Ite ms` header3 |
| D25 | Resident model released after a job because a stale pre-preset model selection was reported (`… changed to timechat_int4`); next chat reloaded it | Run history uses fixed column widths and a 90-character preview; no wrapped `Ite ms` header5 |
| D29 | Chat UI stream lagged minutes behind the worker (per-token full re-render); Stop/Send during the lag acted on a finished job | Run history uses fixed column widths and a 90-character preview; no wrapped `Ite ms` header9 |


## Dataset clip captions (new in v1.6.0)

| Check | Result |
|---|---|
| Preset `Dataset clips - video + audio captions (Qwen3-Omni + Whisper)` over `temp/qa_dataset_clips` (recursive, next-to-source) | 4 clips: `video_caption/<stem>.txt`, `audio_caption/<stem>.txt` (Whisper text such as `lifted off from Cape Canaveral atop Titan Centaur rockets.`), merged `<stem>.txt` beside each clip including `sub ünicode/fırlatma 日本語.*`; silent clip → merged caption = video caption only; `done=4 … audio_captions=3, no_speech=1 in 90 s` |
| Rerun with overwrite off | `Caption output already exists: …` for all 4 clips (skipped) |
| Rerun over a captioned folder | The first v1.6.0 build captioned the caption files themselves (8 inputs, `*_0002.txt` pollution) → fixed: folder scans exclude sidecars and caption-part folders; live rescan shows `4 videos · 0 texts` |
| Preset `Dataset clips - add Whisper audio captions to existing captions` | 3 root clips rebuilt from `video_caption/` in 8.9 s without loading the caption model; audio parts recreated; merged files idempotent. The Unicode subfolder was skipped because the preset turned recursion off → the dataset presets now scan subfolders |
| Preset `Dataset clips - video + sound captions (Qwen3-Omni + Captioner)` | Phase 1 captions (Qwen3 INT4), phase 2 `Sound captions 1/4 … 4/4 with qwen3_omni_captioner_int4` (model switch, one window per 8 s clip), phase 3 merge; `audio_caption/liftoff.txt` = Whisper line + five-paragraph sound description; 198.7 s total |
| Empty transcript-format list (as shipped by the presets) | First build: `Transcript failed … IndexError: list index out of range` for every clip (the client indexed an empty `files` list) → fixed; transcripts inject and audio captions are written |
| Single file with scene detection + Save produced clips (`voyager_1_launch.mp4`, 0–30 s, 4 scenes) | `outputs/0123_qwen3/voyager_1_launch_clips/clip_000N.mp4 + clip_000N.txt + video_caption/ + audio_caption/` with clip-local transcripts; item-level combined files as well |
| Caption Editor | Scanning the clip folder shows video previews, the merged caption, and the read-only **Caption parts (read-only)** video/audio boxes |
| Recover Settings / metadata | `audio_caption_source`, `video_caption_source`, `caption_write_merged`, `batch_save_next_to_source`, `audio_caption_transcript_style` are recorded in `metadata.json` and restored |


## Re-verified on the final v1.6.0 build

| Area | Result |
|---|---|
| Caption tab | Upload/path/folder/ZIP inputs (Unicode, quoted, mixed separators), trim 10–25 s (timestamps offset correctly), single run Qwen3 INT4 (5 scenes, 122 s cold / 76 s warm), GGUF Q4 (25 s, 272 tok/s), TimeChat INT4 (100 s), AVoCaDO INT4 (51 s), Thinking INT4 with the Reasoning tab and `_reasoning.txt`, a SageAttention run recorded `attention: sage`, a torch.compile run (69 s; the recompile limit is now raised), Copy/ZIP/Run history/Open in editor, F9 start, Esc arm, button cancel + confirm |
| Folder batch | `temp/qa_batch_ünicode` recursive: 4 done, 1 unsupported (corrupt) in 91 s with a mirrored Unicode tree; ZIP upload extracted below `outputs/uploaded_batches`; an AVoCaDO batch marks the MP3 unsupported |
| Presets | All 19 shipped presets applied in sequence with **0 console errors** (v1.5.0: 86 tracebacks from GGUF Q8 choices and slider bounds); save/load/delete (with confirmation)/reset/last values with a Unicode preset name; Open/Close All; light/dark toggle |
| Transcribe | JFK WAV via path (3.6 s, correct text), `base` auto-download (147.9 MB at 31 MB/s) → transcribe → `✓ downloaded`, delete model with confirmation, Results ZIP, folder batch (silent videos skipped, corrupt file reported), editor scan of a transcript run with audio preview |
| Chat | Text reply reusing the resident model, a video-attachment reply (`A city street at night … lightning flashing in the sky.`), Save conversation, Clear history, live streaming at worker speed, Stop |
| Caption Editor | scan, next/prev, approve (flag file), autosave, filters, gallery view, find & replace with diff preview, bulk prefix, export approved (caption-only), statistics, regenerate + revert |
| Dataset & Export | clip fitness analyze + plan JSON, TOML generation with Wan/LTX/MiniMax defaults, sub-split (4 clips) |
| Global Settings / Recover / System & Models / Changelog | save settings, theme radio (page reload), recover load + comparison table, environment report, update check, llama.cpp runtime status, model table, Whisper table, log file location |

