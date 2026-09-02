# QA verification log - v1.4.0 (2026-09-02)

This release was verified through the real SECourses Video Captioner Pro interface in Google Chrome on Windows 11,
started with `Windows_Run_Video_Captioner_Pro.bat`, using physical GPU 0 only (NVIDIA GeForce RTX 5090, 32 GB).
The pass covers the v1.4.0 backend/UI/performance/installer work and the second feature round. Automated results
are listed at the end.

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| Python / torch / transformers / gradio | 3.12.10 / 2.13.0+cu130 / 5.16.1 / 6.26.0 |
| GPU used | GPU 0, RTX 5090, driver 610.88 (GPU 1 untouched) |
| llama.cpp | b10621 Windows CUDA build (`llamacpp/b10621`) |
| Test media | `F:\SECourses_Video_Captioner_Pro_TEMP\test_media` (20 s storm video, 78 s Unicode-named launch video, 18 s MP3, WAV, PNG, Unicode batch folders) |

## UI-only pass (before model runs)

| Area | Verified result |
|---|---|
| Startup | Launcher printed the banner with both GPUs and served `http://127.0.0.1:7860`; no browser console errors on load. |
| File path input | A quoted, mixed-separator, non-ASCII path (`"F:/…\test_media/vöyager ünicode 日本語 テスト.mp4"`) was accepted; the probe line showed `video · 01:18.02 · 960 × 720 · 60 fps · aac, 48000 Hz, 2 ch · codec h264` and the player rendered the first frame. (Chrome defers media loading in hidden tabs, so the preview only fills once the tab is visible; not an application issue.) |
| Trim range | Start/End controls and the player trim editor are present under the preview. |
| Processing Pipeline | Sections 4-6 render with every control; `Detect scenes now (preview)` found 9 scenes in the Unicode file; `Preview sampled frames` produced `Plan: 128 frames at 576×416 (~14,976 visual tokens) — showing 16` with timestamped thumbnails. |
| Folder batch | The Unicode folder scan reported `2 videos · 1 audios · 1 images · 0 texts · 4 already captioned in output folder`; kind filters, name filter, recursive, overwrite, and limit controls are present. |
| Presets | Saved `QA ünicode preset 日本` (stored as `QA_ünicode_preset_日本`), it became the selected preset, and Delete removed it with a confirmation message. |
| Caption Editor | Scan of `outputs` listed 60 review items with length/token columns, token-limit warnings, and the first caption preview. |
| Other tabs | Chat, Dataset & Export, Global Settings (logs directory, open presets/logs folder buttons), Recover Settings (recent run auto-discovered), System & Models (environment report, llama.cpp status, all 21 variants ready, unload/refresh buttons) and Changelog all rendered. |
| Caption Editor (round 2) | Gallery view rendered thumbnails for 60 items; Dataset statistics reported 60 items / 58 captioned / 1 failed, character and token ranges, 22.4 % trigger coverage, four duplicate groups, top words. |
| Dataset & Export | Clip fitness analysis over 92 clips (62 ready, 23 would be dropped, 7 sub-split suggestions); Kohya/Musubi TOML generated. |
| Recover Settings | Loaded `outputs/0061_timechat/metadata.json` (16 differences listed) and applied 96 settings to the UI; source paths stayed excluded. |
| Prompt library | Saved, loaded, and deleted `QA prompt ünicode 日本` with status messages. |
| Run history | Listed 40 runs with kind, model, item counts, time, and caption preview. |
| ZIP upload | `qa_v14_upload.zip` was extracted to `outputs/uploaded_batches/<name>_<timestamp>` with Unicode names intact and the folder scan ran (3 files with Scan subfolders). |
| System & Models | Check for updates reported `Up to date (v1.3.2, b0c3021)`; Delete model files showed `Delete BF16 (16.65 GB) from disk?` and **Keep files** left the model untouched. |

## Model runs (GPU 0, app started from the Windows launcher, v1.4.0)

| Run | Input | Verified result |
|---|---|---|
| `qwen3_omni_instruct_int4` single, scene split | uploaded 20 s storm video | 5 scene clips captioned at 13.2-13.4 tok/s (stopped by EOS), 135.8 s total including the 24 s load; `metadata.json` carried 156 settings including every new key; per-clip TXT/JSON under `<stem>_segments`; Copy caption showed "Copied.", Results ZIP wrote `temp/downloads/0062_qwen3.zip` (17.7 KB), Open in Caption Editor scanned the run (6 items). |
| `avocado_bf16` single, cancel | same video | Model switch released the resident INT4 model; BF16 loaded in 15.6 s and generated at 31.8 tok/s; **Cancel → Keep running** left the job running; **Cancel → Yes, cancel** stopped it (`Cancelled: 1 cancelled … in 54.6s`, worker reported `Generation finished (cancelled)`). |
| `qwen3_omni_instruct_int4` folder batch, recursive | Unicode folder with a corrupt MP4, a clip, a nested PNG, a WAV | `done=3, unsupported=1` in 105 s; the corrupt file reported `moov atom not found` and the batch continued; outputs mirrored as `batch_captions/nested 日本語/image テスト.txt`; `batch_0064_qwen3/summary.json` and `run_log.txt` written; a rerun skipped all 3 captioned items in 0.1 s; batch Results ZIP (3.8 MB) created. |
| `qwen3_omni_instruct_gguf_q4` single, scene split | storm video | `llama-server` started with one slot and a 32k context; 5 clips at 263-292 tok/s in 17.5 s; the log records `GGUF frame plan selected 11 of 11 frame(s) at 2 fps` and the precise re-encode fallback for inexact stream copies. |
| `qwen3_omni_instruct_gguf_q4` post-processing + summary | storm video, prefix `ohwx style,`, suffix, replacements `night;evening` / `cars;vehicles`, SRT + VTT + JSONL, Save produced clips, Summarize segments | TXT/JSON/JSONL carry prefix, suffix, and replacements (5 segments each); SRT/VTT cues carry the replacements; 5 clips saved under `<stem>_clips` and shown in the Clips gallery; `<stem>_summary.txt` holds one paragraph plus five `MM:SS-MM:SS Title - sentence` chapters and the JSON has a `summary` field. |
| Chat (`qwen3_omni_instruct_gguf_q4`) | storm video attached by path | Two streamed turns; the second answer used the first ("The weather described in the video is a thunderstorm."); the GGUF frame cap warning was shown; conversation saved as `outputs/0071_chat_qwen3/conversation.{json,md}`. |
| `qwen3_omni_captioner_gguf_q4` | storm video, then 18 s MP3 | Video was refused with `does not support video input. Models that do support it: …` without loading the model; the MP3 produced 461 tokens at 295 tok/s with all five formats, prefix/suffix applied, and `Summary skipped: … cannot take text-only input` logged. |
| `qwen3_omni_thinking_int4` | `test_image.png` by path | Image preview and probe (480 × 270) shown; 782 tokens at 12.15 tok/s in 86.7 s including the load; `test_image_reasoning.txt` written separately from the clean caption; sampling was on and the drawn seed (2677942758) is recorded in the segment usage. |
| `timechat_int4` | storm video, scene split | 5 timestamped segments at 26.9-27.1 tok/s in 99.8 s; TXT/JSON/JSONL/SRT/VTT plus saved clips; prefix, suffix, and replacements applied. |
| `avocado_int4` | storm video, scene split | 5 segments at 26.8-27.0 tok/s in 57.5 s; all formats and clips written. |
| Caption Editor on `0077_avocado` | 6 items | Approve advanced to the next item; Export approved copied `clip_0003.mp4` + `.txt` to `outputs/approved_dataset` and wrote `approved_dataset.zip` (3.7 MiB); **Regenerate selected failed** with `Expected 154 values, got 143` (fix F4.3). |
| Unload model | System & Models | GPU 0 returned to 0.8 GiB used after the AVoCaDO release; model switches also released the previous model automatically each time (17.06, 7.85, 23.10 GiB freed in the log). |
| Global Settings | Save | `app_settings.json` gained `logs_dir` and `ffmpeg_path`; Light theme rendered readable and Dark was restored. |

## Defects found during the GPU pass (fix rounds F3/F4)

| # | Finding | Status |
|---|---|---|
| F3.1 | Preset Delete had no confirmation and left the dropdown empty | fixed; verified: `⚠ Delete preset "QA_F3_ünicode"?` bar, Yes → `Deleted QA_F3_ünicode; loaded Default - Qwen3-Omni Instruct video.` |
| F3.2 | ZIP upload with a single top-level folder needed "Scan subfolders" manually | fixed; verified: input folder became `…\qa_v14_upload_<ts>\zipped` and the scan found 1 video · 1 audio |
| F3.3 | Model variant dropdown accepted typed custom values (`Unknown model variant`) | fixed; verified: typed text is rejected and the INT4 selection is kept |
| F3.4 | Task/prompt preset showed its raw id after loading a universal preset | fixed; verified: `Model-native · Qwen3-Omni — dense audiovisual video` after Load |
| F3.5 | Retry failed ignored `unsupported` (unreadable) items | fixed; verified: after the batch (3 skipped, 1 unsupported) the button was enabled and re-ran exactly 1 item |
| F3.6 | Input mode could diverge from the selected input tab (slow folder scan overwrote the mode; stale cached inputs) | fixed; verified: the folder batch ran from the Folder batch tab after tab switches (`Starting batch job with 4 resolved input(s)`) |
| F3.7 | Case-insensitive replacements lowercased sentence starts (`Cars` → `vehicles`) | fixed; covered by `tests/test_captions_post.py` (three casings) |
| F3.8 | Chat status line lost the final tokens/speed for replies shorter than the throttle window | fixed; verified: `Chat finished: 7 new tokens in 1.8s` and `Tokens: 7 · Speed: 10.63 tok/s · Context: 14 / 32,768` |
| F4.1 | Live log panel appeared to stop updating after a cancelled job | root cause was Chrome/Gradio pausing timers while the tab is hidden (zero timer requests while `document.hidden`, immediate catch-up when visible); F4 still made worker log mirroring sink-independent and added cursor self-healing; F6 adds a note under the panel |
| F4.2 | First Start after a model change was ignored | fixed; verified: selecting TimeChat INT4 and clicking Start 3 s later started the job (`Status: timechat_int4: ready (100%)` → generation) |
| F4.3 | Editor regenerate: `Expected 154 values, got 143` | fixed (inputs wired after all tabs register); the regenerate prompt-preset dropdown was then found empty for every item (blocks the feature) → fix round F7 |
| F4.4 | Recover Settings recent-run list stale until Refresh | fixed; verified: the tab now opens on the newest run (`0080_timechat`) |
| F5.1 | Status line omitted `unsupported`/`cancelled` counts | fixed in F5 (unit-tested formatting) |
| F6.1 | Regenerate with an incompatible model/media pair raised a Gradio value error | fixed in F6 (no job is submitted; a clear status explains the incompatibility) |
| F6.2 | Live log / meter pause while the browser tab is hidden | a note under the panel explains the pause |
| F7 | Regenerate prompt-preset dropdown was empty after scan/selection/model mirror | fixed; verified: after scanning `0077_avocado` with Qwen3 INT4 the dropdown showed `Training captions · Wan 2.2 T2V — dense paragraph` and **Regenerate selected** started a real regeneration (worker loaded the model and generated at ~16 tok/s) |

## Automated results

| Stage | Result |
|---|---|
| Start of session (v1.3.2 + round 1 uncommitted) | 333 passed, 7 skipped |
| After round 2 (F1 backend, F2 UI) | 375 passed, 7 skipped |
| After fix rounds F3 + F4 | 385 passed, 7 skipped |
| Final tree (F5, F6, F7 included) | **393 passed, 7 skipped** in 86 s (`venv\Scripts\python.exe -m pytest tests -q`) |
