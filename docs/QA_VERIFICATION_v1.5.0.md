# QA verification log - v1.5.0 (2026-09-04)

Every feature of the application was exercised through the real interface in Google Chrome on Windows 11, first on the
v1.4.1 build (baseline defect hunt) and then on the integrated v1.5.0 build started from `Windows_Run_Video_Captioner_Pro.bat`
(port 7860). Physical GPU 0 (RTX 5090, 32 GB) was the only GPU used. Implementation and fixes were delegated to
codex (`gpt-5.6-sol`) tasks W1/W2 (Whisper backend + UI/pipeline), B (baseline defects), F (Transcribe-tab defects found on
the integrated build), and G (regressions); every task ran CPU-only tests, and the orchestrator verified each result in Chrome.

## Environment

| Item | Value |
|---|---|
| OS / Python / torch / transformers / gradio | Windows 11 Pro 10.0.26200 / 3.12.10 / 2.13.0+cu130 / 5.16.1 / 6.26.0 |
| New runtime | faster-whisper 1.2.1, ctranslate2 4.8.2 (bundled cuDNN 9), onnxruntime 1.29.0, nvidia-cublas-cu12 12.9.2.10 |
| GPU | GPU 0, RTX 5090 (GPU 1 untouched) |
| Test media | `temp/qa_media` (20 s storm video, 18 s MP3, PNG, 11 s JFK WAV, 5:14 talk-radio MP3/MP4, 28:01 tutorial MP3/MP4 VP9+Opus), `temp/qa_batch_ünicode` (Unicode names, nested folders, a corrupt MP4) |
| Automated suite | `pytest tests -q`: **493 passed, 8 skipped** (baseline v1.4.1: 398 passed, 7 skipped) |

## Baseline pass on v1.4.1 (defects found → fixed in v1.5.0)

| # | Finding | Fix (verified on v1.5.0) |
|---|---|---|
| B1 | Caption Editor **Regenerate selected** on a scene segment re-ran the whole source with scene detection (100 s) and overwrote the segment with the combined 5-segment text | Regenerates only the clip window: `Regenerating clip 2 (00:05.105-00:09.510) of video20s.mp4`, 39.6 s, single caption replaced, Keep/Revert work |
| B2 | Segment rows all showed the parent file name | Rows show `video20s_segments/clip_0002.txt · 00:05.1–00:09.5` |
| B3 | **Export approved** copied the whole 20 s video for every approved segment | Exports `video20s_clip_0002.mp4` cut to 4.40 s + its caption |
| U4 | After a scan the first caption showed but "No preview selected." stayed | First item selected with its video preview; placeholder hidden |
| U5 | Editor find & replace defaulted to partial words (`night` → `eveningtime`) | Whole word on by default |
| B9 | Typing an unknown value into **Task / prompt preset** raised `KeyError` (red toast) | Value rejected, valid label restored, no traceback |
| B11 | The first Chat send unloaded and reloaded the same resident variant (17 GB churn, 16 s) | Chat after captioning reuses the resident model (`Chat finished: 23 tokens in 2.8s`, no reload); reload reasons are now logged |
| B14 | **Write plan JSON** always failed with the default folders | `Wrote …\outputs\clip_fitness\outputs_20260904_191731.json` |
| U7 | Progress bar showed `ready (100 %)` right after Start | Starts at `Preparing input` |
| U8 | Results ZIP download box squeezed the button row | Own row (Caption and Transcribe tabs) |
| U10 | Editor regeneration runs listed as `batch · - · 0 items` in Run history | Correct kind/model/item; whisper runs listed as `transcribe · large-v1` |
| U12 | Enter in the Chat message box inserted a newline | Enter sends, Shift+Enter adds a line (info text says so) |
| U13 | Save-as kept the name of a deleted preset | Cleared after delete |
| — | Static-analysis items: unload VRAM report timing, 3-second model-folder walk in System & Models, uncancellable Caption-tab downloads/scene previews, untyped `output_formats`/`system_prompt`/`trim_end_s`, Reset leaving the dropdown, chat wipe on programmatic model change | All addressed by task B with unit tests |
| B17 | Selecting a browser-unplayable video (VP9/Opus MP4, 28 min) re-encoded it for the preview for 3 minutes and delayed the probe and Start | First-frame poster + note appear in 1.6 s; Start is never delayed |
| B18 | (regression from B) the literal text `None` became the system prompt | `_cast` maps None → "", handlers return "", metadata records `""` |

Verified as working on the baseline and again on v1.5.0: upload/path/folder inputs with quoted mixed-separator Unicode paths,
probe line and previews, scene split, single caption run (Qwen3-Omni INT4, 5 segments), Copy caption, Results ZIP, Open in
Caption Editor, Run history, folder batch over a Unicode tree with a corrupt file (`unsupported`, batch continues), cancel
with confirmation (`cancelled=1`), rerun skipping finished items, preset save/load/delete/reset/last values with Unicode
names, prompt library, Processing Pipeline previews (scene detection, sampled frames), F9/Esc hotkeys, Chat two turns +
save, editor autosave/approve/reject/find & replace/bulk/export/statistics, Dataset analysis/TOML/sub-split, Global
Settings save, Recover load/apply, System & Models unload/update check, light and dark themes.

## Whisper transcription (new in v1.5.0)

| Run | Input | Result |
|---|---|---|
| Transcribe, upload | `test2.mp3` (5:14) large-v1 float16 | 20.1 s; console ETA/segment tail; SRT/VTT/TXT/LRC/TSV/JSON + metadata.json (202 settings) + run_log.txt in `outputs/0089_whisper`; Results ZIP; Copy; Open Output/Last Transcript |
| Reference parity | same file, same parameters through the reference Whisper app's venv | identical lowercase transcript style and text for the first 8 segments |
| Automatic download | switch to `large-v3-turbo` (not present) and Start | 1.62 GB downloaded at up to 74 MB/s with progress in the console and the panel, model loaded, transcribed in 36.8 s; dropdown shows `✓ downloaded` afterwards |
| JFK WAV by path | `jfk.wav` | "And so my fellow Americans, ask not what your country can do for you. Ask what YOU CAN DO FOR YOUR COUNTRY!" in 3.6 s |
| 28-minute audio | `test5_audio.mp3`, large-v1 | 158 s (10.6× realtime); on the integrated build the page stayed responsive, ended at `Complete: 1 transcribed … in 158.4s`, SRT preview capped, Start enabled again |
| Cancel | second 28-minute run | Cancel → confirmation bar → **Yes, cancel** → `Cancelled: 0 transcribed · 0 skipped · 0 failed in 58.4s`, worker exited, next Start ran to completion |
| Trim | `End seconds = 600` | Only the first 10 minutes were transcribed (41 s) |
| Presets | `Transcribe - Whisper best quality (large-v1)` | Applied instantly; model/decoding controls follow |
| Caption + transcript stage | `test2.mp4` (VP9) trimmed to 40 s, Qwen3-Omni GGUF Q4 | Whisper stage ran first (10 segments, 2.3 s), wrote `test2_transcript.srt/.txt`, injected the clip-local text; the caption quotes the transcript verbatim; llama-server loaded in 12.3 s, 237 tok/s, 43 s total |
| System & Models | Whisper speech models table | 17 aliases with size/downloaded/bytes; Download / Verify / Delete / Refresh |

## Defects found on the integrated build (fixed by tasks F/G, re-verified)

- Long transcripts jammed the browser (thousands of live table rows, full JSON with every word, uncapped SRT preview) so the
  final state never rendered and Cancel/Start stopped working → throttled/bounded live views, summary JSON, capped SRT preview
  (now a 12–16 line textbox), final status yielded before heavy views, cancel timer no longer races the confirmation bar.
- Segments table numbered from 2 → from 1. `Context: —` removed from the Transcribe status line; download progress uses the full
  bar. Audio probe no longer shows `n/a · n/a`. GPU dropdown label shortened. Preset renamed from the example name "aa" to
  `Transcribe - Whisper best quality (large-v1)`.
- Transcript injection and summary are now logged (`Transcript injected into the prompt for clip …`) and stored in
  `metadata.json["extra"]["transcript"]`.

## Not changed / notes

- Whisper large-v1 with the shipped defaults produces lowercase, lightly punctuated text on talk-radio audio; this matches the
  reference application for identical parameters (large-v3 / large-v3-turbo produce capitalized, punctuated output).
- Editor edits update the `.txt` sidecars; `.json` sidecars keep the model's original text.
- Data-parallel batches on GPU 1 were not exercised (GPU 1 is excluded from testing on this machine).
