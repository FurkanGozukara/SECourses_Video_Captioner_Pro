# QA verification log - v1.7.0 (2026-09-05)

Every tab of the v1.6.0 build was exercised as a real user in Google Chrome on Windows 11 (RTX 5090, GPU 0
only), started from `Windows_Run_Video_Captioner_Pro.bat --no-browser --server-port 7860`. Defects and
improvements found in that pass were implemented directly (no delegated tasks) and re-verified on the v1.7.0
build in the same browser session.

## Environment

| Item | Value |
|---|---|
| OS / Python / torch / transformers / gradio | Windows 11 Pro 10.0.26200 / 3.12.10 / 2.13.0+cu130 / 5.16.1 / 6.26.0 |
| GPU | GPU 0, RTX 5090 32 GB (GPU 1 untouched) |
| Test media | `temp/qa_media` (20 s storm video, 11 s JFK WAV, 18 s MP3, PNG), `temp/qa_batch_ünicode` (Unicode names, nested folders, corrupt MP4, WAV, PNG), `temp/qa_dataset_clips` |
| Automated suite | `pytest tests -q`: v1.6.0 baseline 562 passed, 8 skipped; v1.7.0 final 576 passed, 8 skipped (adds `tests/test_v17_fixes.py`, 14 tests, including a node-driven check of the browser-side pick classifier) |

## Real-user pass on v1.6.0 (what worked)

| Area | Result |
|---|---|
| Caption single file | File path input with preview and probe line; run with Qwen3-Omni INT4, whole mode, 60-token cap, prefix/suffix, two word replacements, all five output formats: 40.9 s, `finish_reason=length`, every value honoured in the files and `metadata.json` |
| Folder batch | `temp/qa_batch_ünicode` recursive with AVoCaDO INT4: 2 done, 3 unsupported (corrupt MP4, PNG, WAV) with model-specific messages; mirrored Unicode tree and `captions_index.json`; skip-existing then Overwrite |
| Backends | Qwen3 GGUF Q4 (217 tok/s, EOS), TimeChat INT4 (28.9 tok/s), AVoCaDO INT4 (28.8 tok/s), Whisper large-v1 (JFK text correct, 13.6x realtime) |
| Caption Editor | Open in Caption Editor handoff, row selection, typed edit autosaved to disk, Approve flag file, find & replace preview and apply, export approved, dataset statistics |
| Chat | Single-turn AVoCaDO video Q&A ("The weather is dark and rainy."), tokens/speed/context line |
| Presets | Save with a Unicode name, reload, Reset, Delete with confirmation, Load Last Values |
| Other tabs | Global Settings save, theme toggle (both directions, radio in sync), Recover load + Apply to UI (205 settings), System & Models update check, SHA verification of TimeChat INT4, Dataset fitness plan + Musubi TOML, Changelog, hotkeys F4/F9/Esc |

## Defects found on v1.6.0 and fixed in v1.7.0

| # | Finding | Fix (verified on v1.7.0) |
|---|---|---|
| 1 | Loading any universal preset with `vram_preset=auto` (every shipped preset) triggered the automatic VRAM tier plan afterwards, which replaced the preset's `fps`, `max_frames`, `max_pixels`, `max_new_tokens`, attention, and block-swap values. `LTX 2.5 short` (512 tokens, 48 frames) came back as 4,096 tokens and 128 frames; a saved user preset with 1,200 tokens came back as 2,048 | Preset loads (dropdown, Load, Load Last Values, delete fallback) keep every stored value and only swap a variant the detected GPU cannot run; startup and Reset still apply the tier plan. Live: LTX preset shows 512 tokens / 48 frames and the note `Preset values applied as saved` |
| 2 | Picking a model, VRAM tier, pixel preset, or task preset with the keyboard (typing plus Enter, or arrow keys) changed the value but none of the reactions ran (family defaults, tier plan, prompt render, identity line) because Gradio 6 fires `select` only for mouse picks | Reactions listen to `input` with a browser-side pick marker (`change` stamps the clock, a blur-only `input` is ignored); `model_constraints` no longer re-asserts a stale token value. Live: arrow-key model picks update the plan note, token budget, prompt, and pixel preset |
| 3 | "Use video audio" could be unchecked for TimeChat; the pipeline then rejected every video as `does not support video input` although TimeChat always synthesises an audio timeline | Toggle locked on for families with `requires_audio_track`; the runner treats such videos as `video_audio` regardless of the toggle |
| 4 | Precision line said `Checkpoint: 19.9 GB` while the dropdown said `17.8 GB` for the same INT4 build; System & Models reported on-disk sizes in GiB labelled GB | Both use the measured decimal GB |
| 5 | Live log and console showed third-party noise (`Successfully loaded: 'mslk.dll'`, torch Enum deprecation, `rope_parameters`, unused generation flags) | Suppressed from console and live log, kept in the daily log file |
| 6 | Editing a caption in the editor left the sibling `.json` with the old text | The JSON sidecar's `text`/`caption` field follows the edit |
| 7 | `torch.compile` toolchain probe polled the server every second forever; Whisper printed a bare file path to the console; Reset said "defaults (defaults)" | Timer stops once the result is known; console names the written transcript; wording fixed |
| 8 | At idle the browser's main thread was 100 % busy (long tasks of 75-150 ms back to back) and three estimate handlers ran about seven times per second. Cause: Gradio 6.26's frontend re-dispatches a whole `change` event when a deferred (`always_last`, the default for `change`) listener finishes, so any control with two or more such listeners loops forever after two overlapping changes; sixteen controls qualified (Maximum new tokens, Sampling FPS, Maximum pixels, Context length, Model variant with eleven, GPU, Use video audio, ...). Every keyboard pick therefore took about 2.5 s to settle | Token budget, block-swap preview, and auto-split ceiling refresh through one shared listener; every other listener on those controls runs with `trigger_mode="multiple"`; `tests/test_v17_fixes.py::test_at_most_one_deferred_listener_per_event` guards the invariant. Live: no long tasks at idle, requests drop to the timer polls |
| 9 | After visiting the Transcribe, Chat, Caption Editor, and System & Models tabs and returning to Caption, all their timers kept polling: 8.2 requests/s and the main thread 70 % busy, because Gradio keeps hidden tab content mounted | The tab bar's `select` event activates only the chosen tab's timers (`gate_tab_timers` in `vcap/ui/app.py`; the per-Tab `select` never fires for the tab open at load in Gradio 6.26); hidden tabs stop polling, the editor autosave drops to a 3 s background tick so a last edit is still flushed. Live: back on Caption after the same tour, 4 requests/s and long tasks only from the Caption polls |

## Backend values exposed for the first time

| Control | Location | Previously |
|---|---|---|
| GPU layers (-ngl, 0 = fit automatically) | 1. Model → llama.cpp (GGUF) options | `VCAP_LLAMACPP_GPU_LAYERS` |
| MoE expert layers on CPU (--n-cpu-moe) | 1. Model → llama.cpp (GGUF) options | `VCAP_LLAMACPP_N_CPU_MOE` |
| Cap the CUDA allocator at the VRAM free at load | 1. Model → Block swap & offload plan | `VCAP_VRAM_HARD_CAP` |

All three are registered settings (saved in presets and run metadata, recoverable); the environment variables still override them.

## Final run on the v1.7.0 build

After the last edit the app was restarted and `temp/qa_media/video20s.mp4` was captioned from the File path input with Qwen3-Omni Instruct GGUF Q4 selected by keyboard (typing `GGUF Q4` + Enter), the new GPU-layer and CPU-MoE controls at their default 0: llama-server started with the automatic fit, 5 segments captioned at 217 tok/s (EOS), `Job finished: done=1 ... in 36.41s`. At idle the Caption tab now issues only its own four polls per second; after visiting every other tab and returning, their timers are stopped (editor autosave at its 3 s background tick).

## Interface additions

- Global Settings → **Keyboard shortcuts** reference.
- Editing Maximum pixels by hand switches the Pixel preset to Custom (or to the matching preset).
- Result buttons read **Open Caption** / **Open Transcript** so they stay on one line.
