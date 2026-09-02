# SECourses Video Captioner Pro — v1.4.0 goal plan

Date: 2026-09-02. Orchestrator: Claude (planner, verifier, Chrome QA). Implementers: codex (gpt-5.6-sol, YOLO) tasks A, P, U, I running in parallel with strict file ownership (see `temp/codex_v14/CONTRACT.md`).

## 1. Where v1.3.2 stands (verified 2026-09-02)

- Full automated suite: 284 passed, 5 skipped (real-GPU tests) in 94 s.
- task.txt compliance audit (code-verified, not README-based): every requirement area is implemented. Remaining partials: Captioner has one prompt preset by design (prompt-free model); TimeChat has 3 presets (single-task specialist); GGUF benchmark matrix incomplete; editor regenerate preset list not filtered by model; chat tab has no preset picker; no hierarchical "chapters + summary" stage; editor has a table but no thumbnail gallery.
- Parameter plumbing audit: no dead control, but several controls are only partially honored (GGUF ignores attention/block-swap/compile/use_cache and collapses the frame budget to 8–16 stills; sampling knobs are dropped unless `do_sample and temperature > 0`; `audio_sample_rate` never reaches the model path; `max_frames = 0` becomes the family maximum instead of "audio only"; the family clamp on `max_frames` is silent). Hard-coded backend values that users reasonably want: seed, context carry-over word count, re-encode codec/CRF/preset/audio bitrate, total pixel budget, GGUF frames/JPEG quality/threads/batch sizes/flash-attn/cache-reuse/context-tier clamp, OOM retry count, fade threshold, quality-analysis frame count, pinned RAM budget, max caption characters, logs directory.
- Decode-speed audit: TimeChat/AVoCaDO decode at the same ~34 tok/s for BF16, INT8 and INT4, which proves a fixed host-side cost per token dominates. The per-token `StoppingCriteria` flushes the console and dispatches two unthrottled UI events every token; GGUF streams the SSE response one byte at a time (`iter_lines(chunk_size=1)`) and emits per-chunk UI events; `--no-webui`/`-np 1` are not passed; dense gate/up projections are not fused for ConvRot; the Hadamard lookup takes a lock on every linear call. StaticCache + CUDA graphs would raise VRAM and is excluded.
- Robustness: worker isolation uses `CUDA_VISIBLE_DEVICES=<index>` but never sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so on mixed multi-GPU machines the CUDA index can differ from the NVML/nvidia-smi index shown in the picker.
- Linux: llama.cpp b10621 ships no Ubuntu CUDA binary, so GGUF is unusable on Massed Compute/RunPod without a manual build. Installers otherwise work but do not follow the proven Upscaler v8 Python 3.12 provisioning flow.

## 2. Goals for v1.4.0

1. **Every user-configurable value is a Gradio control, and every control is honored by every backend.** Expose the hard-coded values above, make the GGUF path honor the frame budget, make GGUF-irrelevant controls visibly disabled, warn on every silent clamp, and harmonize spec defaults with UI defaults.
2. **Faster decoding with bit-identical output and no extra VRAM** on all backends: throttle per-token host work (Transformers), fix SSE reading and flags (GGUF), fuse gate/up and hoist the Hadamard lookup (ConvRot INT8/INT4). Measure before/after on GPU 0 for every family and publish the table.
3. **Cancel asks for confirmation** with explicit Confirm / Keep running buttons (mouse and `Esc`), and it demonstrably stops the worker.
4. **Features a regular user expects next:** Unload model button; Open in Caption Editor after a run; batch file-kind and name filters; seed; max caption length; long-video chapters + summary stage; TimeChat flatten variants (motion+camera, audiovisual, speech-only SRT, chapters); editor thumbnail gallery, dataset statistics, trainer token-limit warning, regenerate-all-filtered, ZIP export; chat prompt-preset picker; sampled-frame preview; presets/logs folder buttons.
5. **Cloud installers fixed** to the Upscaler v8 flow (fresh install only), plus an automatic Linux llama.cpp CUDA build so GGUF works on Massed Compute/RunPod.
6. **Verified like a user in Chrome**: every tab, every model family and quant path, single + batch + Unicode paths, presets, cancel, recover, editor, dataset, settings, health; then GPU 0 released.

## 3. Work breakdown

| Task | Owner files | Summary |
|---|---|---|
| A — backend parameters & pipeline | `vcap/pipeline/{job,runner,chat,worker,client}.py`, `vcap/core/*`, `vcap/models/{registry,offload,loader,downloads,vram_presets,attention,omni_common,timechat,avocado,qwen3_omni}.py`, `vcap/prompts/*`, `secourses_app.py` | New spec fields and their use; max_frames=0; clamp warnings; audio rate; seed; encode settings; summary stage; TimeChat flatten presets; batch filters; `CUDA_DEVICE_ORDER`; default harmonization |
| P — performance | `vcap/models/llamacpp_backend.py`, `vcap/models/quant/convrot.py`, `omni_common.py::_stopping` only, `runner.py::_Emitter.progress/_emit_running_item` only, `tools/bench/*` | Throttled progress; SSE fix; GGUF flags/frames/seed/options; gate/up fusion; Hadamard hoist; before/after benchmark table |
| U — UI | `vcap/ui/**`, new `vcap/core/caption_stats.py`, `vcap/core/archive.py` | All new controls with descriptions and best defaults; GGUF-aware enabling; cancel confirm; unload; open-in-editor; editor/dataset/chat features; prompt-edit preservation |
| I — installers | `Massed_Compute_Install.sh`, `RunPod_Install_*.sh`, instruction txts, `Windows_Run_*.bat`, `vcap/models/llamacpp_install.py`, README install sections | Upscaler-v8 flow; Linux llama.cpp build; docs |
| Final (Claude) | version, changelog, README, BENCHMARKS, PLAN, QA doc | Integration, full pytest, Chrome QA of everything, GPU release |

## 4. Verification plan (Chrome, GPU 0 only)

1. Restart via `Windows_Run_Video_Captioner_Pro.bat`; confirm no console errors; dark and light theme.
2. Single file: upload, file path, Unicode path, trim, scene split, all output formats, prefix/suffix/replace/trigger, max caption chars, seed reproducibility (sampled run twice), summary stage, open output / last caption / reveal clip / open in editor.
3. Cancel: start a long job, click Cancel → Confirm; verify worker stops and status says cancelled; Keep running leaves the job untouched; `Esc` path.
4. Batch: Unicode folder, kinds filter, name filter, recursive, skip/overwrite, limit, mirrored outputs, per-item errors continue.
5. Models: TimeChat BF16/INT8/INT4, AVoCaDO INT4/INT8, Qwen3 Instruct INT4/INT8/GGUF Q4/GGUF Q8, Thinking INT4/GGUF Q4 (reasoning file), Captioner INT4/GGUF Q4 (audio) — each produces a valid caption; model switch releases VRAM; Unload button frees VRAM.
6. Chat: preset picker, media attach, two turns, stop, save.
7. Editor: scan outputs, gallery, filters, token-limit flag, autosave, approve/reject, regenerate one + regenerate filtered, find/replace, bulk edit, statistics, export approved + ZIP.
8. Dataset: fitness analysis, plan JSON, TOML, sub-split tool.
9. Settings: dirs incl. logs dir, notifications, theme persistence; Recover: load + apply; Health: env report, llama.cpp status, unload.
10. Presets: save (Unicode name) / load / delete / reset / last-used autoload; shipped presets protected.
11. Speed table before/after; final `pytest tests -q`; GPU 0 emptied.

## 5. Status (2026-09-02, integration)

- Tasks A, U, I: complete (see `temp/codex_v14/codex_{A,U,I}_log.txt` final reports). Task P: code complete; its final
  benchmark table is in `docs/QUANT_PERF.md` ("v1.4.0 host-overhead removal").
- Round 2 (tasks F1 backend, F2 UI; contract `temp/codex_v14/CONTRACT_F.md`, audit `AUDIT_EXPOSURES.md`): complete.
  Full suite after integration: 375 passed, 7 skipped.
- Task S (static KV cache + CUDA graphs): measured; see `docs/QUANT_PERF.md` ("v1.4.0 static cache / CUDA graph probe").
- Fix rounds from the Chrome verification: F3 (`temp/codex_v14/TASK_F3.md`: preset delete confirmation, ZIP
  single-folder descent, dropdown custom values, prompt-preset label, retry of unreadable items, input-mode race,
  case-preserving replacements, chat final stats) and F4 (`TASK_F4.md`: live-log panel after cancel, first Start after
  a model change, editor regenerate input count, recent-run refresh), F5 (status counts), F6 (incompatible
  regenerate pairs, hidden-tab note), F7 (regenerate prompt-preset population). Final suite: 393 passed, 7 skipped.
- Chrome verification: `docs/QA_VERIFICATION_v1.4.0.md`.
