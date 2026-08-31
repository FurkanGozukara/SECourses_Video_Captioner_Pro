# QA verification log — v1.1.0 (2026-08-31)

Hands-on verification of every feature area of SECourses Video Captioner Pro against `task.txt`, performed on the release machine (Windows 11, RTX 5090 = GPU 0, Python 3.12, torch 2.13+cu130, transformers 5.16, gradio 6.26). The app was launched through `Windows_Run_Video_Captioner_Pro.bat` and driven in Google Chrome like a user; CPU-only checks ran through pytest (`161 passed`).

## Defects found by testing and fixed in v1.1.0

| Area | Defect (verified) | Fix |
|---|---|---|
| Generation | Qwen3-Omni folders carry no `eos_token_id`, so every caption ran to `max_new_tokens` (a 20 s clip took ~6.5 min for a 250-token caption) | explicit `<\|im_end\|>`/`<\|endoftext\|>` stop ids on load + generate, `finish_reason` in metadata |
| Packaging | `.gitignore` pattern `models/` also ignored the `vcap/models/` **source package** (v1.0.0 commit had none of it) | root-anchored patterns (`/models/`, `/outputs/`, …) |
| Post-processing | trigger word `ohwx` was injected into every caption/SRT cue by default | default `Prompt only / none`; presets explicit; cues never get prefix/suffix/trigger |
| Cancel | one click cancelled, bare `Esc` cancelled | two-stage confirm (6 s), Esc arms/confirms only while a job runs on the Caption tab |
| Theme | no System mode, dark not enforced on light-OS browsers; light mode had invisible checkboxes and a black log box | Dark (default) / Light / System with OS listener; CSS variables for light mode |
| VRAM plans | the auto plan overrode an explicitly selected variant, then applied INT8 offload settings to INT4 (7.9 tok/s); Qwen3 GGUF hit the "<8 GB" placeholder | keep the selected variant, use the tier plan tuned for its precision, skip unsupported tiers |
| Sliders | `max_new_tokens` ceiling shrank below a loaded value → Gradio rejected later events (`Value 9216 is greater than maximum 8192`) | global ceiling; backends clamp per family |
| torch.compile | CUDA-graphs default corrupted `DynamicCache` (TimeChat/Qwen3) and the eager fallback did not rescue the item | Inductor `default` is the default, CUDA-graph modes removed, eager restore + one retry |
| Outputs | empty `*_reasoning.txt`/`.srt` files, duplicate run_log lines, `<stem>_segments` copies for single segments, clips persisted with "Save produced clips" off | fixed in runner/captions_post |
| Batch | subfolders flattened, trim applied to batch items, sidecar `.txt` captioned, scan count vs skip mismatch | `source_root` mirroring, batch-safe trim, sidecar exclusion, output-side scan |
| Editor | invalid regex / no-selection handler arity errors, `0` treated as a max limit, run-dir items had no media, export skipped approved items | fixed, media resolved via `metadata.json`, caption-only export mode |
| Attention | Sage/xFormers were SDPA aliases shown as available | real `AttentionInterface` adapters (validated vs SDPA) with safe fallback; FA2 load failure falls back |
| Downloader | status protocol mismatch (no progress fraction in UI), GGUF absent from the menu, stale sizes | JSON `VCAP_STATUS`, six GGUF menu entries, human-readable status |
| Misc | multi-GPU not exposed, sampling strategy dead, chat mode missing, global settings not persisted, Recover reset theme/paths, hotkeys unscoped | all implemented (see changelog v1.1.0) |

## Browser runs performed after the fixes (GPU 0)

| Test | Result |
|---|---|
| TimeChat INT4, 20 s storm video, scene split, SRT on | 6 clips, every generation stopped by EOS (~470 tok / 13 s), job 87.6 s, txt+json+srt+clips, no trigger prefix, run_log de-duplicated |
| AVoCaDO INT4, same video | 6 clips in 41.3 s, coherent caption; two-stage cancel verified on a second run (armed label → cancelled, partial clip captions kept) |
| Qwen3-Omni Captioner INT4, 18 s MP3 | prompt-free audio captions per chunk (LTX preset 5 s clips); cancelled after 2 chunks — partial results saved |
| Qwen3-Omni Thinking INT4, PNG image | reasoning split to `*_reasoning.txt` and the hidden Reasoning tab, EOS, 56.6 s, peak 16.9 GB; prompt preset auto-adapted to image |
| Qwen3-Omni Instruct GGUF Q4 (llama-server) | 6 clips in 29.7 s at ~270 tok/s |
| Qwen3 Instruct INT4 folder batch (unicode folder with `sub1 dir/`, `sub2/`, image, wav, sidecar txt) | 4/4 done in 87 s, subfolders mirrored, sidecar excluded, `summary.json`; rerun → 4 skipped in 0.1 s; Rescan shows "4 already captioned" |
| torch.compile (Inductor default) TimeChat INT8 | compiled, EOS, 1 done / 0 failed in 58 s (compile adds ~12 s; decode is not faster than eager on this stack) |
| Chat tab, Qwen3 INT4 + image, two turns | "…test pattern…" then "Color bars, timer, and checkered squares." (5 words, context kept), EOS, conversation saved to `outputs/0024_chat_qwen3` |
| Presets | save/load/delete with unicode name; default presets refuse deletion; last-used auto-loads |
| Editor / Dataset / Recover / Health | scan, approve, find&replace diff preview, TOML generation, sub-split (6 clips), clip-fitness verdicts, recover run folder + compare table, llama.cpp/compile cards |
| Theme | fresh browser → dark; Light/System/Dark switching; light-mode checkbox borders and log box fixed |

## Known limitations / follow-ups

- Hugging Face folder `MonsterMMORPG/Wan_GGUF/Video_Captioner_Pro/` was still empty during QA, so real first-use downloads could not be exercised; the downloader was validated with `--status/--list/--verify` on local files and unit tests of the status protocol.
- torch.compile gives no decode speed-up for these models on torch 2.13 (Inductor default: TimeChat 20.9 vs 31.7 tok/s eager); it stays opt-in.
- The Gradio dropdown option list fades out slowly after a selection (cosmetic, Gradio 6 behaviour).
- TimeChat timestamps inside very short clips can exceed the clip length (model behaviour); SRT cues are offset per clip but not clamped.
- BF16 Qwen3-Omni (63 GB) is not runnable on a single 32 GB GPU and remains unmeasured.

Superseded by QA_VERIFICATION_v1.2.0.md after the Hugging Face upload completed; the empty-folder limitation is resolved.
