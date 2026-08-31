# QA verification log - v1.2.0 (2026-09-01)

This release was verified through the real SECourses Video Captioner Pro interface in Google Chrome on Windows, using physical GPU 0 (NVIDIA GeForce RTX 5090). The pass covered fresh-install downloads, every model family and quantization path, mixed-media batches, output generation, recovery, editing, and system controls. The final automated regression suite completed with `180 passed`.

## Fresh-install download verification

The Hugging Face catalog previously requested `MonsterMMORPG/Wan_GGUF/Video_Captioner_Pro/<key>`, while the uploaded model folders were stored at the repository root as `<key>/`. That made every first-use download fail as unavailable despite the upload being complete.

| Check | Verified result |
|---|---|
| Catalog layout | Catalog entries now use the uploaded root folders in `MonsterMMORPG/Wan_GGUF`. All 17 uploaded folders were compared with the local quantized outputs; file names and byte sizes matched. |
| Alternate layout | The downloader probes both the root and `Video_Captioner_Pro/`-prefixed forms and caches a successful remote index for 24 hours. |
| File integrity | Hugging Face downloads validate expected size and SHA-256 for every catalog file before the model is marked ready. |
| GGUF fallback | Qwen3-Omni GGUF files fall back from the primary `ggml-org`/`mradermacher` source to `MonsterMMORPG/Wan_GGUF/<folder>/<file>`, then normalize the result to the expected flat local layout. Resume and cancellation remain active. |
| Disk preflight | Hugging Face and GGUF downloads require the remaining bytes plus 5% headroom. Insufficient space is reported as a clear `VCAP_STATUS` error before transfer begins. |
| Clean live run | `models/timechat_int4` was deleted, then **Start Captioning** automatically downloaded 16 files (6.0 GB) at 90-178 MB/s with 32 Xet streams, verified SHA-256 at about 1.6 GB/s, loaded the model, and completed a caption. Progress appeared in both the UI live log and console. |
| Bulk GGUF fetch | Thinking Q4, Instruct Q8, Captioner Q8, and Thinking Q8 downloaded sequentially to a verified ready state. The four downloads totaled 121 GB. |

## Full model x quantization matrix

All completed generations stopped at EOS unless a row explicitly says otherwise. Times include loading.

| Variant | Input | Result |
|---|---|---|
| `timechat_bf16` | 20 s video | 1,119 tokens at 33.7 tok/s; peak 16.7 GiB |
| `timechat_int8` | 20 s video, scene split into 5 clips | About 26 tok/s; SRT, clips, and sidecars verified |
| `timechat_int4` | 20 s video after a fresh-install download | About 25 tok/s |
| `avocado_bf16` | 20 s video, whole-file mode | 249 tokens at 30 tok/s |
| `avocado_int8` | 20 s video, whole-file mode | 330 tokens at 22 tok/s |
| `avocado_int4` | Video trimmed to 5-15 s and split into 3 clips | About 25 tok/s; trim preserved in metadata and timestamps |
| `qwen3_omni_instruct_bf16` (63.4 GB) | Image | Loaded through Accelerate CPU offload in 97.5 s; peak 51.6 GiB with WDDM spill; 0.42 tok/s; stopped at the configured length cap |
| `qwen3_omni_instruct_int8` | Image | 234 tokens at 4.3 tok/s; peak 30.2 GiB |
| `qwen3_omni_instruct_int4` | Unicode mixed batch: 2 videos, image, and WAV in a subfolder; sidecar excluded | 4/4 completed in 100.7 s with per-modality prompt adaptation; rerun skipped all 4 in 0.09 s |
| `qwen3_omni_instruct_gguf_q4` | 20 s video split into 5 clips | 30.6 s total; about 270 tok/s decode |
| `qwen3_omni_instruct_gguf_q8` | 20 s video, whole-file mode | Server completed in 15.5 s; peak 28.0 GiB; 9.6 tok/s |
| `qwen3_omni_thinking_int8` | Image | 497 tokens at 4.6 tok/s; separate reasoning file verified |
| `qwen3_omni_thinking_int4` | Image | 725 tokens at 12.4 tok/s; 2.4 KB reasoning file |
| `qwen3_omni_thinking_gguf_q4` | 20 s video | 784 tokens at 163 tok/s; reasoning verified |
| `qwen3_omni_thinking_gguf_q8` | 20 s video | 1,281 tokens at 26.5 tok/s; reasoning verified |
| `qwen3_omni_captioner_bf16` (63.4 GB) | 15 s MP3 | EOS after more than 450 tokens at about 0.5 tok/s; 1,347.7 s total including the CPU-offload load; audio tower functional under CPU offload |
| `qwen3_omni_captioner_int8` | 15 s MP3 | 492 tokens at 4.5 tok/s |
| `qwen3_omni_captioner_int4` | 15 s MP3 with prefix, suffix, replacement, and all 5 output formats | TXT, JSON, JSONL, SRT, and VTT verified with post-processing applied |
| `qwen3_omni_captioner_gguf_q4` | 15 s MP3 | 447 tokens at 300 tok/s |
| `qwen3_omni_captioner_gguf_q8` | 15 s MP3 | 481 tokens at 41.5 tok/s |
| `qwen3_omni_thinking_bf16` | Not run | Skipped because it uses the same proven 63.4 GB loader/offload path as Instruct BF16, while its reasoning path was covered by INT4, INT8, and both GGUF variants. Another 63 GB run would not have covered a new code path. |

The matrix exercises every model family and BF16, INT8, INT4, GGUF Q4, and GGUF Q8 execution paths. All six GGUF family/quantization combinations completed end to end. The Instruct and Captioner BF16 runs establish that both visual and audio towers function through automatic CPU offload on the 32 GB test GPU.

## Browser workflow verification

| Area | Verified result |
|---|---|
| Theme | A fresh browser opened in Dark mode. Light mode remained readable. System mode followed live OS color-scheme changes in both directions, and the selected mode re-synchronized from local storage after reload. |
| Cancellation | The first click armed cancellation, the control disarmed after 6 seconds, and a second click during the armed window cancelled the run. Partial outputs were retained and the summary recorded `cancelled=1`. |
| Input preview | Upload chips, media preview, duration/resolution/FPS/audio probe text, and video trim controls were present and synchronized. |
| Batch | A Unicode folder with nested content and a caption sidecar scanned correctly. Recursive mode updated counts, source paths remained authoritative instead of Gradio temporary copies, nested folders were mirrored to the batch output, and a rerun skipped completed items. |
| Task presets | Preset choices, descriptions, and rendered prompts followed model-family and modality changes atomically. With no selected input, each family uses its primary prompt modality rather than producing an empty dropdown. |
| Chat | A media path was attached, two streamed turns retained context (answer `8`), and the conversation was saved under `outputs/0047_chat_qwen3`. |
| Caption Editor | A 50-item scan, selection, media preview, live character/word/token counts, on-disk autosave, approval flags/counts, regex preview and apply, contains filtering (50 to 8), pagination, and approved media/caption export were verified. |
| Dataset & Export | Clip fitness reported `will be dropped: 50f < 81f`; Musubi TOML generation detected the video folder automatically. |
| Recover Settings | A recent run produced a per-key comparison, and **Apply to UI** restored the model. Machine and theme keys stayed excluded. Runtime aliases now restore the compile-mode dropdown and recursive batch checkbox without unknown-key warnings. |
| Presets | A Unicode preset saved, appeared as current, and deleted from disk. Protected starred defaults refused deletion. The last-used preset loaded on a fresh page. |
| System & Models | The environment report, GPU table, pinned llama.cpp b10621 runtime, per-variant GGUF readiness, MSVC 14.44 toolchain, and Triton status were verified. |
| VRAM plans | The 32 GB plan displayed precision-specific placement messages, including the INT8 tail-offload note and GGUF tier configuration. **Show all variants** exposed the 63.4 GB BF16 entries. |

## Release regressions closed

| Observation | v1.2.0 disposition |
|---|---|
| Fresh-page model changes emptied the task-preset choices and both prompt fields after inputs were cleared. | No-input modality now falls back to the selected family's primary modality; choices and a valid selection remain populated, and prompt-bearing families render their user prompt. |
| Recover Settings warned that `compile_mode` and `recursive` were unknown. | Runtime metadata aliases map to the registered `torch_compile_mode` and `batch_recursive` controls. |
| Single-file status showed `ETA: ETA 56s`. | The duplicated prefix was removed and per-segment ETA was added. |
| Consecutive split messages repeated in `run_log.txt`. | Adjacent persisted log lines are deduplicated. |
| SRT/WebVTT cues could extend beyond their source clip. | Cue windows are clamped to the real clip interval with a minimum cue duration. |
| Replacement chips updated only after focus changed. | The preview now refreshes while typing. |
| Windows emitted a warning for the unsupported `expandable_segments` allocator setting. | The flag was removed on Windows and is stripped from child-worker environments. |

## Retracted automation observations

Two apparent defects were traced to Chrome-driver operations rather than application behavior and were retracted as app bugs:

- **Uncommitted dropdown fills:** the driver inserted a value into a dropdown's editable field without committing the option, so Gradio correctly received no selection/change event. Repeating the flow with a committed user-style selection synchronized the UI.
- **Programmatic-empty textarea sync:** the driver emptied a textarea through programmatic DOM mutation without dispatching the normal Gradio input event. Keyboard clearing and normal UI input synchronized correctly.

No release blocker remained after the live matrix, workflow pass, focused regressions, and full automated suite.
