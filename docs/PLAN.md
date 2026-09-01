# SECourses Video Captioner Pro — Master Plan (v1)

Owner/orchestrator: Claude (planner, verifier). Implementers: codex (gpt-5.6-sol, YOLO) tasks.
Knowledge base: `scratchpad/reports/*.md` (reference-app analyses + model research). Every implementer MUST read the reports relevant to its task and open the referenced source files to copy proven code.

## 0. Environment facts (do not re-derive)
- Working root (distribution folder): `G:\SECourses_Video_Captioner_Pro_v1\` — holds `Windows_*.bat`, `Massed_Compute_Install.sh`, `RunPod_Install_*.sh`, instruction txts, `video_caption_requirements.txt`, `Models_Downloader.py` (canonical downloader lives HERE, outside the git repo, exactly like the Upscaler app; the app resolves it via `APP_DIR.parent / "Models_Downloader.py"` with env override `SECOURSES_VCAP_DOWNLOADER`).
- Git repo (app code): `G:\SECourses_Video_Captioner_Pro_v1\SECourses_Video_Captioner_Pro\` → remote `https://github.com/FurkanGozukara/SECourses_Video_Captioner_Pro` (branch `main`, no commits yet). Never commit `venv/`, `models/`, `outputs/`, `temp/`, `logs/`, `presets/` (user presets), media.
- venv: `SECourses_Video_Captioner_Pro\venv` — Python 3.12.10, torch 2.13.0+cu130, gradio 6.26.0, flash_attn 2.8.3 (sm86/sm89/sm120a), sageattention 2.2.0, xformers 0.0.35, torchao 0.18.0, triton-windows 3.7.1, transformers 5.16.x (installed by us). NO pinned versions in requirements except what already exists.
- GPU for all tests/benchmarks: **GPU 0 = RTX 5090 32 GB (sm120)**. GPU 1 (RTX 3090) must NOT be used.
- ffmpeg/ffprobe on PATH (n8.1). Chrome at `C:\Program Files\Google\Chrome\Application\chrome.exe`.
- Temp/work folder for everything non-repo: `F:\SECourses_Video_Captioner_Pro_TEMP\` (`originals/` = HF downloads of the 5 base repos, `converted/`, `logs/`, `test_media/` incl. unicode-named files & `batch_ünicode_folder/`). Scratch for small files: the Claude scratchpad dir.
- Models local layout (runtime): `SECourses_Video_Captioner_Pro\models\<model_key>\` (e.g. `models\qwen3_omni_instruct_int8\`). Each folder is a self-contained HF-loadable dir: `config.json`, `generation_config.json`, `preprocessor_config.json`, tokenizer files, `chat_template.*`, plus ONE weight file `model.safetensors` (bf16 single-file, or int8/int4 ConvRot single-file with `_quantization_metadata` header) or `model.gguf`.
- HF layout (user uploads after we produce files): `MonsterMMORPG/Wan_GGUF/Video_Captioner_Pro/<model_key>/<same files>` — 1:1 mirror of the local folder so `Models_Downloader.py` maps trivially. Downloader must verify size+sha (LFS etag) and resume.

## 1. Product scope (what "done" means)
A Gradio 6 desktop-style app (Windows + Linux, RTX 3000+ and cloud GPUs) that captions **video, audio, image, text** with:
1. Models: TimeChat-Captioner-GRPO-7B (Qwen2.5-VL-7B based, video/image), AVoCaDO (Qwen2.5-Omni-7B thinker, audiovisual video), Qwen3-Omni-30B-A3B **Instruct / Thinking / Captioner** (video+audio+image+text; Captioner = audio captioning). Variants per model: **bf16 / INT8 ConvRot / INT4 ConvRot(W4A8)** produced by us + **GGUF** variants found in non-gated repos (only if actually runnable for the modality; text-only GGUF must be labeled). Model dropdown shows sizes: `Qwen3-Omni Instruct — INT8 ConvRot (32.4 GB)`.
2. Single-file mode: ONE upload point (`gr.File(file_count="multiple")`) that auto-detects type and shows the right preview (video player / audio player / image); also a **file path textbox** that behaves identically; mixed multi-file allowed; per-file capability check with clear message "Model X does not support audio-only input; use Qwen3-Omni …". Optional trim (start/end seconds) applied before processing. Text-prompt-only queries allowed for Qwen3-Omni.
3. Batch folder mode: same pipeline and settings as single (zero duplicated logic); light scan shows counts per type; skip-already-processed with overwrite checkbox; output named as input (`clip.mp4 → clip.txt/.json/.srt`), recursive optional; errors continue and are logged; per-file status + ETA.
4. Preprocessing (shared): frame sampling (uniform / fps target / max_frames / keyframe / adaptive), resolution cap (max_pixels, keep aspect ratio; model-aware), trim, scene detection (PySceneDetect: threshold, min/max scene length, merge-short, fade), model max-duration auto-cap/auto-split with warnings, stream-copy vs re-encode split toggle, sub-split with overlap for target trainer, auto-reject (too short / mostly black / static / low sharpness / silent), save clips+captions so the output is a ready dataset.
5. Post-processing: prefix, suffix, trigger word injection, replace words (`a;b` pairs, `;` and newline separated, shown as key→value chips), regex option, whole-word/case options; output formats: `.txt` always, `.json` (prettified) when structured, `.srt/.vtt` for timestamped outputs (TimeChat timestamps, ASR), `.jsonl` dataset lines, `reasoning.txt` for Thinking (hidden in UI by default), `metadata.json` (every parameter + processing time + model + timings), `run_log.txt`.
6. Outputs: single runs → `outputs/0001_<model_short>/` never overwriting; batch → user output folder (default `outputs/batch_<timestamp>/` or same folder as input when chosen) with metadata/logs under `outputs/…` to avoid polluting the target folder. "Save every processed file" (intermediate clips/frames/audio) global toggle, default off.
7. VRAM presets: auto-detected tier (6/8/10/12/16/24/32/48/80 GB) → per-model parameter set (quant variant, max_pixels, max_frames, fps, offload/block-swap/expert-offload, attention, max_new_tokens). Only variants that fit are offered at a tier. Runtime knobs: quant selector, attention backend (SDPA/Flash2/Sage/xFormers/eager + auto + fallback), torch.compile toggle (warn first run slow), keep-model-loaded + idle unload timer, OOM auto-recovery (drop resolution/frames one notch, retry, log), live VRAM/RAM meter, peak VRAM per run in metadata, GPU picker (single id) + multi-GPU data-parallel batch splitting (one worker per GPU).
8. Subprocess mode (default ON): pipeline runs in worker subprocess(es) (JSON-lines protocol, log streaming, tree-kill cancel with confirm). Stages: (A) media prep (ffmpeg/scene split, CPU) and (B) captioning (GPU) — B consumes A's files; persistent B worker when "keep model loaded" is on (idle timer). In-app mode uses the same pipeline code in-process.
9. Caption Editor / Analyzer tab: source folder (default `outputs/`), scan → table/gallery; player + caption editor side-by-side; prev/next hotkeys; autosave-on-edit toggle; regenerate selected with different prompt/model; find & replace (regex opt) across folder; bulk prefix/suffix/trigger; approve/reject flags → export approved only; diff view on re-caption; filters (length, token count, contains, failed, empty, flag).
10. Dataset tools: target-trainer clip fitness (Wan / Hunyuan / LTX 2.x / MiniMax H3 / custom) with frame-count suggestions and "will be dropped" warnings, resolution bucket preview (480p/720p/1080p, keep-AR / letterbox / crop / area-resize), sub-split with overlap 0/0.5/1s, kohya/musubi dataset TOML export (video+image datasets), export approved-only.
11. Presets: universal preset (all tabs, all settings except file/folder paths), default presets in read-only `presets_default/` (cannot be overwritten/deleted from UI), user presets in `presets/`, save/load/delete, load-or-save marks last-used, last-used auto-loads at start. Task/prompt presets per model (see §6) + fully custom prompt text.
12. Timings: per step, elapsed, ETA, processed/left for single & batch, tokens/s; on Gradio AND console. Model downloads & loading show progress on both too.
13. Metadata Recover tab: load any `metadata.json` → apply all settings to UI.
14. Robustness: non-ASCII paths, `/`, `\\`, `//`, quoted paths; all `open()` utf-8; subprocess env `PYTHONUTF8=1`; continue on per-file failure; typed errors with plain-language messages; every parameter has a description (`info=`) and model-appropriate ranges; presets validated/clamped.
15. Installers: `Windows_Install_and_Update.bat`, `Windows_Run_Video_Captioner_Pro.bat`, `Windows_Download_Models_and_or_Resume.bat`, `Massed_Compute_Install.sh`, `RunPod_Install_Video_Captioner_Pro.sh`, instruction txts — all pointing at this app/repo.
16. Verified by really running the app in Chrome (single, batch, editor, presets, cancel, downloads, both themes) and a speed/VRAM benchmark table (bf16 where it fits on 32 GB, int8, int4, gguf).

## 2. Architecture

### 2.1 Package layout (repo root = `SECourses_Video_Captioner_Pro/`)
```
secourses_app.py                 # entry: parse args (--share --server --port --no-browser --windows-int8-defaults-style flags), build UI, launch
vcap/
  __init__.py                    # APP_DIR, VERSION, dirs (models/outputs/temp/logs/presets/presets_default)
  core/
    paths.py                     # normalize_path (quotes, ~, env, / \ // UNC), sanitize_filename (keep unicode), natural sort, allowed drive roots, collision-safe naming
    logs.py                      # AppLogger: console + ring buffer (UI) + per-run run_log.txt; console progress line renderer (copy SwarmUI console_manager.py); utf-8 stdio setup
    progress.py                  # ProgressTracker (steps, ETA, tok/s), dual sink (gr.Progress + console + log), throttled UI yields
    gpu.py                       # NVML enumeration without CUDA context, VRAM/RAM meter, tier detection, oom text detection, cuda cache clear
    media.py                     # MediaInfo (ffprobe json), kind detection (video/video_no_audio/audio/image/text), preview-safe transcode cache, audio extraction (16k mono wav), frame extraction, duration/fps
    scene_split.py               # PySceneDetect detection (progress), merge-short, min/max length, cap-to-model-limit, sub-split w/ overlap, ffmpeg copy vs precise split (copy from Upscaler chunking.py), manifest
    preprocess.py                # trim, frame sampling plans, resolution cap math (max_pixels/min_pixels/AR), auto-reject analyzers (black/static/sharpness/silence)
    outputs.py                   # run dir allocation `NNNN_<model_short>` (atomic mkdir), batch layout, write_outputs(formats), metadata.json, run_log.txt, open_folder/reveal_file cross-platform
    presets.py                   # PresetStore: defaults dir (read-only) + user dir, save/load/delete/list, last-used, forward-compat merge, path keys excluded
    registry.py                  # SettingsRegistry: ordered (section,key,component,default,meta) triples — single source of truth for presets/metadata/handlers
    captions_post.py             # prefix/suffix/trigger/replace (single-pass regex), TimeChat JSON→flatten, SRT/VTT builders, JSON prettify, jsonl
    export.py                    # kohya/musubi TOML (video + image datasets), approved-only export, dataset copy
    clip_fitness.py              # trainer targets (Wan 81 @16fps, Hunyuan 129, LTX 8k+1 up to 121/257 @25fps, MiniMax H3 17n+5), bucket preview, warnings
    subprocess_runner.py         # Popen w/ utf-8, JSON-lines event protocol, log streaming generator, tree-kill (psutil + taskkill/killpg), cancel tokens, confirm-arm
  models/
    registry.py                  # MODEL_SPECS: family → variants (key, label, quant, size_gb, files, hf_subfolder), capabilities, limits, param schema, prompt presets ids, vram table
    base.py                      # BaseCaptioner (load/unload/caption/capabilities), MediaInput, CaptionResult dataclasses
    loader.py                    # single-file safetensors loading into transformers model (bf16 direct; int8/int4 convrot via QuantLinear swap; gguf via dequant-on-load or on-the-fly), attention selection, device placement/offload
    quant/convrot.py             # Hadamard, ConvRotInt8Linear / ConvRotInt4W4A8Linear (torch._int_mm path + bf16 fallback for small M), metadata parsing
    quant/gguf.py                # gguf reader → dequant to bf16 tensors (ComfyUI-GGUF dequant.py port) for LLM weights
    offload.py                   # VRAM budget, layer-wise offload (accelerate device_map/max_memory) + optional block-swap hooks + MoE expert offload for Qwen3-Omni
    attention.py                 # backend registry (auto/sdpa/flash2/sage/xformers/eager), availability probe, fallback, sdpa_kernel ctx
    vram_presets.py              # tier tables per model → settings
    timechat.py                  # Qwen2_5_VLForConditionalGeneration wrapper (qwen_vl_utils), prompts, timestamped script parsing
    avocado.py                   # Qwen2_5OmniThinkerForConditionalGeneration wrapper (qwen_omni_utils), use_audio_in_video
    qwen3_omni.py                # Qwen3OmniMoeThinkerForConditionalGeneration wrapper (thinker-only, no talker), Instruct/Thinking/Captioner; <think> separation
    downloads.py                 # ensure_model(key) → subprocess Models_Downloader.py --ensure <key> with streamed progress (console+UI) + cancel
  prompts/
    presets.py                   # PROMPT_PRESETS: id, label, description, applies_to (models/modalities), system, user template with {{TRIGGER}} {{LANGUAGE}} etc., output_format hint, post-processor
  pipeline/
    job.py                       # JobSpec (settings dict → typed), InputItem, SegmentPlan
    runner.py                    # run_job(spec, sinks): probe→capability→trim→split→preprocess→caption→post→save; error-continue; OOM recovery; multi-GPU split
    worker.py                    # subprocess entry `python -m vcap.pipeline.worker` (persistent JSON-lines server: load/caption/unload/ping)
  ui/
    theme.py                     # gr.themes + CSS (colored buttons: hue classes generated; light+dark), hotkeys head JS
    components.py                # preset bar, media input block (single upload point + auto preview), replace-pairs chips, log panel, VRAM meter, buttons
    app.py                       # Blocks assembly + tabs wiring, gr.Timer for meters/logs
    tabs/caption_tab.py          # Caption (Upload files | File path | Folder batch sub-modes; shared settings)
    tabs/editor_tab.py           # Caption Editor / Analyzer
    tabs/dataset_tab.py          # Clip fitness + export/TOML
    tabs/settings_tab.py         # Global settings
    tabs/recover_tab.py          # Metadata recover
    tabs/health_tab.py           # GPU/health/models status + download buttons
    tabs/changelog_tab.py
presets_default/*.json           # shipped read-only presets
tools/
  quantize/merge_shards.py       # streaming shard merge → single bf16 safetensors
  quantize/convrot_quantize.py   # INT8 / INT4-W4A8 ConvRot converter for HF transformers models (LLM/VLM/MoE aware exclusions)
  quantize/verify.py             # logits/caption comparison bf16 vs quant
  bench/benchmark.py             # speed & VRAM table (bf16/int8/int4/gguf) on GPU 0
  find_gguf.py                   # scans HF for GGUF variants of our base models (non-gated)
docs/                            # PLAN.md, MODEL_FILES.md, GGUF_BACKEND.md, QUANT_REPORT.md, and reports/
tests/                           # pytest for paths/presets/captions_post/scene_split/outputs/registry (CPU-only)
```

### 2.2 Settings registry (one source of truth)
Every UI control is registered `reg(section, key, component, default, *, in_preset=True, in_metadata=True, model=None)`. Handlers receive `*values` and rebuild dicts via the registry order (Image Captioner ORDER/DEFAULTS + Upscaler `TAB_CONFIGS` pattern; Musubi `reg()` pattern). Paths (`input_path`, `batch_input_folder`, `batch_output_folder`, `editor_folder`) are `in_preset=False`. Presets = JSON `{"_meta": {...}, "settings": {key: value}}` with forward-compat merge and clamping to the current control ranges/choices.

### 2.3 Pipeline (single == batch)
`run_job(JobSpec)` where `JobSpec.inputs: list[InputItem]` (1..N). Steps per item → segments → captions. Batch folder mode just builds inputs from a folder scan and sets output naming mode `mirror_input_names`. Everything else identical. Events: `log`, `progress(step, i, n, eta, tok/s)`, `item_done(status)`, `result`. Sinks: console (always), UI ring buffer (Timer-polled), run_log.txt.

Worker protocol (subprocess mode): parent spawns `python -u -m vcap.pipeline.worker --gpu 0` once; sends JSON lines `{"cmd":"run_job","spec":{...}}`; worker streams `{"ev":"log"|"progress"|"item"|"result"|"error"}` lines; `{"cmd":"unload","unless_variant":...}` returns an `unloaded` event, alongside `{"cmd":"ping"}` and `{"cmd":"exit"}`. Keep-model-loaded = keep worker alive; idle timer sends `unload`/`exit`. Cancel = tree kill (psutil children + `taskkill /T /F` / `killpg`) then worker restart on next run. Media prep (ffmpeg/scene detection) runs in the parent process thread pool (CPU) or in the worker before GPU work — decision: run prep in the worker too (single code path), but model loading is deferred until the first caption call so prep never holds VRAM.

### 2.4 Model interface
```python
class BaseCaptioner:
    spec: ModelSpec
    def load(self, variant_key, device, attention, offload_plan, compile_flag) -> LoadReport(peak_vram, seconds)
    def unload(self)
    def caption(self, media: MediaInput, prompt: PromptSpec, gen: GenParams, pre: PreprocessParams, cb: Callbacks) -> CaptionResult
    # capabilities: {"video","video_audio","audio","image","text","multi_image"}; limits: max_duration_s, default_fps, max_frames, max_pixels, min_pixels, max_new_tokens_cap, context_tokens
```
CaptionResult: `text`, `raw_text`, `reasoning` (Thinking), `structured` (parsed JSON e.g. TimeChat), `segments` (list of (start,end,text) for SRT), `usage` (prompt/generated tokens), `timing`, `peak_vram`.

### 2.5 Quantized single-file format (our own)
- bf16: standard HF single `model.safetensors` (merged shards, same key names) → loads with `from_pretrained(folder)` directly.
- INT8 ConvRot: `model.safetensors` where quantized Linear layers have `<name>.weight` int8 `[out,in]`, `<name>.weight_scale` fp32 `[out,1]`, header `__metadata__["_quantization_metadata"]` = JSON `{"format_version":"1.0","scheme":"int8_convrot","group_size":256,"layers":{name:{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}}}`; excluded layers stay bf16. Loader builds the model on `meta` device, swaps quantized Linears with `ConvRotInt8Linear`, loads with `assign=True`. Kernel: rotate act → per-token int8 → `torch._int_mm` → epilogue; for M < 32 (decode) use dequant-to-bf16 matmul path (cached dequant? no — compute on the fly per call; benchmark both) — implementer must benchmark decode and pick the faster path automatically per M.
- INT4 ConvRot (W4A8): `<name>.weight` int8 `[out,in//2]` packed nibbles, `<name>.weight_scale` fp32 `[out,1]`, scheme `int4_convrot_w4a8`; unpack to int8 on the fly then same int8 path (or dequant bf16 for small M).
- Exclusions (bf16): embeddings, lm_head, norms, biases, MoE routers/gates, vision/audio patch-embed & merger/projector, first vision block, audio encoder conv front-end, `in_features % 256 != 0`; quantize attention q/k/v/o, MLP gate/up/down, MoE experts, vision/audio transformer blocks ≥1.
- GGUF: for LLM-only text weights of a base model (e.g. Qwen2.5-VL-7B GGUF) — only expose if the vision/audio towers can be paired (mmproj) — otherwise mark "text-only GGUF (no video/audio)". The registry lists found repos with sizes.

### 2.6 VRAM presets (per model, initial targets; tune after benchmarks)
| Tier | TimeChat / AVoCaDO (7B) | Qwen3-Omni-30B-A3B thinker |
|---|---|---|
| 6 GB | INT4 + CPU offload of most layers, max_pixels 128*28*28, max_frames 32, fps 1 | not offered (or INT4 + heavy offload "experimental") |
| 8 GB | INT4 + partial offload, 200k px, 48 frames | INT4 + heavy offload (experts on CPU), 128*28*28, 32 frames, slow |
| 10–12 GB | INT4 (fits ~5.5 GB) / INT8 + partial offload, 297920 px (TimeChat default) | INT4 + expert offload |
| 16 GB | INT8 (~9.5 GB) full, 297920–401408 px | INT4 (~18–20 GB) + light offload |
| 24 GB | bf16 (17.9 GB) or INT8, full res, 64–128 frames | INT4 full (fits) |
| 32 GB | bf16 | INT8 (~33 GB) + light offload, or INT4 full |
| 48 GB | bf16 | INT8 full |
| 80 GB+ | bf16 | bf16 (~61 GB) |
Every preset sets: variant, attention (flash2 if available else sdpa), max_pixels, max_frames, fps, max_new_tokens, offload plan (`gpu_layers`/`expert_offload`), batch size 1.

## 3. Model-specific facts (from research; see reports 10/11/12 for full detail — they are authoritative)
### TimeChat-Captioner-GRPO-7B (report 10) — **Qwen2.5-Omni-7B thinker fine-tune (NOT Qwen2.5-VL)**
`config.json` architectures `Qwen2_5OmniForConditionalGeneration`, all 1,346 tensors prefixed `thinker.` (no talker). 17.9 GB bf16. Load with **`Qwen2_5OmniThinkerForConditionalGeneration`** + `Qwen2_5OmniProcessor` (transformers ≥5.15 crashes the full-omni `generate()` for talker-less checkpoints → always the thinker class). Inputs: video **with audio (mandatory; mux silent track if missing)**; image untested. Prompt: NO system prompt (template injects "You are a helpful assistant."), user prompt verbatim `Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.` (6 alternative paraphrases in report). Video element first, then text. `fps=2.0`, `max_pixels=297920`, `max_frames=160` (80 for speed); pass `fps=2.0` scalar + `use_audio_in_video=True` to processor and generate. Greedy, `max_new_tokens=9216` (≥8192). Context 32,768 → recommended clip ≤ 60 s (hard ceiling ~80 s) → auto-split. Output: raw JSON array of segments with keys `timestamp ("MM:SS-MM:SS"), segment_detail_caption, camera_state, video_background, storyline, shooting_style, speech_content, acoustics_content` (~99% parse rate; keep raw on failure) → app produces `.json` (pretty), `.txt` (flattened), `.srt` (timestamps). Single-task specialist: prompt changes break the JSON format; offer "TimeChat 6D raw" and flatten presets only.
### AVoCaDO (report 11) — Qwen2.5-Omni-7B thinker-only (same class/processor as TimeChat)
17.9 GB. `qwen_omni_utils`/own decoder; `use_audio_in_video=True` (audio mandatory → pre-check, else visual-only prompt); `fps=2.0` scalar to processor; `max_pixels=401408`, `total_pixels=20070400` (per video dict); greedy; `max_new_tokens=2048`. Max ~100 s (sweet spot 10–60 s) → auto-split. System prompt `You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.` + 7 official A/V prompts + visual-only + UGC structured + dialogue-extraction/audio-caption post prompts (verbatim in report 11). Output: prose (no timestamps). Both 7B models: batch size 1, ~18 GB weights bf16.
### Qwen3-Omni-30B-A3B (report 12)
Only 30B-A3B exists (Instruct 70.5 GB incl. talker+code2wav 7.08 GB → drop; Thinking/Captioner 63.4 GB thinker-only). Class **`Qwen3OmniMoeThinkerForConditionalGeneration`** (`base_model_prefix="thinker"`), `Qwen3OmniMoeProcessor(min_pixels=..., max_pixels=...)` (pixel caps only honored at `from_pretrained` or via `size={"shortest_edge","longest_edge"}` per call), `qwen_omni_utils.process_mm_info(..., image_patch_size=16)` or our own decoder. Text MoE 48 layers × 128 experts (8 active, inter 768, hidden 2048); **transformers 5.x fuses experts into 3-D params and runs a Python expert loop → ~1–6 tok/s**; our wrapper must implement a grouped experts forward (sort tokens by expert, per-expert matmul on slices; int8/int4 experts via our ConvRot modules). Audio 13 tok/s (40 min fits 32k); video `fps/2 × tokens_per_frame_group` (cap 768 tok/group by default → ~40 s at fps 2; use `size.longest_edge = 256*32*32` → ~2 min; fps 1 → ~4 min). Images uncapped by default → set cap 1280*32*32. `fps` must be a scalar to processor and match sampling. No system prompt for captioning. Prompts (verbatim): audio `Give the detailed description of the audio.`; ASR `Transcribe the <source_language> audio into text.`; S2TT `Listen to the provided <source_language> speech and produce a translation in <target_language> text.`; lyrics `Transcribe the song lyrics into text without any punctuation, separate lines with line breaks, and output only the lyrics without additional explanations.`; SFX `What happened in the audio?`; music analysis prompt; video `Describe the video.`; scene changes `How the scenes in the video change?`; OCR `Extract the text from the image.`; joint `Describe the audio, image and video.` Generation: Instruct greedy `max_new_tokens 4096–8192`; Captioner (audio-only, prompt-free, ≤30 s, single audio) sampling t=0.6 top_p=0.95 top_k=20 `2048–8192`; Thinking t=0.6/0.95/20 `16384–32768`, `<think>…</think>` split (tokens not special → survive decode) → `reasoning.txt`. KV 96 KiB/token. **GGUF path:** llama.cpp ≥ b8775 supports Qwen3-Omni audio+vision via `ggml-org/Qwen3-Omni-30B-A3B-{Instruct,Thinking}-GGUF` (Q4_K_M 18.56 GB + `mmproj-…-Q8_0.gguf` 1.33 GB; Q8_0 32.5 GB; bf16 61.1 GB) and `mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF` (Q2_K…Q8_0 + `mmproj-Q8_0.gguf`) → implement a `llama-server` backend (download prebuilt CUDA binaries per OS + GGUF files; OpenAI-compatible `/v1/chat/completions` with `input_audio`/`input_video`; text-only GGUF repos must be labeled and are not offered by default). Community AWQ-4bit (`cyankiwi/...-AWQ-4bit` 20.5 GB Thinking/Captioner, 27.6 GB Instruct) are optional extra variants (need `compressed-tensors`).

## 4. UI design
- Theme: `gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate", font=GoogleFont("Inter"), font_mono=GoogleFont("JetBrains Mono"))`; dark baseline + light overrides; theme radio persisted (localStorage + `?__theme=`); generated hue button classes `.vc-btn-<hue>` (copy Upscaler `_build_sec_btn_css` pattern; Musubi `.mbtn` palette) — every button on a tab a different hue; Start (emerald), Cancel (red), Download (sky), Open Output (teal), Open Last Caption (violet), Reveal Clip (amber), Save preset (green), Load (blue), Delete (rose), Reset (orange), etc.
- Top bar: title + Patreon link, GPU summary, universal preset bar (dropdown numbered, save-as textbox, Save/Load/Delete/Reset, status), theme toggle.
- Tabs: 🎬 Caption | ✏️ Caption Editor | 📦 Dataset & Export | ⚙️ Global Settings | 🔁 Recover Settings | 🩺 System & Models | 📜 Changelog.
- Caption tab layout: left column = input block (sub-tabs: Upload Files / File Path / Folder Batch) with previews, trim controls, scan summary; right rail = Model (dropdown w/ sizes, variant, VRAM preset, attention, quant, offload, compile, keep-loaded), Task preset + prompts (system/user editable, template vars), Model params (accordion, per-model visibility), Preprocessing accordion, Scene detection accordion, Post-processing accordion, Output formats; bottom = action buttons row + progress/ETA/timings + VRAM/RAM meter + live log (autoscroll) + result panel (caption textbox with copy, JSON view, SRT view, reasoning hidden accordion, clip gallery for splits).
- Editor tab: folder box + Scan (auto default outputs) → `gr.Dataframe` list (file, caption preview, length, tokens, flag, status) + filters; selected item: `gr.Video/Audio/Image` preview + caption `gr.Textbox` (autosave toggle, Save button) + Approve/Reject buttons + Prev/Next (hotkeys ←/→, Ctrl+S); tools accordion: Find & Replace (regex, preview count), Bulk prefix/suffix/trigger, Regenerate selected (model/prompt override, subprocess), Diff view (old vs new HTML), Export approved.
- Every parameter has `info=` description; ranges per model from the param schema.

## 5. Output formats & files
Single run dir `outputs/0001_timechat/`:
```
input_copy.<ext> (only if "save every processed file")   clips/ (when split: clip_0001.mp4 + clip_0001.txt/.json …)
caption.txt  caption.json (if structured)  caption.srt/.vtt (if timestamps)  reasoning.txt (thinking)  metadata.json  run_log.txt
```
Batch: `<out>/<stem>.txt` (+ `.json/.srt` as configured, + `<stem>_clips/` when split); `outputs/batch_0003_timechat/metadata.json + run_log.txt + summary.json`.
metadata.json: app version, timestamp, model (key, variant, files+sha), all settings (registry), per-item results (paths, timings, tokens, peak VRAM, status/error), processing_time_seconds, gpu name.

## 6. Prompt/task presets (initial list; per-model applicability)
Wan 2.2 T2V dense · Wan T2V sparse · Wan I2V motion-only · Hunyuan dense cinematic · LTX 2.5 short physical · MiniMax H3 performance+sound · Character LoRA (trigger, drop identity) · Motion LoRA (trigger then movement) · TimeChat 6D raw JSON · TimeChat → Wan flatten · AVoCaDO aligned A/V · AVoCaDO visual-only · AVoCaDO structured (UGC) · Audio SFX bed · Audio caption (Captioner) · ASR clean · ASR + timestamps (SRT) · ASR + translate ({{TARGET_LANGUAGE}}) · Lyrics · Booru-ish tags · Negative/avoid list · No-speech visual · Screen text include (OCR) · Chapters & summary · Search/index metadata JSON · Closed captions/SDH · Custom (free text). Template variables: `{{TRIGGER}}`, `{{LANGUAGE}}`, `{{TARGET_LANGUAGE}}`, `{{CAPTION_LENGTH}}`, `{{AVOID}}`, `{{SUBJECT_CLASS}}`.

## 7. Work breakdown (codex tasks; parallel where independent)
- T1 Core foundation: `vcap/core/{paths,logs,progress,gpu,media,outputs,presets,registry,subprocess_runner}.py` + tests. (reports 01,03,05,02)
- T2 Scene split + preprocess + clip fitness + captions_post + export: `core/{scene_split,preprocess,clip_fitness,captions_post,export}.py` + tests. (reports 05,06)
- T3 Models layer: registry/base/loader/attention/offload/vram_presets + TimeChat + AVoCaDO + Qwen3-Omni wrappers + downloads bridge. (reports 10,11,12,07,04)
- T4 Quantization tools: merge_shards, convrot_quantize (int8/int4-W4A8), quant loaders (`models/quant/*`), verify, bench. (report 04, 07) — run on GPU 0.
- T5 Pipeline + worker protocol + multi-GPU. (report 03,05)
- T6 UI: theme/CSS/components/app + Caption tab. (reports 08,05,02)
- T7 UI: Editor tab, Dataset tab, Settings, Recover, Health, Changelog. (reports 08,02)
- T8 Models_Downloader.py rewrite for our catalog + bat/sh installers + docs. (reports 01,05)
- T9 Prompt presets + default presets JSON. (reports 10,11,12)
- T10 Integration, Chrome-driven QA, benchmark, fixes. (Claude + codex)
Order: T1,T2,T4(start conversions early),T9 in parallel → T3,T5 → T6,T7,T8 → T10.

## 8. Quality bars for implementers
- No pinned versions; no network at import; no torch import in the Gradio parent process except through the worker (in-app mode imports lazily).
- All file I/O utf-8; paths via `core.paths.normalize_path`; never `os.startfile` without try/except; Linux `xdg-open`.
- Every long op is a generator yielding throttled UI updates (≤ 8 Hz) AND printing to console; cancel handlers `queue=False`.
- Errors: never crash the batch; collect per-item errors; show `gr.Warning` for user mistakes; write to run_log.
- Type hints, docstrings on public functions, small modules, no dead code, no duplicated pipelines.

## 9. torch.compile (addendum)
`vcap/models/torch_compile.py` (T3) provides robust C++ toolchain detection and graceful fallback, ported from Ultimate Image Captioner `joycaption/torch_compile.py` + `torch_compile_workers.py` and Upscaler `runner.py:1491-1615` (`_find_vcvars`/`_capture_vcvars_env`) and Musubi's MSVC handling: probe `cl.exe` → vswhere → vcvars64.bat env capture (cached); Linux gcc/g++ probe. Fallback ladder when C++ tools are missing: full Inductor (MSVC/gcc) → Triton-only Inductor (log "C++ build tools not found — Triton-only fallback") → `backend="cudagraphs"` → eager with a clear reason. Compile only the resident text decoder; never compile offloaded/block-swapped layers; numerics guards on; `TORCHINDUCTOR_WORKER_START=spawn` on Windows; UI shows the probe status next to the toggle plus a compile-mode dropdown and "Clear compile caches" button; worker applies the captured env (never the Gradio parent).

## 10. Interactive chat
The Chat tab uses the Caption tab's registered model, GPU, VRAM, attention, offload, compile, and model-lifetime settings. `PipelineClient.chat()` sends full text history plus first-turn media through the persistent worker's `chat` command, so captioning and chat share the same one-model cache. Qwen3-Omni Instruct/Thinking support streamed multimodal multi-turn chat; TimeChat and AVoCaDO are limited to single-turn video Q&A; the prompt-free Captioner is not exposed as a chat model. Conversation context is rendered with the native chat template and oldest middle turns are removed above 90% of the model window while retaining the first media turn. Saved conversations use `outputs/NNNN_chat_<model_short>/conversation.json` and `conversation.md`.

## 11. Post-release fixes (v1.1.0)

### 11.1 Settings and persistence contracts

- Caption registry additions are `show_all_variants: bool`, `gpu_indices: list[int]`, `sampling_strategy: "fps" | "uniform" | "keyframe" | "adaptive"`, and `context_carry_over: bool`. `gpu_index` and `gpu_indices` remain machine-specific and are excluded from universal presets; `gpu_indices` remains available to metadata.
- Chat registry additions are `chat_system_prompt: str`, `chat_temperature: float`, `chat_top_p: float`, `chat_top_k: int`, `chat_max_new_tokens: int`, and `chat_enable_thinking: bool`. They are preset-owned but omitted from caption-run metadata.
- Global registry additions are `theme_mode`, `outputs_dir`, `temp_dir`, `models_dir`, `save_processed_files`, and `scan_subfolders`. Theme is browser-local (`dark | light | system`) and excluded from presets/metadata. The other five values are written atomically as UTF-8 to repository-root `app_settings.json`; `VCAP_OUTPUTS_DIR`, `VCAP_TEMP_DIR`, and `VCAP_MODELS_DIR` retain environment-variable precedence.
- Protected presets are read only from `presets_default/`; writable presets and `.last_used_preset.txt` live in `presets/`. Saving or loading updates the last-used marker, and application startup resolves last-used, shipped default, then first available preset.

### 11.2 Worker chat protocol

The persistent JSON-lines worker accepts:

```json
{"cmd":"chat","payload":{"settings":{},"history":[],"media":[],"generation":{},"system_prompt":""}}
```

It streams `log`, `status`, and `delta` events. A `delta` includes incremental `delta` and `reasoning_delta` text plus full `text` and `reasoning` snapshots. The terminal response is `{"ev":"chat_result","result":{...}}`; the result includes token counts, `finish_reason`, prefill/decode/total timing, token rate, peak VRAM, cancellation state, warnings, dropped-turn count, context-token count, and retained history. Cooperative chat cancellation stops only the active generation and retains the resident model/worker.

### 11.3 Downloader status protocol

The distribution-level downloader emits one UTF-8 line per state/progress update:

```text
VCAP_STATUS {"key":"timechat_int4","state":"downloading","fraction":0.423,"bytes_done":2735890432,"bytes_total":6467930328,"message":"Downloading model.safetensors"}
```

`state` is `downloading | verifying | ready | error | skipped | missing`; progress fields may be `null`. The bridge accepts this JSON protocol, the legacy text protocol, and plain-percent lines. Both the console and Gradio consume the same normalized progress payload. The catalog also exposes six third-party Qwen3-Omni GGUF Q4/Q8 entries through the same menu, status, ensure, and verify commands.

### 11.4 Pipeline and metadata contracts

- `OutputSpec.source_root: str | None` is serialized under `output.source_root`. Batch layout computes each media file's safe relative parent from this root so recursive outputs and skip checks mirror the input tree; true same-parent stem collisions still receive numeric suffixes.
- `PreprocessSpec.sampling_strategy` reaches `PreprocessParams.sampling_strategy` and survives OOM retries. `JobSpec.context_carry_over` appends at most the last 60 words of the previous final segment only for AVoCaDO and Qwen3-Omni Instruct/Thinking.
- `TokenUsage.finish_reason` and each segment's serialized `usage.finish_reason` are exactly `eos | length | cancelled`. Top-level metadata also exposes the distinct finish reason(s), sampling strategy, context-carry flag, normalized source root, and `processing_time_seconds`.
- Progress payloads preserve existing fields and add processed/remaining/total counts plus job/item elapsed time and ETA. Batch completion also writes compact one-line `summary.json` item/count data.

### 11.5 torch.compile runtime recovery

- The user-facing mode registry defaults to Inductor `default` and offers `max-autotune-no-cudagraphs`; direct `cudagraphs` and `reduce-overhead` are hidden because token-by-token `DynamicCache` mutation is not replay-safe.
- A Dynamo, Inductor, or CUDA-graph error restores every compiled decoder forward to its original eager callable without reloading weights. The pipeline clears CUDA caches and retries the same segment once from fresh model inputs.
- A failed `(model family, requested mode)` pair is process-local disabled after recovery, so later segments and model reuse remain eager. Toolchain discovery still degrades from full Inductor through Triton-only and CUDA-graphs compatibility paths to eager.
