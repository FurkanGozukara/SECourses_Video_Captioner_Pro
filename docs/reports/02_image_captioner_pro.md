# Ultimate Image Captioner Pro — Architecture & Feature Harvest

Root: `G:\Ultimate_Image_Captioner_Pro_v1\Ultimate_Image_Captioner_Pro\`
Downloader: `G:\Ultimate_Image_Captioner_Pro_v1\HF_model_downloader.py`

## 1. File/module layout (LOC)

Total project Python: 19,033 LOC

| Path | LOC | Purpose |
|---|---|---|
| `app.py` | 370 | Gradio `Blocks` shell: topbar, universal preset panel, 7 tabs, theme switch, launch args, `allowed_paths` discovery. |
| `joycaption/common.py` | 649 | Shared utils: paths, image discovery, caption save/finalize, replace-pairs, numbered outputs, metadata, ETA/progress line, VRAM text, TF32/cache toggles. |
| `joycaption/model_catalog.py` | 198 | Model registry (`ModelSpec` frozen dataclass + `MODEL_SPECS` dict) + readiness/weight-completeness validation. |
| `joycaption/model_downloads.py` | 75 | Lazy first-run download: shells out to `HF_model_downloader.py`, per-model threading lock, post-download validation. |
| `joycaption/lazy_engines.py` | 852 | Lazy wrappers + `_ModelSwitchRegistry` that auto-frees other engines when a new one activates. |
| `joycaption/presets.py` | 101 | `UniversalPresetStore`: save/load/delete/last-used JSON presets (atomic tmp+replace). |
| `joycaption/qwen_presets.py` | 179 | Loads 35 JSON system presets from disk, `{{VARIABLE}}` template rendering. |
| `joycaption/vram.py` | 177 | GPU detection, VRAM presets (6→80 GB), per-family + per-model auto settings. |
| `joycaption/attention.py` | 131 | Attention backend registry, aliases, load kwargs, runtime `sdpa_kernel` context. |
| `joycaption/torch_compile.py` | 550 | torch.compile environment probe (MSVC/vswhere bootstrap on Windows), settings, `CompileConfig` kwargs. |
| `joycaption/torch_compile_workers.py` | 280 | Portable parallel TorchInductor worker pool (spawn on Windows) + warmup. |
| `joycaption/styles.py` | 737 | `CUSTOM_CSS` (all colored buttons, layout). |
| `joycaption/subprocess_runner.py` | 147 | Named subprocess worker launcher, live stdout tee, cancel via `taskkill /T /F`. |
| `joycaption/worker.py` | 370 | `python -m joycaption.worker <command> <payload.json> <result.json>` child entry. |
| `joycaption/engines/qwen.py` | 1,792 | `QwenEngine`: load/quantize, true batched generate, JSON finalize+repair-retry, single/ZIP/folder flows, spawn workers. |
| `joycaption/tabs/shared.py` | 194 | `TabUI` dataclass, reusable torch.compile controls, reusable replace-pair chip widget. |
| `joycaption/tabs/qwen.py` | 1,437 | Qwen Vision tab. |
| `joycaption/tabs/output_browser.py` | 508 | "Saved Outputs" dataset review: paginated index, wildcard search, editor, sidecar audit. |

Data dirs: `system_presets/qwen/*.json` (35 read-only presets), `presets/*.json` (user presets), `outputs/NNNN/` (numbered runs), `model_files_*/` (weights).

## 2. Feature list

### Global shell (`app.py:166-334`)
- GPU summary panel: `gpu_summary_html()` renders `GPU 0: <name> - 23.99 GiB -> 24 GB` (`vram.py:82-90`).
- Universal Preset panel spanning all tabs: Preset dropdown with numbered choices `1. name` (`app.py:152-153`), `Save As` textbox, Theme radio, buttons Save (blue) / Load (green) / Reset (yellow) / Delete (red). Preset autoload on `demo.load` → `load_startup_preset` restores last-used (`app.py:256-268, 325`). `preset_dropdown.change` also loads. Reset clears last-used and restores every tab's `DEFAULTS`.
- Theme switch via JS reload with `?__theme=dark|light` (`app.py:316-323`).
- `demo.queue(default_concurrency_limit=1)`.
- `allowed_paths` auto-discovery: all Windows drive letters via `ctypes.windll.kernel32.GetLogicalDrives()` or POSIX `/proc/self/mountinfo` + `/Volumes` (`app.py:50-113`).

### Qwen tab (`tabs/qwen.py`)
- `gr.Image(type="filepath")`, buttons Caption Image (purple) / Cancel (red) / Open Outputs (slate).
- Generated caption textarea (editable, monospace) + custom Copy button (`qwen.py:371-444`).
- Settings rail: Preset dropdown (35 system presets), Format (json/txt/tags/qc), Extension, Beautify JSON, System Prompt, Prompt; Template Variables accordion (Trigger Phrase, Language, Caption Length, Dataset Goal, ...); Model dropdown, VRAM Preset (6/8/10/12/16/24/32/48/64/80 GB), "Run single and batch in subprocess, then terminate it", Quantization radio (bf16/fp16/int8/nf4), Device ID textbox (`0,1` splits across GPUs), Temperature, Top-p, Top-k, Repetition Penalty, Max New Tokens, Image Long Edge, Save image copy, Unload after run, Allow TF32, Clear CUDA cache, Low CPU memory loading, Attention Backend (8 choices), Enable Thinking, torch.compile block, Compact JSON, JSON Repair Retries, Remove newlines, Text Prefix, Text Suffix, replace-pair widget.
- Uploaded Files to ZIP: multi-file upload, Batch Size 1-16, Create Caption ZIP / Cancel / Open Outputs, `gr.File` ZIP download.
- Folder Batch: Input Folder, Output Folder, Process subfolders, Overwrite captions, Append captions, Batch Size, Start / Cancel / Open Batch Output.

### Saved Outputs tab (`tabs/output_browser.py`) — dataset review tool
- Output Day dropdown filter, Search with wildcard support (`*.json`, `*portrait*`, via `fnmatchcase`, searches paths, stems and caption body) (`output_browser.py:97-121`).
- Pagination: Previous / Page / Next / Per Page (10/25/50/100) (`_page_payload`, `197-241`).
- `gr.Dataframe` (Day, Folder, Caption, Image, Type, Modified) with `show_search="search"`, row-click selects.
- Caption/JSON Editor (24 lines) + Save Edit (JSON validated before write `405-418`).
- Audit Sidecars button: images with no caption, captions with no image, invalid JSON (`420-446`).
- Auto-refresh on tab select (`454-460`).

Absent: no `gr.Gallery`, no `gr.Progress()`, no explicit OOM catch/retry, no idle unload timer, no keyboard nav.

## 3. Model abstraction layer

```python
# joycaption/model_catalog.py:11-26
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo_id: str
    subdir: str
    family: str            # "joycaption" | "qwen"  -> which tab/engine
    architecture: str      # "legacy_siglip" | "llava" | "Qwen3VLForConditionalGeneration" | ...
    startup_default: bool = False
    supports_thinking: bool = False
    required_files: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return BASE_DIR / self.subdir

MODEL_SPECS: dict[str, ModelSpec] = { "joycaption_pre_alpha": ModelSpec(...), ... }
def qwen_model_choices() -> list[tuple[str, str]]:
    return [(spec.label, spec.key) for spec in QWEN_MODEL_SPECS]
```

### Param exposure — ORDER / DEFAULTS / settings_from_values triple
Every tab declares a flat `ORDER: list[str]` and matching `DEFAULTS: dict`. Components built into `components: dict[str, Component]`, then `ordered_components = [components[k] for k in ORDER]`. Handlers zip back:
```python
# joycaption/tabs/qwen.py:1012, 1069-1079 + tabs/shared.py:36-37
ordered_components = [components[key] for key in ORDER]
def run_single(image, *values):
    settings = settings_from_values(ORDER, values)   # dict(zip(ORDER, values))
    yield from engine.caption_single(image, settings)
```
Same pair powers the universal preset store (`TabUI(key, order, defaults, inputs)`, `shared.py:28-33`; split/merge at `app.py:126-149`).

Ranges (Qwen tab `qwen.py:938-968`): temperature Slider 0.0-2.0 step 0.01; top_p 0-1 step 0.01 (0.8); top_k 1-100 (20); repetition_penalty 0.8-1.5 step 0.01 (1.0); max_new_tokens 64-8192; image_long_edge 256-1536; json_retries 0-3; batch sizes 1-16.
Sampling auto-gated: `do_sample = temperature > 0`, top_p/top_k only added when sampling (`engines/qwen.py:604-627`); temperature clamped `max(t, 1e-5)`.

### Model-specific show/hide on model change
```python
# joycaption/tabs/qwen.py:1056-1067
def apply_model_defaults(vram_preset, model_key):
    settings = qwen_vram_settings(vram_preset, model_key)
    spec = get_model_spec(model_key)
    return (settings["model_quantization"], settings["image_long_edge"],
            settings["attention_backend"], settings["file_batch_size"],
            settings["folder_batch_size"], settings["max_new_tokens"],
            gr.update(value=False, interactive=spec.supports_thinking))
```
Capability toggled via `interactive=` rather than `visible=` so layout never jumps. All handlers `queue=False, show_progress="hidden"`.

### VRAM auto-config (`vram.py:117-177`)
Buckets ≤8 → nf4/512px/`sdpa_cudnn`/2048 tok; ≤12 → nf4/768; ≤16 → int8/768/4096; ≤24 → bf16/1024/4096; else bf16/1280/6144. Model-aware override for 27B/30B: nf4 up to 32 GB, int8 up to 64 GB, bf16 only at 80 GB.

### Attention backend registry (`attention.py:8-61`)
8 UI choices → 3 HF `attn_implementation` values (`eager`/`sdpa`/`flash_attention_2`); 4 of them force a PyTorch kernel via `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION | EFFICIENT_ATTENTION | CUDNN_ATTENTION | MATH)` context manager around `model.generate` (`attention.py:105-121`, used `qwen.py:643`). ~25 aliases normalized. FA2 pre-validated: rejects int8/nf4 and missing `flash_attn` (`attention.py:88-93`).

## 4. Model loading / unloading / first-run download

### `_ModelSwitchRegistry` (`lazy_engines.py:27-58`) — weakrefs; `activate(key)` frees all other engines' models:
```python
def activate(self, key: str) -> None:
    with self._lock:
        if self._active_key == key: return
        for owner_key, owner_ref in self._owners.items():
            owner = owner_ref()
            if owner is None: stale.append(owner_key)
            elif owner_key != key: owners_to_clear.append(owner)
        self._active_key = key
    for owner in owners_to_clear:
        clear = getattr(owner, "clear_models", None)
        if callable(clear): clear()
```
Unload = `del model` → `gc.collect()` → `torch.cuda.empty_cache()` (`engines/qwen.py:460-466`).
Model caching key: reload only if quant/device/optimization key changed (`qwen.py:509-519`).
Quantization (`qwen.py:487-499`): bf16/fp16 → dtype; int8 → `BitsAndBytesConfig(load_in_8bit=True)`; nf4 → `load_in_4bit, nf4, compute bf16, double_quant`.
All models load with `local_files_only=True` — downloads never happen inside transformers.

### Readiness check (`model_catalog.py:162-198`)
`model_readiness_error()` returns human string: missing dir, missing required file, missing `config.json`, or incomplete shards — parses `model.safetensors.index.json` and verifies every `weight_map` file exists.

### First-run download (`model_downloads.py:45-75`)
Double-checked locking per model key → `subprocess.run([sys.executable, downloader, "--model", key, "--target-root", BASE_DIR])` → re-validates → raises "press the caption button again to resume" on failure. UI streams: "*<label>* is not installed. Downloading it now with the resumable model downloader. Captioning will begin automatically when validation finishes."

### `HF_model_downloader.py` core (1,300 LOC)
`MODEL_DOWNLOAD_SPECS` dict (26-87) with `include_prefixes`; `DOWNLOAD_CONFIG` (95): 16 connections, 10 MB threshold/chunks, 5 retries, 2→30 s backoff; `RobustDownloader` (124); `prefetch_metadata()` (184) uses `model_info(files_metadata=True)`; `sha256_cache.json` + `verified_files_cache.json`; `supports_range()` probe; `download_parallel()` (710) `.partK` files resume; `download_file()` (1039) decision tree: exact size + verified → SKIP; sha match → SKIP; larger → delete & redownload; smaller → RESUME; sha mismatch → delete + failed. Console: 40-char bar, `[VERIFYING]` line. CLI: `--model`, `--all-models`, `--list-models`, `--target-root`, `--dry-run`.

## 5. Caption post-processing & save formats

```python
# joycaption/common.py:203-220
def finalize_caption_text(caption, remove_newlines=True, prefix="", suffix="",
                          replace_pairs=None, replace_case_sensitive=False,
                          replace_single_word=False) -> str:
    if remove_newlines:
        caption = " ".join(str(caption).split())
    caption = apply_replace_pairs(str(caption), replace_pairs,
                                  case_sensitive=replace_case_sensitive,
                                  single_word=replace_single_word)
    return f"{prefix or ''}{caption}{suffix or ''}"
```
`apply_replace_pairs` (249) escapes needle; `single_word` wraps in `(?<!\w)…(?!\w)`; case-insensitive default. `normalize_replace_pairs` accepts JSON strings, dicts or 2-tuples.
Legacy path adds `cut_off_last_sentence` and `remove_repeating_sentences` (`common.py:390-402`).

JSON pipeline (`engines/qwen.py:676-757`): fence strip → extract candidate → parse → validate → repair-retry loop feeding failing JSON + warnings at `temperature=0.0` → strip URL keys (anti-hallucination) → `_postprocess_json_caption` applies prefix/suffix to first primary text key and replace-pairs everywhere except `text`.

Save: single → `save_numbered_generation` (`common.py:274-316`) creates `outputs/0001/` (first free 4-digit, `next_numbered_output_dir` 190) with `<stem>.txt|.json`, optional image copy, `metadata.json`.
`metadata.json` keys (`qwen.py:786-807`): `generation_type, engine, model_label, model_path, source_image_path, preset_id, system_prompt, prompt, output_format, caption_raw, caption_final, json_warnings, settings, elapsed_seconds, generated_tokens, generation_elapsed_seconds, tokens_per_second, vram_before, vram_after, optimizations, saved_at` + output paths.
Folder batch → sidecar next to image; subfolders preserved via `relative_to`.
Write primitive (`common.py:329-365`): `save_caption_file` exists+overwrite → "w"; exists+append → "a"; exists+neither → returns `None` (skipped).

## 6. UI design
- Theme: `gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="zinc", font=GoogleFont("Inter"), font_mono=GoogleFont("JetBrains Mono"))` (`app.py:116-123`).
- CSS: 737-line `CUSTOM_CSS` at `demo.launch(css=...)`. 24 semantic `elem_classes` each with hex background + `filter: brightness(1.06)` hover (`styles.py:5-95`). Palette: save `#2563eb`, load `#059669`, reset `#ca8a04`, delete/cancel `#dc2626`, refresh `#0d9488`, qwen-caption `#6d28d9`, zip `#0891b2`, folder `#16a34a`, json-build `#7c3aed`, open-folder `#475569`.
- Status HTML convention (`common.py:71-73`): `html_message("error"|"success"|"info", msg)` → `<div class="jc-error|jc-success|jc-info">`.
- Custom JS via `js_on_load=` on `gr.HTML` (Gradio 6): drag overlay, copy button, chip remover. Communication back via `trigger("click", {...})` + `gr.EventData._data` (`shared.py:130-140`, `qwen.py:1149-1153`). Hidden sync textboxes `.jc-hidden-sync { display: none }`.
- Replace-pair chip widget (`shared.py:104-194`): renders `find -> replace [X]` chips in HTML, backed by `gr.State` list, event-delegated remove.

## 7. Progress & cancel
- Generators yield tuples into `gr.HTML` status; fast handlers `queue=False, show_progress="hidden", show_progress_on=[]`.
- ETA line (`common.py:507-553`):
```
Device 0: 12/40 local processed, 3 local skipped, 0 local failed, 25 local remaining.
Total 24/80 processed, 6 skipped, 0 failed, 50 remaining. ETA 4m 12s.
Overall 0.42 img/s. Last batch 2 image(s) in 4.31s (2.16s/image, 0.46 img/s, token speed 38.20 tok/s).
```
- CMD: `log_event(msg, scope)` → `[HH:MM:SS] [Model] message`; `ConsoleProgressStoppingCriteria` (`qwen.py:388-421`) piggybacks HF `StoppingCriteria` (always returns False) to print `\r`-updated token progress `token generation 47% (241/512, 38.20 tok/s)`; `hf_logging.disable_progress_bar()`.
- Cancel: `BatchStopFlag` (`common.py:105-113`) checked per iteration; `cancel_active_workers` (`subprocess_runner.py:49-79`) registry of live Popens by name; Windows `taskkill /PID n /T /F`, else `terminate()`; cancelled PIDs tracked so nonzero exit reported as "Worker was cancelled." Cancel buttons `queue=False`.
- `ThinkingBudgetCriteria` (`qwen.py:423-443`) per-row tracking of `</think>` so each batch item gets full answer budget after reasoning; `max_new_tokens` temporarily raised by `MAX_THINKING_TOKENS`.

## 8. Batch loop (`engines/qwen.py:1348-1362, 1412-1440, 1457-1460`)
```python
for offset in range(0, len(chunk), batch_size):
    if self.stop_flag.value: break
    batch_paths = chunk[offset : offset + batch_size]
    work_items = []
    with aggregate_lock:
        for path in batch_paths:
            relative = path.relative_to(input_dir) if process_subfolders else Path(path.name)
            output_image_path = output_dir / relative
            output_caption_path = output_image_path.with_suffix(extension)
            if output_caption_path.exists() and not overwrite and not append:
                aggregate["skipped"] += 1; local_skipped += 1
                continue
            work_items.append((path, output_image_path, output_caption_path))
    try:
        images = [load_rgb_image(p, long_edge) for p, _, _ in work_items]
        raw_outputs = worker_engine.generate_captions(images, settings_batch)
        for (path, out_img, out_cap), image, image_settings, raw in zip(...):
            final, parsed, warnings = worker_engine._finalize_output(image, raw, image_settings)
            actual_caption = save_caption_file(out_cap, final, overwrite=overwrite, append=append, ...)
            copy_image_if_needed(path, out_img, bool(settings.get("save_image", True)))
            if actual_caption: aggregate["processed"] += 1
            else:              aggregate["skipped"]   += 1
    except Exception as exc:
        with aggregate_lock:
            aggregate["failed"] += len(work_items)
        batch_queue.put(("progress", f"{line} Failed batch: {format_exception(exc)}"))
```
Two-phase skip: pre-pass before loading the model builds the queue and counts skips (`qwen.py:1179-1191`) so the model is never loaded for a complete folder.

## Ideas to steal
1. ORDER + DEFAULTS + TabUI triple.
2. Universal cross-tab preset with last-used autoload (`.last_used_preset.txt`) and atomic writes.
3. `_ModelSwitchRegistry` with weakrefs.
4. Two-phase skip with pre-scan.
5. `ConsoleProgressStoppingCriteria` for token-rate progress in terminal.
6. `batch_progress_line()` formatter.
7. `ThinkingBudgetCriteria`.
8. Subprocess mode as VRAM guarantee; `run_worker` tees child stdout live, keeps 80-line tail.
9. torch.compile numerics guard (`torch_compile.py:500-508`): `emulate_precision_casts`, `eager_numerics.division_rounding`, `use_pytorch_libdevice`; `require_compile_environment()` bootstraps MSVC via vswhere on Windows.
10. Windows-safe Inductor worker pool (`TORCHINDUCTOR_WORKER_START=spawn`).
11. `allowed_paths` = every mounted drive.
12. Model readiness via `weight_map` verification.
13. `js_on_load=` HTML components + hidden sync textbox for custom widgets.
14. Saved Outputs audit + wildcard search + page-scoped scanning.
15. Numbered output dirs with full `metadata.json` including raw and final caption, settings, VRAM before/after, tok/s.
16. Prompt templating with `{{VARIABLE}}` + Template Variables accordion; presets are plain JSON on disk with `order` field (`qwen_presets.py:90-179`).
17. Anti-hallucination JSON scrubbing (URL keys).
