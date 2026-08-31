# SECourses Premium Upscaler Pro v8.2 — Architecture Map for Reuse

Root: `G:/SECourses_Premium_Upscaler_Pro_v8/`; App: `G:/SECourses_Premium_Upscaler_Pro_v8/SECourses_Premium_Upscaler_Pro/`
Stack: Gradio 6.24.0, PySceneDetect (`scenedetect[opencv]`), ffmpeg/ffprobe CLI, huggingface_hub ≥ 1.0 + `hf_xet`. ~95,000 LOC app-owned.

## 1. File / module layout

### Root
| File | LOC | Purpose |
|---|---:|---|
| `Models_Downloader.py` | 3526 | Standalone multi-connection resumable HF downloader with SHA verify, Xet fast path, CLI menu + `--ensure-*` single-model modes. **Copy this whole file.** |
| `Windows_Run_SECourses_Upscaler_Pro.bat` | — | Activates venv, sets `TEMP/TMP`, `MODELS_DIR`, `HF_HOME`, `TRANSFORMERS_CACHE`, `PYTORCH_CUDA_ALLOC_CONF`, `python secourses_app.py --windows-int8-defaults`. |
| `Windows_Download_Models_and_or_Resume.bat` | — | Validates venv python, runs `Models_Downloader.py --all`. |

### App root: `secourses_app.py` (2468) — entry point: env bootstrap, theme + all CSS/JS, builds `gr.Blocks`, mounts 13 tabs, wires shared-state sync, launches.

### `ui/` — 19,419 LOC (tab builders; no business logic)
`seedvr2_tab.py` 2844 (canonical tab to copy) · `output_tab.py` 2585 · `flashvsr_tab.py` 2401 · `universal_preset_section.py` 693 (**generic preset UI + wiring reused by every tab**) · `resolution_tab.py` 445 (scene-split / chunking settings + standalone splitter) · `changelog_tab.py` 326 · `media_preview.py` 313 (browser-safe image/video input preview) · `shared_layouts.py` 268 · `health_tab.py` 257 · `queue_tab.py` 193 · `model_tab_common.py` 129 (`sync_signature`, log-tail helpers) · `shared_components.py` 92 (`warn_cancel_confirmation`).

### `shared/` — 38,728 LOC (core, UI-free)
| File | LOC | Purpose |
|---|---:|---|
| `chunking.py` | 4377 | Scene detect / split / process / concat / resume / salvage. |
| `runner.py` | 3236 | Subprocess & in-app model execution, cancel, vcvars capture, CLI arg building. |
| `preset_manager.py` | 821 | File-backed universal preset store (atomic writes, last-used tracking). |
| `universal_preset.py` | 845 | Tab registry (`TAB_CONFIGS`), `values_to_dict`/`dict_to_values`, defaults merge, shared-state sync. |
| `path_utils.py` | 751 | Path normalize, natural sort, ffprobe dims/fps/duration, collision-safe naming, sanitize, metadata emit. |
| `audio_utils.py` | 650 | Audio stream detection/mux/passthrough. |
| `gpu_utils.py` | 622 | NVML/nvidia-smi GPU enumeration (no CUDA context), GPU picker, cache clear. |
| `output_run_manager.py` | 585 | `outputs/0000,0001,…` allocation, `run_context.json`, batch item dirs. |
| `health.py` | 521 | ffmpeg/CUDA/driver/VS-BuildTools/disk checks. |
| `pre_flight_checks.py` | 481 | Pre-run validation. |
| `model_manager.py` | 594 | In-app model cache / delayed loading / VRAM release. |
| `batch_processor.py` | 384 | Generic `BatchJob`/`BatchProgress` engine with per-file error isolation + ETA. |
| `video_codec_options.py` 396 · `video_encoder.py` 319 · `video_fps_utils.py` 231 · `ffmpeg_utils.py` 136 | Codec tables, encode arg builders, ffmpeg scale wrapper. |
| `processing_queue.py` 351 · `queue_state.py` 127 | GPU-resource-aware FIFO app queue. |
| `run_logs.py` 416 · `logging_utils.py` 46 · `command_logger.py` 342 | Run summaries, `run_summary.json`, `executed_commands/*.json`. |
| `error_handling.py` 345 · `input_validation.py` 299 · `ui_validators.py` 248 · `input_detector.py` 383 | Validation + typed errors + input kind detection. |
| `frame_utils.py` 429 · `preview_utils.py` 130 · `chunk_preview.py` 125 | Frame extraction, thumbnails, chunk gallery. |
| `oom_alert.py` | 262 | VRAM OOM text detection → flashing HTML banner + guidance. |
| `gradio_compat.py` | 325 | Scans installed gradio for feature compatibility. |
| `model_downloads.py` | 378 | Bridge: `ensure_<model>()` → subprocess `Models_Downloader.py --ensure-*` with streamed logs + cancel. |
| `path_dialogs.py` | 81 | Native Tk file picker (headless-safe). |
| `process_control.py` | 57 | `terminate_process_tree()` — psutil descendants + taskkill/SIGKILL. |
| `progress_utils.py` 251 · `progress_tracker.py` 204 | `ProgressTracker`/`ChunkProgressTracker` with ETA. |
| `int8_convrot.py` 854 · `int8_convert_engine.py` 370 · `int8_calibration.py` 304 · `int8_layer_policy.py` 187 | Quantization. |

### `shared/services/` — per-tab business logic; each exports `<TAB>_ORDER`, `<tab>_defaults()`, `build_*_callbacks()`: `seedvr2_service.py` 7219 · `resolution_service.py` 945 · `global_service.py` 371 · `output_service.py` 308 …
### `shared/models/` — per-model metadata registries (`seedvr2_meta.py` etc.) driving UI defaults/guardrails.

## 2. UI architecture

### Tab construction — `secourses_app.py:2065-2303`
```python
with gr.Tab("🎬 SeedVR2", render_children=True) as tab_seedvr2:
    seedvr2_ui = seedvr2_tab(...)
tab_seedvr2.select(
    fn=_make_tab_sync("seedvr2"),
    inputs=[shared_state, tab_sync_seedvr2],
    outputs=seedvr2_ui["inputs_list"] + [seedvr2_ui["preset_dropdown"], seedvr2_ui["preset_status"], tab_sync_seedvr2],
    queue=False, show_progress="hidden", trigger_mode="always_last",
)
```
`_make_tab_sync` (`1995-2049`) hashes `{tab, preset name, settings, presets}` with `sync_signature()` and returns `gr.skip()` when unchanged — key anti-flicker trick. Changelog uses `render_children=False`. Global Settings rendered last (`render_global_settings_tab()` `1825-1987`).

### Theme + light/dark
- `gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate", font=GoogleFont("Inter"), font_mono=GoogleFont("JetBrains Mono")).set(body_text_size="16px", button_large_text_size="18px", …)` — `381-403`.
- Dark is baseline. Light corrections in `_LIGHT_THEME_CSS` (`294-326`) using `body:not(.dark) …` with `!important`. Note: Gradio 6 injects custom CSS twice (raw + prefixed with `.gradio-container .contain`), and `body:not(.dark)` can't get that prefix — hence `!important`.
- Theme persistence: `gr.Radio(["dark","light"])` with pure-JS `change` handler (`fn=None, js=...`, `1853-1874`) writing `localStorage["secourses_theme_mode"]`, setting `?__theme=`, toggling `document.body.classList`. On load, `theme_bootstrap_head` (`1637-1667`) restores localStorage.

### Colored buttons — the pattern to copy
1. Shape/behavior: `.action-btn` (radius, weight, hover lift, focus ring, disabled desaturation, keyframes) — `779-810`.
2. Color: 20 generated hue classes `.sec-btn-<hue>` from `(dark, mid, light)` hex triples:
```python
_SEC_BTN_HUES = {"red": ("#991b1b","#dc2626","#f87171"), "emerald": ("#065f46","#059669","#34d399"), ...}  # 20 hues

def _build_sec_btn_css() -> str:
    rules = []
    for name, (dark, mid, light) in _SEC_BTN_HUES.items():
        r, g, b = _hex_to_rgb(mid); lr, lg, lb = _hex_to_rgb(light)
        rules.append(f"""
    .sec-btn-{name} button, button.sec-btn-{name} {{
      background: linear-gradient(135deg, {dark} 0%, {mid} 55%, {light} 100%) !important;
      border-color: rgba({lr},{lg},{lb},0.72) !important; color: #f8fafc !important;
      box-shadow: 0 10px 24px rgba({r},{g},{b},0.30), inset 0 1px 0 rgba(255,255,255,0.18) !important;
    }}""")
    return "\n".join(rules)

CUSTOM_CSS = CUSTOM_CSS + _build_sec_btn_css() + _LIGHT_THEME_CSS   # :1371
```
Usage: `gr.Button("Upscale", variant="primary", size="lg", elem_classes=["action-btn","action-btn-upscale","sec-btn-emerald"])`. Semantic classes: `action-btn-upscale` (green sweep), `action-btn-preview`, `action-btn-optimize` (pulsing red), `action-btn-cancel` (red), `action-btn-open` (teal), `action-btn-clear` (orange). Preset row hues (`universal_preset_section.py:321`): `{"save":"green","load":"sky","reset":"amber","delete":"rose"}`.

### JS injection
`demo.launch(head=CUSTOM_HEAD + theme_bootstrap_head)` — `CUSTOM_HEAD` (`1372-1432`) IIFE with install guard. Per-event JS via `js=` on `.click()`.

### Progress: Gradio + CMD simultaneously (four layers)
1. `gr.Progress` — `def run(*args, progress=gr.Progress())`; `progress(fraction, desc=...)`.
2. Generator streaming — background thread + `queue.Queue` bridge (`seedvr2_service.py:6461-6486`):
```python
progress_queue = queue.Queue()
def processing_thread():
    try:
        result = _process_single_file(..., lambda msg: progress_queue.put(("progress", msg)))
        progress_queue.put(("complete", result))
    except Exception as e:
        progress_queue.put(("error", str(e)))
proc_thread = threading.Thread(target=processing_thread, daemon=True); proc_thread.start()
# main generator drains the queue, throttled at ui_update_throttle = 0.12s, yields UI tuples
```
3. CMD mirror (`runner.py:739-744`):
```python
def log_output(message: str, force_console: bool = True):
    if force_console: print(message, end='', flush=True)   # CMD
    if on_progress:   on_progress(message)                 # UI log
```
and in the read loop `print(line, flush=True); on_progress(line + "\n")` (`runner.py:929-931`).
4. Animated HTML indicator — `_processing_indicator()` (`seedvr2_service.py:5459-5468`) `.processing-banner` div into a `gr.Markdown(elem_classes=["runtime-progress-box"])` with CSS-pinned height (`--runtime-progress-height: 74px`) so buttons never jump.

Windows UTF-8 fix at top of `secourses_app.py:35-45`: wraps stdout/stderr in `TextIOWrapper(..., encoding='utf-8', errors='replace', write_through=True)`.

## 3. Preset system
One universal preset = one JSON file containing every tab. Folder `presets/`; defaults from code (`get_all_defaults()`).
Layout (`shared/preset_manager.py:19-43`): `presets/.last_used_preset.txt` + `presets/<name>.json` → `{"_meta":{version:"2.0",format:"universal",...},"global":{},"seedvr2":{},...}`.

### Generic component registry — `shared/universal_preset.py:97-153`
`TAB_CONFIGS` maps tab name → `{"order": <TAB>_ORDER, "defaults_fn": <tab>_defaults, "needs_model_arg": bool}`. Tab builds `inputs_list` in exactly the same order. Conversion positional:
```python
def values_to_dict(tab_name, values):
    order = TAB_CONFIGS[tab_name]["order"]
    if len(values) != len(order):
        raise ValueError(f"{tab_name}: Expected {len(order)} values, got {len(values)}")
    return dict(zip(order, values))

def dict_to_values(tab_name, data, defaults=None):
    order = TAB_CONFIGS[tab_name]["order"]; defaults = defaults or {}
    normalized = _normalize_tab_settings(tab_name, data, defaults)
    return [normalized.get(k, defaults.get(k)) for k in order]
```
### Save (atomic) — `preset_manager.py:687-729`
```python
def save_universal_preset(self, preset_name, data):
    safe_name = _sanitize_name(preset_name.strip())
    preset_path = self._universal_preset_path(safe_name)
    data.setdefault("_meta", {})
    data["_meta"].update({"version":"2.0","format":"universal","last_modified":datetime.now().isoformat()})
    data["_meta"].setdefault("created_at", data["_meta"]["last_modified"])
    tmp_path = preset_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    tmp_path.replace(preset_path)
    self.set_last_used_universal_preset(safe_name)
    self._invalidate_universal_presets_cache()
    return safe_name
```
`list_universal_presets()` caches 5 s keyed on dir mtime (`652-685`). `delete_universal_preset()` (`756-774`) unlinks and clears last-used if it pointed there. Last-used (`776-804`). Startup auto-load `load_startup_universal_preset()` (`secourses_app.py:1437-1540`).
Save-from-any-tab: `wire_universal_preset_events()` (`universal_preset_section.py:389-537`) registers one `gr.on()` over every `release`/`input`/`change` of every component, calling `sync_tab_to_shared_state()` returning `gr.skip()` when unchanged.
Exclusions: runtime state (`queue_state.py:_RUNTIME_SEED_KEYS`: `last_*`, `*_chunk_preview`, …) never persisted. Guardrails on load (`_coerce_tab_values_for_ui` `68-85`).

## 4. Global settings
`GLOBAL_DEFAULTS` (`secourses_app.py:131-150`) / `GLOBAL_ORDER` (13 keys): `output_dir, temp_dir, theme_mode, telemetry, face_global, face_strength, queue_enabled, global_gpu_device, mode (subprocess|in_app), models_dir, hf_home, transformers_cache, pinned_reference_path`. Global GPU dropdown at page top outside tabs (`1799-1809`). Sharing: `global_settings` dict by reference; `shared_state` `gr.State` (`1694-1787`); `apply_global_settings_live` (`global_service.py:114-268`) validates/creates dirs, updates `runner.temp_dir/output_dir`, `runner.set_mode()`, exports `MODELS_DIR`/`HF_HOME`/etc.

## 5. Scene detection / split / merge — `shared/chunking.py`
### Detection (`51-276`)
```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
video = open_video(video_path)
fps = float(getattr(video, "frame_rate", None) or 30.0)
min_scene_frames = max(1, int(round(float(min_scene_len) * fps)))
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=float(threshold), min_scene_len=min_scene_frames))
if fade_detection:
    scene_manager.add_detector(ThresholdDetector(threshold=12, min_scene_len=min_scene_frames, fade_bias=0.0))
scene_manager.detect_scenes(video=video, show_progress=False, callback=_detect_callback)
scene_list = scene_manager.get_scene_list(start_in_scene=True)
ranges = [(float(s.get_seconds()), float(e.get_seconds())) for s, e in scene_list]
```
Progress obtained via `callback=` (retried without on TypeError), monkey-patched `video.read()` counter, and a 0.2 s poller thread reading `video.position`. Returns `[]` on failure → fallback.
Params (`resolution_service.py:27-43`): `auto_detect_scenes=True, auto_chunk=True, frame_accurate_split=True, chunk_size=0, chunk_overlap=0.0, scene_threshold=27.0, min_scene_len=1.0`. `apply_overlap_to_scenes()` (`279-314`), `fallback_scenes()` (`317`) fixed-length from ffprobe duration.

### Cutting — `split_video()` (`367-926`)
Boundaries snapped once to nearest frame, stored as integer microseconds, consecutive chunks share identical boundary (`509-603`). `SEEK_LEAD_SEC=1.0` / `READ_MARGIN_SEC=1.0`.
```python
def _split_copy():   # fast, bit-exact, keyframe-limited
    cmd = ["ffmpeg","-y","-ss",start_str,"-i",video_path,"-t",dur_str,
           "-c","copy","-avoid_negative_ts","make_zero","-movflags","+faststart",str(out)]

def _split_precise_lossless(pix_fmt, with_audio, encoder="libx264"):  # frame-accurate
    cmd = ["ffmpeg","-y","-ss",seek_str,"-t",read_limit_str,"-copyts","-i",video_path,"-map","0:v:0"]
    if with_audio: cmd += ["-map","0:a?"]
    cmd += ["-vf", f"trim=start={start_abs_str}:end={end_abs_str},setpts=PTS-STARTPTS"]
    if with_audio: cmd += ["-af", f"atrim=start={start_abs_str}:end={end_abs_str},asetpts=PTS-STARTPTS"]
    if src_is_cfr and src_r_fps_str: cmd += ["-r", src_r_fps_str]
    cmd += _lossless_video_args(encoder)          # libx264 -preset ultrafast -qp 0 | libx265 lossless=1
```
Selection (`847-885`): `precise=True` → lossless first; `precise=False` → stream copy first, accepted only if `_chunk_is_exact()`, else lossless fallback. Audio fallback ladder: copy → copy+AAC → video-only. Manifest per chunk `{index, file, start_us, end_us, start_frame, end_frame, expected_frames, split_mode}`.
`_verify_split_coverage()` (`1253-1335`) sums decodable frames vs source. `concat_videos()` (`1915-2363`), `_merge_stream_copy_is_safe()` (`1628-1647`), `_pick_merge_fps()` median never snapping NTSC.
Standalone splitter (`resolution_service.py:496-870`): generator producing `outputs/<stem>_scene_split/scene_0001.mp4…` with detection in worker thread feeding a `queue.Queue`, UI yield every 0.5 s.

## 6. Cancel flow
Confirm via checkbox (`ui/seedvr2_tab.py:743-758`): `cancel_confirm = gr.Checkbox("Confirm cancel (subprocess mode only)")` + `cancel_btn` (`variant="stop"`, `sec-btn-red`).
```python
def handle_cancel_with_confirmation(confirm_checked, state):
    if not confirm_checked:
        message = warn_cancel_confirmation()          # gr.Warning toast
        return (gr.update(value=f"WARNING: {message}", visible=True), gr.update(value=message), gr.update(), gr.update(), gr.update(value=False), state)
    status_upd, log_text, vid_upd, img_upd = service["cancel_action"]()
    return status_upd, log_text, vid_upd, img_upd, gr.update(value=False), state   # auto-unticks
```
`Runner` (`shared/runner.py:181`) launches CLIs with `stdout=PIPE, stderr=STDOUT, bufsize=1, text=True, encoding="utf-8", errors="replace"`, `creationflags=CREATE_NEW_PROCESS_GROUP` / `preexec_fn=os.setsid` (`855-889`). Reader: blocking `readline()` on Windows, `select.select` on POSIX (`900-935`).
Kill — `Runner.cancel()` (`263-415`):
```python
descendants = psutil.Process(proc.pid).children(recursive=True)   # snapshot BEFORE signaling
if platform.system() == "Windows":
    proc.send_signal(signal.CTRL_BREAK_EVENT); proc.wait(timeout=2.0)
    ... proc.terminate() ... proc.kill() ...
    subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"], check=False)
else:
    proc.terminate() (SIGTERM, 2s) → proc.kill()
_reap_descendants()
finally: self._active_process = None; clear_cuda_cache()
```
`force=True` → `terminate_process_tree(proc, grace_sec=0.25)` (`shared/process_control.py`). Partial-output salvage (`seedvr2_service.py:3037-3189`). Metadata emitted for cancelled runs with `"status": "cancelled"`.

## 7. Batch folder processing
`shared/batch_processor.py` (`BatchJob`, `BatchProgress`, `BatchProcessor`); per-tab loop `seedvr2_service.py:5935-6330`.
- Discovery `_list_media_files`; ordering `sort_windows_names` (`path_utils.py:32`) natural sort on all platforms.
- Skip/overwrite (`6131-6180`), `overwrite_existing_batch` checkbox:
```python
overwrite_existing = bool(seed_controls.get("overwrite_existing_batch_val", False))
target_path = batch_output_folder / sanitize_filename(input_file.name)
if target_path.exists() and not overwrite_existing:
    job.status = "skipped"; job.output_path = str(target_path); return True
if overwrite_existing: target_path.unlink(missing_ok=True)
```
Videos → `outputs/<input_stem>/` (`batch_item_dir`, `output_run_manager.py:110-117`) claimed with `mkdir(exist_ok=False)`.
- Continue-on-failure: every job in try/except returning False + `job.error_message`; `BatchProcessor.process_batch` (`batch_processor.py:152-269`) tallies completed/failed/skipped.
- ETA (`batch_processor.py:202-207`): `avg_time_per_file = elapsed / processed; remaining = avg * (total - processed)`. Chunk-level (`seedvr2_service.py:6892-6902`): `eta_s = (elapsed_total_s / max(1e-6, overall_fraction)) - elapsed_total_s`, `finish_local = time.strftime("%H:%M:%S", ...)`.
- Batch metadata: `batch_images_metadata.json` / `batch_videos_summary.json` + per-video `run_metadata.json`.

## 8. Output numbering, metadata, logs, open-folder
```python
# output_run_manager.py:51-73
def allocate_sequential_run_dir(output_root: Path, width: int = 4, start: int = 0) -> Path:
    ensure_dir(output_root)
    existing = _iter_numeric_children(output_root, width=width)
    next_idx = max(max(existing) + 1 if existing else int(start), int(start))
    while True:
        cand = output_root / f"{next_idx:0{width}d}"
        try:
            cand.mkdir(parents=False, exist_ok=False); return cand
        except FileExistsError:
            next_idx += 1
```
`collision_safe_path()` / `collision_safe_dir()` (`path_utils.py:298-375`).
| File | Writer | Content |
|---|---|---|
| `<run_dir>/run_context.json` | `output_run_manager.py:201, 290` | created_at, input_path, model, mode, status |
| `<out>/run_metadata.json` | `emit_metadata()` (`path_utils.py:667-701`) | appends to JSON array: returncode, settings, command, status, timestamp, error_logs (last 50 lines) |
| `<out>/run_summary.json` | `RunLogger.write_summary()` (`logging_utils.py:13-45`) | |
| `executed_commands/<tab>/<ts>_<tab>.json` + `all_commands.jsonl` | `CommandLogger.log_command()` (`command_logger.py:54-138`) | argv, settings, returncode, execution time |
```python
# shared/services/global_service.py:344-356
def open_outputs_folder(path: str):
    path_obj = Path(path); path_obj.mkdir(parents=True, exist_ok=True)
    try:
        if platform.system() == "Windows":  os.startfile(path_obj)
        elif platform.system() == "Darwin": subprocess.run(["open", str(path_obj)])
        else:                               subprocess.run(["xdg-open", str(path_obj)])
        return gr.update(value=f"Opened: {path_obj}")
    except Exception as exc:
        return gr.update(value=f"Could not open folder: {exc}")
```

## 9. Models_Downloader.py — download engine
Trigger flow (lazy, first-selection): `shared/model_downloads.py` — each `ensure_*()` checks required files; else:
```python
command = [sys.executable, "-u", str(app_dir.parent / "Models_Downloader.py"), *arguments]
process = subprocess.Popen(command, cwd=..., stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env={**os.environ, "PYTHONUNBUFFERED":"1"})
for line in process.stdout:
    _emit(line, on_progress)        # print() to CMD + on_progress() to Gradio log
```
`cancel_event` watcher thread calls `terminate_process_tree(process)`. Call sites inside runners right before launching the model (`runner.py:452` etc.).
CLI: `--ensure-seedvr2 <file> [--int8-convrot]`, `--ensure-flashvsr`, `--ensure-gan <file>`, `--all`.
Repo layout: all under `MonsterMMORPG/Wan_GGUF` (one repo, many subdirs) + upstream repos; targets relative to `PROJECT_DIR` (`Models_Downloader.py:89-110`).
Classes: `RemoteFile` (`485-498`), `_digest_from_etag()` (`501`), `_Ranges` (`810-895`) resume map, `_SpeedMeter` (`699`), `_Link` (`731`), `_Progress` (`788`), `RobustDownloader` (`941`) — `ensure_file`, `download_file`, `download_many`, `download_repo`, `_run_pieces`, `_fetch_piece`, `_stream_whole_file`, `_try_run_xet`, `compute_digest`, `_check_free_space`, `_install`, `show_progress_line`. `DOWNLOAD_CONFIG` (`371-446`) env-overridable via `SECOURSES_DL_<KEY>`.
Resumability: `<target>.part` + `<target>.part.json`, `_write_state` fsyncs staging then atomic state write; `_read_state` (`1405-1465`) validates identity. Verification `_already_complete` (`2223-2246`), `ensure_file` (`2248-2372`) retries once from scratch on mismatch; `file_attempts=6` counted only on zero-progress attempts; `RangeNotSupported` → single-stream.
Progress (`1029-1069`): non-TTY prints whole lines every 5 s (`non_tty_progress_interval`); TTY `\r` in-place bar. Line format: `[DOWNLOADING] file: 42.3% (5.1 GB/12.0 GB) 118.4 MB/s ETA 1m 2s | 16 conns @ 7.4 MB/s`. `get_model_choice()` (`2822`), `download_models()` (`2987`), `main()` (`3250`).

## 10. VRAM / attention / optimization options
- Attention backend dropdown `["sdpa","flash_attn_2","flash_attn_3","sageattn_2","sageattn_3"]`; auto-default `_get_default_attention_mode()` (`268-332`) detects compute capability via NVML/nvidia-smi + `importlib.util.find_spec()` — never imports torch in parent Gradio process.
- BlockSwap `blocks_to_swap` slider 0-36, `swap_io_components`; offload devices; VAE tiling; `int8_convrot`; FP8; GGUF; torch.compile (`compile_backend`, `compile_mode`, `compile_fullgraph`, `compile_dynamic`, cache limits) with Windows `Runner._find_vcvars()` + `_capture_vcvars_env()` (`runner.py:1491-1615`); keep model loaded (`cache_dit`, `model_manager.py`); GPU picker (`shared/gpu_utils.py`: `get_gpu_info()` NVML, `auto_select_global_gpu_device`, `resolve_global_gpu_device`, `clear_cuda_cache`); `_enforce_seedvr2_gpu_visibility` (`runner.py:1428`) sets `CUDA_VISIBLE_DEVICES`.
- OOM: `shared/oom_alert.py` `is_vram_oom_text()`, `extract_oom_snippet()`, `build_vram_oom_html()`, `maybe_set_vram_oom_alert()`; flashing `.vram-oom-banner` (CSS `409-505`) via `gr.Timer(5.0)` poll (`secourses_app.py:2386-2437`).
- Auto Tune: `autotune_search.py` frontier-bisect over `batch_size` × `blocks_to_swap` with VRAM watchdog.
- Allocator: `PYTORCH_CUDA_ALLOC_CONF` → `PYTORCH_ALLOC_CONF` (`secourses_app.py:22-24`, `runner.py:761-764`).

## 11. Path handling
- `normalize_path()` (`path_utils.py:108-112`) — `expanduser(expandvars(...))` then `Path(...).resolve()`.
- `_normalize_directory_path()` (`global_service.py:37-72`) — accepts `/`, `//`, `\`; preserves UNC; requires absolute; UI info "Supports /, //, and \ separators".
- `_ensure_writable_dir()` (`75-85`) — mkdir + `.secourses_write_test` probe.
- `sanitize_filename()` (`path_utils.py:476-522`) — strips separators/NULs, replaces `<>:"|?*`, trims dots/spaces, prefixes `file_` for reserved names, caps 200 chars; keeps non-ASCII.
- `filename_natural_sort_key()` + `sort_windows_names()` (`16-56`).
- `_clean_path()` (`ui/media_preview.py:230-245`) strips quotes.
- Encoding: every `open()` `encoding="utf-8"`; subprocess pipes `errors="replace"`; Windows child env `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8` (`runner.py:816-818`); downloader `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (`Models_Downloader.py:66-77`).
- `_scan_disk_roots()` + `_build_launch_allowed_paths()` (`secourses_app.py:201-227`).

## 12. Other reusables
- Launch (`secourses_app.py:2444-2464`): `demo.queue(default_concurrency_limit=4)`; `launch(inbrowser=os.environ.get("SECOURSES_NO_BROWSER","0")!="1", show_error=True, allowed_paths=..., share=..., theme=..., css=..., head=..., favicon_path=...)`; CLI `parse_known_args` (`--share --server --port`).
- Concurrency: heavy handlers `concurrency_limit=32, concurrency_id="app_processing_queue", trigger_mode="multiple"`; light handlers `queue=False, show_progress="hidden", trigger_mode="always_last"`.
- `gr.skip()` + signature caching everywhere — `sync_signature()` (`ui/model_tab_common.py:49-54`).
- App queue: `shared/processing_queue.py` FIFO per GPU-resource lane; `shared/queue_state.py` deep-copies state at click time.
- Health: `shared/health.py:collect_health_report()`; `shared/gradio_compat.py`.
- Media probing: `get_media_dimensions/duration_seconds/fps` (`path_utils.py:129-236`) via ffprobe; `_parse_fraction_to_float`.
- File preview: `ui/media_preview.py` — `pick_preview_paths()`, `safe_video_preview_path()` checks browser-playable (container, codec) pairs, remuxes/transcodes to content-hash cache `gradio.utils.get_upload_folder()/secourses-video-previews/<sha256>/preview.mp4` with atomic `os.replace`.
- ffmpeg wrappers: `ffmpeg_utils.py:scale_video()`, `video_codec_options.py` (`get_codec_choices`, `get_pixel_format_choices(codec)`, `ENCODING_PRESETS`, `AUDIO_CODECS`, `build_ffmpeg_video_encode_args`), `audio_utils.py` (`has_audio_stream`, `ensure_audio_on_video`).
- Native file picker: `shared/path_dialogs.py:get_any_file_path()` Tk guarded by `_display_available()`.
- Errors: `shared/error_handling.py` typed errors, `safe_execute`, `format_user_error`, `check_ffmpeg_available`, `check_disk_space`.
- Changelog tab: `CHANGELOG_ENTRIES = [(title, markdown), ...]` newest-first, one Accordion each.

## Recommended copy order
1. `shared/preset_manager.py` + `shared/universal_preset.py` + `ui/universal_preset_section.py`.
2. `secourses_app.py` 238-289 (`_SEC_BTN_HUES` + `_build_sec_btn_css`), 294-326 (`_LIGHT_THEME_CSS`), 408-1368 (`CUSTOM_CSS`), 1637-1667 (theme bootstrap), 1995-2049 (`_make_tab_sync`).
3. `shared/process_control.py` + `Runner.__init__/cancel/_run_seedvr2_subprocess` (`runner.py:181-415, 712-1084`).
4. `shared/chunking.py:51-926` + `_verify_split_coverage`.
5. `Models_Downloader.py` (whole) + `shared/model_downloads.py`.
6. `shared/output_run_manager.py`, `shared/path_utils.py`, `shared/batch_processor.py`, `shared/command_logger.py`, `shared/logging_utils.py`, `shared/gpu_utils.py`, `shared/oom_alert.py`, `ui/media_preview.py`, `ui/shared_layouts.py`, `ui/model_tab_common.py`, `shared/services/global_service.py`.
