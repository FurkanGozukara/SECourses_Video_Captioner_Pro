# TRELLIS.2 Premium — Subprocess Stage Pipeline Report

Root: `G:\Trellis2_v6_3\Trellis_2_3D_Generator` (app), `G:\Trellis2_v6_3\HF_model_downloader.py` (downloader).

## 1. Files to copy from

| Path | LOC | Role |
|---|---|---|
| `G:\Trellis2_v6_3\Trellis_2_3D_Generator\app_premium.py` | 8863 | Gradio app: CSS/theme, preset system, cancel system, subprocess launcher + log streaming, pipeline orchestration, batch runner. |
| `G:\Trellis2_v6_3\Trellis_2_3D_Generator\subprocess_stage.py` | 3758 | The stage worker. One CLI entrypoint (`--stage --payload --result`), ~24 `stage_*(payload)->dict` functions, JSON in / JSON out. Never imports Gradio. |
| `G:\Trellis2_v6_3\Trellis_2_3D_Generator\subprocess_utils.py` | 104 | Non-overwriting numbered run dirs + indexed filenames + safe relpath. Copy verbatim. |
| `G:\Trellis2_v6_3\HF_model_downloader.py` | 1793 | Standalone CLI HF downloader: 16-connection ranged chunks, resume, sha256 + verified caches, single-line console progress. Not wired to Gradio. |

Adding a stage = adding a `stage_x(payload)` function + one `elif` in `main()`.

## 2. The subprocess stage pipeline

### 2.1 Launch — `subprocess.Popen` + JSON job file + JSON result file
`app_premium.py:1543` `_iter_subprocess_stage(stage, payload, work_dir, log_path, *, session)` is a generator that yields `{"type":"log","text":...}` events and finally `{"type":"result","result":{...}}`.
- payload → `work_dir/<stage>.payload.json` (`1553-1555`); result ← `work_dir/<stage>.result.json` (`1554`, read `1623`)
- argv: `python -u subprocess_stage.py --stage X --payload P --result R` (`1558-1568`)

Interpreter resolution (`279-289`):
```python
def _resolve_subprocess_python() -> str:
    candidates = [os.environ.get("TRELLIS_SUBPROCESS_PYTHON"), LOCAL_SUBPROCESS_PYTHON, sys.executable, DEFAULT_SUBPROCESS_PYTHON]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return str(candidate)
    return sys.executable
```
(Add a POSIX `bin/python` candidate when porting.)

### 2.2 Launcher + log streaming (`app_premium.py:1570-1621`)
```python
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, cwd=APP_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **popen_kwargs)
    _register_active_subproc(session, proc, stage)
    try:
        with log_path.open("a", encoding="utf-8") as lf:
            for line in proc.stdout:
                clean_line = _clean_status_text(line.rstrip("\n"))
                lf.write(clean_line + "\n"); lf.flush()
                yield {"type": "log", "text": clean_line}
    finally:
        _unregister_active_subproc(session, proc)
        if proc.poll() is None:   # generator closed early (Gradio cancel)
            try: proc.terminate(); proc.wait(timeout=5)
            except Exception:
                try: proc.kill(); proc.wait(timeout=5)
                except Exception: pass
    rc = proc.wait()
    if not result_path.exists():
        if _is_cancel_all(session): raise UserCancelled(f"Cancelled during stage {stage!r}.")
        raise RuntimeError(f"Stage {stage!r} failed (rc={rc}) and produced no result file.")
```
stdout+stderr merged into one PIPE, iterated line-by-line; child lines echoed verbatim into UI; coarse `progress(p, desc=...)` set by parent at fixed per-stage constants. The `finally` handles Gradio closing the generator on cancel.

### 2.3 Inputs/outputs between stages — files on disk
No shared memory or pickle-over-pipe. Every stage writes artifacts into `outputs/<NNNN>/` and the next stage receives absolute paths in its JSON payload. Serializers in `subprocess_stage.py`: `_save_cond` (398, `torch.save({k: v.cpu()})`), `_save_npz_sparse` (293, `np.savez_compressed`), `_write_json` (236), RNG state saved/restored (`783-821`) so per-process seeding matches single-process.

Handoff pattern (`app_premium.py:3822-3832`, `4064-4088`):
```python
        cond_512_path   = run_dir / "02_cond_512.pt"
        coords_path     = run_dir / "04_coords.pt"
        shape_slat_path = run_dir / "05_shape_slat.npz"
        ...
            shape_payload = {"model_repo": ..., "seed": int(seed), "cond_512_path": str(cond_512_path),
                             "coords_path": str(coords_path), "shape_slat_path": str(shape_slat_path), ...}
            shape_result = yield from _stage("sample_shape_slat", shape_payload, 0.40)
```
Numbered prefixes (`02_`, `04_`, `05_`) make the run dir self-documenting. `run.json` (3835-3889) records every parameter. Cross-process app state in `gr.State` holds only paths.

### 2.4 VRAM minimization
1. Process death is the unload — each stage is a short-lived process.
2. Per-stage selective model loading via ignore lists (`subprocess_stage.py:455-472`).
3. In-process unload before spawning — `unload_global_pipelines()` (`app_premium.py:1738-1764`).
`_log_vram_usage(label)` (`subprocess_stage.py:77-86`) prints `[VRAM] label: X.XXGB allocated, Y.YYGB reserved`.

### 2.5 In-process vs subprocess
Global checkbox `subprocess_mode` (`6492-6495`, default True) threaded as bool into each function which has `if subprocess_mode:` + duplicate single-process path. Extraction always staged (`4977-4986`).

### 2.6 Cancel / kill tree
State (`2661-2672`): per-session `threading.Event`s `_CANCEL_ALL`/`_CANCEL_BATCH`, `_ACTIVE_SUBPROCS: Dict[session, Popen]`, guarded by `_CANCEL_LOCK`.
```python
# app_premium.py:2762-2799
def _terminate_process(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is not None: return
    except Exception: return
    try:
        if os.name == "nt": proc.terminate()
        else:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except Exception: proc.terminate()
        proc.wait(timeout=3)
    except Exception: pass
    try:
        if proc.poll() is None:
            if os.name == "nt": proc.kill()
            else:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except Exception: proc.kill()
            proc.wait(timeout=3)
    except Exception: pass
```
Windows caveat: not a tree kill — add Job Object or `taskkill /T /F` (matters if stages spawn ffmpeg).
Two-click confirm UX — `_cancel_processing_click` (`7310-7370`): first click arms for 7 s and relabels button; second click within window calls `_cancel_now`. Wired with `queue=False, show_progress="hidden"`. `_cancel_now` (`2802-2832`) clears flags again if nothing was running. `end_session` (`1778-1791`) kills live child on browser unload.

### 2.7 Error propagation
Worker (`subprocess_stage.py:3740-3752`): exception → `{"ok": false, "error_type", "error", "traceback"}` to result file, exit 1. Parent (`app_premium.py:1617-1628`): no result + cancel → `UserCancelled`; no result → RuntimeError rc; `ok=False` → `RuntimeError(f"Stage failed: {error_type}: {error}\n{traceback}")`. `_cancelled_exit` (`3925-3931`) yields clean "CANCELLED by user.". OOM fallback: `_is_cuda_oom_error` (`subprocess_stage.py:245-253`, checks `torch.OutOfMemoryError` and string "out of memory"+"cuda") → `empty_cache()` + retry cheaper.

## 3. VRAM preset system
No GB-threshold table / GPU auto-detection. Two named presets `_builtin_ui_presets()` (`1304-1342`): `best`, `low_vram` (chunked/tiled switches, smaller texture). User picks manually; dropdown `.change` loads. For a real GB table: return more entries keyed by detected tier and set initial `value=` from `torch.cuda.get_device_properties(0).total_memory`.

## 4. Preset save/load/delete/last-used
`PRESETS_DIR = APP_DIR/presets`, one `<name>.json` each, `.last_used_ui_preset.txt` marker (`1118`). `UI_PRESET_VERSION`/`UI_PRESET_FORMAT` (`1116-1117`).
| Function | Line |
|---|---|
| `_sanitize_preset_name` | 1121 |
| `_list_ui_presets` (builtins first) | 1131 |
| `_set_last_used_ui_preset` / `_get_last_used_ui_preset` | 1140 / 1149 |
| `_default_ui_config` (single source of truth) | 1163 |
| `_merge_ui_config` (forward-compat) | 1345 |
| `_save_ui_preset` (rejects builtin names; atomic `.json.tmp` + replace) | 1365 |
| `_load_ui_preset` / `_delete_ui_preset` (refuses builtins) | 1392 / 1410 |
Component registration: two positional lists `_CONFIG_KEYS` (`8378-8493`) and `_CONFIG_COMPONENTS` (`8495-8610`) — fuse into one list of triples when porting. `_values_to_ui_config` (8612), `_ui_config_to_values` (8618) with clamping/validation.

## 5. Progress reporting + ETA
`_log` closure (`3676-3692`) — four sinks from one call:
```python
    def _log(msg: str, p: Optional[float] = None) -> str:
        nonlocal status
        msg = _clean_status_text(msg)
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        status = (status + "\n" if status else "") + line   # UI textbox buffer
        print(line, flush=True)                             # CMD
        if _log_file_path[0] is not None:                   # outputs/<run>/running_logs.txt
            try:
                with open(_log_file_path[0], "a", encoding="utf-8") as f: f.write(line + "\n")
            except Exception: pass
        if p is not None: progress(p, desc=msg)             # Gradio progress bar
        return status
```
Child lines throttled to one UI yield per 0.6 s (`3915-3918`). `_trim_status` (`1499-1522`) bounds to 200 lines / 20000 chars. `yield from _stage(...)` generator-that-returns pattern. Batch ETA (`3072-3093`): mean-of-completed extrapolation; `_scaled_progress` (3139) rescales per-item progress into global bar.

## 6. HF_model_downloader.py
`MODEL_CONFIGS` (26-79); `DOWNLOAD_CONFIG` (81-88): 16 connections, 10 MB chunks, 5 retries; `RobustDownloader` (90) with `Accept-Encoding: identity` (106), token env (109-116). `download_file` (891-993) returns downloaded/skipped/failed. Resume: chunk-level `.part<N>` (663-744, raises on HTTP 200), whole-file (1300-1317), cross-run pre-scan (1159-1170). Two caches `sha256_cache.json` and `verified_files_cache.json` (mtime tolerance 1 s, 353-374). `verify_file_sha256` (606-662). Console-only progress (237-305) with `log()` clearing progress line first (306-311). Dependency discovery from downloaded pipeline configs (1494, 1527).

## 7. Colored buttons / CSS / theme
`APP_THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate", radius_size="lg", font=...)` (`819-841`). CSS string (`409-773`), `head` script (`775-805`). Launch introspects `demo.launch` signature and only passes supported kwargs (`8850-8863`). Hero buttons with `elem_id` + `elem_classes` (`6679-6690`); CSS uses `:where(#id button, button#id, .class button, button.class)` (579-586, 656-722) with glow keyframes (564-577).

## 8. Cross-platform paths, non-ASCII, temp
- `APP_DIR`-derived `TMP_DIR`, `OUTPUTS_DIR`, `PRESETS_DIR` (256-269). Per-session scratch `TMP_DIR/<session_hash>/subprocess/<run_id>/`; `gr.Blocks(delete_cache=(600,600))`.
- `allocate_run_dir` (`subprocess_utils.py:21-48`): max numeric child + `mkdir(exist_ok=False)` retry loop; `next_indexed_path` (51-87).
- `_resolve_user_path` (2879-2897) strips quotes; `_sanitize_folder_name` (2900-2916) replaces `<>:"/\|?*`, strips trailing spaces/dots, guards reserved names; non-ASCII preserved.
- `_discover_allowed_paths_all_drives()` (320-383).
- Encoding: worker `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (`subprocess_stage.py:15-21`); parent `PYTHONIOENCODING=utf-8` + pipe encoding; `_clean_status_text` (1463-1481) mojibake repairer with `_STATUS_TEXT_REPLACEMENTS` (1425-1460).
- Env before torch import: `PYTORCH_ALLOC_CONF=expandable_segments:True` (only if unset), `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

## Carry over unchanged
JSON-payload/JSON-result stage contract; `allocate_run_dir` + numbered artifact prefixes; `yield from _stage(...)` with 0.6 s throttle; `_clean_status_text` + forced UTF-8 stdio; atomic preset write with `_merge_ui_config`.
## Fix while porting
Windows tree kill; POSIX python candidate; fuse key/component lists.
