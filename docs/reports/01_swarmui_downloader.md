# SwarmUI Model Downloader — Progress + Download Architecture

Root: `F:/SwarmUI_Model_Downloader_v155/`

## 1. File layout and LOC

| Path | LOC | Role |
|---|---|---|
| `F:/SwarmUI_Model_Downloader_v155/Downloader_Gradio_App.py` | **5,529** | Whole Gradio UI, catalog rendering, queue wiring, progress plumbing, CSS/theme |
| `F:/SwarmUI_Model_Downloader_v155/utilities/HF_model_downloader.py` | **3,894** | **The download core.** `RobustDownloader` + sparse-range resume + SHA cache |
| `F:/SwarmUI_Model_Downloader_v155/utilities/model_catalog_data.py` | 3,780 | Pure data: model entry dicts + `models_structure` |
| `F:/SwarmUI_Model_Downloader_v155/utilities/url_downloader.py` | **1,098** | CivitAI / HF / generic URL parsing → delegates to `RobustDownloader` |
| `F:/SwarmUI_Model_Downloader_v155/utilities/folder_manager.py` | 617 | Destination folder resolution |
| `F:/SwarmUI_Model_Downloader_v155/utilities/console_manager.py` | **326** | **The console progress renderer.** Multi-row ANSI live block |
| `F:/SwarmUI_Model_Downloader_v155/utilities/parallel_download_manager.py` | **304** | **Resizable FIFO worker pool** |
| `F:/SwarmUI_Model_Downloader_v155/utilities/hf_token_manager.py` | 305 | Token normalize/resolve/validate |
| `F:/SwarmUI_Model_Downloader_v155/utilities/xet_environment.py` | 61 | Sets `HF_XET_CHUNK_CACHE_SIZE_BYTES` **before** any HF import |

The four files worth copying wholesale: **`console_manager.py`, `parallel_download_manager.py`, `xet_environment.py`**, and the `RobustDownloader` half of `HF_model_downloader.py`.

## 2. The download core

### HTTP backend
Plain **`requests`** — no httpx, no aiohttp. `huggingface_hub` is used **only for metadata** (`hf_hub_url`, `get_hf_file_metadata`, `HfApi`, `list_repo_files`), never for bytes. `hf_xet` is used natively as an alternate byte source into the same staging file.

Session setup, `RobustDownloader.__init__`, lines 460–487:
- `Retry(total=2, connect=2, read=0, status=2, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), respect_retry_after_header=True, raise_on_status=False)` — `read=0` deliberate; stream read failures are resumed byte-for-byte by the downloader itself.
- `HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, pool_block=True)` where `pool_size = max(20, num_connections+4)`.
- **`Accept-Encoding: identity` is set globally** (line 486) — byte ranges must refer to the uncompressed representation. Repeated on every range request.
- `_new_transfer_session()` (1010) gives each worker thread its own 2-connection session.

### Config (`DEFAULT_DOWNLOAD_CONFIG`, lines 75–92)
```python
"num_connections": 16, "chunk_size": 1048576,          # 1 MB
"max_retries": 6, "retry_delay": 1, "max_retry_delay": 15,
"connect_timeout": 15, "read_timeout": 30,
"http_status_retries": 2, "progress_window": 8,        # seconds of speed smoothing
"keep_partial_on_cancel": True,
"disk_reserve_bytes": 256*1024*1024,
"piece_size": 32*1024*1024,  "state_flush_interval": 2.0,
"xet_concurrency": 32, "xet_range_flush_size": 32*1024*1024,
```

### Resume — sparse staging file plus an interval map
- `_RangeMap` (130–183): thread-safe merged half-open intervals with `add`, `add_many`, `missing(start,stop)`, `total`, `snapshot()`.
- Staging: `<final>.part` (`_sparse_staging_path`, 1131). State: `<final>.part.json` (`_sparse_state_path`, 1135).
- `_create_sparse_staging` (1239) truncates to full size, calls `_enable_sparse_file` (Windows `DeviceIoControl FSCTL_SET_SPARSE` = `0x000900C4`, line 306) and `_reserve_blocks` (Linux `libc.fallocate`, line 283).
- `_write_sparse_state` (1153) fsyncs the staging inode first, then writes state to `.tmp`, fsyncs, `os.replace`, then `_fsync_parent`.
- `_read_sparse_state` (1190) refuses to resume unless `version == 2`, `identity` matches exactly, and staging logical size equals expected size.
- `_resume_identity` (1139): for HF, `{source, repo_id, filename, commit_hash, sha256, size}`.
- Range requests strictly validated: `_fetch_sparse_piece` (2552) rejects any response whose `Content-Range` isn't exactly `bytes {offset}-{stop-1}/{size}` (2620–2629) and rejects non-identity `Content-Encoding` (2630).
- Signed-URL expiry: on 401/403/410 mid-transfer, refreshes HF metadata under a lock (2597–2607), then raises to retry.

### Retries / backoff
`_retry_delay(attempt)` (590): `min(retry_delay * 2**attempt, max_retry_delay)`. `_wait_before_retry(delay)` (595) uses `cancel_event.wait(delay)` so backoff is interruptible. No backoff if the attempt made forward progress (2675).

### SHA256 verification and caching
- `utilities/sha256_cache.json` — `{"repo/file": sha}` from HF etag/LFS metadata.
- `utilities/verified_files_cache.json` — `{"repo/file": {sha256, size, mtime, verified_at}}`.
- `_load_shared_json_cache` (185) returns one shared dict per path; `_save_shared_json_cache` (203) writes via `NamedTemporaryFile` + `fsync` + `os.replace`.
- `is_file_verified` (728) matches sha + size + mtime within 1.0 s tolerance.
- `verify_file_sha256` (1581) hashes with 8 MB reads and a live progress line every 0.2 s. On mismatch `refetch_sha256_from_hf` (657).
- Verification happens on the staging file before install (`_finish_sparse_download`, 3009).

### Atomic rename
`_install_sparse_staging` (1260): `os.replace(staging, filepath)` + `_fsync_parent` + delete state, with 5 retries at 0.25/0.5/1/2 s on `PermissionError`.

### Disk-space checks
`_has_enough_disk_space` (614) — `shutil.disk_usage(dir).free` vs `required + 256 MB`. `_has_sparse_space` (1254) subtracts already-allocated blocks via `_allocated_file_bytes` (369) (`GetCompressedFileSizeW` on Windows, `st_blocks*512` on POSIX). `_is_disk_full_error` (607) checks `errno.ENOSPC` and `winerror in (39, 112)`.

### Cancellation
Single `threading.Event` (`self.cancel_event`) checked at attempt start, every `iter_content` block, inside `_run_sparse_http` wait loop (2771), inside Xet stream loop (2937, also `session.sigint_abort()`). `keep_partial_on_cancel=True` flushes and preserves the range map.

### Parallel downloads (two levels)
1. Within one file — `_run_sparse_http` (2682): work list = `ranges.missing()` split by 32 MB `piece_size`; `workers = min(num_connections, 32, len(work))`; `ThreadPoolExecutor(thread_name_prefix="swarm-range")`.
2. Across files — `ParallelDownloadManager` (`parallel_download_manager.py`), a resizable FIFO pool; `_worker_loop` (242) only dequeues while `len(self._active) < self._parallel_limit`. Deduplication by key (`_download_task_key` in app 1510).

### Speed calculation (sliding window), `_run_sparse_http` 2749–2783
```python
samples = deque([(started, initial)])
...
samples.append((now, current))
window = max(1.0, float(self.config.get("progress_window", 8)))
while len(samples) > 2 and samples[1][0] <= now - window:
    samples.popleft()
sample_time, sample_bytes = samples[0]
speed = (current - sample_bytes) / max(0.001, now - sample_time)
```
Stall detector at 2789: no byte for `max(60, read_timeout*2)` s → abort and retry.

### HF URL handling
`get_hf_remote_file` (922) resolves an immutable `HFRemoteFile(repo_id, filename, size, sha256, commit_hash, location, xet_file_hash, xet_refresh_route)`; `refresh_hf_remote_file` (972) re-resolves by `commit_hash` and raises if size/sha changed. Auth: `_headers_with_optional_hf_auth` (997) attaches Bearer only for `huggingface.co`/`hf.co`; `_request` (1036) retries once without token on 401/403.

## 3. Progress plumbing

### Core idea — one string per active download pushed to two sinks
`RobustDownloader.show_progress_line` (557):
```python
def show_progress_line(self, text: str):
    console_manager.show_progress_line(text, progress_key=self.console_progress_key)
    if self.ui_progress_callback:
        try:
            self.ui_progress_callback(text)
        except Exception:
            pass
```
`console_progress_key` is the job id — same key identifies console row and UI row.

### Producer line format (`print_progress`, 877)
```python
speed_str = self.format_bytes(speed_bytes_per_sec) + "/s" if speed_bytes_per_sec else "0 B/s"
trailing_text = status_text or f"{speed_str} | ETA {eta_str}"
# Metrics first; long filenames last so a narrow CMD window never truncates the percentage/speed/ETA.
line = (
    f"[{percent:5.1f}%] {self.format_bytes(current)} / {self.format_bytes(total)} | "
    f"{trailing_text} | {filename}"
)
self.show_progress_line(line)
```

### Rate limiting — 0.5 s at the source
`download_single` (3351–3357), `_run_sparse_http` (2781), `_run_sparse_xet` (2962) all use `if now - last_progress >= 0.5`. `verify_file_sha256` uses 0.2 s.

### Console renderer — `console_manager.py` (custom ANSI, no tqdm/rich)
State is `OrderedDict` of `progress_key -> text` per stream, plus `rendered_line_count`. Three modes:
1. TTY with cursor support — one stable row per download, updated in place. `_update_existing_line_locked` (163):
```python
keys = list(state.lines)
line_index = keys.index(progress_key)
distance_from_bottom = len(keys) - line_index - 1
text = _truncate_to_width(state.lines[progress_key])
if distance_from_bottom:
    stream.write(f"\x1b[{distance_from_bottom}A")
stream.write("\r\x1b[2K" + text)
if distance_from_bottom:
    stream.write(f"\x1b[{distance_from_bottom}B\r")
    last_line = _truncate_to_width(state.lines[keys[-1]])
    if last_line:
        stream.write(f"\x1b[{len(last_line)}C")
```
2. TTY without VT — `\r` + one visible row.
3. Redirected / not a TTY — prints at most once per second per key, plus always on 100%.

Windows VT enabled once via `_enable_windows_cursor_movement` (74) — `GetConsoleMode`/`SetConsoleMode` with `ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)`.

`console_manager.log(msg)` (309) clears the live block, prints the log line, re-renders the block:
```python
def log(msg, stream=None):
    stream = stream or sys.stdout
    with _lock:
        state = _state_for(stream)
        if state.rendered_line_count and _isatty(stream):
            _clear_progress_block_locked(stream, state)
            print(msg, file=stream, flush=False)
            _render_progress_block_locked(stream, state)
        else:
            print(msg, file=stream, flush=True)
```
Module guarded by one module-level `threading.RLock`.

### Gradio side — shared state + polling timer (NOT generator yields)
```python
_live_progress_statuses = {}          # job_id -> (label, progress_text)
_live_progress_lock = threading.Lock()
_status_revision = 0                  # bumped on every log/progress change
```
`_render_live_progress_html` (1402) regexes the percent out of the same console string and renders a real `<progress>` element per job:
```python
for label, progress_text in live_progress:
    text = str(progress_text or "Waiting for progress data...").strip()
    match = re.search(r"(?:\[\s*)?(\d+(?:\.\d+)?)%", text)
    percent = max(0.0, min(100.0, float(match.group(1)))) if match else None
    progress_value = f' value="{percent:.1f}"' if percent is not None else ""
    percent_text = f"{percent:.1f}%" if percent is not None else "Working"
    rows.append(
        '<div class="live-progress-row">'
        f'<div class="live-progress-heading"><strong>{escape(str(label))}</strong>'
        f'<span>{escape(percent_text)}</span></div>'
        f'<progress max="100"{progress_value}></progress>'
        f'<div class="live-progress-detail">{escape(text)}</div>'
        '</div>'
    )
```
The pump — 2 Hz `gr.Timer` with per-browser signature (`Downloader_Gradio_App.py:4870–4892`):
```python
log_timer = gr.Timer(0.5, active=True)

def update_log_display(previous_signature):
    log_text, live_html, queue_label, signature = get_status_display_snapshot()
    if signature == previous_signature:
        return gr.skip(), gr.skip(), gr.skip(), signature
    return log_text, live_html, queue_label, signature

log_timer.tick(
    update_log_display,
    [status_signature_state],
    [log_output, live_progress_output, queue_status_label, status_signature_state],
    show_progress=False, queue=False,
)
```
`get_status_display_snapshot` (1425):
```python
def get_status_display_snapshot():
    with log_lock:
        history_snapshot = list(log_history)
    with _live_progress_lock:
        live_progress = list(_live_progress_statuses.values())
    with _status_revision_lock:
        revision = _status_revision
    queue_label = get_download_status_label()
    signature = f"{revision}|{queue_label}"
    return (_compose_log_output(history_snapshot, None), _render_live_progress_html(live_progress), queue_label, signature)
```
`_handle_downloader_log` (1464) calls `add_log(message, mirror_to_console=False)` because the downloader already printed to console. `add_log` (1478) drops exact-duplicate consecutive messages and caps history at 100 lines.

Job start seeds both sinks (`download_worker`, 1867–1872):
```python
starting_status = "[STARTING] Preparing target and fetching file metadata..."
console_manager.show_progress_line(f"{starting_status} | {model_name}", progress_key=job_id)
_set_live_progress_status(starting_status, job_id=job_id, label=model_name)
```
Log textbox `js_on_load` script (3235–3289) implements follow-tail-vs-preserve-scroll with a `#swarm-follow-log` checkbox; CSS `overflow-anchor: none`.

## 4. UI design
- Theme (3019): `gr.themes.Ocean(primary_hue="emerald", secondary_hue="sky", neutral_hue="slate", radius_size="lg", spacing_size="md", text_size="md", font=("Segoe UI Variable","Segoe UI","system-ui","sans-serif"), font_mono=("Cascadia Mono","Cascadia Code","Consolas","ui-monospace","monospace"))`.
- Gradio 6: theme and CSS go to `launch()`: `gradio_app.launch(inbrowser=True, share=..., allowed_paths=..., theme=SWARM_THEME, css=CUSTOM_CSS)` at 5509. `Blocks(title=..., analytics_enabled=False, fill_width=True)` at 3048.
- CSS: `CUSTOM_CSS` ~750 lines, 2262–3017; uses Gradio CSS vars (`--border-color-primary`, `--background-fill-secondary`, `--font-mono`).
- Layout: `gr.Sidebar(open=True, width=380)` (3071); `gr.Tabs()` (3403). Sidebar resizable via JS persisting to `localStorage`.
- Colored buttons: `variant="stop"` cancel; category accent colors via CSS custom props per group.
- Model lists with sizes: raw HTML buttons `⬇️ {name} ({size})`, `get_model_size_display` (396) → `" (12.34 GB)"`; group headers `"3 files · 24.10 GB"`.
- HTML→Gradio bridge (3449–3460): hidden `gr.Textbox(elem_id="swarm-html-action")` + hidden `gr.Button(elem_id="swarm-html-trigger")`; JS writes `data-swarm-action` JSON into the textbox and clicks the hidden button.

## 5. Preset / last-used state
`last_settings.json` in CWD: `{"path", "comfy_ui_structure", "forge_structure", "lowercase_folders"}`; `save_last_settings` (212), `load_last_settings` (237); validates path on load.

## 6. Cross-platform path handling
- `os.path.normpath`/`os.path.join`; case-insensitive directory resolution on POSIX (`resolve_target_directory` 1264).
- Dedupe keys use `os.path.normcase(os.path.abspath(os.path.expanduser(path)))`.
- Platform branches: `_reserve_blocks` (Linux fallocate), `_enable_sparse_file` (Windows), `_allocated_file_bytes`, `_fsync_parent` (POSIX only), `_is_disk_full_error`.
- App startup reconfigures `sys.stdout` for cp1252/cp437 consoles.
- `get_available_drives` (5395) + `allowed_paths` (5481–5505).

## Excerpts

**Download core loop** (`HF_model_downloader.py:3323–3357`, `download_single`):
```python
with open(filepath, mode) as f:
    for chunk in response.iter_content(chunk_size=int(self.config["chunk_size"])):
        if self.cancel_event and self.cancel_event.is_set():
            response.close()
            self.clear_progress_line()
            self.log(f"[CANCELLED] {filename} - download cancelled")
            if self._keep_partial_on_cancel():
                self.log(f"[RESUME] Preserved {self.format_bytes(os.path.getsize(filepath))} for the next attempt.")
            else:
                try:
                    if os.path.exists(filepath): os.remove(filepath)
                except OSError: pass
            return False
        if chunk:
            remaining = file_size - (resume_pos + downloaded)
            if len(chunk) > remaining:
                raise IOError(f"server sent {len(chunk)} bytes with only {remaining} bytes remaining")
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_update >= 0.5:
                total = resume_pos + downloaded
                speed = downloaded / max(0.001, now - start_time)
                self.print_progress(total, file_size, start_time, filename, speed)
                last_update = now
```

**SHA verification cache** (`HF_model_downloader.py:728–767`):
```python
def is_file_verified(self, repo_id, filename, filepath, expected_sha) -> bool:
    if not expected_sha: return False
    cache_key = f"{repo_id}/{filename}"
    with _CACHE_LOCK:
        cached_info = dict(self.verified_cache.get(cache_key, {}))
    if not cached_info or not os.path.exists(filepath): return False
    current_size = os.path.getsize(filepath)
    current_mtime = os.path.getmtime(filepath)
    return (cached_info.get('sha256') == expected_sha and
            cached_info.get('size') == current_size and
            abs(cached_info.get('mtime', 0) - current_mtime) < 1.0)

def mark_file_verified(self, repo_id, filename, filepath, sha256):
    cache_key = f"{repo_id}/{filename}"
    if os.path.exists(filepath):
        with _CACHE_LOCK:
            self.verified_cache[cache_key] = {'sha256': sha256, 'size': os.path.getsize(filepath),
                                              'mtime': os.path.getmtime(filepath), 'verified_at': time.time()}
            self.save_verified_cache()
```

## What to copy, ranked
1. `utilities/console_manager.py` — self-contained (326 LOC, stdlib only).
2. `utilities/parallel_download_manager.py` — self-contained (304 LOC).
3. The dual-sink callback pattern: one `progress_key` per job; `show_progress_line` fans out to console + UI callback; 0.5 s throttle at producer; revision counter + `gr.State` signature + `gr.Timer(0.5)` + `gr.skip()` at consumer.
4. Sparse staging + `_RangeMap` + fsync'd `.part.json` resume model.
5. `utilities/xet_environment.py` + "configure before importing huggingface_hub" ordering.

Caveats: `HF_model_downloader.py` carries ~700 lines of dead legacy code (`_download_with_hf_xet_legacy` 1467, `merge_chunks` 1820/1908, `_download_parallel_legacy` 2200, `_parallel_manifest_*` 1347–1425); `get_model_size_display` (396) prints `DEBUG:` lines on cache miss.
