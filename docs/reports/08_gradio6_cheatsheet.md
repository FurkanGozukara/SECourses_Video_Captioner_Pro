# Gradio 6.26 API Cheat Sheet
Source of truth: `C:/Users/Furkan/Videos/gradio` (v6.26.0). Line refs from that tree.

Environment constraints (`requirements.txt`): `python >= 3.10`, `fastapi>=0.115.2,<1.0`, **`starlette>=1.0.1,<2.0`**, `pydantic>=2.0,<=3.0`, `huggingface_hub>=1.16.0,<2.0`, `gradio_client==2.6.1`, `numpy<3`, `pandas<4`, `pillow<13`.

## 1. Gradio 5 → 6 breaking changes
Doc: `guides/11_other-tutorials/gradio-6-migration-guide.md`.

### 1.1 App-level params moved `Blocks()` → `launch()`
`theme`, `css`, `css_paths`, `js`, `head`, `head_paths` (impl `gradio/blocks.py:2708-2713`).
```python
with gr.Blocks(title="Captioner Pro", fill_width=True) as demo:   # NO theme/css here
    ...
demo.launch(theme=gr.themes.Soft(), css=CSS, head=HEAD_JS)
```
Passing them to `Blocks()` still works (stashed and re-applied, `blocks.py:1144-1159`, `2784-2791`) but warns. `Blocks.__init__` (`1089-1097`): `gr.Blocks(analytics_enabled=None, mode="blocks", title="Gradio", fill_height=False, fill_width=False, delete_cache=None)`.

### 1.2 `show_api` → `footer_links` (launch) and `api_visibility` (events)
- `launch(show_api=...)` removed → `launch(footer_links=[...])`, entries `"api" | "gradio" | "settings" | "runs"` (`blocks.py:2681-2684`); `footer_links=[]` hides footer.
- Event `show_api=` removed, `api_name=False` removed → `api_visibility: "public"|"undocumented"|"private"` (`gradio/events.py:642`).

### 1.3 Event-listener signature (`gradio/events.py:616-646`)
```python
.click(fn=None, inputs=None, outputs=None, api_name=None, api_description=None,
       scroll_to_output=False, show_progress="full", show_progress_on=None,
       queue=True, batch=False, max_batch_size=4, preprocess=True, postprocess=True,
       cancels=None, trigger_mode=None, js=None,
       concurrency_limit="default", concurrency_id=None,
       api_visibility="public", time_limit=None, stream_every=0.5,
       key=None, validator=None)
```
Default `api_name` = function name (not `/predict`).

### 1.4 `buttons=` unification
`show_copy_button`, `show_download_button`, `show_share_button`, `show_fullscreen_button`, `show_reset_button`, `show_copy_all_button`, `show_export_button` removed → single `buttons: list[str | gr.Button]`.
| Component | `buttons` accepts |
|---|---|
| `gr.Textbox` | `"copy"` \| `gr.Button` — only shown if `show_label=True` |
| `gr.Markdown` | `"copy"` |
| `gr.Dataframe` | `"fullscreen"`, `"copy"` |
| `gr.Image` | `"download"`, `"share"`, `"fullscreen"` \| `gr.Button` |
| `gr.Video` / `gr.Audio` | `"download"`, `"share"` \| `gr.Button` |
| `gr.Gallery` | `"share"`, `"download"`, `"download_all"`, `"fullscreen"` \| `gr.Button` |
| `gr.Slider` | `"reset"` |
| `gr.JSON` / `gr.Code` | `"copy"`, `"download"` |
Custom toolbar buttons (`guides/04_additional-features/16_custom-buttons.md`):
```python
rerun = gr.Button("Re-caption", size="sm")
tb = gr.Textbox(label="Caption", buttons=["copy", rerun])
rerun.click(recaption, inputs=[vid], outputs=tb)
```

### 1.5 `visible` is tri-state — key for hidden hotkey buttons
`visible: bool | Literal["hidden"] = True`. `visible=False` → removed from DOM entirely (`js/atoms/src/Block.svelte:215`); `visible="hidden"` → stays in DOM with `display:none`. For JS-driven hidden buttons use `visible="hidden"`.

### 1.6 `gr.Dataframe` `row_count` / `col_count` restructured (`components/dataframe.py:63-113`)
```python
row_count=(5, "fixed")   →   row_count=5, row_limits=(5, 5)
row_count=(5, "dynamic") →   row_count=5, row_limits=None
col_count=(3, "fixed")   →   column_count=3, column_limits=(3, 3)
```

### 1.7 Removed component params
| Removed | Replacement |
|---|---|
| `Image/Video(mirror_webcam=, webcam_constraints=)` | `webcam_options=gr.WebcamOptions(mirror=True, constraints={...})` |
| `Audio/Video(min_length=, max_length=)` | `validator=` + `gr.validators.is_audio_correct_length` / `is_video_correct_length` |
| `WaveformOptions(show_controls=)` | `show_recording_waveform=` |
| `gr.Video` tuple `(video, subtitle)` return | `return gr.Video(value=path, subtitles=srt)` |
| `Client(hf_token=)` | `Client(token=)` |

### 1.8 `gr.HTML(padding=)` / `gr.Markdown` default `container=False, padding=False`.

### 1.9 Unchanged
`gr.update()` (accepts `visible="hidden"`), `every=` on components AND `gr.Timer`, `gr.State`, `.change`/`.input`, `gr.on`, `js=`, `head=`, `queue()`, `concurrency_limit`, `cancels=`, `trigger_mode`, `scale`, `min_width`, `variant`, `Button(size=, icon=)`, `Textbox(info=)`, `Dropdown(allow_custom_value=)`, `Video(sources=)`, `Row(equal_height=)`, `Column(variant=)`, `Accordion(open=)`, `Tabs(selected=)`, `Tab(id=)`, `Timer(active=)`. New: `gr.skip()`, `gr.validate(is_valid, message)`, `gr.Success()`, `gr.api()`, `gr.Draggable`.

## 2. Theming, dark mode, per-button colors
### 2.1 Themes
`Base, Default, Origin, Citrus, Monochrome, Soft, Glass, Ocean, Ember, Neon, Cyberpunk, Mario`.
```python
theme = gr.themes.Soft(
    primary_hue="blue", secondary_hue="slate", neutral_hue="gray",
    spacing_size="sm", radius_size="md", text_size="md",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_dark="#3b82f6",
    button_primary_background_fill_hover="*button_primary_background_fill",
    body_background_fill="#ffffff", body_background_fill_dark="#0b0f19",
    block_background_fill_dark="#111827", loader_color="#2563eb",
)
theme.custom_css = ".my-widget { border-radius: 12px; }"
demo.launch(theme=theme)
```
Convention `<element>_<subelement>_<property>_<state>[_dark]`; `*` references another var. Var list: `guides/11_other-tutorials/css-variables-reference.md` (`--body-background-fill`, `--body-text-color`, `--background-fill-primary/-secondary`, `--block-background-fill`, `--block-border-color`, `--border-color-primary`, `--color-accent`, `--button-primary-background-fill`...).

### 2.2 Dark mode
`js/app/src/routes/[...catchall]/+page.svelte:120-166`: explicit `theme_mode` → `?__theme=` param → system. Adds class **`dark` on `document.body`**. Force dark: `?__theme=dark`. CSS: `.dark .my-class { ... }`.

### 2.3 Per-button colors
`elem_classes` lands directly on the `<button>` (`js/button/shared/Button.svelte:47,70`); `elem_id` becomes `id`. Built-in `.primary/.secondary/.stop` are Svelte-scoped (specificity (0,2,0)) — use `!important` or double the class:
```python
CSS = """
.btn-go, .btn-go.btn-go { background: #16a34a !important; border-color: #15803d !important; color: #fff !important; }
.btn-go:hover { background: #15803d !important; }
.dark .btn-go, .dark .btn-go.btn-go { background: #22c55e !important; border-color: #16a34a !important; color: #04160a !important; }
"""
go = gr.Button("Caption All", variant="primary", elem_classes=["btn-go"], elem_id="btn_go")
demo.launch(css=CSS)
```
`gr.Button(variant=)`: `"primary" | "secondary" | "stop" | "huggingface"`; `size` `"sm"|"md"|"lg"`; `icon: str | Path`; `link`/`link_target`. External files in CSS via `/gradio_api/file=`.

## 3. Media components
### 3.1 `gr.File` (`components/file.py:44-70`)
`gr.File(value=None, file_count="single"|"multiple"|"directory", file_types=[".mp4", "video"], type="filepath"|"binary", height=None, allow_reordering=False, buttons=None)`. Return: `"filepath"`+single → `str`; multiple/directory → `list[str]`. Never `FileData` on the Python side.

### 3.2 `gr.Video` (`components/video.py:62-93`)
```python
gr.Video(value=None, format=None, sources=["upload","webcam"], height=None, width=None,
         include_audio=None, autoplay=False, loop=False, streaming=False, buttons=["download","share"],
         webcam_options=..., watermark=..., subtitles=None, playback_position=0.0, interactive=None,
         visible=True|False|"hidden")
```
Events: `change, clear, start_recording, stop_recording, stop, play, pause, end, upload, input`. Preprocess returns `str` filepath.
**Built-in trim UI only when `interactive=True`** (`js/video/shared/Player.svelte:315`; `VideoControls.svelte` trim icon + `VideoTimeline.svelte` with draggable handles). Trimming runs client-side ffmpeg.wasm; trimmed blob uploaded as `video.mp4` and `.change()` fires with the new path (`Player.svelte:128-133`). No timestamps exposed.
```python
vid = gr.Video(label="Clip", interactive=True, height=420)
vid.change(on_new_clip, vid, gr.Textbox())
```
`format="mp4"` triggers ffmpeg re-encode on preprocess whenever extension differs (`video.py:192-240`) — **leave `format=None`**. New: `subtitles=` (srt/vtt/json), `playback_position` (two-way).

### 3.3 `gr.Audio` (`components/audio.py:83-113`)
`gr.Audio(value=None, sources=["upload","microphone"], type="numpy"|"filepath", format="wav"|"mp3"|None, editable=True, recording=False, streaming=False, autoplay=False, loop=False, buttons=["download","share"], waveform_options=gr.WaveformOptions(...), subtitles=None, playback_position=0.0)`. Use `type="filepath"`. Built-in trim editor with `editable=True` (wavesurfer Regions, `js/audio/shared/WaveformControls.svelte:16,63-120`); trim re-uploads and dispatches `change`. `gr.WaveformOptions`: `waveform_color`, `waveform_progress_color`, `trim_region_color`, `show_recording_waveform=True`, `skip_length=5`, `sample_rate=44100`.

### 3.4 `gr.Gallery` (`components/gallery.py:63-118`)
`gr.Gallery(value=None, format="webp", file_types=None, columns=2, rows=None, height=None, allow_preview=True, preview=None, selected_index=None, object_fit="contain"|"cover"|..., fit_columns=True, type="filepath"|"pil"|"numpy", sources=[...], buttons=[...])`. Holds videos (`GalleryVideo`, `gallery.py:38-49`, `297-303`); videos always returned as paths. Events: `select, upload, change, delete, preview_close, preview_open`.
```python
gal = gr.Gallery(columns=6, object_fit="cover", height=360, allow_preview=True)
def pick(evt: gr.SelectData):
    return evt.index, evt.value
gal.select(pick, None, [idx_num, meta])
clear_sel.click(lambda: gr.Gallery(selected_index=None), None, gal)
```

### 3.5 `gr.Image`: `format="webp"`, `image_mode="RGB"`, `sources`, `type="numpy"|"pil"|"filepath"`, `buttons=["download","share","fullscreen"]`, `placeholder`, `alt_text`.
### 3.6 `gr.MultimodalTextbox`: `sources`, `file_types`, `file_count`, `lines`, `max_lines`, `submit_btn`, `stop_btn`; value `{"text": str, "files": list[str]}`.

### 3.7 Serving big local videos without copying
Guide `guides/04_additional-features/08_file-access.md`:
1. **`gr.set_static_paths([...])`** (`gradio/utils.py:1377-1402`) — served directly from disk, NOT copied to cache. Use for huge videos.
2. `launch(allowed_paths=[...])` — permits serving, but returned files are still copied into cache.
3. Cache — files copied if in CWD, `tempfile.gettempdir()`, or `allowed_paths`.
```python
gr.set_static_paths([ROOT])
demo.launch(allowed_paths=[ROOT], max_file_size="8gb")
gr.HTML(f"<video controls width=640 src='/gradio_api/file={abs_path}'></video>")
```
Cache location `GRADIO_TEMP_DIR`. Eviction `gr.Blocks(delete_cache=(86400, 86400))`. `blocked_paths` beats everything.

## 4. Progress, streaming, cancel
### 4.1 `gr.Progress` (`gradio/helpers.py:673-905`)
```python
def caption_all(files, progress=gr.Progress()):
    progress(0.0, desc="loading model")
    for f in progress.tqdm(files, desc="Captioning", unit="clips"):
        ...
    progress(None)
```
`gr.Progress(track_tqdm=True)` auto-instruments tqdm; `progress((3, 10), desc=...)`; nested calls pass same object.
### 4.2 Generators
```python
def run(files):
    for i, f in enumerate(files):
        yield cap, f"{i+1}/{len(files)}", gr.skip()
    yield final_caption, "done", results_df
```
### 4.3 `show_progress`: `"full"`, `"minimal"`, `"hidden"`; `show_progress_on=[comp,...]`.
### 4.4 `gr.Timer` polling
```python
timer = gr.Timer(1.0, active=False)
timer.tick(poll, None, [vram, logs], show_progress="hidden", api_visibility="private")
start.click(lambda: gr.Timer(active=True), None, timer)
```
Always `show_progress="hidden"` on tick handlers. `every=` form: `gr.Textbox(value=tail_fn, inputs=[path_state], every=timer)`.
### 4.5 Queue (`blocks.py:2556-2596`)
`demo.queue(default_concurrency_limit=1, max_size=32, status_update_rate="auto", api_open=None)`; per-event `concurrency_limit` int/None/"default"; `concurrency_id="gpu"` on all GPU-bound events.
### 4.6 Cancellation
```python
run_evt = run_btn.click(caption_stream, inputs, outputs)
stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_evt]).then(lambda: "Cancelled", None, status)
```
Queued and **iterating generators** are cancelled; running non-generator calls finish. Error if `cancels=` targets an event with `queue=False`. `trigger_mode`: `"once"` (default), `"multiple"`, `"always_last"` (default for `.change`/`.key_up`).
### 4.7 Streaming media out: `streaming=True` + `autoplay=True`, yield chunks ≥1 s.

## 5. Global keyboard shortcuts
```python
HOTKEYS = """
<script>
document.addEventListener('keydown', function (e) {
  const t = e.target.tagName.toLowerCase();
  if (t === 'input' || t === 'textarea' || e.target.isContentEditable) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const map = { 'ArrowLeft': 'hk_prev', 'ArrowRight': 'hk_next', 's': 'hk_save' };
  const id = map[e.key];
  if (!id) return;
  const btn = document.getElementById(id);
  if (btn) { e.preventDefault(); btn.click(); }
}, false);
</script>
"""
with gr.Blocks() as demo:
    prev = gr.Button("prev", elem_id="hk_prev", visible="hidden")   # MUST be "hidden", not False
    next_ = gr.Button("next", elem_id="hk_next", visible="hidden")
demo.launch(head=HOTKEYS)
```
`js=` on events: string JS fn taking inputs returning outputs, runs before `fn` (`fn=None` for JS-only); `js=True` transpiles a lambda.
```python
btn.click(None, inputs=[cap], outputs=[], js="(t) => { navigator.clipboard.writeText(t); return []; }")
```

## 6. `gr.Dataframe` (`components/dataframe.py:63-113`)
```python
df = gr.Dataframe(value=pd.DataFrame(...), headers=["file", "caption", "dur"],
    datatype=["str", "markdown", "number"], type="pandas", row_count=None, row_limits=None,
    column_count=None, column_limits=None, interactive=True, wrap=True, line_breaks=True, max_chars=None,
    max_height=500, column_widths=["30%", "55%", "15%"], pinned_columns=1, static_columns=[0, 2],
    show_row_numbers=True, show_search="none"|"search"|"filter", buttons=["copy", "fullscreen"], label="Captions")
```
Return: `"pandas"` → `pd.DataFrame`; `"polars"`; `"numpy"`; `"array"` → `list[list]`. Browser-side sorting does not affect values. Events: `change, input, select, edit`.
```python
def on_select(evt: gr.SelectData): return evt.index[0]        # (row, col)
def on_edit(evt: gr.EditData): ...                             # evt.index, evt.value, evt.previous_value
df.select(on_select, None, row_idx); df.input(persist_edits, df, None)
```

## 7. `gr.Textbox` (`components/textbox.py:70-101`)
`gr.Textbox(value="", type="text", lines=1, max_lines=None, max_length=None, placeholder=None, label=None, info=..., autofocus=False, autoscroll=True, text_align=None, rtl=False, buttons=["copy"], submit_btn=False|True|"Send", stop_btn=False|True|"Stop", html_attributes=..., interactive=True, visible=...)`. Events: `change, input, select, submit, focus, blur, stop, copy`. Live log: `gr.Textbox(lines=20, max_lines=20, autoscroll=True, interactive=False)`.

## 8. Layout & components
```python
gr.Row(variant="default"|"panel"|"compact", equal_height=False, scale=None, height=None, min_height=None, max_height=None, visible=...)
gr.Column(scale=1, min_width=320, variant=...)
gr.Group(); gr.Accordion(label, open=True); gr.Sidebar(label=None, open=True, width=320, position="left"|"right")
gr.Draggable(orientation="column"|"row")
gr.Tabs(selected=None)  # EVENTS: change, select
gr.Tab(label, visible=True, interactive=True, id=None, scale=None, render_children=False)
gr.Walkthrough(selected=None); gr.Step(label, visible=True, interactive=True, id=None)
```
Programmatic tab switching:
```python
with gr.Tabs(selected="caption") as tabs:
    with gr.Tab("Caption", id="caption"): ...
    with gr.Tab("Export", id="export", interactive=False) as export_tab: ...
goto.click(lambda: gr.Tabs(selected="export"), None, tabs)
unlock.click(lambda: gr.Tab(interactive=True), None, export_tab)
def which(evt: gr.SelectData): return evt.value
tabs.select(which, None, status)
```
Others: `gr.Markdown(value, sanitize_html=True, line_breaks=False, height/max_height, container=False, padding=False, buttons=["copy"])`; `gr.HTML(value, html_template="${value}", css_template="", js_on_load=..., head=None, server_functions=[...], autoscroll=False, min_height, max_height, **props)`; `gr.Label`; `gr.Number(precision, minimum, maximum, step, info)`; `gr.Slider(minimum, maximum, value, step, precision, buttons=["reset"])`; `gr.Radio(choices, type)`; `gr.CheckboxGroup(choices, show_select_all=False)`; `gr.Dropdown(choices, multiselect, allow_custom_value, max_choices, filterable, type)`; `gr.Code(language, lines, max_lines)`; `gr.JSON(open, show_indices, max_height, buttons)`; `gr.HighlightedText`; `gr.State(value, time_to_live, delete_callback)`; `gr.BrowserState(default_value, storage_key, secret)`; `gr.DownloadButton(label, value, variant, size)`; `gr.DeepLinkButton`; `gr.Navbar`. `gr.Examples(..., cache_examples=True, cache_mode="lazy")`. `@gr.render(inputs=..., triggers=...)` with `key=` + `preserved_by_key=`. Multipage `with demo.route("Second", "/second")`.

## 9. `launch()` (`gradio/blocks.py:2657-2762`)
```python
demo.launch(inline=None, inbrowser=False, share=None, debug=False, max_threads=40, auth=None, prevent_thread_lock=False,
    show_error=False, server_name=None, server_port=None, height=500, width="100%", favicon_path=None,
    ssl_keyfile=None, ssl_certfile=None, quiet=False, footer_links=None, run_history=None,
    allowed_paths=None, blocked_paths=None, root_path=None, app_kwargs=None, state_session_capacity=10000,
    max_file_size=None, enable_monitoring=None, strict_cors=True, ssr_mode=None, pwa=None, mcp_server=None,
    i18n=None, theme=None, css=None, css_paths=None, js=None, head=None, head_paths=None) -> (app, local_url, share_url)
```
Recommended local:
```python
demo.queue(default_concurrency_limit=1, max_size=64).launch(
    server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True,
    allowed_paths=[DATA_ROOT, OUT_ROOT], max_file_size="8gb", footer_links=[], enable_monitoring=False,
    ssr_mode=False, pwa=False, theme=THEME, css=CSS, head=HOTKEYS, favicon_path="assets/icon.png")
```
Env vars: `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME`, `GRADIO_TEMP_DIR`, `GRADIO_ALLOWED_PATHS`, `GRADIO_BLOCKED_PATHS`, `GRADIO_ANALYTICS_ENABLED`, `GRADIO_SSR_MODE`, `GRADIO_DEFAULT_CONCURRENCY_LIMIT`.

## 10. Gotchas
1. Starlette 1.x, `fastapi<1.0`, `pydantic<=3.0`, `huggingface_hub>=1.16,<2`.
2. `gr.update()` works; accepts `visible="hidden"`.
3. `visible=False` deletes DOM node; `"hidden"` keeps it.
4. `show_*_button` params are hard errors → `buttons=[...]`.
5. Textbox `buttons=` only with `show_label=True`.
6. `api_name=False`/`show_api=` hard errors → `api_visibility=`.
7. Default API names changed.
8. Custom-CSS specificity — use `!important`.
9. `gr.Video(format="mp4")` invokes ffmpeg on every upload — leave None.
10. Dark mode = `dark` class on `<body>` + `?__theme=`.
11. `gr.Progress`, `gr.Info/Warning/Success` need the queue.
12. `raise gr.Error(msg, duration=10, visible=True, title="Error", print_exception=True)`; `gr.Warning/Info/Success(message, duration=10, title=...)`; `duration=None` sticky.
13. `gr.Request`: `.headers`, `.client.host`, `.query_params`, `.session_hash`, `.username`.
14. Cache growth: `gr.Blocks(delete_cache=(3600, 3600))` + `demo.unload(cleanup_fn)` + `gr.State(time_to_live=...)`.
15. `gradio_client` 2.x: `handle_file(path_or_url)`, `Client(token=...)`.
16. `gr.HTML` is a mini custom-component framework (`html_template`/`css_template`/`js_on_load`/`server_functions`/`head`/`**props`, `trigger(...)`, `watch(...)`, `upload(...)`) — see `guides/03_building-with-blocks/06_custom-HTML-components.md`.
