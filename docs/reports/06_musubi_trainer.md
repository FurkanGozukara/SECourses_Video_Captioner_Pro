# SECourses Musubi Trainer — architecture report

Root: `G:/SECourses_Musubi_Trainer_v29/SECourses_Musubi_Trainer` (`<APP>`).

## 1. Layout
- `<APP>/gui.py` 201 — entry: 15 `gr.Tab`s, theme+CSS at `launch()` (`gui.py:146-167`).
- `<APP>/musubi_tuner_gui/common_gui.py` 3336 — path normalization, TOML config writers (`SaveConfigFile:1802`, `SaveConfigFileToRun:2130`), subprocess env, streaming runner, file pickers.
- `dataset_config_generator.py` 865 — folder-tree → musubi/kohya dataset TOML (image, WAN, LTX-2, MiniMax H3) + validator.
- `class_command_executor.py` 223 — `CommandExecutor`: Popen wrapper, `kill_command` tree kill.
- `class_configuration_file.py` 100 — preset Load/Save/Open widget.
- `class_accelerate_launch.py` 206 — multi-GPU flags.
- `class_training.py` 226 — attention checkboxes.
- `image_captioning_gui.py` 1095 + `class_image_captioning.py` 755 (Qwen2.5-VL image captioner; closest analog).
- `model_quantizer_gui.py` 4379; `<APP>/convert_to_quant/` 17,454 LOC quantizer backend.
- `<APP>/assets/style.css` 18 KB.

## 2. Preset system
- Presets are config TOMLs. `ConfigurationFile` (`class_configuration_file.py:10-100`): `config_dir` default `<APP>/presets`; dropdown `allow_custom_value=True` + buttons Open (amber) / Save (sky) / Load (violet). No delete button anywhere.
- "Last used" = auto-load on dropdown change (`qwen_image_lora_gui.py:4852-4862`). No last-used file.
- Defaults live at repo root as `*_defaults.toml`; protection is a name blacklist (`class_tab_config_manager.py:61-72`), not a filesystem lock.
- Component registry in newer tabs:
```python
# ltx2_lora_gui.py:1285-1289
    registrations = []
    def reg(key, component):
        registrations.append((key, component))
        return component
# ltx2_lora_gui.py:2447
    settings_list = [registration_map[key] for key in LTX2_PARAM_KEYS]
```
- Exclusions on save (`SaveConfigFile`, `common_gui.py:1802-1806`): `["file_path","save_as","headless","print_only"]`; `store_true_params` (`common_gui.py:2286-2322`) omitted when False; `zero_to_none_params` (`2245-2267`); empty `FILE_PATH_PARAMETERS` dropped (`2151-2182`).

## 3. Replace-words, prefix/suffix, dataset TOML
### Replace words — multiline Textbox, one `orig;replacement` per line (no Dataframe)
UI (`image_captioning_gui.py:237-257`):
```python
self.replace_words = gr.Textbox(
    label="Replace Words",
    info="Word pairs to replace in captions. Format: orgword;replaceword (each line is one replacement pair). Applied after prefix/suffix.",
    placeholder="Line 1: man;ohwx man\nLine 2: person;ohwx person\nLine 3: he;ohwx man\nLine 4: woman;ohwx woman\nLine 5: girl;ohwx woman",
    value=self.config.get("image_captioning.replace_words", ""), lines=5, max_lines=5)
self.replace_case_insensitive = gr.Checkbox(label="Case Insensitive Replace", value=True)
self.replace_whole_words_only = gr.Checkbox(label="Replace Whole Words Only", value=True)
```
Parser (`class_image_captioning.py:272-312`):
```python
if replace_words:
    lines = replace_words.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and ";" in line:
            parts = line.split(";", 1)
            if len(parts) == 2:
                org_word, replace_word = parts[0].strip(), parts[1].strip()
                if org_word:
                    if replace_whole_words_only:
                        pattern = r'\b' + re.escape(org_word) + r'\b'
                        caption_text = re.sub(pattern, replace_word, caption_text, flags=re.IGNORECASE) if replace_case_insensitive else re.sub(pattern, replace_word, caption_text)
                    else:
                        if replace_case_insensitive:
                            caption_text = re.sub(re.escape(org_word), replace_word, caption_text, flags=re.IGNORECASE)
                        else:
                            caption_text = caption_text.replace(org_word, replace_word)
# prefix/suffix AFTER replacements
if prefix: caption_text = prefix + caption_text
if suffix: caption_text = caption_text + suffix
```
Flags: the `info=` says "after prefix/suffix" but code applies replacements before — label is wrong. Replacements cascade (each rule re-scans modified text) — use single-pass alternation regex in new app.

### Caption tooling
Single image: `gr.Image(type="filepath")` + editable output Textbox + Copy (JS clipboard `image_captioning_gui.py:505-514`) + Save as text. Batch: recursive `scan_subfolders`, `overwrite_existing_captions` vs `append_existing_captions`, `extension` (default `.txt`), `output_format` text|jsonl, `copy_images`, `gr.Progress()` (`889`), stop button (`924-931`). Extension validation (`class_image_captioning.py:358`). No video captioning.

### Dataset TOML generation — `dataset_config_generator.py`
| Function | Line | Notes |
|---|---|---|
| `parse_repeat_count` | 15 | `^(\d+)_(.+)$` → (repeats, name); `0_x` → 1 + WARNING |
| `get_image_files` / `get_video_files` | 92 / 97 | images `.png .jpg .jpeg .webp .bmp .gif`; videos `.mp4 .avi .mov .webm .mkv .flv .wmv` |
| `create_caption_files` | 109 | |
| `generate_dataset_config_from_folders` | 655 | images; `control_directory`, `no_resize_control`, `control_resolution` |
| `generate_wan_dataset_config_from_folders` | 132 | images or videos; writes `frame_stride`+`frame_sample` |
| `round_frames_to_ltx2` | 296 | floor to `8k+1` |
| `generate_ltx2_dataset_config_from_folders` | 311 | `frame_stride` only if `slide`, `frame_sample` only if `uniform`; `target_fps` default 25.0 |
| `round_frames_to_minimax_h3` | 470 | floor to `17n+5`; range 124–345 |
| `generate_minimax_h3_dataset_config_from_folders` | 489 | videos only, batch_size 1, resolution ×32 |
| `save_dataset_config` | 810 | `toml.dump` + strip trailing array commas |
| `validate_dataset_config` | 826 | |
Core builder (`181-190, 252-278`):
```python
config = {"general": {"resolution": list(resolution), "caption_extension": caption_extension,
                      "batch_size": batch_size, "enable_bucket": enable_bucket, "bucket_no_upscale": bucket_no_upscale},
          "datasets": []}
directory_type = "video_directory" if has_videos else "image_directory"
dataset_entry = {directory_type: validate_path_for_toml(subdir_path), "num_repeats": repeat_count}
if has_videos:
    dataset_entry["target_frames"]    = [num_frames]
    dataset_entry["frame_extraction"] = frame_extraction   # head|chunk|slide|uniform|full
    dataset_entry["frame_stride"]     = frame_stride
    dataset_entry["frame_sample"]     = frame_sample
    dataset_entry["max_frames"]       = max_frames
    if source_fps: dataset_entry["source_fps"] = source_fps
dataset_entry["cache_directory"] = validate_path_for_toml(os.path.join(subdir_path, cache_directory_name))  # unique per dataset
```
Real VIDEO dataset TOML (`<APP>/dataset_tomls/wan_dataset_config_20260801_215555.toml`):
```toml
[[datasets]]
video_directory = "G:/SECourses_Musubi_Trainer_v29/test_video_dataset/1_ohwx"
num_repeats = 1
target_frames = [ 17]
frame_extraction = "head"
frame_stride = 1
frame_sample = 1
max_frames = 129
cache_directory = "G:/SECourses_Musubi_Trainer_v29/test_video_dataset/1_ohwx/cache_dir"

[general]
resolution = [ 960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
```
IMAGE dataset TOML:
```toml
[[datasets]]
image_directory = "G:/data/2_myconcept"
num_repeats = 2
cache_directory = "G:/data/2_myconcept/cache_dir"

[general]
resolution = [ 1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
```
Filenames: `wan_dataset_config_{ts}.toml`, `ltx2_dataset_{ts}.toml`, `minimax_h3_dataset_{ts}.toml`; sink `<APP>/dataset_tomls/`.

## 4. Block swap / VRAM / quant / attention
- `blocks_to_swap`: `gr.Number` (`class_model.py:109-115`) or `gr.Slider(0..MAX)` (`ltx2_lora_gui.py:1612-1620`); companions `use_pinned_memory_for_block_swap`, `block_swap_h2d_only`, `block_swap_ring_size`.
- No in-GUI VRAM preset dropdown; presets are shipped TOMLs + README table (`Demo_Training_Configs_.../README.md:16-33`):
```
| Krea_2_LoRA_Demo.toml            | 12 GB+ | blocks_to_swap 26 | 12.6 GB | 3.45 s/step |
| Krea_2_LoRA_Demo_High_VRAM.toml  | 24 GB+ | 16                | 16.7 GB | 2.37 |
  blocks_to_swap = 8 -> 19.9 GB (24 GB) ; 0 -> 22.7 GB (32 GB)
| MiniMax_H3_LoRA_Demo_24GB.toml   | 24 GB+ | 40 (H2D ring 2) | 384x384 | 124 frames |
| MiniMax_H3_LoRA_Demo_Lowest_VRAM | lowest | 48 (ring 1)     | 256x256 |
```
- Validation (`minimax_h3_lora_gui.py:702-714`): fp8 flags rejected; `blocks_to_swap` range; `block_swap_h2d_only` requires `blocks_to_swap > 0` and `gradient_checkpointing`.
- Attention: checkboxes `sdpa`, `flash_attn`, `sage_attn`, `xformers`, `split_attn`, `flash3` (`class_training.py:36-77`). Newer tabs enforce exactly one (`ltx2_lora_gui.py:742-746`); Qwen auto-repairs to sdpa (`qwen_image_lora_gui.py:2595-2620`). FA fallback contract in `class_training.py:6-11`. `XFORMERS_FORCE_DISABLE_TRITON=1` on Windows (`common_gui.py:2808-2809`). Text-encoder attention dropdown `["", "sdpa", "flash_attention_2", "eager"]` (`minimax_h3_lora_gui.py:1772-1775`).

## 5. ConvRot INT8 integration report summary (`ConvRot_INT8_Integration_Report.html`)
- Third base-weight mode beside BF16/FP8 for Krea 2 LoRA training (upstream musubi-tuner PR #1008). Frozen DiT weights INT8-quantized after regular Hadamard rotation (groups of 256); forward runs real INT8 matmul (`W_rot = W @ Hᵀ`, `x_rot = x @ H`, `y = x_rot @ W_rotᵀ == x @ Wᵀ`).
- Inference-side implementation `<APP>/convert_to_quant/convert_to_quant/utils/convrot.py:18-64` (`build_hadamard`, Kronecker, cached), `rotate_weight:66`, `rotate_activation:97`, `find_max_compatible_group_size:129`.
- Kernel: custom Triton (`musubi-tuner/src/musubi_tuner/modules/convrot_int8_kernels.py`, vendored from comfy-kitchen; `convrot_int8_utils.py` with `ConvRotInt8Quantizer` streaming safetensors loader, `ConvRotInt8LinearFn` autograd, `apply_convrot_int8_monkey_patch()`). Inference: `<APP>/convert_to_quant/convert_to_quant/comfy/int8_kernels.py` (1242 lines) registered via `comfy/quant_ops.py:14-27` with PyTorch fallback.
- Modules: 224 per-block Linears; everything else BF16. Monkeypatch keeps `nn.Linear` class, replaces `forward`.
- Loader: `comfy_quant` metadata layout; `_LAYOUT_REGISTRY` keyed on `(torch_op, layout_type)`; `formats/int8_conversion.py:19` (`convert_int8_to_comfy_quant`).
- Speed (isolated Linears, 4352 tokens): RTX 5090 BF16 16.33 ms / FP8 18.67 / ConvRot INT8 6.79 (2.40×); RTX 3090 57.83 / 62.64 / 22.92 (2.52×). Rel. error vs BF16: FP8 2.52e-02, ConvRot 1.38e-02.
- Quality: mean |Δloss| vs BF16 ConvRot bwd=bf16 0.273% < FP8 0.289% < ConvRot bwd=int8 0.319%.
- No-Triton fallback prints `NO TRITON -> slow dequantized BF16 fallback`.
- File naming (`model_quantizer_gui.py:2113-2183` `_default_output_name`): `{base}_{simple_|learned_}int8_convrot_{dynamic_}g{group}.safetensors`; Klein `{base}-comfy-{int8|int4}-convrot-hq-measured.safetensors`; LTX `{base}-comfy-int8-convrot-hq-22gb-video.safetensors`. Presets `PRESET_INT8_CONVROT = "INT8 ConvRot (Balanced / Native / Recommended)"` (`54`), JSON layer plans in `<APP>/model_quantizer_presets/`.

## 6. Subprocess launch, streaming, kill, progress
- Training inherits parent stdout (CMD window); Gradio gets only coarse status. Only `gr.Timer` is `model_quantizer_gui.py:3868`.
- Launch (`class_command_executor.py:51-72`): `subprocess.Popen(run_cmd, **kwargs)` no stdout capture; wrapper `.bat`/`.sh` when caching precedes training (`qwen_image_lora_gui.py:2828-2886`).
- Live-streaming variant (`common_gui.py:2728-2789`):
```python
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **utf8_subprocess_options(env))
try: proc.stdout.reconfigure(newline="")   # keep tqdm's \r
except (AttributeError, ValueError): pass
tail = deque(maxlen=tail_lines)
try:
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        stripped = line.rstrip("\r\n")
        if stripped.strip(): tail.append(stripped)
finally:
    proc.stdout.close(); returncode = proc.wait()
```
On failure maps tail against `CACHING_ERROR_HINTS` (`2690-2725`) via `explain_subprocess_failure:2718`.
- Kill tree (`class_command_executor.py:74-134`): psutil `parent.children(recursive=True)` → `child.kill()` each → `parent.kill()`; `_cancel_requested` flag prevents reporting cancel as failure. Wired `queue=False`.
- Captioning ETA (`class_image_captioning.py:429-456`): 10-sample moving average, `progress(current/total, msg)` + `print(f"\r[{i}/{n}] ...", end='')`.
- Command archiving `save_executed_script` (`common_gui.py:3210-3286`) → `<APP>/cli_executed_commands/{type}_{name}_{NN}.bat.txt`.

## 7. Download_Train_Models.py (3407 LOC)
`MODEL_CONFIGS` (`45`), `DOWNLOAD_CONFIG` (`289-314`) retries 5, backoff 2→30 s, `hash_chunk_size` 16 MiB, `range_connections` 16, aria2 16 conns; `DOWNLOAD_BACKENDS = ("auto","aria2","ranges","hub")`; `STATE_VERSION = 3`, `.model_download_cache`. `RemoteFileInfo` (`340`) with `digest_spec()` classifying ETag as sha256/git-sha1. Typed errors (`365-393`). Resume: state JSON (`_state_path:586`, `_read_state:593`, `_write_json_atomic:606`, `_state_matches:632`), aria2 (`_probe_aria2:1223`, `ARIA2_PROGRESS_RE` `330-336`), pure-Python ranges (`_range_layout:1950`, `_download_range_chunk:2053` strict 206 + Content-Range validation). `_compute_digest:872` (git blob header for sha1). Stage dir + `_atomic_replace:1933`; local reuse by size index (`_build_local_file_index:1004`, `_try_local_reuse:1095`). Progress `show_progress_line:482` lock-guarded `\r`, 5 s non-TTY interval.

## 8. Colored buttons / theme CSS
Theme at launch (`gui.py:154-160`): `gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", radius_size=gr.themes.sizes.radius_lg)`, `css=read_file_content("./assets/style.css")`.
`.mbtn` system (`style.css:258-326`):
```css
.gradio-container button[class*="mbtn-"] {
    --tx: #ffffff;
    background: linear-gradient(135deg, var(--g1), var(--g2)) !important;
    color: var(--tx) !important; border: none !important;
    border-radius: var(--btn-radius) !important; font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(var(--gs), 0.28), inset 0 1px 0 rgba(255,255,255,0.18) !important;
    transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
.gradio-container button[class*="mbtn-"]:hover:not([disabled]) { filter: brightness(1.07) saturate(1.05); transform: translateY(-1px); }
.gradio-container button[class*="mbtn-"]:active:not([disabled]) { filter: brightness(.94); transform: translateY(0) scale(.985); }
.gradio-container button[class*="mbtn-"][disabled] { opacity:.55; filter: saturate(.55); cursor:not-allowed; box-shadow:none !important; }
.dark .gradio-container button[class*="mbtn-"] { box-shadow: 0 2px 12px rgba(var(--gs),.20) !important; }
.gradio-container button.mbtn-red     { --g1:#ef4444; --g2:#dc2626; --gs:239,68,68; }
.gradio-container button.mbtn-amber   { --g1:#fbbf24; --g2:#f59e0b; --gs:245,158,11; --tx:#451a03; }
.gradio-container button.mbtn-lime    { --g1:#a3e635; --g2:#84cc16; --gs:132,204,22; --tx:#1a2e05; }
.gradio-container button.mbtn-cyan    { --g1:#22d3ee; --g2:#06b6d4; --gs:6,182,212;  --tx:#083344; }
/* + orange gold green emerald forest teal sky blue navy indigo violet purple fuchsia plum pink rose slate stone */
```
Usage `elem_classes=["mbtn", "mbtn-emerald"]`. Hero header `.app-hero` (`53-142`).

## 9. GPU picker / multi-GPU
Manual text entry only: `class_accelerate_launch.py:100-138` `multi_gpu` checkbox, `gpu_ids` textbox ("0,1"), `validate_gpu_ids` (`114-126`, advisory only), `num_processes`. Emission `AccelerateLaunch.run_cmd` (`149-206`).

## 10. Cross-platform paths
```python
# common_gui.py:350-401
def normalize_path(path: str) -> str:
    if not path: return path
    path = path.strip()
    while path and path[0] in ('"', "'"): path = path[1:]
    while path and path[-1] in ('"', "'"): path = path[:-1]
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    try:
        return str(Path(path).resolve()).replace("\\", "/")
    except (OSError, RuntimeError):
        return os.path.abspath(os.path.normpath(path)).replace("\\", "/")
```
`validate_path_for_toml:404`; encoding constants `SUBPROCESS_TEXT_ENCODING="utf-8"`, `errors="replace"`, `PYTHONIOENCODING="utf-8:backslashreplace"` via `utf8_subprocess_options:2843-2852` with `PYTHONUTF8=1`; configs read `utf-8-sig`; script generation branches `.bat` vs `.sh` (`3289-3336`); Tk pickers guarded by `is_display_available()`; `PORTABLE_MODEL_PATH_KEYS` (`40-59`) resolve against `$MUSUBI_TRAINING_MODELS_DIR`.

## Flags before copying
1. `replace_words` info text wrong. 2. Replacements cascade. 3. No preset delete / last-used persistence. 4. Default presets not truly protected. 5. No live log in browser. 6. `validate_gpu_ids` dead. 7. Unbounded log file — add rotation.
