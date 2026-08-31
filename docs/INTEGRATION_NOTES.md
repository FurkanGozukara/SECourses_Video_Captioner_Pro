# Verified integration probes (run on this machine, transformers 5.16.1)

These were executed for real against the downloaded checkpoints' processors on 2026-08-30. Build on them; do not re-derive.

## Video/audio input WITHOUT qwen-omni-utils readers
`torchvision.io.read_video` does not exist in torchvision 0.28; `torchcodec` fails to load (no ffmpeg shared libs); `decord` has no py3.12 Windows wheel. Therefore the app decodes media itself (PyAV frames + ffmpeg audio) and feeds the processors directly. **Both processors accept this:**

### Qwen2_5OmniProcessor (TimeChat + AVoCaDO folder)
```python
proc = Qwen2_5OmniProcessor.from_pretrained(model_dir)
conv = [{"role":"user","content":[{"type":"video"},{"type":"text","text":PROMPT}]}]  # placeholder items are enough for the template
text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
# frames: list[PIL.Image] OR np.uint8 [T,H,W,C]; dims multiples of 28; audio: np.float32 mono 16 kHz
inputs = proc(text=text, audio=[audio_np], videos=[frames], fps=2.0, use_audio_in_video=True,
              return_tensors="pt", padding=True)
# -> input_ids, attention_mask, feature_attention_mask, input_features, pixel_values_videos,
#    video_grid_thw, video_second_per_grid   (video+audio interleaving happened: 355 vs 303 tokens in the probe)
```
- Video-only: omit `audio=` and pass `use_audio_in_video=False` (works).
- TimeChat chat template auto-injects `You are a helpful assistant.` when no system message — correct, keep it.
- The 5.16 warning about `cap_pixels_per_frame` is expected; pass `cap_pixels_per_frame=False` if the processor `__call__` accepts it (pins current behavior; our frames are pre-resized anyway).

### Qwen3OmniMoeProcessor (Qwen3-Omni-30B-A3B-* folder)
```python
proc = Qwen3OmniMoeProcessor.from_pretrained(model_dir)   # optionally min_pixels=/max_pixels= here
# frames dims multiples of 32. Probe results:
proc(text=..., audio=[a4s], videos=[frames8], fps=2.0, use_audio_in_video=True, ...)  # OK -> 628 tokens
proc(text=..., videos=[frames8], fps=2.0, use_audio_in_video=False, ...)              # OK -> 574
proc(text=..., audio=[a4s], ...)                                                       # audio-only OK -> 70 (≈13 tok/s * 4 s + text)
proc(text=..., images=[img1080p], size={"shortest_edge":4*32*32,"longest_edge":1280*32*32}, ...)  # OK -> 1238 (cap applied per call)
```
- No system prompt auto-injected (template renders straight to user turn) — matches report 12.
- `fps` must be a SCALAR and match the actual sampling fps (else silent A/V misalignment).

## Other environment facts
- `scenedetect` 0.7.1 has no `[opencv]` extra — depend on `scenedetect` + `opencv-python` separately.
- `qwen-omni-utils` 0.0.9's video readers are unusable here, but its `smart_resize` math and constants remain the reference.
- venv has: transformers 5.16.1, accelerate 1.14, safetensors 0.8, av 18.1, librosa 1.0, soundfile 0.14, audioread, psutil 7.2.2, nvidia-ml-py, gguf 0.19, einops, opencv-python 5.0, scenedetect 0.7.1, tomli-w, imageio(-ffmpeg), sentencepiece, tiktoken, protobuf. Do NOT add torchcodec/decord.
- GPU 0 = RTX 5090 32 GB (sm120), GPU 1 = RTX 3090 (DO NOT USE). ffmpeg n8.1 on PATH.
- The GitHub remote `origin` (SECourses_Video_Captioner_Pro) is reachable; repo currently has no commits.
