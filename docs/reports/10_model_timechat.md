# TimeChat-Captioner — Deep Research Report

**Researched:** 2026-08-30
**Model:** https://huggingface.co/yaolily/TimeChat-Captioner-GRPO-7B (public, **not gated**)
**Code:** https://github.com/yaolinli/TimeChat-Captioner (BSD-3-Clause, 56 stars, last push 2026-06-29)
**Paper:** arXiv:2602.08711 — *"TimeChat-Captioner: Scripting Multi-Scene Videos with Time-Aware and Structural Audio-Visual Captions"*, ICML 2026 (LANCO @ Peking University + Kuaishou/Kling)
**Live demo:** https://huggingface.co/spaces/hugging-apps/timechat-captioner (ZeroGPU, running)

---

## ⚠️ HEADLINE CORRECTION — the base model is NOT Qwen2.5-VL

The task brief assumed **Qwen2.5-VL-7B-Instruct**. It is not.

**TimeChat-Captioner-GRPO-7B is a full fine-tune of `Qwen/Qwen2.5-Omni-7B`** — an *audio-visual omni* model.

```
config.json  →  "architectures": ["Qwen2_5OmniForConditionalGeneration"]
                "model_type": "qwen2_5_omni"
HF tags      →  base_model:Qwen/Qwen2.5-Omni-7B, base_model:finetune:Qwen/Qwen2.5-Omni-7B
```

This changes essentially every integration decision:

| | Qwen2.5-VL assumption | Reality (Qwen2.5-Omni) |
|---|---|---|
| Model class | `Qwen2_5_VLForConditionalGeneration` | `Qwen2_5OmniForConditionalGeneration` |
| Processor | `Qwen2_5_VLProcessor` | `Qwen2_5OmniProcessor` |
| Helper lib | `qwen_vl_utils` / `process_vision_info` | **`qwen_omni_utils` / `process_mm_info`** |
| Audio | none | **mandatory** — the video's audio track is a required input |
| Structure | single tower | **thinker** (VLM+audio) + talker (TTS, *weights absent here*) |
| Context | 128k | **32,768** (`max_position_embeddings`) |

There is **no `qwen_vl_utils` / `process_vision_info` code path anywhere** in this project. Anything built around Qwen2.5-VL needs to be re-plumbed.

---

## 1. Architecture, model files, configs, license

### 1.1 Repo-level facts (HF API)

| Field | Value |
|---|---|
| `id` | `yaolily/TimeChat-Captioner-GRPO-7B` |
| `sha` | `46ca41be720031a2fd0959b4902147a71c22fc10` |
| `createdAt` / `lastModified` | 2026-02-09 / 2026-02-11 |
| `gated` / `private` | `false` / `false` |
| `pipeline_tag` | `video-text-to-text` |
| `library_name` | `transformers` |
| `downloads` / `likes` | 616 / 5 |
| `safetensors.total` | **8,931,813,888 params, all BF16** |
| `usedStorage` | 17,875,483,023 bytes (~16.6 GiB) |
| **`license`** | **NOT DECLARED in the HF model card metadata** (see §1.7) |

### 1.2 Top-level `config.json`

```jsonc
{
  "architectures": ["Qwen2_5OmniForConditionalGeneration"],
  "model_type": "qwen2_5_omni",
  "dtype": "bfloat16",
  "enable_audio_output": false,   // ← talker NOT built at load time
  "enable_talker": true,          // ← legacy/unused flag; the code reads enable_audio_output
  "hidden_size": 3584,
  "eos_token_id": 151645,
  "pad_token_id": 151643,
  "transformers_version": "4.57.1"
}
```

### 1.3 `thinker_config` (the only part with weights)

`model_type: qwen2_5_omni_thinker`, `architectures: ["Qwen2OmniNaViTThinkerForConditionalGeneration"]`
`position_id_per_seconds: 25`, `seconds_per_chunk: 2`, `user_token_id: 872`
Token ids: `audio 151646`, `image 151655`, `video 151656`, `vision_bos 151652`, `vision_eos 151653`, `audio_bos 151647`, `audio_eos 151648`, `vision_pad 151654`.

**`thinker_config.text_config`** (`qwen2_5_omni_text`) — a Qwen2.5-7B decoder:

| Key | Value |
|---|---|
| `hidden_size` / `intermediate_size` | 3584 / 18944 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` / `num_key_value_heads` | 28 / 4 (GQA) |
| `vocab_size` | 152064 |
| `max_position_embeddings` | **32768** |
| `rope_theta` | 1,000,000.0 |
| `rope_scaling` | `{"mrope_section": [16,24,24], "rope_type": "default"}` |
| `layer_types` | 28 × `full_attention` (no sliding window) |
| `sliding_window` / `use_sliding_window` | `None` / `false` |
| `rms_norm_eps` | 1e-06 |
| `hidden_act` | `silu` |
| `tie_word_embeddings` | `false` |

**`thinker_config.vision_config`** (`qwen2_5_omni_vision_encoder`) — Qwen2.5-VL-style NaViT ViT:

| Key | Value |
|---|---|
| `depth` | 32 |
| `hidden_size` / `embed_dim` | 1280 |
| `intermediate_size` | 3420 |
| `num_heads` | 16 |
| `out_hidden_size` | 3584 |
| `patch_size` / `spatial_patch_size` | 14 |
| `temporal_patch_size` | 2 |
| `spatial_merge_size` | 2 |
| `window_size` | 112 |
| `fullatt_block_indexes` | `[7, 15, 23, 31]` |
| `tokens_per_second` | 25 |
| `in_channels` | 3 |

**`thinker_config.audio_config`** (`qwen2_5_omni_audio_encoder`) — Whisper-large-style encoder:
`d_model 1280`, `encoder_layers/num_hidden_layers 32`, `encoder_attention_heads 20`, `encoder_ffn_dim 5120`, `num_mel_bins 128`, `max_source_positions 1500`, `n_window 100`, `output_dim 3584`, `activation gelu`, `scale_embedding false`.

**`talker_config`** is present in `config.json` (`qwen2_5_omni_talker`, hidden 896, 24 layers, vocab 8448, mrope `[16,24,24]`) — **but there are zero talker weights in the checkpoint** (see §1.5). It exists only because it was copied from the base config.

### 1.4 Complete file list with sizes (`/tree/main`)

| File | Bytes | Notes |
|---|---:|---|
| `.gitattributes` | 1,570 | |
| `README.md` | 5,284 | model card |
| `added_tokens.json` | 579 | |
| **`args.json`** | 24,809 | full ms-swift GRPO training args (very informative) |
| `chat_template.jinja` | 1,281 | |
| `config.json` | 20,406 | |
| `generation_config.json` | 117 | |
| `merges.txt` | 1,671,853 | |
| `model-00001-of-00004.safetensors` | 4,985,055,536 | |
| `model-00002-of-00004.safetensors` | 4,991,496,832 | |
| `model-00003-of-00004.safetensors` | 4,991,496,936 | |
| `model-00004-of-00004.safetensors` | 2,895,740,064 | |
| `model.safetensors.index.json` | 119,061 | |
| `preprocessor_config.json` | 667 | |
| `special_tokens_map.json` | 833 | |
| **`spk_dict.pt`** | 259,544 | **byte-identical to Qwen2.5-Omni-7B's** (same LFS sha256 `6a05609b28f5…`) — must be downloaded, see §5.4 |
| `tokenizer.json` | 11,421,870 | |
| `tokenizer_config.json` | 5,160 | |
| `trainer_state.json` | 311,071 | |
| `training_args.bin` | 12,241 | |
| `video_preprocessor_config.json` | 1,322 | |
| `vocab.json` | 2,776,833 | |

**Total weights: 17,863,789,368 B ≈ 16.64 GiB.** Full repo ≈ 16.6 GiB.

### 1.5 Weight map — thinker only

`model.safetensors.index.json`: **1,346 tensors, 100 % prefixed `thinker.`**

```
thinker.visual      518 tensors   (ViT)
thinker.audio_tower 489 tensors   (Whisper encoder)
thinker.model       338 tensors   (Qwen2.5 LLM)
thinker.lm_head       1 tensor
talker.*              0   ← ABSENT
token2wav.*           0   ← ABSENT
metadata: {"total_parameters": 8931813888, "total_size": 17863627776}
```

**Consequence:** the model is text-out only. Speech synthesis is impossible. `model.disable_talker()` (or equivalent) is mandatory. Because `config.enable_audio_output == false`, transformers never *builds* the talker at load, so `disable_talker()` is actually a no-op safety call — but see the §5 landmine.

### 1.6 Tokenizer / processor / templates

**`tokenizer_config.json`** — `Qwen2Tokenizer` (fast), `model_max_length 32768`, `eos <|im_end|>`, `pad <|endoftext|>`, `bos None`, `unk None`, `clean_up_tokenization_spaces false`, `processor_class Qwen2_5OmniProcessor`, extra special tokens `<|IMAGE|> <|VIDEO|> <|AUDIO|> <|vision_bos|> <|vision_eos|> <|audio_bos|> <|audio_eos|>`.

**`chat_template.jinja`** (verbatim, newlines shown):

```jinja
{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
```

> **Note:** unlike stock Qwen2.5-Omni, this template's default system message is the plain `You are a helpful assistant.` — **not** the Omni "You are Qwen, a virtual human…" speech-mode system prompt. Training used `system: None` (confirmed in `args.json`), so the default is what the model saw. **Do not inject a custom system prompt.**

**`preprocessor_config.json`** (image + audio front-end):

```json
{ "processor_class": "Qwen2_5OmniProcessor",
  "image_processor_type": "Qwen2VLImageProcessor",
  "min_pixels": 3136, "max_pixels": 12845056,
  "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
  "image_mean": [0.48145466,0.4578275,0.40821073],
  "image_std":  [0.26862954,0.26130258,0.27577711],
  "feature_extractor_type": "WhisperFeatureExtractor",
  "sampling_rate": 16000, "feature_size": 128, "n_fft": 400, "hop_length": 160,
  "chunk_length": 300, "n_samples": 4800000, "nb_max_frames": 30000,
  "dither": 0.0, "padding_side": "right", "padding_value": 0.0,
  "return_attention_mask": true }
```

> These `min_pixels 3136` / `max_pixels 12845056` are the **stock Qwen defaults and are NOT what inference uses.** The real resolution cap (297,920) is imposed per-request by `qwen_omni_utils`, not by this file. See §2.4.

**`video_preprocessor_config.json`**: `Qwen2VLVideoProcessor`, **`do_sample_frames: false`** (frames are pre-sampled by `qwen_omni_utils`), `fps: null`, `max_frames: 768`, `min_frames: 4`, `min_pixels: 3136`, `max_pixels: 12845056`, `size: {longest_edge: 12845056, shortest_edge: 3136}`, `resample: 3` (bicubic), `do_convert_rgb/do_normalize/do_rescale/do_resize: true`, `rescale_factor: 1/255`.

**`generation_config.json`** — this is the whole file:

```json
{ "_from_model_config": true,
  "eos_token_id": [151645, 151643],
  "transformers_version": "4.57.1" }
```

**There are no sampling parameters.** `do_sample` therefore defaults to `False` → **inference is greedy / deterministic**. (The `temperature 1.0 / top_p 0.99 / top_k 50` in `args.json` are GRPO *rollout* params used during training, not inference settings.)

### 1.7 License — read carefully

- **HF model repo: no `license` field at all** in the card metadata (`cardData` = only `base_model`, `library_name`, `pipeline_tag`). Weights are therefore formally unlicensed on HF.
- **GitHub repo: BSD-3-Clause**, "Copyright (c) 2026, Language Computing and Machine Learning Group (LANCO) @ Peking University". Added in the single commit `9014445` after GitHub issue #2 asked about it.
- **Author's own statement (GitHub issue #2, by `yaolinli`):** code is BSD-3-Clause and permits *academic research, commercial use, modification and redistribution*; **model weights additionally require compliance with Qwen2.5-Omni's Apache 2.0 license**.
- Verified: `Qwen/Qwen2.5-Omni-7B` ships `license: other` + `license_name: apache-2.0` and its `LICENSE` file is a verbatim **Apache License 2.0**. So the upstream weight license is genuinely Apache-2.0 (permissive, commercial-OK).

**Practical read:** commercial use is intended and permitted, but the HF weights repo has no explicit license tag — if that matters legally, cite the GitHub LICENSE + the author's issue-#2 statement.

---

## 2. Exact inference recipe from the repo

### 2.1 Environment (verbatim from `readme.md` / HF card / dataset card — all three identical)

```bash
conda create -n timechatcap python=3.12
conda activate timechatcap
pip install torch torchvision
pip install transformers==4.57.1          # ← hard pin
pip install accelerate
pip install flash-attn --no-build-isolation
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-omni-utils[decord] -U    # → 0.0.9 (2026-02-10)
```

`qwen-omni-utils` 0.0.9 deps: `av, librosa, packaging, pillow, requests` + optional `decord`.
Video backend priority in `qwen_omni_utils`: **torchcodec → decord → torchvision** (override with `FORCE_QWENVL_VIDEO_READER=decord`). Audio is always read via **librosa + audioread/FFmpeg**, so **ffmpeg must be on PATH**.

Training used **ms-swift v3.11.0.dev0** (vendored at `ThirdPartyLib/ms-swift`), which pins `transformers>=4.33,<4.58` — irrelevant for inference.

### 2.2 Code path

| What | Where |
|---|---|
| Model + processor load | `Infer/inference.py` → `main()`, lines ~197-206 |
| **Single-video caption generation** | `Infer/inference.py` → **`process_single_video(video_path, model, processor)`**, lines 37-130 |
| Prompt constant | `Infer/inference.py` lines 26-36 |
| Multi-GPU launcher | `Infer/infer.sh` (data-parallel: N independent processes, `CUDA_VISIBLE_DEVICES=$GPU_ID`, disjoint `--start_index/--end_index` slices, then `cat rank_*.jsonl > merged_result.jsonl`) |
| Resume logic | `Infer/inference.py` → `load_existing_results()` (skips clips whose `prediction` is non-empty and != `"FAILED"`) |
| Eval (SODA_M, F1, mIoU) | `Eval/eval_sodam.py`, `Eval/eval_time.py` (both call **Gemini-2.5-Pro as the checklist judge** — needs GCP creds) |

Loading (verbatim):

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
model.disable_talker()
```

### 2.3 THE EXACT PROMPT

`Infer/inference.py` lines 26-36 — this is the only prompt in the entire repo:

```python
MAX_PIXELS = 297920
VIDEO_MAX_PIXELS = 297920

# The prompt used to instruct the model.
PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated."
)
```

Assembled as one string, **verbatim**:

> `Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.`

**System prompt: NONE.** `args.json` → `"system": null`. The chat template auto-inserts `You are a helpful assistant.` Do not override it.

**There is no explicit "produce six-dimension timestamped screenplay JSON" instruction anywhere.** The structured output is *entirely baked in by SFT+GRPO*. The prompt is a plain "describe the video" request; the schema emerges from the weights. This is important: **you cannot change the prompt much without losing the JSON format.**

#### The 7 training-time prompt paraphrases (extracted from the SFT set)

I pulled the head of `data/sft_v3_merged_all.jsonl` from `yaolily/Timechat-OmniCaptioner-42K` and found exactly **7 distinct paraphrases** (each prefixed with the `<video>` placeholder), used as prompt augmentation:

1. `Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.`  ← **the one the repo ships**
2. `Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.`  ← **the one used for GRPO** (`data/grpo_v5_keypoints.jsonl` uses only this one)
3. `Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.`
4. `Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.`
5. `Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized.`
6. `Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.`
7. `Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.`

Any of these is safe. Prompt #1 (the shipped one) is the recommended default.

**Modality ordering caveat:** training data placed `<video>` **before** the text. `Infer/inference.py` matches this (video content item first, then text). **The model card / HF card / dataset card snippet has them reversed (text first)** — a cosmetic inconsistency in the docs. Prefer `Infer/inference.py`'s ordering (video → text).

### 2.4 Video ingestion — every knob explained

The conversation item (verbatim from `Infer/inference.py`):

```python
{
    "type": "video",
    "video": video_path,
    "max_pixels": MAX_PIXELS,        # 297920
    "max_frames": 160,               # reduce to 80 for faster speed
    "fps": 2.0,
    "video_max_pixels": VIDEO_MAX_PIXELS,   # 297920
},
```

I read `qwen_omni_utils` 0.0.9 source (`v2_5/vision_process.py`) to establish exact semantics:

**Module constants:**
```python
MAX_RATIO = 200;  SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4;      IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128;    VIDEO_MAX_TOKEN_NUM = 768
FPS = 2.0;  FRAME_FACTOR = 2;  FPS_MIN_FRAMES = 4;  FPS_MAX_FRAMES = 768
MODEL_SEQ_LEN = int(os.environ.get('MODEL_SEQ_LEN', 128000))
```

**Keys `fetch_video()` actually reads:** `fps`, `nframes`, `min_frames`, `max_frames`, `min_pixels`, `max_pixels`, `total_pixels`, `resized_height`, `resized_width`, `sample_fps` (frame-list mode), `video_start`/`video_end` (audio side).

> **🔴 `video_max_pixels` IS A NO-OP.** It is *never* read by `qwen_omni_utils`. It's an **ms-swift environment-variable name** (`VIDEO_MAX_PIXELS`) that leaked into the inference snippet. It is silently ignored in the transformers path — in the repo, in the model card, and in the HF Space. Harmless, but do not expect it to do anything.

**Frame count** (`smart_nframes`):
```
nframes = total_frames / video_fps * fps          # = duration * 2.0
nframes = min(min(max(nframes, min_frames=4), max_frames=160), total_frames)
nframes = floor_to_multiple_of(nframes, 2)
```
→ at `fps=2.0, max_frames=160` the sampler saturates at **80 seconds of video**.

**Per-frame resolution** (`fetch_video`, with `image_factor = 14*2 = 28`):
```
VIDEO_FRAME_MIN_PIXELS = 128 * 28 * 28 = 100,352
VIDEO_FRAME_MAX_PIXELS = 768 * 28 * 28 = 602,112
total_pixels  (default) = 128000 * 28 * 28 * 0.9 = 90,316,800
cap = max(min(602112, total_pixels/nframes*2), min_pixels*1.05)
max_pixels = min(user_max_pixels, cap)     # 297,920 wins for any nframes ≤ 606
resized_h, resized_w = smart_resize(h, w, factor=28, min_pixels=100352, max_pixels=297920)
```
**`MAX_PIXELS = 297920` = 380 × 28 × 28 → ≤ 380 visual tokens per sampled frame.**

**Actual resize results (computed):**

| Source | Resized | px | tokens/frame |
|---|---|---:|---:|
| 1080p / 720p 16:9 | 392 × 700 | 274,400 | **350** |
| vertical 9:16 | 700 × 392 | 274,400 | **350** |
| 4:3 | 448 × 616 | 275,968 | **352** |

**Token budget** (video tokens = `nframes/2 × tokens_per_frame`; audio ≈ 25 tok/s):

| Clip | nframes | video tok | audio tok | total | vs 32,768 ctx |
|---|---:|---:|---:|---:|---|
| 30 s @ fps 2 | 60 | 10,500 | 750 | ~11,250 | comfortable |
| **60 s @ fps 2** | 120 | 21,000 | 1,500 | **~22,500** | **the sweet spot** |
| 80 s @ fps 2 | 160 (capped) | 28,000 | 2,000 | ~30,000 | + 8k output ⇒ **overflow** |

This is exactly why the docs say ~60 s. **The real ceiling is the 32,768-token context**, shared between input *and* the up-to-8192-token output.

**Training-time env vars** (`Train/script/sft_single_node.sh`) — the ground truth for what the model saw:

```bash
export MAX_PIXELS=297920
export VIDEO_MAX_PIXELS=297920
export VIDEO_TOTAL_PIXELS=20070400     # = 25,600 tokens total video budget
export FPS=2
export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=True
```

**Stay at `fps=2.0` and `max_pixels=297920`.** Both were fixed across SFT and GRPO; deviating is off-distribution.

### 2.5 Audio path

`process_mm_info(conversation, use_audio_in_video=True)` → `process_audio_info()` runs:

```python
assert _check_if_video_has_audio(path), "Video must has audio track when use_audio_in_video=True"
```

then `librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)`. **A video with no audio stream raises an `AssertionError` and the whole call fails.** (The ms-swift training template has the same hard assert: `assert False,"use_audio_in_video为True,但是没有音频"`.) You must pre-check with `av`/`ffprobe` and mux a silent track if absent — see §6.3.

Audio is chunk-interleaved with video at `seconds_per_chunk = 2.0`, `position_id_per_seconds = 25` (processor defaults), producing `<|vision_bos|><|audio_bos|>…<|audio_eos|><|vision_eos|>` blocks.

### 2.6 Generation call

`Infer/inference.py`:
```python
text_ids = model.generate(
    **inputs,
    use_audio_in_video=True,
    return_audio=False,
    thinker_max_new_tokens=8192,
    talker_max_tokens=8192,     # ← typo: real param is talker_max_new_tokens; inert
    use_cache=True,
)
```
Model card variant uses `thinker_max_new_tokens=9216, talker_max_tokens=9216` (9216 = the GRPO `max_completion_length`).

| Param | Value | Source |
|---|---|---|
| `thinker_max_new_tokens` | **8192** (`inference.py`) / **9216** (model card) | either is fine; 9216 matches training |
| `temperature` / `top_p` / `top_k` | **unset ⇒ greedy** (`do_sample=False`) | `generation_config.json` has no sampling keys |
| `use_cache` | `True` | |
| `return_audio` | `False` | ⚠ deprecated in transformers 5 — see §5 |
| `repetition_penalty` | unset (1.0) | |

Decode: `processor.decode(text_ids[0][inputs.input_ids[0].size(0):], skip_special_tokens=True)`.

### 2.7 Output schema & post-processing

Post-processing is a **bare `json.loads`** — the model emits raw JSON with **no markdown fence, no preamble**:

```python
parsed = json.loads(response)
if isinstance(parsed, list):
    item["prediction_json"] = parsed
```

**Schema — a JSON array of segment objects, 8 string keys each:**

```jsonc
[
  {
    "timestamp":               "00:00-00:10",   // "MM:SS-MM:SS", relative to clip start, 0-based
    "segment_detail_caption":  "...",           // detailed events / actions
    "camera_state":            "...",           // shot sizes, angles, pans/tilts, cuts
    "video_background":        "...",           // setting, location, atmosphere
    "storyline":               "...",           // narrative function of the segment
    "shooting_style":          "...",           // post-production / editing technique
    "speech_content":          "...",           // transcript w/ speaker labels (+ translation if non-EN)
    "acoustics_content":       "..."            // "1) Tone of speech: ... 2) Background sounds or music: ..."
  },
  ...
]
```

Key order in real outputs is exactly: `timestamp, segment_detail_caption, camera_state, video_background, storyline, shooting_style, speech_content, acoustics_content`.

**Confirmed by `Train/modified/plugin.py::check_dict_format()`** — the GRPO format reward requires *all eight* keys, each a `str`:
```python
required_keys = ["timestamp", "segment_detail_caption", "camera_state", "video_background",
                 "storyline", "shooting_style", "speech_content", "acoustics_content"]
```

**⚠️ "Six dimensions" vs. reality:** the paper/project page name **six** dimensions (Detailed Events, Visual Background, Camera State, Shot Editing Style, Dialogue Content, Acoustics Content). The model emits **seven** caption fields — the extra one is **`storyline`**, which the benchmark GT (`human_json_anno`) does *not* contain and the checklist scorer does *not* evaluate. So: 6 scored dimensions + `storyline` (free bonus) + `timestamp`.

**Timestamp parsing** (`Eval/eval_time.py::unify_timestamp_format`) accepts `"MM:SS-MM:SS"`, `"HH:MM:SS-HH:MM:SS"`, `"SS-SS"`, mixed, or `[start, end]` floats. The model always produces `MM:SS-MM:SS` in practice.

**Empirical output stats** (I parsed all 1,122 lines of `Eval/example_pred.jsonl`, which are real TimeChat-Captioner outputs on OmniDCBench):

| Metric | Value |
|---|---|
| Clips | 1,122 |
| Valid JSON | **1,113 / 1,122 = 99.2 %** (9 failures, incl. degenerate 6-char outputs) |
| Segments per clip | min 2, **mean 6.98**, max 31 |
| Output length | min 6, **mean 8,654 chars**, max 36,749 chars (≈9k tokens — hits the cap) |
| Clip duration | min 13 s, **mean 58.1 s**, max 70 s |

**Budget ~7 segments and ~2,200 output tokens for a typical 60-second clip.** Always wrap `json.loads` in try/except and keep the raw string.

---

## 3. Other tasks the model supports

**In the repo: none.** There is exactly one prompt (`PROMPT` in `inference.py`) and one task. `grep` over `Infer/`, `Eval/`, `Train/` finds no grounding, QA, or alternate-task prompts.

The paper reports two **downstream evaluations**, but both are *derived from the same dense-caption output*, not separate prompted tasks:

| Task | How it's done | Reported |
|---|---|---|
| **Audio-visual reasoning / Omni VideoQA** | The model captions the video; a **frozen judge LLM (Gemini-2.5-Pro) answers benchmark questions using ONLY the caption as evidence** — no video. Measures caption informativeness, not the model's QA skill. | DailyOmni **52.8**, WorldSense **22.6** |
| **Temporal grounding** | The `timestamp` fields of the dense caption are matched against the query span. | Charades-STA **R1@0.7 = 48.3** |

The author confirms this protocol in GitHub issue #1 (explaining why their Qwen3-Omni number differs from Qwen's official 75.8 on DailyOmni): *"方式二（ours）：衡量video captioning模型的表现。用Captioner模型的caption输出，喂给一个judge model (Gemini-2.5-Pro)，让这个judge model只看caption不看原视频回答omni question"*.

**Bottom line: it's a single-task specialist.** Do not expect free-form VQA, chat, or instruction following — it was full-finetuned (not LoRA) on captioning only and will likely emit the JSON schema regardless of what you ask.

Base-model capabilities (image input, speech recognition, chat) are architecturally present but were **not preserved by design** — `freeze_vit: true` but `freeze_llm: false, freeze_aligner: false, train_type: full` means the LLM was fully overwritten.

---

## 4. Known limitations

| Limitation | Detail |
|---|---|
| **Max video length** | **~60 s recommended** (official note: *"limit video input to around 1 minute. Please segment longer videos into around 60-second clips"*). Hard ceiling ~80 s at `fps=2, max_frames=160`. Longer videos silently break timestamps (see §5.5). |
| **Max frames** | 160 (`max_frames`); readme suggests 80 for speed. Library ceiling `FPS_MAX_FRAMES=768`, video-processor `max_frames: 768` — unusable here because of the 32k context. |
| **Context** | **32,768 tokens** total (input + output). This, not frame count, is the binding constraint. |
| **Audio is MANDATORY** | `assert _check_if_video_has_audio(path)` in `qwen_omni_utils`. Silent / audio-less videos **crash**. Not optional — the model was trained with `use_audio_in_video=True` on every sample. |
| **No speech output** | Talker + token2wav weights absent (§1.5). Text only. |
| **VRAM (bf16)** | Weights **17.86 GB**. Repo recommendation: *"We recommend using a GPU with at least 60GB of memory for inference on 60-second video clips."* The overhead is the ViT forward over 120 frames + a ~22.5k-token KV cache + FA2 workspace. On 24 GB it will OOM at 60 s/160 frames; the HF Space works only because ZeroGPU allocates a large slice. **Mitigations:** `max_frames=80`, `fps=1.0`, or `max_pixels` ≈ 150k, each roughly halving activation cost. |
| **Batch inference** | **Not supported by the repo** — `process_single_video()` is strictly batch-size 1; multi-GPU is process-level data parallelism. transformers explicitly raises `NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")` — that guard only fires for audio output, so *text-only* batching is theoretically possible, but it is untested here and the left-padding + interleaved audio/video chunking makes it fragile. **Recommendation: batch = 1.** |
| **Image input** | Architecturally supported (`<|IMAGE|>` token, `Qwen2VLImageProcessor`, `image_token_index 151655`). **Never trained or evaluated on images.** Expect it to emit a timestamped JSON array for a still image. Not recommended. |
| **Greedy only** | No sampling params in `generation_config.json`. Deterministic; no diversity knob without explicitly passing `do_sample=True` (off-distribution). |
| **~0.8 % malformed JSON** | 9/1122 outputs unparseable, including near-empty degenerate outputs. Needs retry / fallback. |
| **Non-English speech** | `speech_content` transcribes source-language dialogue and adds an English translation inline (e.g. `Lu Yan: 境州之行如何? (How was your trip to Jing?)`). Benchmark is Chinese movies + YouTube. |
| **Quantized / GGUF** | **None exist.** HF search for `TimeChat-Captioner` returns exactly 1 repo; `?filter=gguf` returns `[]`; `search=timechat gguf` returns `[]`. No AWQ/GPTQ/INT8/MLX either. Also no **SFT-only** checkpoint was published (`args.json` references an internal SFT path; only the GRPO model is public). llama.cpp does not support `qwen2_5_omni`, so a GGUF is unlikely to appear. |
| **Other sizes** | None. Only 7B. Author `yaolily`'s other public models are unrelated (GenS, LLaVA ablations). |
| **Eval reproduction** | `Eval/eval_sodam.py` / `eval_time.py` need a **Gemini-2.5-Pro** GCP service-account JSON (hardcoded path in the file) — you cannot reproduce the metrics without Google Cloud credentials. |
| **`ffmpeg` required** | audioread/librosa shell out to FFmpeg for the audio track. |

---

## 5. Compatibility with the CURRENT latest `transformers`

**Latest release as of today (2026-08-30): `transformers` 5.16.1**, published 2026-08-26. (5.16.0 on 2026-08-26; 5.15.1 on 2026-08-19; 5.15.0 on 2026-08-10.) The repo pins **4.57.1** — roughly 20 minor releases and a full major version behind.

I verified everything below against **actual source**: the `transformers-5.16.1-py3-none-any.whl` from PyPI, plus the raw `modeling_qwen2_5_omni.py` / `processing_qwen2_5_omni.py` at git tags `v4.57.1`, `v5.0.0`, `v5.13.0`, `v5.14.0`, `v5.15.0`, `v5.15.1`, `v5.16.0`, `v5.16.1`. A second reviewer then reproduced all of it **empirically** in two isolated venvs (4.57.1 and 5.16.1), instantiating the real `TimeChat-Captioner-GRPO-7B` config on `torch.device("meta")` and running the real processor — no weights downloaded. Both analyses agree.

**Preprocessing is byte-for-byte identical between 4.57.1 and 5.16.1.** Hashed on both:

| tensor | SHA (both versions) |
|---|---|
| chat-template string | `4edbd5c3e09d29f4` |
| `input_ids` (1, 1253) | `d0d93f39d4edb808` |
| `input_features` | `2432552f2e1dbc83` |
| `pixel_values_videos` | `f5154db06c2ee3b6` |
| `video_grid_thw` / `video_second_per_grid` | `[[8,20,28]]` / `[1.0]` |

So the *only* incompatibility is in `generate()`.

### 5.1 What still works ✅

| Item | Status in 5.16.1 |
|---|---|
| `transformers/models/qwen2_5_omni/` | present (config, modeling, modular, processing) |
| `Qwen2_5OmniForConditionalGeneration` | in `modeling_qwen2_5_omni.__all__` |
| `Qwen2_5OmniProcessor` | `__all__ = ["Qwen2_5OmniProcessor"]` |
| Top-level import `from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor` | works — v5 `__init__.py` uses `define_import_structure(Path(__file__).parent / "models", prefix="models")`, auto-exporting every model's `__all__` |
| `AutoModelForCausalLM` mapping | `("qwen2_5_omni", "Qwen2_5OmniForConditionalGeneration")` registered |
| `model.disable_talker()` | still defined |
| **Processor signature** | **unchanged**: `__call__(self, text, images=None, videos=None, audio=None, **kwargs)` — `audio=` (singular) is still correct; `videos=` still takes decoded tensors; `use_audio_in_video` still a valid `videos_kwargs` |
| `attn_implementation="flash_attention_2"` | still valid. Full 5.16.1 `ALL_ATTENTION_FUNCTIONS`: `flash_attention_2, flash_attention_3, flash_attention_4, flex_attention, sdpa, paged|eager, paged|flash_attention_2/3/4, paged|sdpa`. `"kernels-community/flash-attn"` is *also* accepted but needs `kernels>=0.16,<0.17` — no rename required |
| `torch_dtype=` | still accepted (`modeling_utils.py:4131` — `torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC`), emits a `torch_dtype is deprecated! Use dtype instead!` log line. ⚠ v5's new default is `dtype="auto"`, not float32 |
| `padding=True`, `.to(model.device).to(model.dtype)` on the `BatchFeature` | unchanged (casts floats to bf16, leaves `input_ids` int64) |
| `apply_chat_template(..., tokenize=False)` | still returns a plain `str`. (The v5 note *"`apply_chat_template` returns `BatchEncoding`"* concerns the tokenizing path only.) v5 dropped `Qwen2_5OmniProcessor`'s custom override, which only emitted a system-prompt warning — output text is bit-identical |
| `processor(audio=...)` | **still singular `audio=`**, never renamed to `audios=` |
| No `video_metadata` requirement | `qwen_omni_utils` pre-decodes frames and sets `do_sample_frames=False`; the processor never needs metadata |
| KV cache | no break. PR [#47872](https://github.com/huggingface/transformers/pull/47872) *"Fix Qwen2.5-Omni / Qwen3-Omni-MoE generation with a compilable cache"* (merged 2026-08-19) shipped in 5.16.0 |
| `transformers/models/deprecated/` | contains **only** `__init__.py` — Qwen2.5-Omni was never deprecated or moved |

**NEW in 5.16.x — a video-token-count warning you must decide about.** Every video call now logs:

> `Qwen2VL video processing does not apply the per-frame pixel cap the reference implementation (qwen-vl-utils) applies, so some videos cost far more tokens than they would there. In v5.22 the capped behavior will become the default and cap_pixels_per_frame will be removed. Pass cap_pixels_per_frame=True to adopt the reference behavior now, or False to keep the current behavior and silence this warning.`

Pass **`cap_pixels_per_frame=False`** to keep today's token counts and silence it. **Plan for v5.22:** the default flips and video token counts change. For *this* model it should be a no-op either way — `qwen_omni_utils` has already capped every frame at 297,920 px before the processor sees it — but pin the behaviour explicitly rather than inherit a future default. On 4.57.1 the kwarg is harmlessly ignored (`Keyword argument 'cap_pixels_per_frame' is not a valid argument for this processor and will be ignored.`), so it is safe in cross-version code.

### 5.2 🔴 HARD BLOCKER — transformers ≥ 5.15.0 crashes this model

`Qwen2_5OmniForConditionalGeneration.generate()` builds `talker_kwargs` **before** the `generate_audio` guard. In 5.15.0 a line was added:

```python
talker_kwargs = {
    "max_new_tokens": talker_max_new_tokens,
    ...
    "pad_token_id": self.talker.codec_pad_token,   # ← ADDED in 5.15.0, UNCONDITIONAL
    ...
}
...
generate_audio = return_audio and self.has_talker   # ← guard comes AFTER
```

This checkpoint has `config.enable_audio_output == false`, so `self.talker` is **never created**. Result:

```
AttributeError: 'Qwen2_5OmniForConditionalGeneration' object has no attribute 'talker'
```

…even with `generation_mode="text"`. Verified line positions:

| Version | `self.talker.codec_pad_token` before the guard? | Verdict |
|---|---|---|
| 4.57.1 | no (only at L3942, after guard@3878) | ✅ works |
| 5.0.0 | no | ✅ |
| 5.13.0 | no (L3965 vs guard@3901) | ✅ |
| **5.14.0** | no (guard@3901) | ✅ **last safe version** |
| **5.15.0** | **yes — L3774 vs guard@3807** | ❌ |
| 5.15.1 / 5.16.0 / 5.16.1 | **yes — L3867 vs guard@3900** | ❌ |

**Root cause (cited):** PR **[huggingface/transformers#47186](https://github.com/huggingface/transformers/pull/47186)** - *"Add support for batched Qwen2.5/3-Omni audio generation"*, merged **2026-08-03**, first shipped in **5.15.0**. Its diff adds exactly one line to `modeling_qwen2_5_omni.py` and `modular_qwen2_5_omni.py`:

```diff
              "eos_token_id": talker_eos_token_id,
+            "pad_token_id": self.talker.codec_pad_token,
```

**Qwen3-Omni-MoE is unaffected** - the same PR touched it but did not add this line (verified absent in 5.16.1's `modeling_qwen3_omni_moe.py`).

**Nobody has reported this upstream.** GitHub issue search on `repo:huggingface/transformers` for `"disable_talker"` returns 0 results; `"has no attribute 'talker'"` returns 0 results. The line is still on `main` today. Meanwhile the official docs at `docs/source/en/model_doc/qwen2_5_omni.md:366` still instruct you to write `text_ids = model.generate(**inputs, return_audio=False)` - which is now broken for any talker-less checkpoint.

**Empirically reproduced** (real TimeChat config, meta device, isolated venvs):

```
##### 5.16.1 #####
AttributeError [generate(..., return_audio=False)]        'Qwen2_5OmniForConditionalGeneration' object has no attribute 'talker'
AttributeError [generate(..., generation_mode="text")]    'Qwen2_5OmniForConditionalGeneration' object has no attribute 'talker'

##### 4.57.1 #####
RuntimeError   [generate(..., return_audio=False)]        Tensor on device meta is not on the expected device cpu!   <-- reached real generation
```

The same break occurs on the **stock** `Qwen/Qwen2.5-Omni-7B` config (`enable_audio_output: true`) the moment you call `disable_talker()` - so this affects every text-only Omni deployment, not just this fine-tune. And "just enable the talker instead" is not an option here: the checkpoint has zero talker/token2wav tensors, so it would build ~2 GB of randomly-initialised modules.

**Workaround (recommended, version-proof):** skip the omni wrapper's `generate` and call the thinker directly. `Qwen2_5OmniThinkerForConditionalGeneration.forward()` accepts `use_audio_in_video` and `video_second_per_grid` in its signature, so `GenerationMixin.generate` forwards them cleanly:

```python
out = model.thinker.generate(**inputs, use_audio_in_video=True,
                             max_new_tokens=8192, use_cache=True, do_sample=False)
```

This bypasses the talker entirely on every version, 4.57.1 → 5.16.1.

### 5.3 Other v5 API changes affecting this code

| 4.57.1 | 5.x | Impact |
|---|---|---|
| `return_audio=False` | `@deprecate_kwarg("return_audio", version="v5", new_name="generation_mode")` - use **`generation_mode="text"`** | `return_audio=False` **is still honoured**: `deprecate_kwarg`'s `raise_if_greater_or_equal_version` defaults to `False`, so at 5.16.1 the action degrades to `Action.NONE` - it silently renames the kwarg with **no warning**. WARNING: **omitting it now defaults to `return_audio=True`** (changed in **5.0.0**: `return_audio = generation_mode != "text" and generation_mode is not False`, replacing v4's `if return_audio is None: return_audio = self.has_talker`), which on a talker-less model raises `ValueError: Cannot use talker when talker module not initialized.` **Always pass it explicitly.** |
| `talker_max_tokens=8192` (repo) | real param is `talker_max_new_tokens` | The repo's `talker_max_tokens` was **always** a typo — it falls into `**kwargs`, is prefix-stripped to `talker_kwargs["max_tokens"]`, and is inert when talker is unused. Just drop it. |
| `torch_dtype=` | `dtype=` | warning only |
| — | `spk_dict.pt` still fetched unconditionally in `from_pretrained` (`if spk_path is None: raise ValueError`) and `speaker_params = self.speaker_map[speaker]` is unconditional in **all** versions | fine here: the repo ships a byte-identical `spk_dict.pt`, so `"Chelsie"` resolves. **Do not delete `spk_dict.pt` from a local copy.** |

**Evidence the HF Space already hit this:** `hugging-apps/timechat-captioner/app.py` (built 2026-07-07, `requirements.txt` = unpinned `transformers`) already uses the v5 idioms — `generation_mode="text"` and `talker_max_new_tokens` instead of `return_audio=False`/`talker_max_tokens`, and `attn_implementation="sdpa"` instead of FA2. It was built against ~5.13, i.e. before the 5.15.0 regression.

### 5.4 🟠 Silent-corruption bug: `fps` is not passed to the processor

Both `Qwen2_5OmniProcessor.__call__` versions (identical in 4.57.1 and 5.16.1) do:

```python
fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
videos_inputs["video_second_per_grid"] = second_per_grid_ts
```

`video_second_per_grid` drives M-RoPE temporal positions — i.e. **it is literally what the timestamps are computed from.**

`Infer/inference.py` never passes `fps` to `processor(...)`, so it silently relies on the `2.0` default.

- At `fps=2.0` and clips ≤ 80 s → correct by luck.
- **For clips > 80 s**, `smart_nframes` clamps to `max_frames=160`, so the *actual* sample rate is `sample_fps = nframes/duration` (e.g. a 100 s clip → 1.6 fps). The processor still assumes 2.0 → **every timestamp is compressed by ~20 %.** No error, no warning.
- The HF Space exposes an **fps slider (0.5–4.0)** and likewise never forwards it → **any setting other than 2.0 produces wrong timestamps.**

**Fix:** use `return_video_kwargs=True` and forward the *actual* sampled fps, **as a scalar** (`process_mm_info` returns it as a one-element list, which fails on **both** versions - 4.57.1: `TypeError: unsupported operand type(s) for /: 'int' and 'list'`; 5.16.1: `StrictDataclassFieldValidationError: Field 'fps' expected int, got list`, because v5's `Qwen2_5_OmniVideosKwargs` dropped its list-typed `fps` override and now inherits the scalar-validated `VideosKwargs.fps`):

```python
audios, images, videos, video_kwargs = process_mm_info(conv, use_audio_in_video=True,
                                                       return_video_kwargs=True)
# video_kwargs == {"do_sample_frames": False, "fps": [2.0]}
inputs = processor(..., fps=float(video_kwargs["fps"][0]), do_sample_frames=False)
```

### 5.5 Recommendation

| Goal | Do this |
|---|---|
| **Maximum fidelity to the paper** | `pip install transformers==4.57.1` (the official pin). Zero surprises. Anything in **4.51.3 - 4.57.6** works as written. |
| **Newer stack, minimal risk** | `transformers>=5.0,<5.15` (**5.14.1**, 2026-07-16, is the last good release). Works - **but you must pass `generation_mode="text"` / `return_audio=False` explicitly**, since the default flipped in 5.0.0. |
| **transformers 5.15 / 5.16 (current latest)** | **Must** call `model.thinker.generate(...)`, not `model.generate(...)`. Everything else works. |

`qwen-omni-utils` 0.0.9 is independent of transformers (it only needs `av/librosa/pillow/torch/torchvision`) and works with all of the above.

---

## 6. Copy-paste-ready inference

### 6.1 Install

```bash
python -m pip install "transformers==4.57.1" accelerate qwen-omni-utils[decord] av librosa
# optional, big speedup: pip install flash-attn --no-build-isolation
# ffmpeg MUST be on PATH (librosa/audioread shells out to it for the audio track)
```

> On Windows, `decord` wheels are flaky — omit `[decord]` and let it fall back to `torchvision`, or install `torchcodec` (preferred by `qwen_omni_utils`). Force a backend with `FORCE_QWENVL_VIDEO_READER=torchvision`.

### 6.2 Minimal standalone script (version-proof: 4.57.1 → 5.16.1)

```python
"""TimeChat-Captioner-GRPO-7B — single-video structured caption.
Reproduces Infer/inference.py::process_single_video, with the fps and
transformers-5.15+ talker fixes applied.
"""
import json, os, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL_ID   = "yaolily/TimeChat-Captioner-GRPO-7B"
VIDEO_PATH = "example_video.mp4"        # <= ~60 s, MUST HAVE AN AUDIO TRACK

# ---- knobs fixed at training time; do not change without reason ----
MAX_PIXELS      = 297920   # 380 visual tokens/frame
MAX_FRAMES      = 160      # 80 => ~2x faster, ~half the VRAM
FPS             = 2.0
MAX_NEW_TOKENS  = 9216     # == GRPO max_completion_length

PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated."
)

# 1) load (thinker weights only: 17.86 GB bf16)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,                     # use torch_dtype= on transformers 4.x
    device_map="cuda",
    attn_implementation="flash_attention_2",  # "sdpa" if flash-attn isn't installed
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
model.disable_talker()
model.eval()

# 2) conversation — VIDEO FIRST, then text (matches the training data's "<video>..." order)
conversation = [{
    "role": "user",
    "content": [
        {"type": "video", "video": VIDEO_PATH,
         "max_pixels": MAX_PIXELS, "max_frames": MAX_FRAMES, "fps": FPS},
        {"type": "text", "text": PROMPT},
    ],
}]
# NOTE: no system message. The chat template injects "You are a helpful assistant."
#       which is exactly what training used (args.json -> "system": null).

# 3) preprocess. return_video_kwargs=True gives us the ACTUAL sampled fps.
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos, video_kwargs = process_mm_info(
    conversation, use_audio_in_video=True, return_video_kwargs=True
)
sample_fps = float(video_kwargs["fps"][0])   # scalar! a list would TypeError

inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True,
    use_audio_in_video=True,
    fps=sample_fps,             # <-- drives video_second_per_grid == the timestamps
    do_sample_frames=False,     # frames are already sampled by qwen_omni_utils
    cap_pixels_per_frame=False, # transformers 5.16+: pins current token counts & silences the
                                # v5.22-default-flip warning. Harmlessly ignored on 4.57.1.
)
inputs = inputs.to(model.device).to(model.dtype)

# 4) generate. Calling the THINKER directly avoids the transformers>=5.15.0
#    AttributeError: ... has no attribute 'talker' (talker_kwargs is built
#    unconditionally in Qwen2_5OmniForConditionalGeneration.generate).
with torch.inference_mode():
    out_ids = model.thinker.generate(
        **inputs,
        use_audio_in_video=True,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,        # greedy — generation_config.json has no sampling params
        use_cache=True,
    )

response = processor.decode(out_ids[0][inputs.input_ids[0].size(0):], skip_special_tokens=True)

# 5) parse: raw JSON array, no markdown fence (~99.2% parse rate in the authors' own run)
try:
    segments = json.loads(response)
except json.JSONDecodeError:
    segments = None

print(response)
if segments:
    for s in segments:
        print(s["timestamp"], "|", s["segment_detail_caption"][:100])
```

<details>
<summary>Drop-in replacement for step 4 if you prefer the official wrapper (transformers ≤ 5.14 only)</summary>

```python
with torch.inference_mode():
    out_ids = model.generate(
        **inputs,
        use_audio_in_video=True,
        generation_mode="text",         # transformers 5.x; use return_audio=False on 4.57.1
        thinker_max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
    )
```
</details>

### 6.3 Required guard: reject / repair audio-less videos

```python
import av, subprocess

def has_audio(path: str) -> bool:
    with av.open(path) as c:
        return any(s.type == "audio" for s in c.streams)

def ensure_audio(src: str, dst: str) -> str:
    """qwen_omni_utils asserts a video has an audio track. Mux a silent one if not."""
    if has_audio(src):
        return src
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-shortest", "-c:v", "copy", "-c:a", "aac", dst,
    ], check=True, capture_output=True)
    return dst
```

### 6.4 Files to vendor from the repo

**None are required.** Inference needs only `transformers` + `qwen_omni_utils`; there is no custom modeling/processing code in the HF repo (no `*.py`, no `trust_remote_code`).

Useful-to-copy-but-optional:

| File | Why |
|---|---|
| `Infer/inference.py` | reference implementation + resume logic; ~90 lines of it is the caption call |
| `Infer/infer.sh` | multi-GPU data-parallel launcher pattern |
| `Eval/eval_time.py::unify_timestamp_format`, `::_normalize_pred_item_list`, `::merge_fields` | robust timestamp/field normalization for downstream consumers |
| `Train/modified/plugin.py::check_dict_format`, `::check_timestamp_list_format` | exact output-validity predicate the model was rewarded on — ideal as an output validator |
| `Train/modified/qwen.py` | only needed if you re-train with ms-swift |

**Do not vendor `ThirdPartyLib/ms-swift`** (that's ~36 MB of training framework; `Train/modified/FileMap` shows only 3 files were patched).

### 6.5 Pre-flight checklist for a 60-second clip

1. Segment source video into **≤ 60 s** clips (the authors' benchmark averages 58.1 s).
2. Verify an **audio track** exists (§6.3) — hard crash otherwise.
3. `fps=2.0`, `max_frames=160`, `max_pixels=297920`. Drop to `max_frames=80` / `fps=1.0` if VRAM-bound.
4. Pass the real `fps` to `processor(...)`, or timestamps drift on any clip > 80 s.
5. Reserve ≥ 9,216 output tokens; total context 32,768.
6. `json.loads` with a try/except + retry; expect ~7 segments and ~8.6k chars.
7. VRAM: 17.9 GB weights + activations; the authors recommend ≥ 60 GB for the full 60 s / 160-frame configuration.
8. Batch size 1. Parallelise across GPUs with separate processes.

---

## Appendix A — GRPO training configuration (from `args.json`)

| Key | Value |
|---|---|
| `rlhf_type` | `grpo` |
| `train_type` | `full` (not LoRA) |
| Base for GRPO | internal SFT checkpoint `sft_v6_40k/.../checkpoint-1250` (never published) |
| `reward_funcs` | `external_timestamp_format_reward`, `external_timestamp_length_reward`, `external_dense_caption_f1_reward`, `external_dense_caption_sodam_reward` |
| `reward_weights` | `0.5 0.5 1.0 1.0` (from `Train/script/grpo_multinode.sh`) |
| `num_generations` | 8 |
| `temperature` / `top_p` / `top_k` | 1.0 / 0.99 / 50 (rollout only) |
| `beta` (KL) | 0.04 |
| `max_completion_length` / `max_length` | 9216 / 32768 |
| `learning_rate` | 1e-5, 1 epoch, cosine, warmup 0.05 |
| `freeze_vit` / `freeze_aligner` / `freeze_llm` | `true` / `false` / `false` |
| `torch_dtype` / `attn_impl` | bfloat16 / flash_attn |
| `deepspeed` | ZeRO-2 |
| `system` | **`null`** |
| `template` | `qwen2_5_omni` |

SFT (from `Train/script/sft_single_node.sh`): lr 5e-5, 2 epochs, `max_length 32768`, `truncation_strategy delete`, `freeze_vit true`, ZeRO-2, 8×GPU, per-device batch 1 × grad-accum 2.

Length reward detail (`TiemstampCaptionLength`): rewards each segment serialising to **< 750 tokens**, and penalises total completions **> 4096 tokens** by `max(0, 1 - (len-4096)/4096)`. That explains the observed ~8.6k-char / ~2.2k-token typical outputs.

## Appendix B — Related artifacts

| Artifact | URL | Notes |
|---|---|---|
| Train set (TimeChatCap-42K) | `yaolily/Timechat-OmniCaptioner-42K` | `data/sft_v3_merged_all.jsonl` (843 MB), `data/grpo_v5_keypoints.jsonl` (50.8 MB); videos as pre-extracted JPG frame lists + separate `.wav` |
| Benchmark (OmniDCBench) | `yaolily/OmniDCBench` | `ours_gt_file.json` (7.2 MB), `ours_gt_file_keypoints.json` (16.6 MB), Movie/Youtube tarballs (~12.6 GB) + `videos_3min` (~39 GB) |
| Demo Space | `hugging-apps/timechat-captioner` | Gradio 6.19, ZeroGPU, `attn_implementation="sdpa"`, unpinned transformers, exposes MCP server |
| Project page | https://timechat-captioner.github.io/ | six-dimension schema descriptions |

## Appendix C — Evidence trail (local paths)

- Repo clone: `C:\Users\Furkan\AppData\Local\Temp\claude\G--SECourses-Video-Captioner-Pro-v1\f5e44159-609b-47ea-bc14-04e9ece9d6fd\scratchpad\research\timechat\repo\`
  - `Infer\inference.py`, `Infer\infer.sh`, `Infer\readme.md`, `readme.md`, `LICENSE`
  - `Train\script\sft_single_node.sh`, `Train\script\grpo_multinode.sh`, `Train\modified\plugin.py`, `Train\modified\qwen.py`, `Train\modified\FileMap`
  - `Eval\eval_time.py`, `Eval\eval_sodam.py`, `Eval\example_pred.jsonl` (1,122 real predictions)
- HF configs: `…\scratchpad\research\timechat\hf\` (`config.json`, `generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer_config.json`, `chat_template.jinja`, `args.json`, `index.json`, `README.md`)
- `qwen_omni_utils` 0.0.9 source: `…\scratchpad\research\timechat\qou\qwen_omni_utils-0.0.9\src\qwen_omni_utils\v2_5\vision_process.py`, `audio_process.py`
- transformers sources: `…\scratchpad\research\timechat\tf\` (`t.whl` = 5.16.1 wheel; `m_v4.57.1.py`, `m_v5.0.0.py`, `m_v5.13.0.py`, `m_v5.14.0.py`, `m_v5.15.0.py`, `m_v5.15.1.py`, `m_v5.16.0.py`, `m_v5.16.1.py`, `proc_v4.57.1.py`, `proc_v5.16.1.py`)
- HF Space source: `…\scratchpad\research\timechat\space\app.py`, `requirements.txt`
- Training-data heads: `…\scratchpad\research\timechat\sft_head.jsonl`, `grpo_head.jsonl`
- Empirical cross-version repro (second reviewer): `…\scratchpad	est1.py` … `test12.py`, isolated venvs `tv4571\` (transformers 4.57.1) and `tv516\` (5.16.1), extracted wheel trees `v4\` / `v5\`, and `modeling.diff`
