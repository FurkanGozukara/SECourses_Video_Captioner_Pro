# AVoCaDO — Audiovisual Video Captioner: Deep Research Report

**Model:** `AVoCaDO-Captioner/AVoCaDO` (HF, **non-gated**, Apache-2.0)
**Code:** https://github.com/AVoCaDO-Captioner/AVoCaDO
**Paper:** arXiv [2510.10395](https://arxiv.org/abs/2510.10395) (v1, 12 Oct 2025 — only version on arXiv)
**Project page:** https://avocado-captioner.github.io/
**Authors:** Kling Team, Kuaishou Technology + CASIA/UCAS + Peking Univ + Nanjing Univ
**HF last modified:** 2025-10-16 · **Downloads:** ~159 · **Dataset:** `AVoCaDO-Captioner/training_set` (non-gated)

> Verification note: all config/weight facts below were read directly from the HF API + raw files; all inference facts were read from a fresh `git clone` of the repo; all transformers-compatibility facts were read from the **transformers v5.16.1 source tree** (`modeling_qwen2_5_omni.py`, `processing_qwen2_5_omni.py`, `configuration_qwen2_5_omni.py`, `core_model_loading.py`, `configuration_utils.py`, `processing_utils.py`, `video_processing_qwen2_vl.py`) and the v5.16.1 docs page. Nothing here is from memory. No weights were downloaded.

---

## 1. Base architecture, files, license

### 1.1 Architecture

| Field | Value |
|---|---|
| `architectures` | `["Qwen2_5OmniForConditionalGeneration"]` |
| `model_type` | `qwen2_5_omni` |
| Base model | `Qwen/Qwen2.5-Omni-7B` (declared `base_model:finetune:` on the HF card) |
| `transformers_version` in config | `4.52.4` |
| `torch_dtype` | `bfloat16` |
| `enable_audio_output` | **`false`** |
| `enable_talker` | `true` *(a leftover key — transformers reads `enable_audio_output`, not this)* |
| License | **Apache-2.0** |

**Thinker** (`thinker_config`, `model_type: qwen2_5_omni_thinker`, `architectures: Qwen2OmniNaViTThinkerForConditionalGeneration`):

- **Text LLM** (`text_config`, `qwen2_5_omni_text`): hidden 3584, 28 layers, 28 heads / **4 KV heads** (GQA), intermediate 18944, vocab 152064, `max_position_embeddings` 32768, RoPE θ=1e6, `mrope_section [16,24,24]`, `tie_word_embeddings: false`, sliding_window 32768 (`use_sliding_window: false`).
- **Vision encoder** (`vision_config`, `qwen2_5_omni_vision_encoder`): depth 32, embed/hidden 1280, 16 heads, intermediate 3420, patch 14, `spatial_merge_size` 2, `temporal_patch_size` 2, `window_size` 112, `fullatt_block_indexes [7,15,23,31]`, `out_hidden_size` 3584, `tokens_per_second` 25.
- **Audio encoder** (`audio_config`, `qwen2_5_omni_audio_encoder`): Whisper-style, `d_model` 1280, 32 encoder layers, 20 heads, ffn 5120, **`num_mel_bins` 128**, `max_source_positions` 1500, `n_window` 100, `output_dim` 3584.
- Alignment: `position_id_per_seconds: 25`, `seconds_per_chunk: 2` (TMRoPE audio/video interleaving).

**Talker + Token2Wav:** `talker_config` and `token2wav_config` (BigVGAN + DiT) are **present in config.json but have NO weights in the checkpoint** — see below. Because `enable_audio_output: false`, transformers never instantiates them.

### 1.2 ★ Key finding — the checkpoint is thinker-only

I parsed `model.safetensors.index.json` (1346 tensors). Every key is prefixed `thinker.`:

| Prefix | # tensors |
|---|---|
| `thinker.visual.*` | 518 |
| `thinker.audio_tower.*` | 489 |
| `thinker.model.*` (text LLM) | 338 |
| `thinker.lm_head.weight` | 1 |
| `talker.*` / `token2wav.*` | **0 — none** |

- `total_size` = **17,863,627,776 bytes** → **8.93 B params** at bf16 (HF card rounds this to "9B").
- Sample keys: `thinker.model.layers.0.self_attn.k_proj.bias`, `thinker.model.embed_tokens.weight`, `thinker.lm_head.weight`, `thinker.visual.*`, `thinker.audio_tower.*`.
- **Consequence:** speech output is impossible with this checkpoint regardless of settings; the talker weights were stripped at release. This also means the thinker-only load path is *lossless* — nothing is skipped.

### 1.3 Full file list with real sizes

| File | Size |
|---|---|
| `model-00001-of-00004.safetensors` | 4,985,055,536 B (4.985 GB) |
| `model-00002-of-00004.safetensors` | 4,991,496,832 B (4.991 GB) |
| `model-00003-of-00004.safetensors` | 4,991,496,936 B (4.991 GB) |
| `model-00004-of-00004.safetensors` | 2,895,740,064 B (2.896 GB) |
| `model.safetensors.index.json` | 119,025 B |
| `config.json` | 15,275 B |
| `generation_config.json` | 69 B |
| `preprocessor_config.json` | 667 B |
| `chat_template.jinja` | 1,281 B |
| `tokenizer.json` | 11,421,870 B |
| `vocab.json` | 2,776,833 B |
| `merges.txt` | 1,671,853 B |
| `tokenizer_config.json` | 5,160 B |
| `special_tokens_map.json` | 833 B |
| `added_tokens.json` | 579 B |
| `spk_dict.pt` | 259,544 B |
| `README.md` | 1,860 B |
| `.gitattributes` | 1,570 B |

**Total download ≈ 17.9 GB.** There is **no** `video_preprocessor_config.json` (matters — see §5.5).

`spk_dict.pt` is vestigial (speaker embeddings for the absent talker) but **must not be deleted**: `Qwen2_5OmniForConditionalGeneration.from_pretrained` hard-requires it and raises `ValueError` if absent (verified in v5.16.1 source, its overridden `from_pretrained`). The thinker-only class does not need it.

### 1.4 `generation_config.json` — effectively empty

```json
{ "_from_model_config": true, "transformers_version": "4.52.4" }
```

**No `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_new_tokens`, `eos_token_id`, or `pad_token_id`.** All decoding behaviour must be supplied at the call site. (`pad_token_id: 151643` and thinker `eos_token_id: 151645` live in `config.json` instead.)

### 1.5 `preprocessor_config.json` (defaults)

```json
{
  "processor_class": "Qwen2_5OmniProcessor",
  "feature_extractor_type": "WhisperFeatureExtractor",
  "image_processor_type": "Qwen2VLImageProcessor",
  "sampling_rate": 16000, "feature_size": 128, "n_fft": 400, "hop_length": 160,
  "chunk_length": 300, "n_samples": 4800000, "nb_max_frames": 30000,
  "dither": 0.0, "padding_value": 0.0, "padding_side": "right", "return_attention_mask": true,
  "image_mean": [0.48145466, 0.4578275, 0.40821073],
  "image_std":  [0.26862954, 0.26130258, 0.27577711],
  "min_pixels": 3136, "max_pixels": 12845056,
  "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2
}
```

- **Audio sample rate: 16 kHz**, 128 mel bins.
- `chunk_length: 300` s and `n_samples: 4800000` (= 300 s × 16 kHz) → the Whisper feature extractor pads/truncates each audio to **300 seconds max**. Note this is a *feature-extractor* ceiling, well above the model's practical ~100 s limit (§4).
- `padding_side: "right"` here is the **audio feature extractor's** padding, not text padding.

### 1.6 `chat_template.jinja` — verbatim

Standard Qwen2.5-Omni template (unmodified). Notable behaviour: **if the first message is not `system`, it auto-injects `system\nYou are a helpful assistant.`** — which is *not* the prompt AVoCaDO was trained with, so always pass the system message explicitly.

```jinja
{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
```

Special tokens (`added_tokens.json`): `<|AUDIO|>`=151646, `<|IMAGE|>`=151655, `<|VIDEO|>`=151656, `<|audio_bos|>`=151647, `<|audio_eos|>`=151648, `<|vision_bos|>`=151652, `<|vision_eos|>`=151653, `<|vision_pad|>`=151654, `<|im_start|>`=151644, `<|im_end|>`=151645, `<|endoftext|>`=151643.

### 1.7 Variants / quantizations — none exist

Searched the HF API by `search=AVoCaDO`, `search=avocado`, and `author=AVoCaDO-Captioner`, plus datasets:

- **The org publishes exactly ONE model:** `AVoCaDO-Captioner/AVoCaDO`. Non-gated.
- **No GGUF. No AWQ/GPTQ/bnb. No 3B/2B/30B variants. No community re-uploads or quants of any kind.** Every other `avocado*` hit on the Hub is unrelated (avocado price datasets, LoRA toys, etc.).
- One dataset: `AVoCaDO-Captioner/training_set` (non-gated, ~1.4K downloads).
- **Practical implication:** bf16 (17.9 GB) is the only distribution format. Any quantization must be done yourself. GGUF is not realistic here — llama.cpp has no Qwen2.5-Omni audio-tower support; bitsandbytes 4-bit/8-bit via `BitsAndBytesConfig` on the thinker is the viable route.

---

## 2. Exact inference recipe from the repo

### 2.1 Code paths

| File | Function | Purpose |
|---|---|---|
| `inference.py` | `load_model_and_processor(model_path)` / `generate_caption(model, processor, file_path, prompt)` | **The reference recipe.** `python inference.py assets/case_1.mp4` |
| `eval_scripts/video-SALMONN2-testset/generate_caption.py` | `chat()` | A+V, per-sample prompts |
| `eval_scripts/UGC-VideoCap/generate_caption.py` | `chat()` | A+V, structured prompt |
| `eval_scripts/Daily-Omni/generate_caption.py` | `chat()` / `worker_proc()` | A+V, multi-GPU, `max_frames: 256` |
| `eval_scripts/WorldSense/generate_caption.py` | `chat()` / `worker_proc()` | A+V, multi-GPU, `max_frames: 256` |
| `eval_scripts/VDC/generate_caption.py` | `collate_fn()` / `generate_captions()` | **Visual-only**, the only *batched* example (bs=2) |
| `eval_scripts/DREAM-1K/generate_caption.py` | `chat()` | Visual-only |

`MODEL_PATH = "path_to_AVoCaDO"  # TODO` in `inference.py` must be filled in. Several eval scripts also have `video_dir = ""  # TODO`.

### 2.2 Dependencies (`environment.yml`, Python 3.10.15)

Load-bearing pins:

| Package | Pinned | Notes |
|---|---|---|
| `torch` | 2.6.0 | |
| `torchvision` | 0.21.0 | fallback video reader |
| `transformers` | **4.52.3** | HF config says 4.52.4 |
| `qwen-omni-utils` | **0.0.8** | latest is 0.0.9 — **behaviour differs, see §5.6** |
| `qwen-vl-utils` | 0.0.8 | |
| `flash-attn` | 2.7.0.post2 | hard-coded `attn_implementation="flash_attention_2"` |
| `accelerate` | 0.34.1 | for `device_map="auto"` |
| `decord` | 0.6.0 | preferred video reader |
| `av` | 13.1.0 | PyAV — audio-track detection |
| `librosa` | 0.11.0 | audio resampling to 16 kHz (pulls in `audioread`) |
| `soundfile` | 0.13.1 | |
| `ffmpeg-python` 0.2.0, `imageio-ffmpeg` 0.5.1, `moviepy` 1.0.3 | | |
| `numpy` | 1.26.3 | <2 |
| `deepspeed`, `vllm==0.6.3`, `xformers`, `peft`, `sglang` (VDC judge) | | training/eval only, **not needed for inference** |

**★ ffmpeg binary is a hard runtime requirement.** `qwen_omni_utils/v2_5/audio_process.py` calls `audioread.ffdec.FFmpegAudioFile(path)` for *every* video when `use_audio_in_video=True`. It shells out to `ffmpeg`. Not a pip package — the executable must be on `PATH`.

**★ Videos without an audio track hard-crash.** Same file:

```python
assert _check_if_video_has_audio(path), "Video must has audio track when use_audio_in_video=True"
```

`_check_if_video_has_audio` opens the file with PyAV and checks for an audio stream. **Pre-check every input** and fall back to `use_audio_in_video=False` for silent videos.

### 2.3 Constants — identical in all 7 scripts

```python
VIDEO_MAX_PIXELS   = 401408     # 512*28*28   -> per-frame cap (~512 visual tokens/frame)
VIDEO_TOTAL_PIXELS = 20070400   # 512*28*28*50 = 25600*28*28 -> whole-video budget
USE_AUDIO_IN_VIDEO = True       # False in VDC and DREAM-1K
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)   # set BEFORE process_mm_info
```

Note the confusing naming: the **env var** `VIDEO_MAX_PIXELS` sets `qwen_omni_utils`' *total* pixel budget (its internal `VIDEO_TOTAL_PIXELS` global), while the **Python constant** `VIDEO_MAX_PIXELS` is the *per-frame* cap passed in the message dict. They are different things.

Other knobs:
- **fps: never set explicitly.** Relies on `qwen_omni_utils` default `FPS = 2.0` — *and* on `Qwen2_5OmniProcessor`'s independent default `fps=2.0` (used to compute `second_per_grid_ts = temporal_patch_size / fps`). These two must agree or the audio/video temporal alignment silently breaks.
- `max_frames: 256` — only in Daily-Omni, WorldSense, VDC. Elsewhere the library default `FPS_MAX_FRAMES = 768` applies.
- `min_frames` default `FPS_MIN_FRAMES = 4`; `FRAME_FACTOR = 2` (frame count always even).
- Audio: 16 kHz mono (`SAMPLE_RATE = 16000` in `audio_process.py`).

### 2.4 The exact loading + captioning code (`inference.py`, verbatim)

```python
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

def load_model_and_processor(model_path: str):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor

def generate_caption(model, processor, file_path, prompt):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "video", "video": file_path, "max_pixels": VIDEO_MAX_PIXELS},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                              do_sample=False, thinker_max_new_tokens=2048)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model_generation = text.split("\nassistant\n")[-1]
    return model_generation
```

### 2.5 Talker: yes, it must be off

`model.disable_talker()` is called in **every single script**. With this checkpoint it is technically a no-op (`enable_audio_output: false` means transformers never builds the talker), but keep it for safety across versions. `return_audio` is never passed in the repo — on transformers 4.52 it defaults to `None`, which is falsy. **On transformers 5.x this changes and breaks — see §5.2.**

### 2.6 Generation defaults

| Parameter | Value |
|---|---|
| Sampling | **`do_sample=False` — greedy.** Every script. |
| `temperature` / `top_p` / `top_k` / `repetition_penalty` | **never set anywhere** (and absent from `generation_config.json`) |
| max new tokens | **`thinker_max_new_tokens=2048`** — note the `thinker_`-prefixed arg, not plain `max_new_tokens` |
| `use_audio_in_video` | passed to **both** `process_mm_info`, `processor()`, **and** `model.generate()` — all three must match |
| dtype / attn | `torch.bfloat16` / `flash_attention_2` |
| num_beams | 1 (default) |

Distractors to ignore: `temperature=0, top_p=0.001` in `eval_scripts/Daily-Omni/evaluation.py` belongs to the **Gemini judge**, not AVoCaDO. The paper's `temperature 1.0` is GRPO rollout sampling during training.

### 2.7 ★ EXACT prompts (verbatim)

#### System prompt (all modes — audiovisual, visual-only, everything)

```
You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.
```

This is the stock Qwen2.5-Omni system prompt, kept unchanged through fine-tuning. Always pass it; omitting it lets the chat template inject `You are a helpful assistant.` instead, which is off-distribution.

#### Audiovisual captioning — 7 paraphrases, selected by `random.choice(prompt_list)`

Identical list in `inference.py`, `Daily-Omni/generate_caption.py`, and `WorldSense/generate_caption.py`:

1. `Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.`
2. `Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.`
3. `Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.`
4. `Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.`
5. `Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.`
6. `Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.`
7. `Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized.`

The randomization is a training-distribution-matching trick (the SFT data was built with prompt paraphrases). For a **deterministic product, pick one and keep it fixed** — #1 or #5 are the most representative.

#### Visual-only captioning (VDC, `USE_AUDIO_IN_VIDEO = False`)

```
Describe every aspect of the video in full detail, covering all the information it contains.
```

(Literally prompt #5 with the audio clause excised — a clean template for visual-only mode.)

#### Visual-only, frame-style (DREAM-1K, `USE_AUDIO_IN_VIDEO = False`)

```
Imagine the video from these frames and describe it in detail.
```

#### Structured audiovisual (UGC-VideoCap) — the only prompt that dictates content sections

```
You are given a short video with both audio and visual content. Write a detailed and coherent paragraph that naturally integrates all modalities. Your description should include: (1) the primary scene and background setting; (2) key characters or objects and their actions or interactions; (3) significant audio cues such as voices, background music, sound effects, and their emotional tone; (4) any on-screen text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's content in a human-like, narrative style.
```

*(video-SALMONN-2 reuses each sample's own annotation prompt: `anno["conversations"][0]["value"].replace("<image>\n", "")`.)*

#### There is no audio-only / dialogue-transcription inference prompt

The paper's Appendix F contains **data-annotation** prompts (used with Gemini-2.5-Pro to *build* the dataset, not to run AVoCaDO). Two are useful as post-processors:

**Audio-caption annotation prompt (Fig. 10)** — shows what "audio description" means in this model's output distribution:
```
You are a professional audio caption writer. Your task is to create a detailed narrative description of an audio in the video. Your description must include the following elements:
Narration / Dialogue: Please accurately transcribe the spoken words (narration or dialogue) from the audio. In addition to the transcription, describe the speaker's tone and emotional delivery during the speech—such as whether the tone is calm, excited, hesitant, enthusiastic, serious, sarcastic, etc.—based on vocal cues like pitch, pace, volume, and emotion.
Music & Sound: Describe the background music's mood and any important sound effects.
The audio caption should be coherent and well-structured. Do not simply give the transcriptions without the speaker's tone and emotions.
```

**Dialogue-extraction prompt (Fig. 14)** — a ready-made post-processor to turn AVoCaDO prose into structured dialogue:
```
You are a highly skilled assistant specializing in extracting conversational dialogue from text. Your task is to carefully analyze the given description of a video and accurately identify and extract all dialogue content within it.
Please directly output the dialogue in the following format without adding any other content. If no dialogue is present, state: "None."
Dialogue format:
Speaker A Description: "Dialogue from speaker A."
Speaker B Description: "Dialogue from speaker B."
Speaker A Description: "Further dialogue..."
The description for each speaker (e.g., "Person in red dress") must align with the given description and should be simplified for brevity. The key is to be concise and clearly distinguish between speakers (e.g., "Man in red shirt" is sufficient).
Video description: {video description}
```

**Fusion prompt (Fig. 11)** — this is the prompt that *defines* AVoCaDO's output style, worth reading to understand the model:
```
You are tasked with fusing the visual caption and audio caption into a single, coherent narrative based on the video content. Follow these strict rules:
1. Preserve every single sentence from both the visual caption and audio caption exactly as they appear.
2. Do NOT omit or delete any sentence in any way.
3. You may reorder the sentences (from both captions) to create a logical and temporally accurate sequence that reflects the video's events.
4. Ensure the integrated narrative flows naturally in time with the video, aligning visual actions with corresponding sounds or spoken content.
```

---

## 3. Output format and post-processing

### 3.1 Format: free-flowing narrative prose. No timestamps, no tags.

Verified across every qualitative example in the paper (Figs. 4, 7, 8) and the project page's full sample: **not one contains a timestamp, a `[SPEAKER]` tag, a section header, or a bullet list.** Temporal alignment is expressed purely through **narrative ordering** — audio and visual events are interleaved in the sentence stream in the order they occur, with transition cues ("The scene transitions to…", "The video then returns to…", "The audio then transitions to…").

Conventions the model reliably follows:

| Element | Rendering |
|---|---|
| **Dialogue** | Straight double quotes, verbatim ASR: `"I'm Aubrey."` |
| **Speaker ID** | By appearance/role, **not** a name tag: "The man with the beard", "The man in the blue shirt", "A male narrator with a deep, smooth, and professional voice". Real names only if spoken aloud. |
| **Tone/emotion** | Annotated inline on nearly every utterance: "in a friendly and slightly breathless tone", "her tone conversational and a little hesitant" |
| **Nested quotes** | Single quotes inside: `"I know a brother said, 'Give me my money back.'"` |
| **Sung lyrics** | `a female vocalist singing "Food Mania Review."` |
| **SFX** | Plain prose: "a sharp, percussive sound effect of hands slapping the table"; "The sound of rain can be heard faintly in the background." |
| **Music** | Mood-described: "an upbeat, energetic, and slightly retro-sounding musical jingle" |
| **On-screen text** | OCR'd and quoted: `"EPISODES & CLIPS"` |
| **Camera** | "The camera remains static throughout the scene", "The camera pulls back" |
| **Paragraphs** | Roughly one per scene / shot-group on longer outputs |

Representative excerpt (paper Fig. 4 / repo `assets/case_2.png`):

> "A static, medium shot shows two people, an older woman and a younger girl… The audio begins with a sharp, percussive sound effect of hands slapping the table. The younger girl speaks in a friendly, clear voice, "I'm Aubrey." The woman replies with equal cheerfulness, "I'm Amy." They then speak together in an enthusiastic, presentational tone, "And you're watching Food Mania Review!"... The scene transitions to a title card… followed by an upbeat, energetic, and slightly retro-sounding musical jingle with a female vocalist singing "Food Mania Review." The video then returns to the original shot… while the music fading."

### 3.2 Length

Paper Fig. 5 (caption token length vs. video duration): **mean 1,437 tokens**, **max 3,982** for videos ≤100 s. The project page's full untruncated sample is 894 words / ~1,100–1,200 tokens. GRPO's length reward `R_L` gives full credit below **2,048** tokens, decays linearly to 0 across 2,048→4,096, and 0 above — which is exactly why inference uses `thinker_max_new_tokens=2048`.

### 3.3 Post-processing in the repo

Exactly one line, everywhere:

```python
model_generation = text.split("\nassistant\n")[-1]
```

`batch_decode(..., skip_special_tokens=True)` strips `<|im_start|>`/`<|im_end|>` but leaves the plain-text role markers, so the decoded string looks like `system\n…\nuser\n…\nassistant\n<CAPTION>`. Splitting on `"\nassistant\n"` and taking the last element yields the caption. No further cleanup, no JSON, no regex.

*(A cleaner alternative: slice the generated ids — `text_ids[:, inputs["input_ids"].shape[1]:]` — and decode only those. This avoids a pathological failure if the caption itself ever contains the literal string `"\nassistant\n"`.)*

**If you need structure** (timestamps, speaker tags), you must add an LLM post-pass — the paper's Fig. 14 prompt (§2.7) is purpose-built for the dialogue half. Ordering is reliably chronological, so approximate timestamps can be back-derived, but the model emits none natively.

---

## 4. Limits

### 4.1 ★ Maximum video duration: ~100 seconds

Verbatim from the paper (Appendix D.2):

> "Qwen2.5-Omni supports a maximum context window of 32K tokens and encodes audio at a rate of 25 tokens per second. In our training and evaluation, to effectively capture video dynamics and preserve the visual detail of each frame, we sample videos at 2 fps, with each frame allocated a maximum of 512 tokens for encoding. Due to the context window constraint, the total number of video tokens is capped at 25,600."

> "**A 100-second high-resolution video consumes 2,500 audio tokens (100s × 25 tokens/s) and the maximum 25,600 video tokens, totaling 28,100 tokens for multimodal input.** When combined with the input text prompt and the generated caption, the total token count approaches the 32K context limit. **To prevent context overflow and ensure the generation of complete and untruncated captions, we constrain our training dataset to videos of 100 seconds or less.**"

And Appendix C:

> "During both training and evaluation, video inputs are sampled at 2 fps, and the resolution of each frame is limited to a maximum of 512 × 28 × 28 pixels. Due to the base model's context window limitation of 32K tokens, the total video tokens is restricted to 25600 × 28 × 28. All training is conducted on 16 NVIDIA H200 GPUs, while evaluation is performed on NVIDIA H20 GPUs."

Token budget at a glance: 25,600 video + 2,500 audio (100 s) = 28,100 input; + prompt + up to 2,048 generated ≈ 30–31K of the 32,768 window.

**★ The failure mode is silent, not an exception.** Past ~50 frames (≈25 s at 2 fps), `qwen_omni_utils` keeps the *total* pixel budget fixed and shrinks each frame:
`max_pixels = max(min(602112, total_pixels / nframes * 2), min_pixels * 1.05)`.
So a 100 s clip still "works" but every frame is downscaled ~4×; a 5-minute clip degrades to near-thumbnail frames and then overflows context and truncates the caption mid-sentence. **Segment long videos and stitch** — there is no chunking, sliding window, or memory mechanism in the model or repo.

Training-data durations (all filtered to ≤100 s): Shot2Story "10s to 40s"; ShortVideo "under 30 seconds to over 5 minutes, with most being less than one minute". Eval benchmarks: video-SALMONN-2 testset "30 to 60 seconds, average 51 seconds"; UGC-VideoCap "each under 60 seconds". **The sweet spot is 10–60 s.**

Frame-count ceilings: `max_frames: 256` in three eval scripts (= 128 s at 2 fps); library default `FPS_MAX_FRAMES = 768`.

### 4.2 VRAM at bf16

| Component | Estimate |
|---|---|
| Weights (thinker only — all there is) | **17.9 GB** |
| KV cache | 28 layers × 4 KV heads × 128 dim × 2 (K+V) × 2 B = **~57 KB/token** → ~1.7 GB at 30K tokens |
| Vision-tower activations | **the peak driver** — ~2,048 raw patches/frame before merge; 4 full-attention blocks (`fullatt_block_indexes [7,15,23,31]`) over the whole sequence |
| Audio tower | modest (32 layers × 1280 dim, ≤1500 mel positions/window) |

Practical guidance:
- **24 GB (4090/3090):** works for short clips (~10–30 s) with flash-attn and a reduced `total_pixels`. Tight-to-failing at the full 25,600-token budget.
- **32–48 GB (A6000/L40S/5090):** comfortable at the full 100 s / 25,600-token budget.
- **80 GB+:** headroom for batching.
- The authors evaluated on **H20 (96 GB)** — they never had to tune for consumer VRAM, so no low-VRAM path exists in the repo.
- **Levers if OOM:** lower `total_pixels` (e.g. 10,035,200 = half); lower per-frame `max_pixels` (e.g. `256*28*28 = 200704`); cap `max_frames`; drop fps to 1.0 (but then **also pass `fps=1.0` to the processor** or alignment breaks); 4-bit `BitsAndBytesConfig` on the thinker (~6 GB weights, quality cost unmeasured).
- No `device_map="auto"` CPU-offload benefit worth chasing — the vision tower is compute-bound.

### 4.3 Image-only and audio-only

- **Architecturally supported** — the thinker keeps the base Qwen2.5-Omni image and audio pathways intact (`<|IMAGE|>`/`<|AUDIO|>` tokens in the template, `Qwen2VLImageProcessor` + `WhisperFeatureExtractor` in the processor, `get_image_features` / `get_audio_features` on the model class). You *can* pass `{"type": "image", ...}` or `{"type": "audio", ...}`.
- **But out of distribution.** AVoCaDO is a video-captioning fine-tune; there are **no image-only or audio-only prompts, scripts, or benchmark results** anywhere in the repo or paper. Expect base-model-ish behaviour with a strong pull toward long narrative captions.
- **Visual-only video *is* well supported and validated** (VDC / DREAM-1K, `use_audio_in_video=False`) and actually *improved* over the base model: VDC Detailed Acc 39.7→47.4, DREAM-1K F1 31.6→35.9. Use the visual-only prompt from §2.7.

### 4.4 Batch inference

Supported but awkward. The **only** batched example is `eval_scripts/VDC/generate_caption.py` (visual-only, `batch_size=2`), and it needs a manual `collate_fn` that **zero-pads shorter videos to the batch's max frame count**:

```python
padding_tensor = torch.zeros((padding_needed, C, H, W), dtype=video.dtype)
padded_video = torch.cat([video, padding_tensor], dim=0)
```

That injects black frames into shorter clips — a real quality hazard. Caveats:
- No batched example exists for `use_audio_in_video=True` (audio/video interleaving makes ragged batching much harder).
- Text padding must be **left**-sided for decoder-only generation. transformers 5.16.1's `Qwen2_5OmniProcessorKwargs._defaults` sets `"padding_side": "left"` — correct. (The `padding_side: "right"` in `preprocessor_config.json` is the audio feature extractor's, unrelated.)
- **Recommendation for production: batch size 1**, and scale with multiple processes/GPUs — which is exactly what the repo's `run_multi_gpu` / `worker_proc` helpers do (one model per GPU, `mp.set_start_method("spawn")`, sharded work, per-rank JSONL, merge at the end).

### 4.5 Other limits

- **No Limitations section exists in the paper** (verified across all 23 pages of arXiv v1). Nothing stated on non-English performance, music, licensing/ethics, or inference compute.
- **Hallucination is not solved:** AVoCaDO's hallucination rate (16.2) is *worse* than several baselines and worse than Gemini-2.5-Pro (13.3). The win is on *missing* content (Miss 41.7→21.1). Detailed captioning trades hallucination for coverage.
- **English-dominant** training data (YouTube-Commons "with English as the majority language", CinePile "English-language films"). Zero multilingual evaluation.
- **Repetition collapse still occurs at ~0.4–1.0%** even after the length reward (down from 3.9%/4.9%). Worth a detect-and-retry guard in a pipeline.
- **Headline results:** video-SALMONN-2 total error 57.1→**37.3** vs. its own base; UGC-VideoCap avg **73.2** (beats Gemini-2.5-Pro's 72.6); Daily-Omni **50.1** and WorldSense **25.7** (vs. base 13.4/8.6).

---

## 5. Compatibility with current transformers (5.16.1, Aug 2026)

**Latest release: `transformers==5.16.1`, published 2026-08-26** (PyPI). The repo targets **4.52.3/4.52.4** — a full major version behind.

**Bottom line: Qwen2.5-Omni is still first-class in 5.16.1** — same class names, same weight layout, `use_audio_in_video` intact, thinker-only loading officially documented. But there are **four concrete breaking changes** that make `inference.py` fail or silently misbehave.

### 5.1 What did NOT change (verified in v5.16.1 source)

- `Qwen2_5OmniForConditionalGeneration`, `Qwen2_5OmniThinkerForConditionalGeneration`, `Qwen2_5OmniProcessor`, `Qwen2_5OmniConfig`, `Qwen2_5OmniThinkerConfig` — all present and exported.
- **Thinker submodule layout is unchanged**: `self.audio_tower`, `self.visual`, `self.model` (text), `self.lm_head`. AVoCaDO's checkpoint keys (`thinker.audio_tower.*`, `thinker.visual.*`, `thinker.model.*`, `thinker.lm_head.weight`) map **1:1** after prefix stripping. No VLM-standardization rename (no `model.language_model` refactor) hit this model.
- `Qwen2_5OmniProcessor.__call__(text, images, videos, audio, **kwargs)` — still `audio=` (singular), and `use_audio_in_video`, `seconds_per_chunk`, `position_id_per_seconds`, `min_pixels`, `max_pixels`, `min_frames`, `max_frames` are all still valid `videos_kwargs`.
- `Qwen2VLVideoProcessor.__init__` still honours `min_pixels`/`max_pixels` via an explicit BC shim mapping them onto `size["shortest_edge"]`/`size["longest_edge"]`.
- `model.generate(..., use_audio_in_video=..., thinker_max_new_tokens=..., do_sample=...)` — signature unchanged; `thinker_*` / `talker_*` / `token2wav_*` prefix routing intact.
- `disable_talker()` / `enable_talker()` still exist.
- `attn_implementation="flash_attention_2"` and `"sdpa"` both supported (`_supports_flash_attn = True`, `_supports_sdpa = True`).
- Auto-mapping `("qwen2_5_omni", "Qwen2VLVideoProcessor")` exists, so the processor builds a video processor even though AVoCaDO ships no `video_preprocessor_config.json`.

### 5.2 ★ BREAKING #1 — `model.generate()` now raises on a talker-less model

In v5.16.1, `Qwen2_5OmniForConditionalGeneration.generate` is decorated `@deprecate_kwarg("return_audio", version="v5", new_name="generation_mode")` and begins:

```python
generation_mode = kwargs.pop("generation_mode", None)
return_audio = generation_mode != "text" and generation_mode is not False
...
if return_audio and not self.has_talker:
    raise ValueError("Cannot use talker when talker module not initialized. ...")
```

With `generation_mode` unset (`None`), `return_audio` evaluates to **`True`**. AVoCaDO's config has `enable_audio_output: false` → `has_talker == False` → **`ValueError`**.

In 4.52 the parameter was `return_audio: Optional[bool] = None`, and `None` is falsy, so the same call worked. **The repo's `inference.py` therefore crashes on transformers 5.x.**

**Fixes (either):** pass `generation_mode="text"` to `generate()`, **or** — better — use the thinker class directly (§5.4), which has no such gate.

### 5.3 ★ BREAKING #2 — `torch_dtype` is deprecated in favour of `dtype`

`modeling_utils.py` v5.16.1:
```python
if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
    logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
```
Still functional, but warns. Use `dtype=torch.bfloat16`.

### 5.4 ★ Thinker-only loading — officially supported, and ideal here

The v5.16.1 docs state it explicitly:

> "Use `Qwen2_5OmniForConditionalGeneration` to generate audio and text output. To generate only one output type, use `Qwen2_5OmniThinkerForConditionalGeneration` for text-only…"
> "To generate only text output and save compute by not loading the audio generation model, we can use `Qwen2_5OmniThinkerForConditionalGeneration` model."

I traced both mechanisms that make this work on AVoCaDO's composite checkpoint:

1. **Config resolution** (`configuration_utils.py` L710-716): `Qwen2_5OmniThinkerConfig.from_pretrained` sees `model_type: "qwen2_5_omni"` ≠ its own `"qwen2_5_omni_thinker"`, then scans the config dict for a nested value whose `model_type` matches — finding `thinker_config`. Resolves correctly, no warning.
2. **Weight prefix stripping** (`core_model_loading.py`, `rename_source_key`, step 3):
   ```python
   if renamed_key.startswith(base_model_prefix) and meta_state_dict.get(re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)) is not None:
       renamed_key = re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)
   ```
   `Qwen2_5OmniThinkerForConditionalGeneration.base_model_prefix == "thinker"`, so `thinker.model.layers.0…` → `model.layers.0…` — an exact hit in the thinker's state dict.

**Because the AVoCaDO checkpoint contains only thinker weights, this load is 100 % clean: zero missing keys, zero unexpected keys.** (On the original `Qwen/Qwen2.5-Omni-7B` you would get "unexpected keys" noise from talker/token2wav; not here.)

VRAM note: with `enable_audio_output: false`, the *full* class also skips building talker+token2wav, so **VRAM is identical either way (~17.9 GB)**. The thinker class is still preferable — it sidesteps §5.2 entirely and doesn't require `spk_dict.pt`.

**Modern thinker-only loading snippet:**

```python
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_PATH = "AVoCaDO-Captioner/AVoCaDO"   # or a local dir

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,                  # `torch_dtype` is deprecated in v5
    device_map="auto",
    attn_implementation="flash_attention_2",   # use "sdpa" if flash-attn isn't installed (Windows)
)
model.eval()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

# generate() here is the plain GenerationMixin one — no `thinker_` prefix, no generation_mode gate
out_ids = model.generate(**inputs, use_audio_in_video=True, do_sample=False, max_new_tokens=2048)
```

Note the argument rename: with the thinker class it is **`max_new_tokens=2048`**, not `thinker_max_new_tokens=2048`.

### 5.5 ★ BREAKING #3 — do NOT use `apply_chat_template(load_audio_from_video=True)` together with `use_audio_in_video=True`

transformers 5.x adds native video+audio loading in `apply_chat_template`, and the official docs example passes **both** `load_audio_from_video=True` and `use_audio_in_video=True`. Tracing the source, that combination looks broken:

- `processing_utils.py` L2163-2169: when `load_audio_from_video=True`, it **mutates the message in place** — `message["content"].append({"type": "audio"})` — so the Jinja template now emits an extra `<|audio_bos|><|AUDIO|><|audio_eos|>` after the video, while only **one** audio array is appended to `batch_audios`.
- `processing_qwen2_5_omni.py`, `replace_multimodal_special_tokens`: token positions are sorted, so `<|VIDEO|>` is handled first. Under `use_audio_in_video=True` that branch **consumes the single audio** via `next(audio_lengths)` to build the interleaved chunk string. The loop then reaches the standalone `<|AUDIO|>` and calls `next(audio_lengths)` again on an exhausted iterator.

I did not execute this (no weights downloaded), and there is no test covering the combination in `tests/models/qwen2_5_omni/`, so treat it as **strongly suspect rather than proven**. Either way, the safe alternative — `use_audio_in_video=False` + `load_audio_from_video=True` — is **wrong for AVoCaDO**: it appends audio as a separate block instead of TMRoPE-interleaving it with the video, destroying exactly the audio↔visual temporal alignment the model was trained for.

**→ Keep using `qwen_omni_utils.process_mm_info` + an explicit `processor(...)` call.** That path I verified line-by-line against the 5.16.1 processor signature and it is unchanged.

### 5.6 ★ BREAKING #4 — `qwen-omni-utils` 0.0.9 silently ignores `VIDEO_MAX_PIXELS`

The repo pins `qwen-omni-utils==0.0.8`; latest is **0.0.9** (2026-02-10). I diffed both wheels:

**0.0.8** (`vision_process.py`):
```python
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
...
total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
```

**0.0.9**:
```python
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))
...
total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9)
```

The `VIDEO_MAX_PIXELS` env var **no longer exists in 0.0.9**. AVoCaDO's `os.environ['VIDEO_MAX_PIXELS'] = '20070400'` becomes a **silent no-op**, and the budget jumps to the default `128000 × 28 × 28 × 0.9 = 90,316,800` — **4.5× AVoCaDO's intended budget**. On a 60 s clip that inflates visual tokens far past the 25,600 the model was trained with, and you'd hit context overflow or badly off-distribution input with no error message.

**Version-proof fix — pass `total_pixels` explicitly in the video element dict.** Both 0.0.8 and 0.0.9 honour `ele.get("total_pixels", ...)`:

```python
{"type": "video", "video": path, "max_pixels": 401408, "total_pixels": 20070400}
```

Do this **in addition to** the env var, so the recipe is correct on either version. Other 0.0.9 constants are unchanged and safe: `FPS = 2.0`, `FRAME_FACTOR = 2`, `FPS_MIN_FRAMES = 4`, `FPS_MAX_FRAMES = 768`, `SAMPLE_RATE = 16000`, per-frame cap still `768*28*28`. Deps also shifted: 0.0.9 requires `av, librosa, packaging, pillow, requests`, with **`decord` demoted to an optional extra** (`pip install qwen-omni-utils[decord]`); without it the reader falls back to torchvision.

### 5.7 Other things to watch

- **fps consistency.** The processor computes `second_per_grid_ts = temporal_patch_size / fps` with its **own** default `fps=2.0`, independently of how `process_mm_info` sampled frames. If you ever change fps, pass `fps=<same value>` to `processor(...)` too, or TMRoPE alignment silently drifts.
- **`do_sample_frames`.** `Qwen2VLVideoProcessor.do_sample_frames = False` (kept False for BC). Irrelevant on the `process_mm_info` path (frames are pre-sampled) — but relevant if you ever switch to `apply_chat_template` video loading.
- **flash-attn on Windows** is painful to build. `attn_implementation="sdpa"` works and is a fine substitute (somewhat higher peak memory in the vision tower). transformers 5.x also supports hub kernels (`"kernels-community/flash-attn"`) as a prebuilt alternative.
- **`numpy<2`** — the repo pins 1.26.3; several pinned deps in `environment.yml` (notably `vllm==0.6.3`, `decord==0.6.0`) predate numpy 2. Don't blindly upgrade.
- `environment.yml` is a **full Linux conda lock file** with Kuaishou-internal packages (`infra-kess`, `dsc-auth`, `ks-kafka-python`, `dapr`) and heavy training deps. **Do not `conda env create` it for inference** — install the ~10 packages in §6.2 instead.

---

## 6. Copy-paste-ready minimal standalone snippet

### 6.1 Files to vendor from the repo

**None.** There are no custom modeling/processing/configuration files — no `modeling_avocado.py`, no `trust_remote_code=True`, nothing on the Hub repo beyond stock config/tokenizer/weights. Everything comes from `transformers` + `qwen_omni_utils` (a pip package).

The only things worth copying out of the GitHub repo are **text constants**, all of which are reproduced verbatim in §2.3 and §2.7:
1. The system prompt
2. The 7 audiovisual prompts (or your chosen fixed one)
3. The visual-only prompts
4. `VIDEO_MAX_PIXELS = 401408`, `VIDEO_TOTAL_PIXELS = 20070400`

The `eval_scripts/DREAM-1K/tarsier/` subtree is a vendored copy of the Tarsier benchmark harness — **evaluation only, irrelevant to inference.**

### 6.2 Install

```bash
pip install "transformers>=5.16.1" accelerate torch torchvision
pip install "qwen-omni-utils[decord]"        # brings av, librosa, pillow, requests
pip install soundfile
# optional, big speedup + lower vision-tower memory; skip on Windows and use attn_implementation="sdpa"
pip install flash-attn --no-build-isolation
# REQUIRED: ffmpeg executable on PATH (audioread shells out to it for video audio tracks)
```

### 6.3 The snippet (transformers 5.16.1 + qwen_omni_utils, thinker-only, one video with audio)

```python
"""
AVoCaDO audiovisual video captioning - minimal standalone inference.
Verified against transformers 5.16.1 / qwen-omni-utils 0.0.8 and 0.0.9.
Requires: ffmpeg on PATH. ~18 GB VRAM at bf16.
"""
import os

# qwen-omni-utils 0.0.8 reads this at import time -> set BEFORE importing it.
# (No-op on 0.0.9; the explicit "total_pixels" below covers both versions.)
VIDEO_TOTAL_PIXELS = 20_070_400   # 512*28*28*50 = 25600*28*28  (whole-video budget)
VIDEO_MAX_PIXELS   = 401_408      # 512*28*28                   (per-frame cap)
os.environ["VIDEO_MAX_PIXELS"] = str(VIDEO_TOTAL_PIXELS)

import sys
import torch
import av
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_PATH = "AVoCaDO-Captioner/AVoCaDO"   # or a local snapshot dir

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
# Prompt #1 of the 7 official audiovisual paraphrases (fixed, for determinism).
AV_PROMPT = (
    "Provide a comprehensive description of all the content in the video, leaving out no details. "
    "Be sure to include as much of the audio information as possible, and ensure that your "
    "descriptions of the audio and video are closely aligned."
)
# Official visual-only prompt (used when the file has no audio track).
VISUAL_ONLY_PROMPT = (
    "Describe every aspect of the video in full detail, covering all the information it contains."
)


def has_audio_track(path: str) -> bool:
    """qwen_omni_utils hard-asserts on this; check first so we can degrade gracefully."""
    try:
        with av.open(path) as c:
            return any(s.type == "audio" for s in c.streams)
    except Exception:
        return False


def load():
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,              # `torch_dtype` is deprecated in transformers 5.x
        device_map="auto",
        attn_implementation="flash_attention_2",   # -> "sdpa" if flash-attn is unavailable
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    return model, processor


@torch.inference_mode()
def caption(model, processor, video_path: str, prompt: str | None = None) -> str:
    use_audio = has_audio_track(video_path)
    if prompt is None:
        prompt = AV_PROMPT if use_audio else VISUAL_ONLY_PROMPT
    if not use_audio:
        print(f"[warn] no audio track in {video_path}; falling back to visual-only.")

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": VIDEO_MAX_PIXELS,       # per-frame cap
                "total_pixels": VIDEO_TOTAL_PIXELS,   # whole-video budget (version-proof)
                # "max_frames": 256,                  # optional hard frame cap (=128 s at 2 fps)
            },
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True,
        use_audio_in_video=use_audio,
        fps=2.0,          # must match qwen_omni_utils' FPS default, or TMRoPE alignment drifts
    )
    inputs = inputs.to(model.device).to(model.dtype)

    out_ids = model.generate(
        **inputs,
        use_audio_in_video=use_audio,
        do_sample=False,          # greedy, exactly as the authors run it
        max_new_tokens=2048,      # thinker class -> plain `max_new_tokens`
    )

    # Decode ONLY the newly generated tokens (safer than splitting on "\nassistant\n")
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


if __name__ == "__main__":
    video = sys.argv[1]
    model, processor = load()
    print(caption(model, processor, video))
```

**If you must keep the full `Qwen2_5OmniForConditionalGeneration` class** (e.g. to stay closer to the repo), change exactly three things — nothing else:

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
    attn_implementation="flash_attention_2")
model.disable_talker()
out_ids = model.generate(
    **inputs, use_audio_in_video=use_audio,
    generation_mode="text",          # <-- REQUIRED on transformers 5.x, else ValueError
    do_sample=False,
    thinker_max_new_tokens=2048,     # <-- note the thinker_ prefix on this class
)
```

---

## 7. Pipeline checklist

1. **~100 s hard ceiling; 10–60 s sweet spot.** Segment longer input and stitch. Failure past the limit is *silent* (resolution decay → truncated caption), not an exception.
2. **Reproduce the recipe exactly** or lose quality: stock Qwen system prompt, one of the 7 official prompts verbatim, greedy, 2048 max new tokens, `use_audio_in_video=True` in all three places, `max_pixels=401408` + `total_pixels=20070400`, fps 2.
3. **`total_pixels` in the dict, not just the env var** — the env var is dead in `qwen-omni-utils` 0.0.9.
4. **On transformers 5.x**: use the thinker class (or pass `generation_mode="text"`); use `dtype=` not `torch_dtype=`; avoid `apply_chat_template(load_audio_from_video=True)`.
5. **Pre-check for an audio track** — silent videos raise an `AssertionError` deep inside `qwen_omni_utils`.
6. **ffmpeg must be on PATH.**
7. **Output is unstructured prose.** No timestamps, no speaker tags. Add an LLM post-pass (paper Fig. 14 prompt) if you need structure.
8. **Batch size 1**; scale with one process per GPU. The bs=2 example pads with black frames and is visual-only.
9. **17.9 GB weights.** 24 GB works for short clips; 32 GB+ for the full budget. No quantized release exists.
10. **Guard against repetition collapse** (~0.4–1 % residual rate) — detect and retry.
