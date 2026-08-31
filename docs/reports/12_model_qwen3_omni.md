# Qwen3-Omni-30B-A3B — Deep Research Report for a Captioning App

Research date: **2026-08-30**. Sources: HF API + raw repo files, `github.com/QwenLM/Qwen3-Omni` @ `e423585` (last commit 2026-04-23), `transformers` v4.57.1 / v5.2.0 / v5.16.1 source, `qwen-omni-utils` 0.0.9, arXiv 2509.17765, llama.cpp master, Alibaba Model Studio docs.
Numbers marked **[measured]** were verified locally by running the real `Qwen3OmniMoeProcessor` (transformers 5.14.1) against the official config files, and by parsing safetensors/GGUF headers over HTTP range requests. **No model weights were downloaded.**

---

## 0. TL;DR for the captioning app

| Question | Answer |
|---|---|
| Sizes available | **Only 30B-A3B.** No 7B/4B Qwen3-Omni exists. Flash and Qwen3.5-Omni are **API-only, no weights**. |
| Best variant for captioning | **Instruct** (general video/audio/image captioning, greedy, fast) · **Captioner** (audio-only, prompt-free, extremely detailed) · **Thinking** (only if you need reasoning; 2–5× more tokens) |
| Talker/code2wav | **Thinking and Captioner already ship without them.** Only Instruct has them (+7.08 GB). `disable_talker()` on Instruct. |
| Disk (bf16) | Instruct **70.53 GB** · Thinking **63.45 GB** · Captioner **63.45 GB** |
| transformers | min **4.57.0**, Qwen recommends **≥5.2.0**, latest **5.16.1** works. **BREAKING:** in 5.x `generate()` returns ONE value, not a `(text_ids, audio)` tuple, when audio is off. The official README snippet is stale. |
| transformers speed | **Terrible** (~1–6 tok/s on A100) — the MoE experts run in a Python for-loop. Use **vLLM** (≈88 tok/s output) for anything batch. |
| Audio token rate | **exactly 13 tokens/second** [measured]. 40 min = 31,200 tokens (that's where the "40 minutes" claim comes from — it's the 32k context). |
| Video token rate | `fps/2 × tokens_per_frame_group`; default cap **768 tok/group** → up to 768 tok/s at fps=2 [measured] |
| Max context | trained **32,768**; `max_position_embeddings` **65,536** |
| GGUF multimodal | **YES since 2026-04-12.** `ggml-org/Qwen3-Omni-30B-A3B-{Instruct,Thinking}-GGUF` ship an mmproj with the **real AuT audio encoder** (verified by header parse). Q4_K_M = 18.56 GB + 1.33 GB mmproj. Ollama and LM Studio still **cannot** do audio. |
| 24 GB card | Yes, via **AWQ-4bit Thinking/Captioner (20.5 GB)** or **GGUF Q4_K_M (18.6 GB + 1.3 GB mmproj)**. bitsandbytes 4-bit is **useless** here (see §5.4). |

---

## 1. Model family — what actually exists (Aug 2026)

### 1.1 Open weights (all non-gated, license `other`/apache-2.0)

| Repo | Contents | Params | Repo size |
|---|---|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | thinker + talker + code2wav | 35.26 B | **70.53 GB** |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | thinker only (CoT) | 31.72 B | **63.45 GB** |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | thinker only, audio-caption finetune of Instruct | 31.72 B | **63.45 GB** |

All three: `model_type: qwen3_omni_moe`, `architectures: ["Qwen3OmniMoeForConditionalGeneration"]`, `dtype: bfloat16`, last modified 2025-09-22.

### 1.2 There is **no** other size

HF API `?author=Qwen&search=omni` returns exactly 7 repos: the 3 above + Qwen2.5-Omni-7B/-3B/-7B-AWQ/-7B-GPTQ-Int4. **The smallest open omni model is still Qwen2.5-Omni-3B.** Qwen published **no** official AWQ/GPTQ/FP8/GGUF for Qwen3-Omni.

### 1.3 Closed / API-only relatives (no weights)

| Model | Weights | Context / max in / max out | Audio | Video |
|---|---|---|---|---|
| `qwen3-omni-flash` (2025-09-15, 2025-12-01) | **API only** | 65,536 / 49,152 / 16,384 | 20 min | 150 s |
| `qwen3-omni-flash-realtime` | API only | 8 dialog turns | — | — |
| `qwen3.5-omni-plus` / `-flash` (2026-03-15) | **API only** | 262,144 / 196,608 / 65,536 | 3 h | 1 h |
| `qwen3-omni-30b-a3b-captioner` (hosted) | open | 65,536 / 32,768 / 32,768 | **40 min** | — |

The tech report confirms Flash is in-house: *"two in-house developed variants, designated Qwen3-Omni-Flash-Instruct and Qwen3-Omni-Flash-Thinking"*.
**Qwen3.5-Omni** (arXiv 2604.15804, 2026-04-17) exists and supersedes Qwen3-Omni — *"scales to hundreds of billions of parameters and supports a 256k context length… over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS)"* — but it is **proprietary, API-only**. For local weights **Qwen3-Omni-30B-A3B is still the newest option**.

---

## 2. Architecture

### 2.1 `config.json` structure

```
Qwen3OmniMoeConfig (model_type: qwen3_omni_moe)
├── thinker_config            (qwen3_omni_moe_thinker)
│   ├── audio_config          (qwen3_omni_moe_audio_encoder)   ← AuT
│   ├── vision_config         (qwen3_omni_moe_vision_encoder)
│   └── text_config           (qwen3_omni_moe_text)            ← MoE LLM
├── talker_config             (qwen3_omni_moe_talker)          ← Instruct ONLY
│   ├── text_config           (talker MoE)
│   └── code_predictor_config (qwen3_omni_moe_talker_code_predictor)
└── code2wav_config                                            ← Instruct ONLY
```

**Thinking and Captioner `config.json` contain NO `talker_config` and NO `code2wav_config`, and set `enable_audio_output: false`.** Instruct sets `enable_audio_output: true`.

Top-level ids: `im_start 151644`, `im_end 151645`, `system 8948`, `user 872`, `assistant 77091`, `tts_bos/eos/pad` (Instruct only).

#### Thinker → `text_config` (the 30B-A3B MoE LLM)
```
hidden_size 2048   num_hidden_layers 48   num_attention_heads 32   num_key_value_heads 4   head_dim 128
num_experts 128    num_experts_per_tok 8  moe_intermediate_size 768  shared_expert_intermediate_size 0
decoder_sparse_step 1 (every layer is MoE)   norm_topk_prob true   use_qk_norm true
vocab_size 152064   max_position_embeddings 65536   rope_theta 1e6
rope_scaling: {interleaved: true, mrope_interleaved: true, mrope_section: [24,20,20]}   ← TM-RoPE
tie_word_embeddings false   rms_norm_eps 1e-6
```
→ **≈3.0 B active parameters per token** (8/128 experts + attention + embeddings).

#### Thinker → `audio_config` (AuT audio encoder)
```
model_type qwen3_omni_moe_audio_encoder
d_model 1280   encoder_layers 32   encoder_attention_heads 20   encoder_ffn_dim 5120
num_mel_bins 128   max_source_positions 1500   downsample_hidden_size 480   output_dim 2048
n_window 50        n_window_infer 800        conv_chunksize 500        activation gelu
```
Paper: AuT is an *"attention-encoder-decoder model… trained from scratch on 20 million hours"*, downsamples the filterbank 8× via Conv2D, *"reducing the token rate to 12.5 Hz"*, and uses *"dynamic attention window sizes, covering attention query patterns ranging from 1 to 8 seconds"*, ~0.6B params.
Decoding those constants: `n_window=50` → `chunk_len = 100` mel frames @10 ms hop = **1 s per chunk → 13 tokens**; `n_window_infer=800` → `800//(50*2) = 8` chunks → **8-second attention window at inference** (this is the "1 to 8 seconds"); `conv_chunksize=500` is a *batch* split of the Conv2D stack (`"Split to chunk to avoid OOM during convolution"`) = 500 s of audio per conv batch — **this is the knob to lower if you OOM on long audio**.

#### Thinker → `vision_config`
```
model_type qwen3_omni_moe_vision_encoder   (SigLIP2-So400M lineage)
depth 27   hidden_size 1152   num_heads 16   intermediate_size 4304   hidden_act gelu_pytorch_tanh
patch_size 16   spatial_merge_size 2   temporal_patch_size 2   image_size 768   out_hidden_size 2048
deepstack_visual_indexes [8, 16, 24]   apply_vit_abs_pos_embed true   tokens_per_second 2
```

#### `talker_config` / `code2wav_config` (Instruct only)
Talker: MoE, `hidden_size 1024`, `num_hidden_layers 20`, `num_experts 128`, `num_experts_per_tok 6`, `moe_intermediate_size 384`, `vocab_size 3072`, `num_code_groups 16`, `speaker_id {chelsie: 2301, ethan: 2302, aiden: 2303}`, `accept_hidden_layer 24`, `position_id_per_seconds 13`, `seconds_per_chunk 2`. Plus a 5-layer `code_predictor`.
Code2Wav: causal ConvNet, `codebook_size 2048`, `num_quantizers 16`, `decoder_dim 1536`, `upsample_rates [8,5,4,3]` → **24 kHz** output.

### 2.2 Shard list & exact sizes

| Variant | Shards | Per shard | Total weights | `total_size` in index |
|---|---|---|---|---|
| Instruct | `model-0000{1..15}-of-00015.safetensors` | 14 × 4.998 GB + 0.554 GB | **70.52 GB** | 70,519,637,090 (35,259,818,545 params) |
| Thinking | `model-0000{1..16}-of-00016.safetensors` | 15 × ~4.000 GB + 3.449 GB | **63.44 GB** | 63,438,410,976 (31,719,205,488 params) |
| Captioner | `model-0000{1..16}-of-00016.safetensors` | 15 × ~4.000 GB + 3.462 GB | **63.44 GB** | 63,438,410,976 |

### 2.3 Per-submodule breakdown **[measured from safetensors headers, Instruct]**

| Module | Params | bf16 size | Share |
|---|---:|---:|---:|
| `thinker.model` (48-layer MoE LLM) | 30.221 B | **60.442 GB** | 85.7 % |
| `talker` | 3.325 B | **6.649 GB** | 9.4 % |
| `thinker.audio_tower` (AuT) | 0.648 B | **1.296 GB** | 1.8 % |
| `thinker.visual` | 0.539 B | **1.077 GB** | 1.5 % |
| `thinker.lm_head` | 0.311 B | **0.623 GB** | 0.9 % |
| `code2wav` | 0.216 B | **0.432 GB** | 0.6 % |
| **Total** | **35.260 B** | **70.520 GB** | |
| **Thinker subtotal** | **31.719 B** | **63.438 GB** | |

Within `thinker.model`, the MoE experts are ≈**29.0 B params (96 %)**; attention is only ≈0.91 B. Remember this for §5.4.

### 2.4 What you can drop for captioning

**Drop `talker` + `code2wav` → saves 3.541 B params = 7.081 GB of weights (bf16).**

- On **Thinking / Captioner**: nothing to drop — they already ship thinker-only. `disable_talker()` is a harmless no-op.
- On **Instruct**: `model.disable_talker()` (deletes both submodules after load) or `from_pretrained(..., enable_audio_output=False)` (never builds them), or load `Qwen3OmniMoeThinkerForConditionalGeneration` directly (never even reads the talker shards).
- The README claims *"about 10GB"* saved; the transformers docs claim *"~2GB"*. **Both are wrong for weights: the true figure is 7.08 GB.** The README's 10 GB presumably includes talker KV/activations at runtime.
- **You cannot drop the vision tower for audio-only work** and vice-versa: they are only 1.08 + 1.30 GB combined (3.4 % of the model), and there is no supported switch. Not worth engineering.

### 2.5 Processor / tokenizer files

Official repos ship **only these** (no `tokenizer.json`, no `special_tokens_map.json`, no `added_tokens.json`, no `video_preprocessor_config.json`):

| File | Instruct | Thinking | Captioner |
|---|---|---|---|
| `config.json` | 13,671 B | 8,528 B | 8,528 B |
| `chat_template.json` | 6,772 B | 5,946 B | 5,946 B |
| `generation_config.json` | 146 B | 113 B | 112 B |
| `preprocessor_config.json` | 603 B | 603 B | 603 B |
| `tokenizer_config.json` | 7,344 B | idem | idem |
| `vocab.json` + `merges.txt` | 2.78 MB + 1.67 MB | idem | idem |

`preprocessor_config.json` (identical in all three):
```json
{"feature_extractor_type":"WhisperFeatureExtractor","image_processor_type":"Qwen2VLImageProcessor",
 "processor_class":"Qwen3OmniMoeProcessor","sampling_rate":16000,"feature_size":128,"n_fft":400,
 "hop_length":160,"n_samples":4800000,"nb_max_frames":30000,"dither":0.0,"padding_value":0.0,
 "padding_side":"right","return_attention_mask":true,
 "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
 "patch_size":16,"merge_size":2,"temporal_patch_size":2,
 "min_pixels":3136,"max_pixels":12845056}
```
Tokenizer = `Qwen2Tokenizer` (slow BPE, converted to fast on load), `model_max_length: 131072`, `eos_token: <|im_end|>`, `pad_token: <|endoftext|>`.

Special tokens (all `special: true` unless noted): `<|im_start|> 151644`, `<|im_end|> 151645`, `<|vision_start|> 151652`, `<|vision_end|> 151653`, `<|image_pad|> 151655`, `<|video_pad|> 151656`, `<|audio_start|> 151669`, `<|audio_end|> 151670`, `<|audio_pad|> 151675`, `<tts_pad> 151671`, `<tts_text_bos> 151672`, `<tts_text_eod> 151673`.
**`<think> 151667` and `</think> 151668` have `"special": false`** → they SURVIVE `skip_special_tokens=True`, so you can split on them. (Important, see §7.2.)

### 2.6 `chat_template`

All three templates are functionally identical (ChatML + multimodal placeholders). Key facts:

- **No system prompt is auto-injected.** Unlike Qwen2.5-Omni, if you pass no system message you get none. **[measured]**
- Content items map to: `image → <|vision_start|><|image_pad|><|vision_end|>`, `video → <|vision_start|><|video_pad|><|vision_end|>`, `audio → <|audio_start|><|audio_pad|><|audio_end|>`, `text → raw`.
- `add_generation_prompt=True` emits `<|im_start|>assistant\n`. Passing `enable_thinking=False` additionally emits `<think>\n\n</think>\n\n` to force an empty reasoning block.
- Assistant history: reasoning is stripped from all turns except the last (`{%- if loop.index0 > ns.last_query_index %}`), so past `<think>` blocks are dropped automatically — good for multi-turn.
- Tools/`<tool_call>` supported.

Rendered example **[measured]**:
```
<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>Describe.<|im_end|>\n<|im_start|>assistant\n
```

---

## 3. transformers inference

### 3.1 Class names (all top-level importable from `transformers`)

```python
Qwen3OmniMoeForConditionalGeneration        # thinker + talker + code2wav  (config_class Qwen3OmniMoeConfig)
Qwen3OmniMoeThinkerForConditionalGeneration # THINKER ONLY  ← use this for captioning
Qwen3OmniMoeTalkerForConditionalGeneration
Qwen3OmniMoeProcessor
Qwen3OmniMoeConfig / Qwen3OmniMoeThinkerConfig / Qwen3OmniMoeTalkerConfig / Qwen3OmniMoeCode2WavConfig
Qwen3OmniMoeThinkerTextModel, Qwen3OmniMoeCode2Wav, Qwen3OmniMoeTalkerModel, ...
```

`Qwen3OmniMoeThinkerForConditionalGeneration` has `base_model_prefix = "thinker"`, so `from_pretrained("<any of the three repos>")` loads **only** the `thinker.*` tensors and skips talker/code2wav shards entirely. **Verified [measured]:** `Qwen3OmniMoeThinkerConfig.from_pretrained(<omni repo>)` correctly extracts the nested `thinker_config` (48 layers / 128 experts / hidden 2048 / audio d_model 1280 / vision depth 27). This is the officially documented text-only path in the transformers docs.

`trust_remote_code` is **not** needed — support is native.

### 3.2 `generate()` signature (transformers 5.x) and the **breaking change**

```python
@torch.no_grad()
def generate(self, input_ids=None, speaker="Ethan", use_audio_in_video=False,
             return_audio=None,
             thinker_max_new_tokens=1024,          # ← DEFAULT IS 1024. ALWAYS OVERRIDE.
             thinker_eos_token_id=151645,
             talker_max_new_tokens=4096, talker_do_sample=True, talker_top_k=50,
             talker_top_p=1.0, talker_temperature=0.9, talker_repetition_penalty=1.05,
             **kwargs)
```
Any `thinker_*` / `talker_*` / `token2wav_*` kwarg has its prefix stripped and is forwarded to the corresponding `.generate()`. So `thinker_return_dict_in_generate=True`, `thinker_do_sample`, `thinker_temperature`, `thinker_top_p`, `thinker_top_k`, `thinker_repetition_penalty` all work.

| transformers | no audio | with audio |
|---|---|---|
| **4.57.x** | `return thinker_result, None` (2-tuple) | `return thinker_result, talker_wavs` (`thinker_result` is the dict) |
| **5.0 – 5.16** | `return thinker_result` (**ONE value**) | `return thinker_result.sequences, talker_wavs` (**tensor**, not dict) |

Verified identical in v5.2.0, v5.8.0, v5.12.0, v5.16.1.

> ⚠️ **The official README/model-card snippet is broken on transformers 5.x.**
> `text_ids, audio = model.generate(..., return_audio=False, thinker_return_dict_in_generate=True)` unpacks a single `ModelOutput` → garbage/exception.
> And `text_ids.sequences[...]` fails when audio *is* generated, because 5.x already returns `.sequences`.
> **Correct 5.x usage:**
> ```python
> out = model.generate(**inputs, return_audio=False, thinker_max_new_tokens=8192)  # tensor [B, L]
> # or, if you want scores/dict:
> out = model.generate(**inputs, return_audio=False, thinker_return_dict_in_generate=True)  # ModelOutput → out.sequences
> ```
> The transformers docs show the correct form: `text_ids = model.generate(**inputs, return_audio=False)`.

`disable_talker()` (5.x source):
```python
def disable_talker(self):
    if hasattr(self, "talker"):   del self.talker
    if hasattr(self, "code2wav"): del self.code2wav
    self.has_talker = False
```
It deletes the modules **after** they were loaded, so peak load-time RAM/VRAM is unaffected. Prefer `enable_audio_output=False` at `from_pretrained` (never builds them) or the Thinker class (never reads their shards).

### 3.3 `from_pretrained` options

```python
model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",                       # NOTE: `torch_dtype` is deprecated in 5.x → use `dtype`
    device_map="auto",                  # accelerate; also accepts max_memory={0:"22GiB","cpu":"64GiB"}
    attn_implementation="flash_attention_2",  # or "sdpa" (default) / "eager"
)
```
- `_supports_flash_attn = True`, `_supports_sdpa = True`, `_supports_attention_backend = True`.
- `_no_split_modules` for the thinker: `["Qwen3OmniMoeAudioEncoder", "Qwen3OmniMoeVisionEncoder", "Qwen3OmniMoeThinkerTextDecoderLayer"]` → `device_map="auto"` splits cleanly at decoder-layer granularity.
- **The audio tower works without flash-attn**: when FA is not requested it splits into per-8-second windows and loops SDPA over them. Correct, but a Python loop of `ceil(secs/8) × 32` calls — for 40 min that's ~9,600 SDPA calls. FA2 is strongly preferred for long audio.
- FA2 requires fp16/bf16 and is painful to build on Windows; `sdpa` is a fine fallback for short clips.

### 3.4 `qwen_omni_utils.process_mm_info` (v0.0.9, released 2026-02-10)

```python
def process_mm_info(conversations, use_audio_in_video, return_video_kwargs=False,
                    return_video_metadata=False, image_patch_size=14):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations, return_video_kwargs=..., ...)
    return (audios,) + vision      # (audios, images, videos[, video_kwargs])
```
Module constants (`qwen_omni_utils/v2_5/vision_process.py`):
```
FPS = 2.0            FRAME_FACTOR = 2      FPS_MIN_FRAMES = 4     FPS_MAX_FRAMES = 768
IMAGE_MIN_TOKEN_NUM = 4      IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128    VIDEO_MAX_TOKEN_NUM = 768
MAX_RATIO = 200      SPATIAL_MERGE_SIZE = 2
MODEL_SEQ_LEN = int(os.environ.get('MODEL_SEQ_LEN', 128000))
SAMPLE_RATE = 16000  (audio_process.py)
```
Per-element keys it honours:
- image: `image` / `image_url`, `min_pixels`, `max_pixels`, `resized_height`, `resized_width`
- video: `video` / `video_url`, `fps` **or** `nframes` (mutually exclusive), `max_frames`, `min_pixels`, `max_pixels`, `total_pixels`, `video_start`, `video_end`
- audio: `audio` / `audio_url` (path, `http(s)://`, `file://`, `data:audio;base64,`, or a mono `np.ndarray`), `audio_start`, `audio_end`
- Video backend order: `torchcodec` → `decord` → `torchvision`; override with `FORCE_QWENVL_VIDEO_READER`.
- Video max pixels default: `total_pixels = MODEL_SEQ_LEN * factor² * 0.9`, then `max_pixels = max(min(768*factor², total_pixels/nframes*2), min_pixels*1.05)`.
- ⚠️ Its **default `image_patch_size=14`** is wrong for Qwen3-Omni (which uses 16). Pass `image_patch_size=16` for correct token budgeting.
- `use_audio_in_video=True` **asserts the video has an audio track** (`"Video must has audio track when use_audio_in_video=True"`). Guard silent videos.

**You can skip `qwen-omni-utils` entirely** on transformers ≥5: `processor.apply_chat_template(..., tokenize=True, return_dict=True, load_audio_from_video=True, fps=2)` does the loading natively (needs `torchcodec` for mp4 audio, see §11).

### 3.5 Multi-file / mixed inputs & batching

- **Mixed modalities in one message: fully supported.** Order matters — Qwen's own eval protocol says *"the text should come **after** multimodal data in the sequence"*:
  ```python
  {"role":"user","content":[{"type":"audio",...},{"type":"image",...},{"type":"video",...},
                            {"type":"text","text":"Describe the audio, image and video."}]}
  ```
- **Batching is supported when `return_audio=False`** (README: *"Batch inference does not support returning audio"*). Use the Thinker class or `disable_talker()`. Text padding is **left-padded** (`text_kwargs: {padding: False, padding_side: "left"}`). Verified [measured]: a batch of a 5 s and a 20 s audio yields `input_ids (2, 272)` with 65 and 260 audio tokens respectively.
- `web_demo.py` caps history at `IMAGE_TURN_LIMIT = 1`, `VIDEO_TURN_LIMIT = 5`, `AUDIO_TURN_LIMIT = 5`.
- vLLM: `limit_mm_per_prompt={'image':3,'video':3,'audio':3}` in the README; the Captioner uses `{'audio':1}`.
- **Multi-turn:** `use_audio_in_video` must be set consistently across all steps *"otherwise unexpected results may occur"*.

---

## 4. Media knobs & token budgeting **[all measured]**

### 4.1 Audio — exactly 13 tokens/second

`_get_feat_extract_output_lengths(L, n_window=50) = ... + (L // 100) * 13`, with 100 mel frames = 1 s.

| Duration | Audio tokens |
|---|---|
| 1 s | 13 |
| 30 s | 390 |
| 60 s | 780 |
| 5 min | 3,900 |
| 10 min | 7,800 |
| 30 min | 23,400 |
| **40 min** | **31,200** ← fits 32,768 |
| 60 min | 46,800 |
| 84 min | 65,520 ← fits 65,536 |

**Audio is NOT truncated** — the processor sets `audio_kwargs: {padding: True, truncation: False}`, so `n_samples: 4800000` (300 s) in `preprocessor_config.json` is *not* a cap. Confirmed: 2400 s → `input_features (1, 128, 240000)` → 31,200 tokens.
**The "40 minutes" claim comes from the tech report** (abstract, §1, conclusion): *"The system can process audio recordings up to 40 minutes per instance for ASR and spoken-language understanding"*. It is exactly the 32k context divided by the token rate — not a hard architectural limit. The paper explicitly says TM-RoPE anchoring *"affords the model the flexibility to support streaming inputs of arbitrary duration."*

### 4.2 Video — `tokens/s = (fps / 2) × tokens_per_frame_group`

`video_grid_thw = (T, H/16, W/16)` with `T = nframes / temporal_patch_size(2)`; tokens `= T·(H/16)·(W/16) / merge_size²(4) = T · (H·W / 1024)`.

Measured on 8-second clips at fps=2 (16 frames):

| Source resolution | Resulting grid | Video tokens | tok/s |
|---|---|---|---|
| 640×480 | `[8, 30, 40]` | 2,400 | 300 |
| 1280×720 | `[8, 40, 72]` | 5,760 | **720** (auto-resized) |
| 1920×1080 | `[8, 40, 72]` | 5,760 | **720** (auto-resized) |
| 3840×2160 | `[2, 40, 72]` (4 frames) | 1,440 | 720 |
| 1920×1080 + `size={"longest_edge":256*32*32}` | `[8, 24, 42]` | 2,016 | 252 |

→ **Video frames ARE capped by default at 768 tokens per temporal group** (`videos_kwargs._defaults: size {shortest_edge: 128*32*32, longest_edge: 768*32*32}`). Max ≈768 tok/s at fps=2, ≈384 tok/s at fps=1.

Practical budgets for a 32,768-token window (leaving room for output):

| fps | `size.longest_edge` | tok/s | Max video length |
|---|---|---|---|
| 2 | default `768*32*32` | 768 | ~40 s |
| 2 | `256*32*32` | 256 | ~2 min |
| 1 | `256*32*32` | 128 | ~4 min |
| 1 | `128*32*32` | 64 | ~8 min |

Qwen's own eval protocol: *"all video data are set to `fps=2` during evaluation."* The DashScope Flash API caps video at **150 s**.

### 4.3 Images — **NOT capped by default** (danger)

| Image | Tokens |
|---|---|
| 768×768 | 576 |
| 1920×1080 | **2,040** |
| 3840×2160 | **8,160** |
| 3840×2160 with `max_pixels=1280*32*32` | 1,222 |

Default `max_pixels = 12,845,056` px = **12,544 tokens** for one image. Set a cap.

> ⚠️ **`min_pixels`/`max_pixels` only work at `from_pretrained()` time.** Passing them to `processor(...)` at call time is **silently ignored** (measured: 3840×2160 still produced 8,160 tokens). At call time you must use `size={"shortest_edge": N, "longest_edge": M}` (values in **pixels**, i.e. `tokens * 32 * 32`).

```python
# correct
proc = Qwen3OmniMoeProcessor.from_pretrained(P, min_pixels=4*32*32, max_pixels=1280*32*32)
# or per call
inputs = proc(..., size={"shortest_edge": 4*32*32, "longest_edge": 1280*32*32})
```

### 4.4 `use_audio_in_video` — the fps trap

With `use_audio_in_video=True` the processor **interleaves** video and audio tokens by comparing
`video_token_indices = t · video_second_per_grid · position_id_per_seconds(13)` against `audio_token_indices = arange(n_audio_tokens)`,
where `video_second_per_grid = temporal_patch_size(2) / fps` and **`fps` defaults to `1.0` in the processor**.

Measured on the same 8 s clip sampled at 2 fps:

| passed to `processor()` | `video_second_per_grid` | correct? |
|---|---|---|
| `fps=2.0` | `1.0` | ✅ |
| *(omitted)* | `2.0` | ❌ **silently 2× misaligned** — same token count, wrong ordering |

> ⚠️ **The official README snippet omits `fps` when calling `processor(...)`, while `qwen_omni_utils` samples at `FPS=2.0` by default.** That is a silent 2× audio/video temporal misalignment for every `use_audio_in_video=True` call. For a video captioner this matters.
> **Fix:** pass the actual sampling fps as a **scalar** — `processor(..., fps=2.0, use_audio_in_video=True)`. Do **not** blindly splat `**video_kwargs` from `return_video_kwargs=True`: it returns `fps` as a *list*, and the processor then does `[fps] * len(videos)` → `2 / [2.0]` → `TypeError`. Use `fps=float(video_kwargs["fps"][0])`.

### 4.5 Context length

- Trained/validated: **32,768** (paper: *"we increased the maximum token length from 8,192 to 32,768"*). All official vLLM examples use `max_model_len=32768`.
- `max_position_embeddings`: **65,536** (used by the multi-GPU vLLM serve examples).
- `tokenizer.model_max_length`: 131,072 (not meaningful here).

---

## 5. VRAM / memory

### 5.1 Official table (README, "Minimum GPU memory requirements")

| Model | Precision | 15 s video | 30 s video | 60 s video | 120 s video |
|---|---|---|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct | BF16 | **78.85 GB** | **88.52 GB** | **107.74 GB** | **144.81 GB** |
| Qwen3-Omni-30B-A3B-Thinking | BF16 | **68.74 GB** | **77.79 GB** | **95.76 GB** | **131.65 GB** |

> *"theoretical minimum memory requirements for inference with `transformers` and `BF16` precision, tested with `attn_implementation="flash_attention_2"`. The Instruct model includes both the thinker and talker components, whereas the Thinking model includes only the thinker part."*

Baseline weights are 63.44 / 70.52 GB, so the 15 s row is essentially weights + ~5–8 GB. Everything beyond that is vision-tower activations and KV.

### 5.2 KV cache (derived)

48 layers × 4 KV heads × 128 head_dim × 2 (K,V) × 2 B = **96 KiB per token**:

| Context | KV cache (bf16) |
|---|---|
| 8,192 | 0.75 GiB |
| 32,768 | **3.0 GiB** |
| 65,536 | **6.0 GiB** |

(Quantizing KV to fp8/q8 halves this; llama.cpp `-ctk q8_0 -ctv q8_0`.)

### 5.3 Real-world speed (GitHub issue #46, Qwen dev confirmed)

| Backend | Model | Throughput |
|---|---|---|
| transformers, A100 80 GB | Thinking | **~5.7 tok/s** |
| transformers, A100 80 GB | Instruct | **~1 tok/s** |
| vLLM | Thinking | **609 tok/s in, 87.6 tok/s out** |

Qwen maintainer BakerBunker: *"Transformers inference is slow because it uses a for loop to access the experts."* Confirmed in the source — `Qwen3OmniMoeThinkerTextExperts.forward` loops `for expert_idx in expert_hit`. **For a production captioner, use vLLM or llama.cpp, not raw transformers.**

### 5.4 ⚠️ bitsandbytes 4-bit / 8-bit is effectively **useless** on transformers ≥5

In transformers 5.x the MoE experts are **fused 3-D `nn.Parameter`s**, not `nn.Linear` modules:
```python
class Qwen3OmniMoeThinkerTextExperts(nn.Module):
    self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2*inter, hidden))   # [128, 1536, 2048]
    self.down_proj    = nn.Parameter(torch.empty(num_experts, hidden, inter))     # [128, 2048, 768]
```
and `replace_with_bnb_linear()` only converts `type(module) is nn.Linear`. **The 29 B params of experts (91 % of the thinker) stay in BF16.** You'd save roughly 3.6 GB out of 63.4 GB.

- In **transformers 4.57.x** the experts were `nn.ModuleList` of `nn.Linear` → bnb 4-bit *did* work there (at the cost of the 5.x accuracy/perf improvements Qwen now recommends). Pin 4.57.1 only if you truly need bnb.
- **torchao** (`Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`) filters on `nn.Linear` too → same limitation.
- The checkpoint itself stores per-expert tensors (`thinker.model.layers.N.mlp.experts.{0..127}.{gate,up,down}_proj.weight`, 393 tensors/layer); transformers fuses them on load.

**Use AWQ/GPTQ (compressed-tensors) or GGUF instead — those quantize the fused experts properly.**

### 5.5 Non-gated quantized repos (safetensors)

All verified non-gated. Sizes are total repo bytes.

| Repo | Format | Base | Size | Notes |
|---|---|---|---|---|
| **`cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`** | compressed-tensors INT4 W4A16, group 32, sym, MSE | Instruct | **27.57 GB** | 15.6k dl, 58 likes. **Only the thinker's Linears are quantized**; `audio_tower`, `visual`, `lm_head`, `embed_tokens`, `talker`, `code2wav` stay BF16 (11,563 entries in `ignore`). Best quality/size for captioning. |
| **`cyankiwi/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit`** | same | Thinking | **20.49 GB** | ← **fits a 24 GB card** |
| **`cyankiwi/Qwen3-Omni-30B-A3B-Captioner-AWQ-4bit`** | same | Captioner | **20.49 GB** | ← **fits a 24 GB card** |
| `cyankiwi/…-Instruct-AWQ-8bit` | INT8 | Instruct | 42.52 GB | |
| `cyankiwi/…-Thinking-AWQ-8bit` / `…-Captioner-AWQ-8bit` | INT8 | | 35.44 GB | |
| `thomasip/Qwen3-Omni-30B-A3B-Instruct-GPTQ-4bit` | GPTQModel, bits=4, group 128, C4 calib | Instruct | 26.40 GB | |
| `jart25/Qwen3-Omni-30B-A3B-Instruct-AWQ-W4A16` | llm-compressor W4A16 | Instruct | 27.57 GB | ships `recipe.yaml` |
| `Saktsant/Qwen3-Omni-30B-A3B-Instruct-AWQ` | AWQ | Instruct | 27.57 GB | |
| **`marksverdhei/Qwen3-Omni-30B-A3B-FP8`** | block-wise FP8 E4M3 128×128 | Instruct | **37.39 GB** | **50.7k downloads** — the most-used quant. Thinker + Talker → FP8; **vision encoder, audio tower, code2wav, embeddings, norms, MoE gates stay BF16**. For vLLM. |
| `mohhaamed/Qwen3-Omni-30B-A3B-FP8` | FP8 | Instruct | 37.39 GB | |
| `sammysun0711/Qwen3-Omni-30B-A3B-Instruct-FP8-Dynamic` | llm-compressor FP8-dynamic | Instruct | 37.32 GB | |
| `openaudio/qwen3_omni_fp8_dynamic` | AngelSlim FP8 | Instruct | 33.60 GB | |
| `cybermotaz/Qwen3-Omni-30B-A3B-Instruct-NVFP4` | **NVFP4** (TensorRT-LLM style, split `thinker/` `talker/` `code2wav/` dirs) | Instruct | **27.57 GB** | needs Blackwell + TRT-LLM |
| `vito95311/Qwen3-Omni-30B-A3B-Thinking-INT8FP16` | INT8/FP16 mix | Thinking | 19.31 GB | ships `qwen_ultimate_offloading.py` |
| `ReopenAI/Qwen3-omni-ASR-GPTQ-Int4` | GPTQ Int4, thinker-only ASR | Thinking | 19.52 GB | ships custom `modeling_qwen3_omni_moe_thinker.py` |
| `huihui-ai/Huihui-Qwen3-Omni-30B-A3B-Thinking-abliterated` | BF16 (uncensored) | Thinking | 63.45 GB | |
| `mlx-community/Qwen3-Omni-30B-A3B-Instruct-{4,5,6,8}bit`/`-bf16` | MLX | Instruct | **21.84** / 29 / 33 / 38.78 / 61 GB | keeps audio + vision + talker + code2wav |
| `pherber3/Qwen3-Omni-30B-A3B-Instruct-4bit-mlx` | MLX 4-bit | Instruct | 17.20 GB | |
| `abnormalmapstudio/Qwen3-Omni-30B-A3B-{Instruct,Thinking,Captioner}-mxfp4-mlx` | MLX MXFP4 | | ~16.2 GB | |

There is **no official Qwen FP8/AWQ/NVFP4 repo** — all of the above are community.

### 5.6 Can 24 / 16 / 12 GB cards run it?

| VRAM | Verdict |
|---|---|
| **24 GB (4090/3090)** | ✅ **Yes.** Best: llama.cpp `Q4_K_M` 18.56 GB + `mmproj-Q8_0` 1.33 GB (+~1.5 GB KV @16k) — full audio/video/image. Or AWQ-4bit Thinking/Captioner (20.49 GB) with a short context. |
| **16 GB** | ⚠️ Only with partial offload — llama.cpp `IQ4_XS` (16.56 GB) or `Q3_K_M` (14.71 GB) + mmproj, `-ngl` tuned so some experts sit in RAM. MoE offload is relatively cheap (only 8/128 experts fire per token) but still 3–10× slower. |
| **12 GB** | ⚠️ `Q2_K` (11.26 GB) + 1.33 GB mmproj won't both fit; needs CPU offload for a good chunk. Quality at Q2_K on a captioning task will be visibly worse. Not recommended. |
| **Multi-GPU** | `device_map="auto"` splits at `Qwen3OmniMoeThinkerTextDecoderLayer` / audio / vision boundaries; vLLM `-tp N`. |
| **CPU-offload tricks in transformers** | `device_map="auto", max_memory={0:"22GiB","cpu":"100GiB"}, offload_folder="offload"` works but combines the slowest possible paths (Python expert loop + PCIe). Expect well under 1 tok/s. **Use llama.cpp instead.** |

---

## 6. GGUF / llama.cpp / MLX — **multimodal GGUF works** (as of 2026-04-12)

This reverses the situation that held for the first ~7 months after release.

### 6.1 llama.cpp status

- **PR [ggml-org/llama.cpp#19441](https://github.com/ggml-org/llama.cpp/pull/19441)** "mtmd: qwen3 audio support (qwen3-omni and qwen3-asr)" by **ngxson** — **MERGED 2026-04-12** (commit `21a4933`). PR body: `- [x] qwen3-omni-moe working (vision + audio input)`.
- Docs PR **#21857** merged 2026-04-13. `docs/multimodal.md` now lists, under **Mixed modalities**:
  ```
  # Qwen3 Omni
  # Capabilities: audio input, vision input
  (tool_name) -hf ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF
  (tool_name) -hf ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF
  ```
  *(verified directly against llama.cpp master)*
- Conversion classes live in `conversion/qwen3vl.py` (the old `convert_hf_to_gguf.py` was split into a package — **grepping that file alone gives a false negative**):
  `@ModelBase.register("Qwen3OmniMoeForConditionalGeneration") class Qwen3OmniMmprojModel(Qwen3VLVisionModel, Qwen25AudioModel): has_audio_encoder = True; has_vision_encoder = True` and `class Qwen3OmniMoeTextModel(Qwen3VLMoeTextModel)` (`MODEL_ARCH.QWEN3VLMOE`).
- Runtime: `tools/mtmd/models/qwen3a.cpp`, `PROJECTOR_TYPE_QWEN3A`, `clip_graph_qwen3a`, `mtmd_audio_preprocessor_qwen3a`; video via ffmpeg (`--video`, `mtmd_helper_support_video()`); `llama-server` accepts OpenAI `input_audio` / `input_video` parts.
- Earlier PRs #18404, #18420, #18501, #19077 were all **closed unmerged** — repos built on them are incompatible.
- Requires a build ≥ **b8775 (2026-04-13)**. Known open issue [#27136](https://github.com/ggml-org/llama.cpp/issues/27136) (2026-08-15, `bug-unconfirmed`, 0 comments) reports audio failing with quantized KV cache (`-ctk q4_0 -ctv q4_0`) — avoid quantized KV until confirmed.

### 6.2 Official GGUF repos (non-gated) — **header-verified by me**

**`ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`** (created 2026-04-13, 45,685 downloads)

| File | Size |
|---|---|
| `Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf` | **18.557 GB** |
| `Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf` | 32.484 GB |
| `Qwen3-Omni-30B-A3B-Instruct-bf16.gguf` | 61.097 GB |
| `mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf` | **1.325 GB** |
| `mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf` | 2.207 GB |

**`ggml-org/Qwen3-Omni-30B-A3B-Thinking-GGUF`** (35,971 downloads) — identical file set and sizes.

**I range-read the mmproj header myself.** `mmproj-…-Instruct-Q8_0.gguf`: 860 tensors, 33 KV pairs:
```
general.architecture = clip      general.type = mmproj
clip.has_audio_encoder  = True   clip.audio.projector_type  = qwen3a
clip.has_vision_encoder = True   clip.vision.projector_type = qwen3vl_merger
clip.audio.block_count = 32   embedding_length = 1280   feed_forward_length = 5120
clip.audio.attention.head_count = 20   num_mel_bins = 128   projection_dim = 2048
clip.vision.block_count = 27  embedding_length = 1152   head_count = 16   patch_size = 16
tensor prefixes: a.* = 522 (audio tower), v.* = 330 (vision), mm.* = 8 (projectors)
```
This is a **1:1 match** with `thinker_config.audio_config` in the HF repo — **the real AuT encoder is in the GGUF, not a stub.** Audio captioning via GGUF is genuinely functional.

Text GGUF: `general.architecture = qwen3vlmoe`, 48 blocks, 128 experts / 8 active, ctx 65536.
**Limitation:** llama.cpp carries the **thinker only** — no talker/code2wav, so no speech output. (Confirmed by size: Instruct bf16 GGUF = 61.097 GB = same as Thinking, vs 70.5 GB for HF Instruct.) Irrelevant for captioning.

### 6.3 Captioner GGUF

| Repo | Files |
|---|---|
| **`mradermacher/Qwen3-Omni-30B-A3B-Captioner-GGUF`** | Q2_K 11.259 · Q3_K_S 13.293 · Q3_K_M 14.712 · Q3_K_L 15.901 · IQ4_XS 16.557 · Q4_K_S 17.456 · **Q4_K_M 18.557** · Q5_K_S 21.081 · Q5_K_M 21.726 · Q6_K 25.093 · Q8_0 32.484 GB · **`mmproj-Q8_0.gguf` 1.325 GB** · `mmproj-f16.gguf` 2.204 GB |
| `presencesw/Qwen3-Omni-30B-A3B-Captioner-GGUF` | Q4_K_M 18.557 GB + `mmproj-…-BF16.gguf` 2.207 GB (header-verified: `has_audio_encoder=True`, `qwen3a`, 860 tensors) |

⚠️ mradermacher's **`-i1-` (imatrix)** repos ship **no mmproj** → text only.

### 6.4 The older third-party GGUFs — all obsolete/broken/text-only

| Repo | mmproj | Reality |
|---|---|---|
| `TrevorJS/Qwen3-Omni-30B-A3B-GGUF` | yes (2.388 GB f16) + `talker-f16` 7.087 GB | Built for his own unmerged fork (dead PRs #18404/#18420). **Incompatible with upstream** — predates the `qwen3a` layout. |
| `vito95311/…-Thinking-GGUF-INT8FP16` | no | 2 × 32.718 GB files (identical size — suspect). README (zh) admits multimodal "may not yet be natively supported". **Text only.** |
| `giangndm/qwen3-30b-omni-text-only-Q4_K_M-GGUF` | no | 18.557 GB. **Text only** (says so in the name). |
| `giangndm/qwen3-30b-omni-text-only` | n/a | safetensors, `model_type: qwen3_moe` — multimodality deliberately stripped. |
| `giangndm/qwen3-30b-omni-audio-encoder` | no | Raw AuT in safetensors (1.296 GB), **not a GGUF**. |
| `rodial/Qwen3-Omni-30B-A3B-Thinking-GGUF` | no | 61.097 GB F16. README is a verbatim copy of Qwen's card touting omni capability — **misleading. Text only.** |
| `phnxsystms/Qwen3-Omni-30B-A3B-Instruct-GGUF` | "yes" but **32.484 GB** (README claims 2.3 GB) | **Wrong file uploaded**; needs his dead fork. |
| `PatrickScully/Qwen3-Omni-30B-A3B-Instruct-GGUF` | — | **Empty repo**, only `.gitattributes` + a stub README. |
| `mradermacher/Melinoe-Qwen3Omni-30B-A3B-Thinking-GGUF` | no (`skip_mmproj`) | 11 quants of a finetune-of-a-finetune. **Text only.** |

### 6.5 Runtime support matrix

| Runtime | Text | Image | **Audio** | **Video** | Notes |
|---|:--:|:--:|:--:|:--:|---|
| **llama.cpp** (`llama-mtmd-cli`, `llama-server`) | ✅ | ✅ | ✅ | ✅ | build ≥ b8775; needs ffmpeg on PATH for video |
| **Ollama** | ⚠️ 3rd-party text-only GGUFs | ❌ | ❌ | ❌ | `ollama.com/library/qwen3-omni` → **404**; [ollama#12376](https://github.com/ollama/ollama/issues/12376) open since 2025-09-23. Ollama runs its own engine — llama.cpp's mtmd work does not transfer. |
| **LM Studio** | ✅ | probably ✅ | ❌ | ❌ | [lms#543](https://github.com/lmstudio-ai/lms/issues/543) "Adding Audio functionality" open since 2026-04-29, no maintainer reply — **LM Studio has no audio-input path at all**. |
| **vLLM / vllm-omni** | ✅ | ✅ | ✅ | ✅ | recommended for throughput; see §6.6 |
| **MLX (mlx-vlm ≥0.7.0rc0)** | ✅ | ✅ | ✅ | ✅ | `mlx_vlm/models/qwen3_omni_moe/` has `audio.py`/`vision.py`/`thinker.py`/`talker.py`/`code2wav.py`. `mlx-community/…-Instruct-4bit` (21.84 GB) retains audio+vision+**talker+code2wav** — more complete than llama.cpp. CLI: `mlx_vlm.generate --model <repo> --audio a.wav --prompt "..."`. ⚠️ LM Studio's MLX backend can't load it ([mlx-engine#243](https://github.com/lmstudio-ai/mlx-engine/issues/243), open). |

Working llama.cpp commands:
```bash
llama-mtmd-cli -hf ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF \
  --audio clip.wav -p "Give the detailed description of the audio."
llama-mtmd-cli -hf ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF --video clip.mp4 -p "Describe the video."
llama-server  -hf ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF -c 32768 --jinja   # OpenAI input_audio / input_video
```

### 6.6 vLLM

README now points at **vLLM-Omni** (`pip install vllm; pip install qwen-omni-utils -U`) with [offline](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_omni/) and online docs. Three stages: Thinker → Talker → Code2Wav; `--modalities text` skips audio generation.
Baseline `LLM(...)` args from the README:
```python
LLM(model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
    tensor_parallel_size=torch.cuda.device_count(),
    limit_mm_per_prompt={'image':3,'video':3,'audio':3},
    max_num_seqs=8, max_model_len=32768, seed=1234)
```
`vllm serve` supports **thinker only**, and **`use_audio_in_video` is not available in `vllm serve`** — *"you can handle this by separately passing video and audio inputs for processing."*
Known issue: [Qwen3-Omni#152](https://github.com/QwenLM/Qwen3-Omni/issues/152) — vLLM hangs on audio longer than 15 minutes.

---

## 7. Thinking vs Instruct vs Captioner

### 7.1 Differences

| | **Instruct** | **Thinking** | **Captioner** |
|---|---|---|---|
| Components | thinker + talker + code2wav | thinker | thinker |
| Outputs | text **+ 24 kHz speech** (3 voices: Ethan/Chelsie/Aiden) | text only | text only |
| Inputs | text, image, audio, video | text, image, audio, video | **audio only, exactly one clip, single-turn** |
| Reasoning | none | `<think>…</think>` CoT | none |
| `generation_config.json` | `talker_*` only → thinker defaults to **greedy** | `max_new_tokens 32768, temperature 0.6, top_p 0.95, top_k 20, repetition_penalty 1.0` | same as Thinking |
| Size | 70.53 GB | 63.45 GB | 63.45 GB |
| Best for | video captioning, ASR, VQA, OCR, general | hard reasoning over media, math/AV-QA | maximally detailed audio captions |

Captioner is *"a downstream audio fine-grained caption model fine-tuned from Qwen3-Omni-30B-A3B-Instruct"*.

### 7.2 Thinking output format

Raw generated text (verified in the cookbooks' recorded outputs):
```
<think>
Okay, so I need to figure out ... reasoning ...
</think>

To solve for \( J(0) \), we follow these steps: ...
```
`<think>` **is generated by the model**, not prefilled by the template (unless you pass `enable_thinking=False`, which prefills an *empty* `<think>\n\n</think>\n\n`).

Because token ids 151667/151668 have `"special": false`, they survive `skip_special_tokens=True`. Split like this:

```python
def split_thinking(text: str):
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()
```
The chat template also parses `message.reasoning_content` or splits on `</think>` for history, and drops reasoning from all but the last assistant turn.

### 7.3 Captioner specifics

From the model card:
> *"Without requiring any additional prompting, the model can automatically parse and describe various types of audio content, ranging from complex speech and environmental sounds to music and cinematic sound effects…"*
> *"**Note**: Qwen3-Omni-30B-A3B-Captioner is a single-turn model that accepts only one audio input per inference. It does not accept any text prompts and supports **audio input only**, with **text output only**. As Qwen3-Omni-30B-A3B-Captioner is designed for generating fine-grained descriptions of audio, excessively long audio clips may diminish detail perception. We recommend, as a best practice, **limiting audio length to no more than 30 seconds**."*

**Prompt = the empty prompt.** The message is literally just the audio:
```python
messages = [{"role": "user", "content": [{"type": "audio", "audio": audio_path}]}]
```

Example output (`cookbooks/omni_captioner.ipynb`, cinematic SFX clip):
> *"The audio clip is a meticulously crafted, high-fidelity, 24-second soundscape designed to evoke a cinematic sense of imminent threat, danger, and dramatic tension. It opens with a single, sharp inhalation… Around the 9-second mark, the music intensifies dramatically: a powerful, low-frequency roar (evocative of a massive engine or an approaching natural disaster) erupts… Throughout, the audio remains pristine: there is no environmental noise, no background voices, and no natural reverberation… Culturally and stylistically, the clip is rooted in Western action, thriller, and sci-fi genres…"*

Another (Mandarin speech): it transcribes the speech verbatim inline, identifies the accent (*"standard Putonghua… retroflex 'r'… neutral, Northern accent"*), describes recording conditions, and summarizes intent.

**Note the Instruct model is nearly as good at this**, with a prompt — `audio_caption.ipynb` runs on **Instruct** and produces comparably rich captions. So for a mixed app, Instruct alone can cover audio + video + image captioning; add Captioner only if you want the maximum-detail audio path.

---

## 8. PROMPTS — verbatim from cookbooks / README

### 8.1 System prompts

**(a) The recommended audio-visual interaction system prompt** (README, "Prompt for Audio-Visual Interaction"). Use this when the audio *is the query* and you want a conversational, speakable answer. **Do NOT use it for captioning** — it explicitly forbids describing video content.

```python
user_system_prompt = "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen."
message = {
    "role": "system",
    "content": [
        {"type": "text", "text": f"{user_system_prompt} You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face."}
    ]
}
```
`web_demo.py` uses exactly this string as `default_system_prompt` **only when `--generate-audio` is set with the transformers backend**; otherwise `default_system_prompt = ''`.

**(b) For captioning / benchmarks — use NO system prompt.** README, "Setting for Evaluation": *"**System Prompt**: No `system prompt` should be set for any evaluation benchmark."*

**(c) Speech-output prompt (transformers docs only):** if you *do* want audio out, the docs say the system prompt must be exactly:
`"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."`

**(d) Persona example** (`audio_visual_interaction.ipynb`, character styling):
```
You are a romantic and artistic AI, skilled at using metaphors and personification in your responses, deeply romantic, and prone to spontaneously reciting poetry.
You are a voice assistant with specific characteristics. 
Interact with users using brief, straightforward language, maintaining a natural tone.
...
```
and a one-liner: `"你是一个北京大爷，说话很幽默，说这地道北京话。"`

### 8.2 Task prompts (all verbatim, all as `{"type":"text","text": ...}` **after** the media)

| Task | Prompt | Source |
|---|---|---|
| **Audio captioning** | `Give the detailed description of the audio.` | `audio_caption.ipynb` |
| Audio captioning (var.) | `Please provide a detailed description of the audio.` | `audio_caption.ipynb` |
| Audio captioning (var.) | `Give a thorough description of the audio.` | `audio_caption.ipynb` |
| **Prompt-free audio caption** | *(no text item at all — Captioner model)* | `omni_captioner.ipynb` |
| **ASR — Chinese** | `请将这段中文语音转换为纯文本。` | README eval + `speech_recognition.ipynb` |
| **ASR — other languages** | `Transcribe the <source_language> audio into text.` | README eval |
| ASR — English | `Transcribe the English audio into text.` | `speech_recognition.ipynb` |
| ASR — French | `Transcribe the French audio into text.` | `speech_recognition.ipynb` |
| **Speech translation (S2TT)** | `Listen to the provided <source_language> speech and produce a translation in <target_language> text.` | README eval |
| S2TT example | `Listen to the provided Chinese speech and produce a translation in English text.` | `speech_translation.ipynb` |
| S2TT example | `Listen to the provided English speech and produce a translation in Chinese text.` | `speech_translation.ipynb` |
| **Lyrics transcription** | `Transcribe the song lyrics into text without any punctuation, separate lines with line breaks, and output only the lyrics without additional explanations.` | README eval |
| **Sound event / SFX** | `What happened in the audio?` | `sound_analysis.ipynb` |
| Sound event / SFX | `What is this sound? In what kind of situation might it occur?` | `sound_analysis.ipynb` |
| Scene inference from sound | `Guess where I am?` | `sound_analysis.ipynb` |
| **Mixed audio analysis** | `Determine which sound effects and musical instruments are present in the audio.` | `mixed_audio_analysis.ipynb` |
| Mixed audio (zh) | `判断说话人的国籍和性别，并告诉我音频里出现的音效是什么？` | `mixed_audio_analysis.ipynb` |
| **Music analysis** | `Describe the style, rhythm, dynamics, and expressed emotions of this piece of music. Identify the instruments used and suggest possible scenarios from which this music might originate.` | `music_analysis.ipynb` |
| Music appreciation | `Write an appreciative description for this piece of music. Identifying its style and genre. Analyze the collaborative patterns of different instruments in audio and explain their impact on the overall atmosphere.` | `music_analysis.ipynb` |
| Music style (zh) | `请分析这是什么风格的音乐？` | `music_analysis.ipynb` |
| **Video description** | `Describe the video.` | `video_description.ipynb` |
| **Video scene transitions / chapters** | `How the scenes in the video change?` | `video_scene_transition.ipynb` |
| Video navigation | `If I want to stop at the window. Which direction should I take?` | `video_navigation.ipynb` |
| **Audio-visual QA** | `What was the first sentence the boy said when he met the girl?` | `audio_visual_question.ipynb` |
| Audio-visual MCQ | `Question: <q>\nChoices: ["A. …", "B. …", "C. …", "D. …"]\nPlease give your answer.` | `audio_visual_question.ipynb` |
| **OCR** | `Extract the text from the image.` | `ocr.ipynb` |
| OCR (zh) | `请提取图片中的文字。` | `ocr.ipynb` |
| **Object grounding** | `Locate the object: bird.` / `Locate the object: A person riding a motorcycle while wearing a helmet.` | `object_grounding.ipynb` |
| Image QA | `What style does this image depict?` / `Based on this image, what do you think will happen next?` | `image_question.ipynb` |
| Multimodal joint | `Analyze this audio, image, and video together.` | README, "Best Practices for the Thinking Model" |
| Multimodal joint (eval) | `Describe the audio, image and video.` | README eval |

### 8.3 Gaps — no official prompt exists for these

Qwen ships **no** official prompt for **chapters/summary** or **structured JSON output**. The closest official artifacts are the scene-transition prompt above and the tool-calling template. If you need JSON, use the standard pattern (put the schema in the system message, keep the media first and the instruction last) and drive it with the model's function-calling tokens (`<tool_call>`, id 151657) if you need hard guarantees. `audio_function_call.ipynb` shows the tool-calling system prompt shape:
```
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{'type': 'function', 'function': {'name': 'web_search', 'description': '...', 'parameters': {...}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
```

### 8.4 Best practice for the Thinking model (README, verbatim)

> *"To achieve optimal performance, we recommend that users include an explicit textual instruction or task description in each round of dialogue alongside the multimodal input. This helps clarify the intent and significantly enhances the model's ability to leverage its reasoning capabilities."*

---

## 9. Recommended generation parameters

### 9.1 Official defaults

| Source | Params |
|---|---|
| `generation_config.json` — **Thinking & Captioner** | `max_new_tokens 32768, temperature 0.6, top_p 0.95, top_k 20, repetition_penalty 1.0` |
| `generation_config.json` — **Instruct** | only `talker_*` keys (`talker_max_new_tokens 4096, talker_temperature 0.9, talker_top_k 50, talker_top_p 1.0, talker_repetition_penalty 1.05`) → the thinker falls back to `do_sample=False` (**greedy**) |
| README eval protocol | *"`Instruct` models use greedy decoding during generation without sampling. For `Thinking` models, the decoding parameters should be taken from the `generation_config.json`."* |

### 9.2 What the code actually uses

| File | Backend | Params |
|---|---|---|
| All cookbooks except captioner (Instruct/Thinking) | vLLM | `SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)` (≈greedy) |
| " | transformers | `thinker_do_sample=False, thinker_max_new_tokens=8192` |
| `omni_captioner.ipynb` | vLLM | `SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192)` |
| " | transformers | `thinker_do_sample=True, thinker_temperature=0.6, thinker_top_p=0.95, thinker_top_k=20, thinker_max_new_tokens=8192` |
| `web_demo.py` | transformers | `thinker_max_new_tokens=32768, thinker_do_sample=True`, UI defaults `temperature 0.6, top_p 0.95, top_k 20` |
| " | vLLM | `max_tokens=16384` |
| `web_demo_captioner.py` | both | `max_tokens/thinker_max_new_tokens=32768`, `temperature 0.6, top_p 0.95, top_k 20` |
| README vLLM quickstart | vLLM | `SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384)` |
| vllm-omni docs | vLLM | thinker `temperature=0.9, top_p=0.9, top_k=-1, max_tokens=1200, repetition_penalty=1.05` |

### 9.3 Recommended for a captioning app

| Variant | Sampling | `max_new_tokens` |
|---|---|---|
| **Instruct** (video/audio/image captioning, OCR, ASR) | greedy: `do_sample=False` (transformers) / `temperature=0, top_p=1, top_k=1` (vLLM) | **4096–8192** |
| **Captioner** (audio) | `temperature=0.6, top_p=0.95, top_k=20, do_sample=True` | **2048–8192** (captions run 300–600 tokens) |
| **Thinking** | `temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0` | **16384–32768** — reasoning eats most of it; 8192 truncates on hard inputs |

> ⚠️ **`thinker_max_new_tokens` defaults to 1024** in `Qwen3OmniMoeForConditionalGeneration.generate()`. Always set it explicitly. (When you use `Qwen3OmniMoeThinkerForConditionalGeneration` directly, the repo's `generation_config.json` applies and `max_new_tokens=32768` for Thinking/Captioner.)
> Also: `repetition_penalty` > 1 tends to damage verbatim ASR — keep it at 1.0 for transcription.

---

## 10. Copy-paste-ready minimal snippets (thinker only, no talker)

Requirements:
```bash
pip install "transformers>=5.2.0" accelerate
pip install qwen-omni-utils -U          # optional; needs ffmpeg on PATH
pip install torchcodec                  # needed if you use the NATIVE apply_chat_template video/audio path
pip install -U flash-attn --no-build-isolation   # optional but strongly recommended (Linux)
```

### 10.1 Shared loader

```python
# qwen3omni_captioner.py
import torch
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"     # or -Thinking / -Captioner

processor = Qwen3OmniMoeProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=4 * 32 * 32,          # 4 visual tokens minimum
    max_pixels=1280 * 32 * 32,       # cap ONE image at 1280 tokens (must be set here, not at call time)
)

# Thinker only: talker + code2wav shards are never read -> saves 7.08 GB VRAM/disk-read on Instruct.
model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",                                     # NOT torch_dtype (deprecated in 5.x)
    device_map="auto",
    attn_implementation="flash_attention_2",          # fall back to "sdpa" if flash-attn is unavailable
).eval()


@torch.no_grad()
def run(conversation, *, fps=None, use_audio_in_video=False,
        max_new_tokens=8192, do_sample=False, **gen_kwargs):
    """Returns the decoded assistant text (reasoning included for the Thinking model)."""
    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video, image_patch_size=16
    )

    proc_kwargs = dict(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True,
                       use_audio_in_video=use_audio_in_video)
    if fps is not None:
        proc_kwargs["fps"] = float(fps)   # MUST be a scalar and MUST match the sampling fps

    inputs = processor(**proc_kwargs).to(model.device).to(model.dtype)

    # NOTE: this is the THINKER class -> plain HF generate(), single return value.
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, **gen_kwargs)
    return processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


def split_thinking(text):
    """Thinking model only: separate <think>...</think> from the final answer."""
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()
```

### 10.2 Video **with** audio

```python
VIDEO = "/path/to/clip.mp4"
FPS = 2.0     # qwen-omni-utils default; must be echoed to the processor

conversation = [{
    "role": "user",
    "content": [
        {"type": "video", "video": VIDEO,
         "fps": FPS,
         # keep the token budget sane; 256 tokens/frame-group ≈ 256 tok/s at fps=2
         "max_pixels": 256 * 32 * 32,
         # optional trimming:
         # "video_start": 0.0, "video_end": 60.0,
        },
        {"type": "text", "text": "Describe the video."},   # text goes AFTER the media
    ],
}]

print(run(conversation, fps=FPS, use_audio_in_video=True, max_new_tokens=4096))
```
Video **without** its audio track (or a silent video): pass `use_audio_in_video=False` and omit `fps` matching concerns — but still pass `fps=FPS` so the grid metadata is right.

### 10.3 Audio file

```python
conversation = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": "/path/to/clip.wav"},   # or http(s)://, file://, data:audio;base64,, or np.ndarray
        {"type": "text", "text": "Give the detailed description of the audio."},
    ],
}]
print(run(conversation, max_new_tokens=4096))
```

**Captioner variant (prompt-free, audio only, ≤30 s):**
```python
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
conversation = [{"role": "user", "content": [{"type": "audio", "audio": "/path/to/clip.wav"}]}]
print(run(conversation, max_new_tokens=2048, do_sample=True,
          temperature=0.6, top_p=0.95, top_k=20))
```

### 10.4 Image

```python
conversation = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/image.jpg", "max_pixels": 1280 * 32 * 32},
        {"type": "text", "text": "Describe this image in detail."},
    ],
}]
print(run(conversation, max_new_tokens=2048))
```

### 10.5 Batch (mixed modalities), thinker only

```python
from qwen_omni_utils import process_mm_info

conversations = [conv_image, conv_audio, conv_text, conv_mixed]
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=False, image_patch_size=16)
inputs = processor(text=text, audio=audios, images=images, videos=videos,
                   return_tensors="pt", padding=True, use_audio_in_video=False
                   ).to(model.device).to(model.dtype)
out_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
print(processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### 10.6 If you must use `Qwen3OmniMoeForConditionalGeneration` (full class)

```python
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto",
    attn_implementation="flash_attention_2",
    enable_audio_output=False,          # never builds talker/code2wav (better than disable_talker())
)
# transformers 5.x: ONE return value when audio is off
out_ids = model.generate(**inputs, return_audio=False,
                         thinker_max_new_tokens=8192, thinker_do_sample=False,
                         use_audio_in_video=True)
text = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
```

### 10.7 Native path (no `qwen-omni-utils`; requires `torchcodec`)

```python
inputs = processor.apply_chat_template(
    conversation,
    load_audio_from_video=True,      # pulls the video's audio track
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    processor_kwargs={"fps": 2, "padding": True, "use_audio_in_video": True},
).to(model.device)
```
(In transformers ≥5.14 processor kwargs must go in `processor_kwargs=`, otherwise you get
`Kwargs passed to processor.__call__ have to be in processor_kwargs dict, not in **kwargs`.)

### 10.8 Files to vendor

**None.** There is no `trust_remote_code`, no custom modeling file, no custom processor in the official repos. Everything is native transformers. Optional pip deps only:

| Dep | Why |
|---|---|
| `qwen-omni-utils>=0.0.9` | convenient loader (URLs, base64, video trimming, fps/frame budgeting). Pure Python, 2 files, ~10 KB — you can vendor `audio_process.py` + `vision_process.py` if you want to fix the `image_patch_size=14` default. |
| `torchcodec` | required by transformers' native `load_video`/`load_audio` for mp4 (torchvision ≥0.26 removed `torchvision.io.read_video`) |
| `av`, `librosa`, `audioread`, `soundfile` | qwen-omni-utils' decoders |
| `ffmpeg` on PATH | both paths, and llama.cpp video |
| `accelerate` | `device_map` |
| `flash-attn` | recommended; Linux only in practice |
| `compressed-tensors` | only if loading the AWQ/W4A16 repos in transformers |

---

## 11. Gotcha checklist (things that will silently bite)

1. **`generate()` return shape changed in transformers 5.x.** `text_ids, audio = model.generate(..., return_audio=False)` from the official README is broken. Use one return value. And `text_ids.sequences` fails when audio *is* on (5.x already returns `.sequences`).
2. **`thinker_max_new_tokens` defaults to 1024.** Captions get truncated mid-sentence if you forget.
3. **`fps` must be passed to `processor()` as a scalar and must match the sampling fps**, or `use_audio_in_video=True` misaligns audio and video by 2× — silently, with the same token count. Don't splat `**video_kwargs` (it gives a list → `TypeError`).
4. **`min_pixels`/`max_pixels` are ignored at call time.** Set them at `from_pretrained()`, or use `size={"shortest_edge":…,"longest_edge":…}` per call.
5. **Images are uncapped by default** (up to 12,544 tokens each); video *is* capped at 768 tokens/frame-group.
6. **`qwen_omni_utils.process_mm_info` defaults `image_patch_size=14`**, but Qwen3-Omni uses 16. Pass 16.
7. **`use_audio_in_video=True` asserts the video has an audio track** — silent videos raise.
8. **transformers is ~1–6 tok/s** because the MoE experts run in a Python loop. Use vLLM / llama.cpp for production.
9. **bitsandbytes 4/8-bit saves almost nothing** on transformers ≥5 (fused expert `nn.Parameter`s). Use AWQ/GPTQ/GGUF/FP8.
10. **torchvision ≥0.26 removed `torchvision.io.read_video`** → transformers' default video backend crashes; install `torchcodec` or force `decord`/`pyav`.
11. **`<think>`/`</think>` are NOT special tokens** (`special: false`) → they survive `skip_special_tokens=True`. Good — split on them.
12. **No system prompt is auto-injected.** For captioning/eval, use none. For speech output, the exact "You are Qwen, a virtual human…" string is required.
13. **Captioner ignores text prompts entirely** and is single-turn, one audio; keep clips ≤30 s for best detail.
14. **The HF model cards are stale** (Sept 2025: `pip install git+github.com/huggingface/transformers`, `VLLM_USE_V1='0'`, `wangxiongts/vllm` fork). The GitHub README (updated 2026-04-23) is the current one, but its code snippet was not updated for the 5.x return change.
15. **vLLM hangs on audio > 15 min** ([Qwen3-Omni#152](https://github.com/QwenLM/Qwen3-Omni/issues/152)); `vllm serve` has no `use_audio_in_video`.
16. **llama.cpp**: use a build ≥ b8775, and avoid quantized KV cache with audio until [#27136](https://github.com/ggml-org/llama.cpp/issues/27136) is confirmed fixed.
