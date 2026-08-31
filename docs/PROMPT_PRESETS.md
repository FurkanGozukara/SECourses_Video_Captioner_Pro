# Prompt and Task Presets

Prompt presets live in `vcap.prompts.presets`. They describe the prompt, compatible model families and media, generation overrides, output contract, and optional post-processor as one immutable record. Shipped universal setting presets in `presets_default/` select these records by `prompt_preset_id`; user presets may override the rendered system or user text without changing the registry.

Model abbreviations in the table are **TC** (TimeChat), **AVo** (AVoCaDO), **Q3-I** (Qwen3-Omni Instruct), **Q3-T** (Qwen3-Omni Thinking), and **Q3-C** (Qwen3-Omni Captioner). AVoCaDO entries outside its model-native group are exposed for video workflows but are marked in the UI as potentially outside its training distribution. TimeChat is intentionally restricted to its unchanged native prompt, and Q3-C accepts only its prompt-free audio preset.

## Preset Registry

| Group | ID | Label | Models | Modalities | Output |
|---|---|---|---|---|---|
| Training captions | `wan22_t2v_dense` | Wan 2.2 T2V — dense paragraph | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `wan_t2v_sparse` | Wan T2V — sparse motion line | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `wan_i2v_motion_only` | Wan I2V — motion only | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `hunyuan_dense_cinematic` | Hunyuan — dense cinematic | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `ltx25_short_physical` | LTX 2.5 — short physical action | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `minimax_h3_performance_sound` | MiniMax H3 — performance and sound | Q3-I, Q3-T, AVo | video+audio | text |
| Training captions | `character_lora` | Character LoRA — trigger first | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `motion_lora` | Motion LoRA — trigger then movement | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `style_lora` | Style LoRA — visual treatment | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `no_speech_visual` | Visual only — ignore speech | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `screen_text_include` | Screen text — OCR aware | Q3-I, Q3-T, AVo | video, video+audio | text |
| Training captions | `image_dense_caption` | Image — dense caption | Q3-I, Q3-T | image | text |
| Training captions | `image_short_caption` | Image — short caption | Q3-I, Q3-T | image | text |
| Model-native | `timechat_6d_raw` | TimeChat — 6D raw JSON | TC | video+audio | timestamped JSON |
| Model-native | `timechat_flatten_wan` | TimeChat — Wan motion paragraph | TC | video+audio | text |
| Model-native | `timechat_to_srt` | TimeChat — detailed events to SRT | TC | video+audio | SRT segments |
| Model-native | `avocado_av_aligned` | AVoCaDO — aligned audiovisual | AVo | video+audio | text |
| Model-native | `avocado_visual_only` | AVoCaDO — visual only | AVo | video | text |
| Model-native | `avocado_structured_ugc` | AVoCaDO — structured UGC paragraph | AVo | video+audio | text |
| Model-native | `avocado_dialogue_extract` | AVoCaDO — dialogue extraction post-prompt | AVo | text | lines |
| Model-native | `qwen3_video_describe` | Qwen3-Omni — describe video | Q3-I, Q3-T | video, video+audio | text |
| Model-native | `qwen3_video_dense` | Qwen3-Omni — dense audiovisual video | Q3-I, Q3-T | video+audio | text |
| Model-native | `qwen3_scene_changes` | Qwen3-Omni — scene changes | Q3-I, Q3-T | video, video+audio | lines |
| Model-native | `qwen3_audio_caption` | Qwen3-Omni — detailed audio | Q3-I, Q3-T | audio | text |
| Model-native | `qwen3_captioner_promptfree` | Qwen3-Omni Captioner — prompt free | Q3-C | audio | text |
| Model-native | `qwen3_thinking_dense` | Qwen3-Omni Thinking — dense evidence | Q3-T | video+audio | text |
| Model-native | `qwen3_joint_describe` | Qwen3-Omni — joint media description | Q3-I, Q3-T | audio, image, video+audio | text |
| Model-native | `qwen3_ocr` | Qwen3-Omni — image OCR | Q3-I, Q3-T | image | text |
| Model-native | `qwen3_image_describe` | Qwen3-Omni — describe image | Q3-I, Q3-T | image | text |
| Audio | `audio_sfx_bed` | Audio — SFX and ambience bed | Q3-I, Q3-T | audio, video+audio | text |
| Audio | `sound_events` | Audio — sound events | Q3-I, Q3-T | audio | text |
| Audio | `music_analysis` | Music — technical analysis | Q3-I, Q3-T | audio | text |
| Audio | `music_appreciation` | Music — appreciation | Q3-I, Q3-T | audio | text |
| Audio | `mixed_audio_instruments` | Audio — effects and instruments | Q3-I, Q3-T | audio | text |
| Transcription | `asr_clean` | ASR — clean transcript | Q3-I, Q3-T | audio, video+audio | text |
| Transcription | `asr_clean_punctuated` | ASR — clean and punctuated | Q3-I, Q3-T | audio, video+audio | text |
| Transcription | `asr_timestamped_srt` | ASR — timestamped SRT | Q3-I, Q3-T | audio, video+audio | SRT segments |
| Transcription | `asr_translate` | ASR — speech translation | Q3-I, Q3-T | audio, video+audio | text |
| Transcription | `lyrics` | Lyrics — line transcription | Q3-I, Q3-T | audio, video+audio | lines |
| Transcription | `closed_captions_sdh` | Closed captions — SDH | Q3-I, Q3-T | audio, video+audio | SRT segments |
| Transcription | `speaker_diarized_transcript` | Transcript — speaker diarized | Q3-I, Q3-T | audio, video+audio | lines |
| Analysis | `chapters_summary` | Analysis — chapters and summary | Q3-I, Q3-T, AVo | video, video+audio | lines |
| Analysis | `search_index_json` | Analysis — search index JSON | Q3-I, Q3-T, AVo | video, video+audio | JSON |
| Analysis | `audiovisual_description_ad` | Accessibility — audio description | Q3-I, Q3-T, AVo | video+audio | SRT segments |
| Tags | `negative_avoid_list` | Negative prompt — avoid list | Q3-I, Q3-T, AVo | video, video+audio | tags |
| Tags | `booru_tags` | Image — Booru-style tags | Q3-I, Q3-T | image | tags |
| Utility | `custom` | Custom — free text | Q3-I, Q3-T, AVo | video, video+audio, audio, image, text | text |

## Template Variables

Rendering is deliberately Jinja-free. `render_prompt()` replaces `{{NAME}}` tokens, uses the defaults below when a known variable is omitted, and replaces unknown variables with an empty string. An empty `{{AVOID}}` disappears cleanly; a non-empty value becomes a complete `Do not mention: ...` sentence.

| Variable | Default | Meaning |
|---|---|---|
| `{{TRIGGER}}` | `ohwx` | Unique concept token used by LoRA captions. |
| `{{LANGUAGE}}` | `English` | Language for captions or metadata. |
| `{{SOURCE_LANGUAGE}}` | `English` | Language spoken in the source audio. |
| `{{TARGET_LANGUAGE}}` | `English` | Language requested for speech translation. |
| `{{CAPTION_LENGTH}}` | `detailed` | Requested answer density, such as `short` or `detailed`. |
| `{{AVOID}}` | empty | Concepts the model must not mention. |
| `{{SUBJECT_CLASS}}` | `person` | Identity-neutral class noun used with a trigger. |
| `{{EXTRA_INSTRUCTIONS}}` | empty | Optional task-specific override appended to the prompt. |

## Post-Processors

The registry may select `timechat_parse`, `timechat_flatten_wan`, `timechat_flatten_full`, `timechat_srt`, `strip_reasoning`, `srt_from_bracketed`, `lyrics_lines`, `tags_normalize`, `json_extract`, or `plain`. Every processor returns a `PostResult` containing display text, an optional structured value, and zero or more `(start_s, end_s, text)` segments. Timestamp segments can be serialized with the shared `to_srt()` and `to_vtt()` helpers.
