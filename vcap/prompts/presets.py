"""Built-in task and prompt presets for every captioning model family."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class PromptPreset:
    id: str
    label: str
    group: str
    description: str
    system_prompt: str | None
    user_prompt: str
    applies_to_models: tuple[str, ...]
    modalities: tuple[str, ...]
    output_format: str
    post_processor: str | None
    generation_overrides: dict[str, Any]
    recommended_media: dict[str, Any]
    tags: tuple[str, ...]


PRESET_GROUPS: tuple[str, ...] = (
    "Training captions",
    "Model-native",
    "Audio",
    "Transcription",
    "Analysis",
    "Tags",
    "Utility",
)

TEMPLATE_VARIABLES: dict[str, dict[str, str]] = {
    "TRIGGER": {
        "description": "Unique token placed in training captions to identify the learned concept.",
        "default": "ohwx",
    },
    "LANGUAGE": {
        "description": "Language requested for descriptive captions and metadata.",
        "default": "English",
    },
    "SOURCE_LANGUAGE": {
        "description": "Language spoken in the source audio.",
        "default": "English",
    },
    "TARGET_LANGUAGE": {
        "description": "Language used for translated speech output.",
        "default": "English",
    },
    "CAPTION_LENGTH": {
        "description": "Requested caption detail or length, such as short or detailed.",
        "default": "detailed",
    },
    "AVOID": {
        "description": "Concepts that the generated caption must not mention.",
        "default": "",
    },
    "SUBJECT_CLASS": {
        "description": "Generic class noun used in place of identity details.",
        "default": "person",
    },
    "EXTRA_INSTRUCTIONS": {
        "description": "Optional task-specific instructions appended to the prompt.",
        "default": "",
    },
}


TIMECHAT_OFFICIAL_PROMPT = (
    "Thoroughly describe everything in the video, capturing every detail. "
    "Include as much information from the audio as possible, and ensure that "
    "the descriptions of both audio and video are well-coordinated."
)

AVOCADO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

AVOCADO_AV_PROMPT = (
    "Provide a comprehensive description of all the content in the video, leaving out no details. "
    "Be sure to include as much of the audio information as possible, and ensure that your "
    "descriptions of the audio and video are closely aligned."
)

AVOCADO_VISUAL_PROMPT = (
    "Describe every aspect of the video in full detail, covering all the information it contains."
)

AVOCADO_UGC_PROMPT = (
    "You are given a short video with both audio and visual content. Write a detailed and coherent "
    "paragraph that naturally integrates all modalities. Your description should include: (1) the "
    "primary scene and background setting; (2) key characters or objects and their actions or "
    "interactions; (3) significant audio cues such as voices, background music, sound effects, and "
    "their emotional tone; (4) any on-screen text (OCR) and its role in the video context; and (5) "
    "the overall theme or purpose of the video. Ensure the output is a fluent and objective paragraph, "
    "not a bullet-point list, and captures the video's content in a human-like, narrative style."
)

AVOCADO_DIALOGUE_PROMPT = """You are a highly skilled assistant specializing in extracting conversational dialogue from text. Your task is to carefully analyze the given description of a video and accurately identify and extract all dialogue content within it.
Please directly output the dialogue in the following format without adding any other content. If no dialogue is present, state: "None."
Dialogue format:
Speaker A Description: "Dialogue from speaker A."
Speaker B Description: "Dialogue from speaker B."
Speaker A Description: "Further dialogue..."
The description for each speaker (e.g., "Person in red dress") must align with the given description and should be simplified for brevity. The key is to be concise and clearly distinguish between speakers (e.g., "Man in red shirt" is sufficient).
Video description: {video description}"""


QWEN_GENERIC = ("qwen3_omni_instruct", "qwen3_omni_thinking")
QWEN_AND_AVOCADO = (*QWEN_GENERIC, "avocado")
VIDEO_MODALITIES = ("video", "video_audio")
QWEN_GREEDY = {"do_sample": False, "max_new_tokens": 8192, "repetition_penalty": 1.0}
QWEN_AUDIO_GREEDY = {"do_sample": False, "max_new_tokens": 4096, "repetition_penalty": 1.0}
GENERIC_VIDEO_MEDIA = {"max_duration_s": 120, "fps": 1.0, "max_pixels": 262144}
AVOCADO_DEVIATION = " This preset may deviate from training distribution when used with AVoCaDO."


PRESETS: tuple[PromptPreset, ...] = (
    PromptPreset(
        id="wan22_t2v_dense",
        label="Wan 2.2 T2V — dense paragraph",
        group="Training captions",
        description=(
            "Produces a single information-dense caption for Wan 2.2 text-to-video training. "
            "It prioritizes observable subject, motion, scene, and cinematography without inventing intent."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write one {{CAPTION_LENGTH}} {{LANGUAGE}} paragraph describing the video for Wan 2.2 text-to-video training. "
            "State the main subject and visible clothing, the scene and spatial layout, the complete physical action in temporal order, camera framing and movement, and the observable lighting and color conditions. "
            "Use concrete present-tense language, preserve meaningful changes over time, and mention sound only when it visibly affects the action. "
            "Output one continuous paragraph with no heading, bullets, labels, speculation, or production commentary. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "wan", "t2v", "dense", "default"),
    ),
    PromptPreset(
        id="wan_t2v_sparse",
        label="Wan T2V — sparse motion line",
        group="Training captions",
        description=(
            "Produces a compact motion-first training caption when dense descriptions overfit the dataset. "
            "The answer is intentionally constrained to one short line." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write exactly one short {{LANGUAGE}} line for sparse Wan text-to-video training. "
            "Name the visible subject, its single dominant physical action, and only the camera motion needed to understand the shot. "
            "Use present tense and concrete verbs, omit secondary details and interpretation, and do not use a list, label, or second sentence. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 256, "repetition_penalty": 1.0},
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "wan", "t2v", "sparse", "motion"),
    ),
    PromptPreset(
        id="wan_i2v_motion_only",
        label="Wan I2V — motion only",
        group="Training captions",
        description=(
            "Describes only what changes after the supplied reference image. "
            "Identity and static appearance are excluded so the caption teaches motion rather than reconstructing the first frame."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Caption this clip for Wan image-to-video training using only subject movement, object movement, environmental motion, and camera movement. "
            "Describe the temporal order, direction, speed, and physically visible result of each important motion in precise present-tense language. "
            "Do not restate identity, face, body, clothing, colors, static objects, background layout, lighting, style, or other information already present in the reference image. "
            "Return one compact paragraph without a heading or list. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 1024, "repetition_penalty": 1.0},
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "wan", "i2v", "motion-only"),
    ),
    PromptPreset(
        id="hunyuan_dense_cinematic",
        label="Hunyuan — dense cinematic",
        group="Training captions",
        description=(
            "Builds a detailed Hunyuan-ready caption with explicit cinematic construction. "
            "It records blocking and camera grammar alongside visible content." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write one {{CAPTION_LENGTH}} {{LANGUAGE}} paragraph for Hunyuan video training that objectively describes the full shot. "
            "Identify subjects, wardrobe, setting, physical blocking, eyelines, entrances and exits, and the ordered action without assigning hidden motives. "
            "Specify shot type, camera angle and movement, plausible lens character only when visually supported, depth of field, composition, key and fill lighting, practical light sources, and significant editing transitions. "
            "Use present tense, no headings or lists, and no poetic or evaluative filler. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "hunyuan", "cinematic", "dense"),
    ),
    PromptPreset(
        id="ltx25_short_physical",
        label="LTX 2.5 — short physical action",
        group="Training captions",
        description=(
            "Targets short LTX 2.5 clips with literal, physically grounded action. "
            "It avoids narrative padding that is not learnable from a five-second shot." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Caption this clip of no more than five seconds in one concise {{LANGUAGE}} paragraph for LTX 2.5 training. "
            "Describe the subject, initial pose, exact physical action, contact with objects or surfaces, resulting motion, and camera movement in temporal order. "
            "Use literal present-tense verbs and observable spatial relations; do not add backstory, emotion without visible evidence, metaphor, mood prose, or events outside the clip. "
            "Keep the answer short and omit headings and lists. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 5, "fps": 2.0, "max_pixels": 262144},
        tags=("training", "ltx", "short", "physical"),
    ),
    PromptPreset(
        id="minimax_h3_performance_sound",
        label="MiniMax H3 — performance and sound",
        group="Training captions",
        description=(
            "Captures performance detail and synchronized sound for MiniMax H3 datasets. "
            "Dialogue is represented as an audible cue rather than silently discarded." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write one coherent {{LANGUAGE}} paragraph for MiniMax H3 training that follows the character performance and the surrounding soundscape together. "
            "Describe posture, gesture, facial performance, gaze, interaction, movement timing, scene, and camera behavior, then align audible dialogue cues, vocal delivery, music, Foley, and environmental sound with the visible events that produce or accompany them. "
            "Quote only speech that is clearly intelligible, identify speakers by visible role, and mark uncertain speech as indistinct rather than guessing. "
            "Use present tense, objective wording, and no headings or bullet list. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=("video_audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 262144},
        tags=("training", "minimax", "performance", "dialogue", "sound"),
    ),
    PromptPreset(
        id="character_lora",
        label="Character LoRA — trigger first",
        group="Training captions",
        description=(
            "Creates character LoRA captions that bind the learned identity to a trigger while withholding fixed identity traits. "
            "Outfit and action coverage can be changed with the extra-instructions field." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write one {{LANGUAGE}} training caption whose very first token is {{TRIGGER}}, immediately followed by the generic class {{SUBJECT_CLASS}}. "
            "Do not describe identity-bound hair, face, facial features, body shape, age, ethnicity, or other permanent anatomy that the trigger should learn. "
            "Describe the visible outfit, accessories, pose, physical action, interaction, scene, and camera only when they vary in this sample. "
            "Follow any supplied outfit or action inclusion override exactly, and return one paragraph without labels or commentary. {{EXTRA_INSTRUCTIONS}} {{AVOID}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 1024, "repetition_penalty": 1.0},
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "lora", "character", "trigger"),
    ),
    PromptPreset(
        id="motion_lora",
        label="Motion LoRA — trigger then movement",
        group="Training captions",
        description=(
            "Binds a repeatable movement pattern to the trigger instead of appearance. "
            "Static scene and identity attributes are deliberately suppressed." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Begin the caption with {{TRIGGER}}, then describe only the learned movement in one {{LANGUAGE}} paragraph. "
            "Record the motion's starting pose, ordered body or object mechanics, direction, speed, rhythm, contacts, follow-through, and any camera movement needed to interpret it. "
            "Do not describe identity, anatomy, clothing, background, lighting, style, emotion, story, or static appearance unless an item physically changes the motion. "
            "Use present tense, no heading, and no list. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 768, "repetition_penalty": 1.0},
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "lora", "motion", "trigger"),
    ),
    PromptPreset(
        id="style_lora",
        label="Style LoRA — visual treatment",
        group="Training captions",
        description=(
            "Separates a reusable visual treatment from sample-specific content for style LoRA training. "
            "It favors stable, observable style vocabulary over artist-name guessing." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write one {{LANGUAGE}} style-training caption beginning with {{TRIGGER}} and then briefly identify the visible content. "
            "Describe observable medium, rendering or capture technique, line and surface treatment, color palette, contrast, lighting design, texture, motion treatment, compositing, and editing rhythm that define the style. "
            "Distinguish recurring style from incidental subject matter, do not guess an artist or studio, and do not use subjective praise or metaphor. "
            "Return one paragraph without labels or a list. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 1536, "repetition_penalty": 1.0},
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "lora", "style", "trigger"),
    ),
    PromptPreset(
        id="no_speech_visual",
        label="Visual only — ignore speech",
        group="Training captions",
        description=(
            "Produces a visual caption even when the file contains dialogue or narration. "
            "It is useful for trainers whose text encoder should not learn speech content." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Describe only the visible video in one {{CAPTION_LENGTH}} {{LANGUAGE}} paragraph. "
            "Cover subjects, clothing, objects, scene, ordered physical action, composition, camera movement, lighting, and visible transitions using objective present-tense language. "
            "Ignore all dialogue, narration, lyrics, music, sound effects, and other audio even when they are clear, and never infer unseen events from sound. "
            "Do not use headings or lists. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("training", "visual-only", "no-speech"),
    ),
    PromptPreset(
        id="screen_text_include",
        label="Screen text — OCR aware",
        group="Training captions",
        description=(
            "Captures legible interface, game, sign, title-card, and subtitle text in visual context. "
            "It is intended for UI and game LoRAs where screen text carries state." + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write one {{LANGUAGE}} paragraph describing the video with special attention to legible on-screen text. "
            "Transcribe visible UI labels, menus, scores, HUD values, signs, title cards, captions, and notifications exactly when readable, preserving case and meaningful line order, and state where each item appears and how it changes. "
            "Also describe the relevant scene, subject action, cursor or control response, camera, and visual transition that gives the text context. "
            "Mark unreadable text as illegible instead of guessing, and output no list or heading. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 524288},
        tags=("training", "ocr", "ui", "game", "screen-text"),
    ),
    PromptPreset(
        id="image_dense_caption",
        label="Image — dense caption",
        group="Training captions",
        description="Creates a grounded, information-dense image training caption with spatial and photographic detail.",
        system_prompt=None,
        user_prompt=(
            "Write one {{CAPTION_LENGTH}} {{LANGUAGE}} paragraph that describes only what is visibly supported by the image. "
            "Identify the primary and secondary subjects, distinctive clothing and objects, poses and interactions, foreground-to-background scene layout, spatial relationships, legible text, composition, viewpoint, lighting, color, focus, and observable medium or photographic treatment. "
            "Use precise concrete nouns and present tense, distinguish certainty from ambiguity, and do not invent names, events, motives, or off-frame content. "
            "Return no heading, list, or analysis. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("image",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 4096, "repetition_penalty": 1.0},
        recommended_media={"max_pixels": 1310720, "min_pixels": 65536},
        tags=("training", "image", "dense"),
    ),
    PromptPreset(
        id="image_short_caption",
        label="Image — short caption",
        group="Training captions",
        description="Produces one compact, literal image caption for datasets that benefit from sparse conditioning.",
        system_prompt=None,
        user_prompt=(
            "Write one short {{LANGUAGE}} sentence describing the image. "
            "Name the main subject, its most important visible action or pose, and the essential setting using specific concrete words. "
            "Omit minor objects, hidden intent, stylistic praise, headings, labels, and any second sentence. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("image",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 128, "repetition_penalty": 1.0},
        recommended_media={"max_pixels": 1310720, "min_pixels": 65536},
        tags=("training", "image", "short"),
    ),

    # Model-native prompts are kept exactly as published in the model reports.
    PromptPreset(
        id="timechat_6d_raw",
        label="TimeChat — 6D raw JSON",
        group="Model-native",
        description="Runs TimeChat with its shipped prompt and preserves the native timestamped eight-key segment array.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video_audio",),
        output_format="timestamped_json",
        post_processor="timechat_parse",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "json", "timestamped"),
    ),
    PromptPreset(
        id="timechat_flatten_wan",
        label="TimeChat — Wan motion paragraph",
        group="Model-native",
        description="Uses the unchanged TimeChat task, then joins detailed events and camera state into one Wan-ready motion paragraph.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video_audio",),
        output_format="text",
        post_processor="timechat_flatten_wan",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "wan", "flatten", "default"),
    ),
    PromptPreset(
        id="timechat_flatten_motion_camera",
        label="TimeChat → motion + camera",
        group="Model-native",
        description="Keeps each segment's detailed motion and camera state in chronological order for I2V and motion datasets.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video",),
        output_format="text",
        post_processor="timechat_flatten_motion_camera",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "motion", "camera", "i2v"),
    ),
    PromptPreset(
        id="timechat_flatten_av",
        label="TimeChat → audiovisual",
        group="Model-native",
        description="Keeps motion, camera state, speech, and acoustics for each segment while dropping unrelated native fields.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video",),
        output_format="text",
        post_processor="timechat_flatten_av",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "audiovisual", "flatten"),
    ),
    PromptPreset(
        id="timechat_speech_only",
        label="TimeChat → speech transcript (SRT)",
        group="Model-native",
        description="Extracts non-empty speech fields as timestamped subtitle cues and a plain-text transcript.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video",),
        output_format="srt_segments",
        post_processor="timechat_speech_only",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "speech", "transcript", "srt"),
    ),
    PromptPreset(
        id="timechat_chapters",
        label="TimeChat → chapters",
        group="Model-native",
        description="Writes one concise MM:SS-MM:SS storyline chapter line for every native segment.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video",),
        output_format="text",
        post_processor="timechat_chapters",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "chapters", "storyline"),
    ),
    PromptPreset(
        id="timechat_to_srt",
        label="TimeChat — detailed events to SRT",
        group="Model-native",
        description="Uses TimeChat's native JSON and converts each timestamped detailed-event caption into an SRT cue.",
        system_prompt=None,
        user_prompt=TIMECHAT_OFFICIAL_PROMPT,
        applies_to_models=("timechat",),
        modalities=("video_audio",),
        output_format="srt_segments",
        post_processor="timechat_srt",
        generation_overrides={"do_sample": False, "max_new_tokens": 9216, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 60, "fps": 2.0, "max_pixels": 297920, "max_frames": 160},
        tags=("native", "timechat", "srt", "timestamped"),
    ),
    PromptPreset(
        id="avocado_av_aligned",
        label="AVoCaDO — aligned audiovisual",
        group="Model-native",
        description="Uses official audiovisual prompt number 1 for deterministic, training-aligned narrative captioning.",
        system_prompt=AVOCADO_SYSTEM_PROMPT,
        user_prompt=AVOCADO_AV_PROMPT,
        applies_to_models=("avocado",),
        modalities=("video_audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 100, "fps": 2.0, "max_pixels": 401408, "total_pixels": 20070400},
        tags=("native", "avocado", "audiovisual", "aligned", "default"),
    ),
    PromptPreset(
        id="avocado_visual_only",
        label="AVoCaDO — visual only",
        group="Model-native",
        description="Uses AVoCaDO's official VDC visual-only prompt with video audio disabled.",
        system_prompt=AVOCADO_SYSTEM_PROMPT,
        user_prompt=AVOCADO_VISUAL_PROMPT,
        applies_to_models=("avocado",),
        modalities=("video",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 100, "fps": 2.0, "max_pixels": 401408, "total_pixels": 20070400},
        tags=("native", "avocado", "visual-only"),
    ),
    PromptPreset(
        id="avocado_structured_ugc",
        label="AVoCaDO — structured UGC paragraph",
        group="Model-native",
        description="Uses AVoCaDO's official UGC-VideoCap prompt to cover scene, action, sound, OCR, and purpose in prose.",
        system_prompt=AVOCADO_SYSTEM_PROMPT,
        user_prompt=AVOCADO_UGC_PROMPT,
        applies_to_models=("avocado",),
        modalities=("video_audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0},
        recommended_media={"max_duration_s": 100, "fps": 2.0, "max_pixels": 401408, "total_pixels": 20070400},
        tags=("native", "avocado", "ugc", "ocr", "structured"),
    ),
    PromptPreset(
        id="avocado_dialogue_extract",
        label="AVoCaDO — dialogue extraction post-prompt",
        group="Model-native",
        description="Keeps the paper's official dialogue-extraction post-prompt for a second text-model pass; it is not an AVoCaDO inference task.",
        system_prompt=None,
        user_prompt=AVOCADO_DIALOGUE_PROMPT,
        applies_to_models=("avocado",),
        modalities=("text",),
        output_format="lines",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.0},
        recommended_media={},
        tags=("native", "avocado", "post-prompt", "dialogue"),
    ),
    PromptPreset(
        id="qwen3_video_describe",
        label="Qwen3-Omni — describe video",
        group="Model-native",
        description="Uses the official minimal video-description prompt from the Qwen3-Omni cookbook.",
        system_prompt=None,
        user_prompt="Describe the video.",
        applies_to_models=QWEN_GENERIC,
        modalities=VIDEO_MODALITIES,
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("native", "qwen3", "video"),
    ),
    PromptPreset(
        id="qwen3_video_dense",
        label="Qwen3-Omni — dense audiovisual video",
        group="Model-native",
        description="Provides a deterministic, dataset-oriented dense audiovisual caption instead of the official minimal request.",
        system_prompt=None,
        user_prompt=(
            "Describe the video in one {{CAPTION_LENGTH}} {{LANGUAGE}} paragraph, integrating visible and audible events in chronological order. "
            "Ground the caption in subjects, clothing, objects, setting, physical action, interactions, camera framing and movement, lighting, readable screen text, speech with speaker cues, music, sound effects, and environmental sound. "
            "Align each sound with the event or interval it accompanies, quote only clearly intelligible speech, and mark uncertain details rather than guessing. "
            "Use objective present tense with no heading, bullet list, analysis, or unsupported intent. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("video_audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("qwen3", "video", "audiovisual", "dense", "default"),
    ),
    PromptPreset(
        id="qwen3_scene_changes",
        label="Qwen3-Omni — scene changes",
        group="Model-native",
        description="Uses the official scene-transition prompt and returns chapter-like prose or lines.",
        system_prompt=None,
        user_prompt="How the scenes in the video change?",
        applies_to_models=QWEN_GENERIC,
        modalities=VIDEO_MODALITIES,
        output_format="lines",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("native", "qwen3", "scenes", "chapters"),
    ),
    PromptPreset(
        id="qwen3_audio_caption",
        label="Qwen3-Omni — detailed audio",
        group="Model-native",
        description="Uses the official Instruct audio-captioning prompt for detailed speech, environment, music, and sound description.",
        system_prompt=None,
        user_prompt="Give the detailed description of the audio.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 600},
        tags=("native", "qwen3", "audio", "caption"),
    ),
    PromptPreset(
        id="qwen3_captioner_promptfree",
        label="Qwen3-Omni Captioner — prompt free",
        group="Model-native",
        description="Sends exactly one audio input and no text or system message, matching the Captioner model's required single-turn distribution.",
        system_prompt=None,
        user_prompt="",
        applies_to_models=("qwen3_omni_captioner",),
        modalities=("audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 8192,
            "repetition_penalty": 1.0,
        },
        recommended_media={"max_duration_s": 30},
        tags=("native", "qwen3", "captioner", "audio", "prompt-free", "default"),
    ),
    PromptPreset(
        id="qwen3_thinking_dense",
        label="Qwen3-Omni Thinking — dense evidence",
        group="Model-native",
        description="Gives Thinking an explicit evidence-grounded audiovisual task and separates its reasoning from the final caption.",
        system_prompt=None,
        user_prompt=(
            "Analyze the provided video and audio carefully, then write one {{CAPTION_LENGTH}} {{LANGUAGE}} caption grounded only in perceptible evidence. "
            "Reason through temporal order, subject and object continuity, physical action, camera changes, speech, music, sound effects, and cross-modal alignment before composing the answer. "
            "In the final answer, use one objective chronological paragraph, quote only intelligible speech, identify uncertainty briefly, and omit the reasoning, headings, and lists. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=("qwen3_omni_thinking",),
        modalities=("video_audio",),
        output_format="text",
        post_processor="strip_reasoning",
        generation_overrides={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": 32768,
            "repetition_penalty": 1.0,
        },
        recommended_media={"max_duration_s": 120, "fps": 1.0, "max_pixels": 262144},
        tags=("native", "qwen3", "thinking", "dense", "reasoning", "default"),
    ),
    PromptPreset(
        id="qwen3_joint_describe",
        label="Qwen3-Omni — joint media description",
        group="Model-native",
        description="Uses the official evaluation prompt for a joint audio, image, and video description.",
        system_prompt=None,
        user_prompt="Describe the audio, image and video.",
        applies_to_models=QWEN_GENERIC,
        modalities=("video_audio", "image", "audio"),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("native", "qwen3", "multimodal", "joint"),
    ),
    PromptPreset(
        id="qwen3_ocr",
        label="Qwen3-Omni — image OCR",
        group="Model-native",
        description="Uses the official OCR prompt and returns extracted image text.",
        system_prompt=None,
        user_prompt="Extract the text from the image.",
        applies_to_models=QWEN_GENERIC,
        modalities=("image",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 4096, "repetition_penalty": 1.0},
        recommended_media={"max_pixels": 1310720, "min_pixels": 65536},
        tags=("native", "qwen3", "ocr", "image"),
    ),
    PromptPreset(
        id="qwen3_image_describe",
        label="Qwen3-Omni — describe image",
        group="Model-native",
        description="Requests a grounded general-purpose image description suitable for review or downstream caption editing.",
        system_prompt=None,
        user_prompt=(
            "Describe the image in {{LANGUAGE}} using one {{CAPTION_LENGTH}} paragraph. "
            "Cover the main subjects, visible actions or poses, important objects, spatial layout, setting, readable text, composition, viewpoint, lighting, color, focus, and observable medium. "
            "State only visible evidence, note genuine ambiguity without guessing, and omit headings, lists, hidden intent, and off-frame events. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("image",),
        output_format="text",
        post_processor="plain",
        generation_overrides={"do_sample": False, "max_new_tokens": 4096, "repetition_penalty": 1.0},
        recommended_media={"max_pixels": 1310720, "min_pixels": 65536},
        tags=("qwen3", "image", "description", "default"),
    ),

    PromptPreset(
        id="audio_sfx_bed",
        label="Audio — SFX and ambience bed",
        group="Audio",
        description="Builds a plot-free production-sound caption focused on events, materials, acoustics, and room tone.",
        system_prompt=None,
        user_prompt=(
            "Describe the audible sound-effects and ambience bed in one {{CAPTION_LENGTH}} {{LANGUAGE}} paragraph without telling a visual plot. "
            "List events in chronological order and identify likely source behavior only when acoustically supported, including impacts, friction, footsteps, mechanisms, weather, voices as nonverbal texture, material qualities, distance, direction, reverberation, dynamics, and persistent room tone. "
            "Separate overlapping layers and note silence or transitions, but do not invent unseen characters, locations, causes, dialogue, or narrative purpose. "
            "Use concrete audio-production vocabulary and no heading or bullet list. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 120},
        tags=("audio", "sfx", "ambience", "room-tone"),
    ),
    PromptPreset(
        id="sound_events",
        label="Audio — sound events",
        group="Audio",
        description="Uses the official Qwen3-Omni sound-event prompt.",
        system_prompt=None,
        user_prompt="What happened in the audio?",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 120},
        tags=("native", "audio", "sfx", "events"),
    ),
    PromptPreset(
        id="music_analysis",
        label="Music — technical analysis",
        group="Audio",
        description="Uses the official Qwen3-Omni prompt for music style, rhythm, dynamics, emotion, instrumentation, and likely context.",
        system_prompt=None,
        user_prompt="Describe the style, rhythm, dynamics, and expressed emotions of this piece of music. Identify the instruments used and suggest possible scenarios from which this music might originate.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 600},
        tags=("native", "audio", "music", "analysis"),
    ),
    PromptPreset(
        id="music_appreciation",
        label="Music — appreciation",
        group="Audio",
        description="Uses the official Qwen3-Omni appreciation prompt for genre, instrumental collaboration, and atmosphere.",
        system_prompt=None,
        user_prompt="Write an appreciative description for this piece of music. Identifying its style and genre. Analyze the collaborative patterns of different instruments in audio and explain their impact on the overall atmosphere.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 600},
        tags=("native", "audio", "music", "appreciation"),
    ),
    PromptPreset(
        id="mixed_audio_instruments",
        label="Audio — effects and instruments",
        group="Audio",
        description="Uses the official mixed-audio prompt to identify sound effects and musical instruments together.",
        system_prompt=None,
        user_prompt="Determine which sound effects and musical instruments are present in the audio.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio",),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 600},
        tags=("native", "audio", "mixed", "instruments", "sfx"),
    ),

    PromptPreset(
        id="asr_clean",
        label="ASR — clean transcript",
        group="Transcription",
        description="Uses Qwen3-Omni's official language-parameterized ASR instruction without adding formatting constraints.",
        system_prompt=None,
        user_prompt="Transcribe the {{SOURCE_LANGUAGE}} audio into text.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 2400},
        tags=("native", "asr", "transcription", "clean", "default"),
    ),
    PromptPreset(
        id="asr_clean_punctuated",
        label="ASR — clean and punctuated",
        group="Transcription",
        description="Adds conservative punctuation and paragraphing rules to the official ASR task.",
        system_prompt=None,
        user_prompt=(
            "Transcribe the {{SOURCE_LANGUAGE}} audio into text. "
            "Preserve the spoken words and their order exactly while adding natural sentence punctuation, capitalization, and paragraph breaks. "
            "Do not summarize, translate, censor, correct grammar, label speakers, describe sounds, or add an introduction, and mark genuinely unintelligible speech as [inaudible]."
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 2400},
        tags=("asr", "transcription", "punctuated"),
    ),
    PromptPreset(
        id="asr_timestamped_srt",
        label="ASR — timestamped SRT",
        group="Transcription",
        description="Requests bracketed timestamp lines that can be deterministically converted to SubRip cues.",
        system_prompt=None,
        user_prompt=(
            "Transcribe the {{SOURCE_LANGUAGE}} speech verbatim and divide it into short subtitle cues. "
            "Output exactly one cue per line in the form [MM:SS-MM:SS] text, using the audible start and end time and continuing to HH:MM:SS when the recording exceeds one hour. "
            "Keep each cue readable, preserve speech order and wording, add only necessary punctuation, use [inaudible] for unclear speech, and output no heading, explanation, or code fence."
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="srt_segments",
        post_processor="srt_from_bracketed",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 2400},
        tags=("asr", "transcription", "timestamps", "srt"),
    ),
    PromptPreset(
        id="asr_translate",
        label="ASR — speech translation",
        group="Transcription",
        description="Uses Qwen3-Omni's official parameterized speech-to-text translation prompt.",
        system_prompt=None,
        user_prompt="Listen to the provided {{SOURCE_LANGUAGE}} speech and produce a translation in {{TARGET_LANGUAGE}} text.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="text",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 2400},
        tags=("native", "asr", "translation", "s2tt"),
    ),
    PromptPreset(
        id="lyrics",
        label="Lyrics — line transcription",
        group="Transcription",
        description="Uses the official lyrics prompt and normalizes the result to non-empty lyric lines.",
        system_prompt=None,
        user_prompt="Transcribe the song lyrics into text without any punctuation, separate lines with line breaks, and output only the lyrics without additional explanations.",
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="lines",
        post_processor="lyrics_lines",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 600},
        tags=("native", "asr", "lyrics", "music"),
    ),
    PromptPreset(
        id="closed_captions_sdh",
        label="Closed captions — SDH",
        group="Transcription",
        description="Produces timestamped dialogue plus useful music and sound-effect cues for deaf and hard-of-hearing viewers.",
        system_prompt=None,
        user_prompt=(
            "Create {{LANGUAGE}} closed captions for deaf and hard-of-hearing viewers as short timestamped cues. "
            "Output exactly [MM:SS-MM:SS] text on each line, transcribe speech faithfully, prefix a concise speaker label when the speaker is unclear from context, and include meaningful non-speech cues such as [music], [door slams], or [applause]. "
            "Describe only sounds needed to follow the program, place each cue at its audible interval, avoid duplicate visual description, and output no heading or explanation."
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="srt_segments",
        post_processor="srt_from_bracketed",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 2400},
        tags=("accessibility", "captions", "sdh", "srt", "timestamps"),
    ),
    PromptPreset(
        id="speaker_diarized_transcript",
        label="Transcript — speaker diarized",
        group="Transcription",
        description="Separates speakers by stable generic labels while preserving wording and turn order.",
        system_prompt=None,
        user_prompt=(
            "Transcribe the {{SOURCE_LANGUAGE}} speech verbatim and diarize each turn with stable labels SPEAKER 1, SPEAKER 2, and so on. "
            "Write one line per turn as [MM:SS-MM:SS] SPEAKER N: text, retaining overlaps as separate lines and using the same label whenever the same voice returns. "
            "Do not infer names, gender, identity, or relationships from voice alone, do not summarize or translate, and mark uncertain speaker assignment or wording explicitly."
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("audio", "video_audio"),
        output_format="lines",
        post_processor="plain",
        generation_overrides=QWEN_AUDIO_GREEDY,
        recommended_media={"max_duration_s": 2400},
        tags=("asr", "diarization", "speakers", "timestamps"),
    ),

    PromptPreset(
        id="chapters_summary",
        label="Analysis — chapters and summary",
        group="Analysis",
        description=(
            "Creates navigable chapter markers and a compact synopsis for long-form video."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Segment the video into meaningful chapters based on changes in topic, activity, location, speaker focus, or scene. "
            "For each chapter output one line as [HH:MM:SS] concise title — one-sentence factual synopsis, using the actual start time and covering the entire video without overlapping chapters. "
            "After the chapter lines, write a section titled Summary containing one compact {{LANGUAGE}} paragraph of the video's main progression and outcome. "
            "Do not invent timestamps, merge unrelated events, or add viewing advice. {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="lines",
        post_processor="plain",
        generation_overrides=QWEN_GREEDY,
        recommended_media={"max_duration_s": 600, "fps": 1.0, "max_pixels": 262144},
        tags=("analysis", "chapters", "summary", "index"),
    ),
    PromptPreset(
        id="search_index_json",
        label="Analysis — search index JSON",
        group="Analysis",
        description=(
            "Extracts searchable people, objects, actions, text, places, sounds, and keywords into strict JSON."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Analyze the media and return exactly one valid JSON object with keys summary, people, objects, actions, locations, ocr_text, speech, sounds, and keywords. "
            "Use a string for summary and arrays of unique concise strings for every other key; record only directly observed or heard evidence, keep actions as present-tense verb phrases, and preserve readable OCR and intelligible speech verbatim. "
            "Use {{LANGUAGE}} except for verbatim text, emit every key even when its value is empty, and do not include markdown fences, comments, trailing commas, confidence prose, or additional keys. "
            "Normalize near-duplicates while retaining terms useful for dataset search. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=("video", "video_audio"),
        output_format="json",
        post_processor="json_extract",
        generation_overrides=QWEN_GREEDY,
        recommended_media={"max_duration_s": 120, "fps": 1.0, "max_pixels": 262144},
        tags=("analysis", "json", "search", "index", "metadata"),
    ),
    PromptPreset(
        id="audiovisual_description_ad",
        label="Accessibility — audio description",
        group="Analysis",
        description=(
            "Writes concise accessibility narration that fits around existing dialogue and important sound."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Write an accessibility audio-description script in {{LANGUAGE}} for the visual information needed to understand the video. "
            "Describe essential actions, gestures, expressions, identities established on screen, setting changes, on-screen text, and visual transitions in chronological order, using brief present-tense phrases that can fit into natural pauses. "
            "Do not repeat dialogue or obvious sound cues, speak over important audio, explain filmmaking technique, infer thoughts, or describe insignificant decoration. "
            "Output one narration cue per line as [MM:SS-MM:SS] text with no heading or commentary. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=("video_audio",),
        output_format="srt_segments",
        post_processor="srt_from_bracketed",
        generation_overrides=QWEN_GREEDY,
        recommended_media={"max_duration_s": 120, "fps": 2.0, "max_pixels": 262144},
        tags=("analysis", "accessibility", "audio-description", "timestamps"),
    ),

    PromptPreset(
        id="negative_avoid_list",
        label="Negative prompt — avoid list",
        group="Tags",
        description=(
            "Generates a concise negative-prompt list limited to visible failure modes and likely unwanted artifacts."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt=(
            "Inspect the media and produce a compact comma-separated negative prompt for recreating the clip or image. "
            "Include only observable defects or clearly relevant failure modes such as blur, compression artifacts, flicker, warped anatomy, duplicate objects, broken motion, illegible text, exposure problems, or unwanted overlays; do not negate desired subject, action, composition, or style. "
            "Use lowercase {{LANGUAGE}} tags, deduplicate synonyms, add any supplied avoid concepts, and output tags only with no sentence, heading, explanation, or numbering. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_AND_AVOCADO,
        modalities=VIDEO_MODALITIES,
        output_format="tags",
        post_processor="tags_normalize",
        generation_overrides={"do_sample": False, "max_new_tokens": 512, "repetition_penalty": 1.0},
        recommended_media=GENERIC_VIDEO_MEDIA,
        tags=("tags", "negative", "avoid", "quality"),
    ),
    PromptPreset(
        id="booru_tags",
        label="Image — Booru-style tags",
        group="Tags",
        description="Produces normalized image tags for tag-conditioned image datasets without natural-language prose.",
        system_prompt=None,
        user_prompt=(
            "Tag the image using concise lowercase Booru-style tags separated by commas. "
            "Cover subject count and class, clearly visible identity-neutral attributes, clothing, pose, action, expression, objects, setting, composition, camera angle, lighting, color, and medium, using underscores for multiword tags when conventional. "
            "Order from most important to least important, avoid duplicates, guesses, artist names, ratings, and unsupported anatomy, and output only the comma-separated tags with no sentence or heading. {{AVOID}} {{EXTRA_INSTRUCTIONS}}"
        ),
        applies_to_models=QWEN_GENERIC,
        modalities=("image",),
        output_format="tags",
        post_processor="tags_normalize",
        generation_overrides={"do_sample": False, "max_new_tokens": 1024, "repetition_penalty": 1.0},
        recommended_media={"max_pixels": 1310720, "min_pixels": 65536},
        tags=("tags", "booru", "image", "dataset"),
    ),

    PromptPreset(
        id="custom",
        label="Custom — free text",
        group="Utility",
        description=(
            "Leaves both prompt fields empty so the user can supply a task without preset wording."
            + AVOCADO_DEVIATION
        ),
        system_prompt=None,
        user_prompt="",
        applies_to_models=("avocado", "qwen3_omni_instruct", "qwen3_omni_thinking"),
        modalities=("video", "video_audio", "audio", "image", "text"),
        output_format="text",
        post_processor="plain",
        generation_overrides={},
        recommended_media={},
        tags=("utility", "custom"),
    ),
)


# The architecture plan uses this longer registry name; keep both public names.
PROMPT_PRESETS = PRESETS


_PRESETS_BY_ID = {preset.id: preset for preset in PRESETS}
if len(_PRESETS_BY_ID) != len(PRESETS):
    raise RuntimeError("Prompt preset ids must be unique")

_GROUP_INDEX = {name: index for index, name in enumerate(PRESET_GROUPS)}
_TEMPLATE_RE = re.compile(r"{{\s*([A-Za-z][A-Za-z0-9_]*)\s*}}")


def list_presets(model_family: str | None = None, modality: str | None = None) -> list[PromptPreset]:
    """List presets in product group order, optionally filtered by family/modality."""

    selected = [
        preset
        for preset in PRESETS
        if (model_family is None or "*" in preset.applies_to_models or model_family in preset.applies_to_models)
        and (
            modality is None
            or modality in preset.modalities
            or (
                model_family == "timechat"
                and modality == "video_audio"
                and "video" in preset.modalities
            )
        )
    ]
    order = {preset.id: index for index, preset in enumerate(PRESETS)}
    return sorted(selected, key=lambda preset: (_GROUP_INDEX[preset.group], order[preset.id]))


def get_preset(preset_id: str) -> PromptPreset:
    """Return a preset by id, raising a descriptive ``KeyError`` when absent."""

    try:
        return _PRESETS_BY_ID[preset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt preset: {preset_id}") from exc


def _avoid_sentence(value: Any) -> str:
    avoid = str(value or "").strip()
    if not avoid:
        return ""
    suffix = "" if avoid[-1] in ".!?" else "."
    return f"Do not mention: {avoid}{suffix}"


def _render_template(template: str | None, variables: dict[str, Any]) -> str | None:
    if template is None:
        return None
    if not template or "{{" not in template:
        return template

    supplied = {str(key).upper(): value for key, value in variables.items()}

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).upper()
        if name not in TEMPLATE_VARIABLES:
            return ""
        value = supplied.get(name, TEMPLATE_VARIABLES[name]["default"])
        if name == "AVOID":
            return _avoid_sentence(value)
        return str(value if value is not None else "").strip()

    rendered = _TEMPLATE_RE.sub(replace, template)
    rendered = re.sub(r"[ \t]+", " ", rendered)
    rendered = re.sub(r" *\n *", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip()


def render_prompt(preset: PromptPreset, variables: dict[str, Any]) -> tuple[str | None, str]:
    """Render system and user templates with defaults and simple token replacement."""

    system = _render_template(preset.system_prompt, variables)
    user = _render_template(preset.user_prompt, variables)
    return system, user or ""


_DEFAULT_PRESET_IDS: dict[tuple[str, str], str] = {
    ("timechat", "video"): "timechat_flatten_wan",
    ("timechat", "video_audio"): "timechat_flatten_wan",
    ("avocado", "video"): "avocado_visual_only",
    ("avocado", "video_audio"): "avocado_av_aligned",
    ("qwen3_omni_instruct", "video"): "wan22_t2v_dense",
    ("qwen3_omni_instruct", "video_audio"): "qwen3_video_dense",
    ("qwen3_omni_instruct", "audio"): "qwen3_audio_caption",
    ("qwen3_omni_instruct", "image"): "qwen3_image_describe",
    ("qwen3_omni_instruct", "text"): "custom",
    ("qwen3_omni_thinking", "video"): "wan22_t2v_dense",
    ("qwen3_omni_thinking", "video_audio"): "qwen3_thinking_dense",
    ("qwen3_omni_thinking", "audio"): "qwen3_audio_caption",
    ("qwen3_omni_thinking", "image"): "qwen3_image_describe",
    ("qwen3_omni_thinking", "text"): "custom",
    ("qwen3_omni_captioner", "audio"): "qwen3_captioner_promptfree",
}


def default_preset_for(model_family: str, modality: str) -> PromptPreset:
    """Return the product default for a supported model-family/modality pair."""

    try:
        return get_preset(_DEFAULT_PRESET_IDS[(model_family, modality)])
    except KeyError as exc:
        raise KeyError(f"No default prompt preset for {model_family!r} and {modality!r}") from exc


__all__ = [
    "AVOCADO_AV_PROMPT",
    "AVOCADO_DIALOGUE_PROMPT",
    "AVOCADO_SYSTEM_PROMPT",
    "AVOCADO_UGC_PROMPT",
    "AVOCADO_VISUAL_PROMPT",
    "PRESETS",
    "PRESET_GROUPS",
    "PROMPT_PRESETS",
    "PromptPreset",
    "TEMPLATE_VARIABLES",
    "TIMECHAT_OFFICIAL_PROMPT",
    "default_preset_for",
    "get_preset",
    "list_presets",
    "render_prompt",
]
