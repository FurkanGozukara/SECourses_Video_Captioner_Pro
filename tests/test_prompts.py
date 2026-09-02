from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from vcap.prompts.postprocess import (
    POST_PROCESSORS,
    json_extract,
    segments_from_text_timestamps,
    strip_reasoning,
    tags_normalize,
    timechat_flatten_full,
    timechat_flatten_wan,
    timechat_parse,
    timechat_srt,
    trim_avocado_trailing_qa,
    to_srt,
    to_vtt,
)
from vcap.prompts.presets import (
    AVOCADO_AV_PROMPT,
    AVOCADO_SYSTEM_PROMPT,
    PRESETS,
    PRESET_GROUPS,
    TEMPLATE_VARIABLES,
    TIMECHAT_OFFICIAL_PROMPT,
    default_preset_for,
    get_preset,
    list_presets,
    render_prompt,
)


EXPECTED_PRESET_IDS = {
    "wan22_t2v_dense",
    "wan_t2v_sparse",
    "wan_i2v_motion_only",
    "hunyuan_dense_cinematic",
    "ltx25_short_physical",
    "minimax_h3_performance_sound",
    "character_lora",
    "motion_lora",
    "style_lora",
    "no_speech_visual",
    "screen_text_include",
    "negative_avoid_list",
    "booru_tags",
    "image_dense_caption",
    "image_short_caption",
    "timechat_6d_raw",
    "timechat_flatten_wan",
    "timechat_flatten_motion_camera",
    "timechat_flatten_av",
    "timechat_speech_only",
    "timechat_chapters",
    "timechat_to_srt",
    "avocado_av_aligned",
    "avocado_visual_only",
    "avocado_structured_ugc",
    "avocado_dialogue_extract",
    "qwen3_video_describe",
    "qwen3_video_dense",
    "qwen3_scene_changes",
    "qwen3_audio_caption",
    "qwen3_captioner_promptfree",
    "qwen3_thinking_dense",
    "qwen3_joint_describe",
    "qwen3_ocr",
    "qwen3_image_describe",
    "asr_clean",
    "asr_clean_punctuated",
    "asr_timestamped_srt",
    "asr_translate",
    "lyrics",
    "closed_captions_sdh",
    "speaker_diarized_transcript",
    "audio_sfx_bed",
    "sound_events",
    "music_analysis",
    "music_appreciation",
    "mixed_audio_instruments",
    "chapters_summary",
    "search_index_json",
    "audiovisual_description_ad",
    "custom",
}

TIMECHAT_SAMPLE = [
    {
        "timestamp": "00:00-00:06",
        "segment_detail_caption": (
            "A woman in a red raincoat steps from the curb, opens a black umbrella, "
            "and walks across the wet street."
        ),
        "camera_state": "A medium-wide handheld shot tracks left beside her.",
        "video_background": "Rain falls on a city intersection at dusk as traffic waits behind her.",
        "storyline": "The pedestrian begins crossing after sheltering under the umbrella.",
        "shooting_style": "The continuous tracking shot uses natural motion and no visible cut.",
        "speech_content": "No intelligible speech is heard.",
        "acoustics_content": "Rain patters on the umbrella over distant engines and a crossing signal.",
    },
    {
        "timestamp": "00:06-00:12",
        "segment_detail_caption": (
            "She reaches the opposite pavement, closes the umbrella, and enters a brightly lit cafe."
        ),
        "camera_state": "The camera pans right, then settles into a static wide shot at the doorway.",
        "video_background": "Warm interior light contrasts with the blue-gray rainy street.",
        "storyline": "The crossing ends as she takes shelter inside.",
        "shooting_style": "A clean match-on-action cut occurs as the door opens.",
        "speech_content": "A barista says, \"Good evening.\"",
        "acoustics_content": "The rain softens when the door closes, followed by a bell and quiet cafe music.",
    },
]


def test_registry_has_every_required_unique_preset_and_valid_contract():
    assert {preset.id for preset in PRESETS} == EXPECTED_PRESET_IDS
    assert len(PRESETS) == len(EXPECTED_PRESET_IDS)
    assert PRESET_GROUPS == (
        "Training captions",
        "Model-native",
        "Audio",
        "Transcription",
        "Analysis",
        "Tags",
        "Utility",
    )

    allowed_modalities = {"video", "video_audio", "audio", "image", "text"}
    allowed_outputs = {"text", "json", "timestamped_json", "srt_segments", "tags", "lines"}
    variables = {name: spec["default"] for name, spec in TEMPLATE_VARIABLES.items()}
    variables.update({"TRIGGER": "testtoken", "AVOID": "watermark", "EXTRA_INSTRUCTIONS": "Prefer literal wording."})

    for preset in PRESETS:
        assert preset.group in PRESET_GROUPS
        assert preset.modalities and set(preset.modalities) <= allowed_modalities
        assert preset.output_format in allowed_outputs
        assert preset.post_processor is None or preset.post_processor in POST_PROCESSORS
        system, user = render_prompt(preset, variables)
        assert system is None or isinstance(system, str)
        assert isinstance(user, str)
        assert "{{" not in (system or "")
        assert "{{" not in user


def test_template_defaults_substitution_unknowns_and_avoid_sentence():
    preset = get_preset("character_lora")
    _, rendered = render_prompt(
        preset,
        {
            "TRIGGER": "zxperson",
            "SUBJECT_CLASS": "performer",
            "LANGUAGE": "German",
            "AVOID": "watermark, real name",
            "EXTRA_INSTRUCTIONS": "Keep the jacket; omit the action.",
        },
    )
    assert rendered.startswith("Write one German training caption")
    assert "zxperson" in rendered
    assert "performer" in rendered
    assert "Do not mention: watermark, real name." in rendered
    assert "Keep the jacket; omit the action." in rendered

    unknown = replace(get_preset("custom"), user_prompt="before {{NOT_DEFINED}} after {{LANGUAGE}}")
    assert render_prompt(unknown, {}) == (None, "before after English")


def test_avoid_placeholder_disappears_cleanly_when_empty():
    _, empty = render_prompt(get_preset("qwen3_video_dense"), {"AVOID": ""})
    _, populated = render_prompt(get_preset("qwen3_video_dense"), {"AVOID": "logos"})
    assert "Do not mention:" not in empty
    assert "{{AVOID}}" not in empty
    assert "Do not mention: logos." in populated


def test_model_and_modality_filtering_and_defaults():
    assert {preset.id for preset in list_presets("timechat")} == {
        "timechat_6d_raw",
        "timechat_flatten_wan",
        "timechat_flatten_motion_camera",
        "timechat_flatten_av",
        "timechat_speech_only",
        "timechat_chapters",
        "timechat_to_srt",
    }
    assert [preset.id for preset in list_presets("qwen3_omni_captioner", "audio")] == [
        "qwen3_captioner_promptfree"
    ]
    assert "booru_tags" in {preset.id for preset in list_presets("qwen3_omni_instruct", "image")}
    assert "booru_tags" not in {preset.id for preset in list_presets("avocado", "image")}
    assert default_preset_for("timechat", "video_audio").id == "timechat_flatten_wan"
    assert default_preset_for("avocado", "video").id == "avocado_visual_only"
    assert default_preset_for("qwen3_omni_captioner", "audio").id == "qwen3_captioner_promptfree"
    with pytest.raises(KeyError):
        default_preset_for("timechat", "audio")


def test_official_prompts_and_generation_settings_are_unchanged():
    assert TIMECHAT_OFFICIAL_PROMPT == (
        "Thoroughly describe everything in the video, capturing every detail. "
        "Include as much information from the audio as possible, and ensure that the descriptions "
        "of both audio and video are well-coordinated."
    )
    assert get_preset("timechat_6d_raw").system_prompt is None
    assert get_preset("timechat_6d_raw").generation_overrides["max_new_tokens"] == 9216
    assert AVOCADO_SYSTEM_PROMPT == (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
        "perceiving auditory and visual inputs, as well as generating text and speech."
    )
    assert get_preset("avocado_av_aligned").user_prompt == AVOCADO_AV_PROMPT
    assert get_preset("qwen3_video_describe").user_prompt == "Describe the video."
    assert get_preset("qwen3_scene_changes").user_prompt == "How the scenes in the video change?"
    captioner = get_preset("qwen3_captioner_promptfree")
    assert captioner.user_prompt == "" and captioner.system_prompt is None
    assert captioner.generation_overrides["temperature"] == 0.6
    assert captioner.recommended_media["max_duration_s"] == 30


def test_timechat_parse_realistic_two_segment_json_and_trailing_comma_repair():
    raw = "```json\n" + json.dumps(TIMECHAT_SAMPLE, ensure_ascii=False, indent=2)[:-2] + ",\n]\n```"
    result = timechat_parse(raw, {})
    assert result.structured == TIMECHAT_SAMPLE
    assert result.segments == [
        (0.0, 6.0, TIMECHAT_SAMPLE[0]["segment_detail_caption"]),
        (6.0, 12.0, TIMECHAT_SAMPLE[1]["segment_detail_caption"]),
    ]
    assert json.loads(result.text) == TIMECHAT_SAMPLE


def test_timechat_truncated_array_keeps_every_complete_object():
    missing_array_close = json.dumps(TIMECHAT_SAMPLE, ensure_ascii=False)[:-1]
    result = timechat_parse(missing_array_close, {})
    assert result.structured == TIMECHAT_SAMPLE
    assert len(result.segments) == 2

    second_object_cut_off = json.dumps(TIMECHAT_SAMPLE, ensure_ascii=False)[:-80]
    result = timechat_parse(second_object_cut_off, {})
    assert result.structured == [TIMECHAT_SAMPLE[0]]
    assert len(result.segments) == 1


def test_timechat_flatteners_and_srt_use_native_fields():
    raw = json.dumps(TIMECHAT_SAMPLE, ensure_ascii=False)
    wan = timechat_flatten_wan(raw, {})
    assert "opens a black umbrella" in wan.text
    assert "tracks left beside her" in wan.text
    assert "00:00" not in wan.text
    assert "\n" not in wan.text

    full = timechat_flatten_full(raw, {})
    assert "Visual:" in full.text
    assert "Camera:" in full.text
    assert "Setting:" in full.text
    assert "Audio:" in full.text
    assert "\n\n" in full.text

    srt = timechat_srt(raw, {})
    assert srt.segments[1][:2] == (6.0, 12.0)
    assert "00:00:06,000 --> 00:00:12,000" in srt.text
    assert TIMECHAT_SAMPLE[1]["segment_detail_caption"] in srt.text


def test_bracketed_timestamp_parsing_and_srt_vtt_formatting():
    source = (
        "[00:01.250-00:03.500] First line\n"
        "continues here\n"
        "[01:02:03-01:02:05] Second line"
    )
    segments = segments_from_text_timestamps(source)
    assert segments == [
        (1.25, 3.5, "First line\ncontinues here"),
        (3723.0, 3725.0, "Second line"),
    ]

    srt = to_srt(segments)
    assert srt == (
        "1\n00:00:01,250 --> 00:00:03,500\nFirst line\ncontinues here\n\n"
        "2\n01:02:03,000 --> 01:02:05,000\nSecond line\n"
    )
    vtt = to_vtt(segments)
    assert vtt.startswith("WEBVTT\n\n00:00:01.250 --> 00:00:03.500")
    assert "01:02:03.000 --> 01:02:05.000" in vtt


def test_thinking_split_exposes_reasoning_and_answer():
    result = strip_reasoning("<think>\nCompare the visible evidence.\n</think>\n\nThe door opens.", {})
    assert result.reasoning == "Compare the visible evidence."
    assert result.text == "The door opens."
    assert result.structured == {
        "reasoning": "Compare the visible evidence.",
        "answer": "The door opens.",
    }

    plain_result = strip_reasoning("No reasoning block.", {})
    assert plain_result.reasoning == ""
    assert plain_result.text == "No reasoning block."


def test_remaining_structured_post_processors():
    tags = tags_normalize("Cat, blue sky, CAT\n#Blue Sky, soft_light", {})
    assert tags.text == "cat, blue sky, soft_light"
    assert tags.structured == ["cat", "blue sky", "soft_light"]

    extracted = json_extract('Result follows: {"objects": ["cup"], "ok": true,} end', {})
    assert extracted.structured == {"objects": ["cup"], "ok": True}
    assert json.loads(extracted.text) == extracted.structured

    expected_processors = {
        "timechat_parse",
            "timechat_flatten_wan",
            "timechat_flatten_motion_camera",
            "timechat_flatten_av",
            "timechat_speech_only",
            "timechat_chapters",
            "timechat_flatten_full",
        "timechat_srt",
        "strip_reasoning",
        "srt_from_bracketed",
        "lyrics_lines",
        "tags_normalize",
        "json_extract",
        "plain",
    }
    assert set(POST_PROCESSORS) == expected_processors


def test_avocado_trailing_qa_cleanup_is_cap_gated_and_conservative():
    caption = (
        "A long audiovisual description of the rainy intersection, moving traffic, "
        "wet reflections, thunder, and lightning. " * 4
    ).strip()
    artifact = caption + "\n\nQuestion: What is shown in the video?\nAnswer: A city street."

    assert trim_avocado_trailing_qa(artifact, hit_token_cap=True) == (caption, True)
    assert trim_avocado_trailing_qa(artifact, hit_token_cap=False) == (artifact, False)
    ordinary = caption + " The narrator asks a question: why is the road wet?"
    assert trim_avocado_trailing_qa(ordinary, hit_token_cap=True) == (ordinary, False)


def test_all_shipped_universal_presets_have_the_versioned_settings_contract():
    required_keys = {
        "model_key",
        "prompt_preset_id",
        "system_prompt",
        "user_prompt",
        "trigger_word",
        "trigger_mode",
        "language",
        "source_language",
        "target_language",
        "caption_length",
        "avoid_list",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
        "do_sample",
        "fps",
        "max_frames",
        "max_pixels",
        "min_pixels",
        "use_audio_in_video",
        "scene_detect_enabled",
        "scene_threshold",
        "scene_min_len_s",
        "scene_max_len_s",
        "merge_short_scenes",
        "split_mode",
        "max_clip_duration_s",
        "sub_split_overlap_s",
        "caption_prefix",
        "caption_suffix",
        "replace_words",
        "output_formats",
        "save_clips",
        "save_reasoning",
        "overwrite_existing",
        "vram_preset",
        "attention_backend",
        "subprocess_mode",
        "keep_model_loaded",
        "idle_unload_minutes",
        "chat_system_prompt",
        "chat_temperature",
        "chat_top_p",
        "chat_top_k",
        "chat_max_new_tokens",
        "chat_enable_thinking",
        "context_tokens",
    }
    # Only Thinking presets spell out the caption-side reasoning switch.
    optional_keys = {"enable_thinking"}
    preset_files = sorted((Path(__file__).parents[1] / "presets_default").glob("*.json"))
    assert len(preset_files) == 13

    for preset_file in preset_files:
        payload = json.loads(preset_file.read_text(encoding="utf-8"))
        assert payload["_meta"]["format"] == "secourses_vcap_preset"
        assert payload["_meta"]["version"] == 1
        keys = set(payload["settings"])
        assert required_keys <= keys <= required_keys | optional_keys, preset_file.name
        if "thinking" in payload["settings"]["model_key"]:
            assert payload["settings"]["enable_thinking"] is True, preset_file.name
        else:
            assert "enable_thinking" not in keys, preset_file.name
        assert payload["settings"]["prompt_preset_id"] in EXPECTED_PRESET_IDS
        assert payload["settings"]["split_mode"] in {"copy", "precise"}
        assert set(payload["settings"]["output_formats"]) <= {"txt", "json", "srt"}
