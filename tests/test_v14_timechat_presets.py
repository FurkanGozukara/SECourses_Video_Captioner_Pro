from __future__ import annotations

import json

from vcap.prompts.postprocess import (
    POST_PROCESSORS,
    timechat_chapters,
    timechat_flatten_av,
    timechat_flatten_motion_camera,
    timechat_speech_only,
)
from vcap.prompts.presets import TIMECHAT_OFFICIAL_PROMPT, get_preset


SAMPLE = [
    {
        "timestamp": "00:10-00:15",
        "segment_detail_caption": "A person opens a door",
        "camera_state": "The camera pans right",
        "video_background": "A bright hallway",
        "storyline": "The visitor enters",
        "shooting_style": "A continuous take",
        "speech_content": "",
        "acoustics_content": "A latch clicks",
    },
    {
        "timestamp": "00:00-00:10",
        "segment_detail_caption": "A person approaches a door",
        "camera_state": "A static wide shot",
        "video_background": "An exterior porch",
        "storyline": "The visitor arrives",
        "shooting_style": "Natural lighting",
        "speech_content": "Hello there.",
        "acoustics_content": "Footsteps and wind",
    },
]


def test_timechat_motion_and_av_flatteners_keep_only_requested_fields() -> None:
    raw = json.dumps(SAMPLE)
    motion = timechat_flatten_motion_camera(raw)
    assert motion.text.index("approaches") < motion.text.index("opens")
    assert "static wide shot" in motion.text
    assert "exterior porch" not in motion.text
    assert "Hello there" not in motion.text
    assert "Footsteps" not in motion.text

    audiovisual = timechat_flatten_av(raw)
    assert audiovisual.text.index("approaches") < audiovisual.text.index("opens")
    assert "Hello there" in audiovisual.text
    assert "Footsteps and wind" in audiovisual.text
    assert "exterior porch" not in audiovisual.text
    assert "Natural lighting" not in audiovisual.text


def test_timechat_speech_and_chapter_variants() -> None:
    raw = json.dumps(SAMPLE)
    speech = timechat_speech_only(raw)
    assert speech.text == "Hello there."
    assert speech.segments == [(0.0, 10.0, "Hello there.")]

    chapters = timechat_chapters(raw)
    assert chapters.text.splitlines() == [
        "00:00-00:10 The visitor arrives",
        "00:10-00:15 The visitor enters",
    ]
    assert chapters.segments == []


def test_timechat_variant_presets_keep_the_official_model_prompt() -> None:
    contracts = {
        "timechat_flatten_motion_camera": "text",
        "timechat_flatten_av": "text",
        "timechat_speech_only": "srt_segments",
        "timechat_chapters": "text",
    }
    for preset_id, output_format in contracts.items():
        preset = get_preset(preset_id)
        assert preset.group == "Model-native"
        assert preset.applies_to_models == ("timechat",)
        assert preset.modalities == ("video",)
        assert preset.output_format == output_format
        assert preset.user_prompt == TIMECHAT_OFFICIAL_PROMPT
        assert preset.post_processor in POST_PROCESSORS
