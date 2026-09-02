from __future__ import annotations

import os
from pathlib import Path

import pytest

from vcap.core import app_settings, media
from vcap.core.captions_post import Segment, clamp_segments_to_window, to_srt, to_vtt


def test_precise_trim_uses_selected_video_and_audio_encoder_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"input")
    target = tmp_path / "trimmed.mp4"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        media,
        "probe_media",
        lambda path: media.MediaInfo(
            Path(path),
            "video",
            duration=8.0,
            has_video=True,
            has_audio=True,
        ),
    )
    monkeypatch.setattr(media, "resolve_video_encoder", lambda _codec: "h264_nvenc")
    monkeypatch.setattr(
        media,
        "run_ffmpeg",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )

    result = media.trim_media(
        source,
        target,
        1.25,
        5.5,
        mode="precise",
        encode_codec="h264_nvenc",
        encode_crf=27,
        encode_preset="slow",
        encode_audio_bitrate="256k",
    )

    assert result == target.resolve()
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p6"
    assert command[command.index("-cq") + 1] == "27"
    assert command[command.index("-b:a") + 1] == "256k"


def test_configured_ffmpeg_path_prefers_environment_then_app_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    env_dir = tmp_path / "environment"
    setting_dir = tmp_path / "setting"
    env_dir.mkdir()
    setting_dir.mkdir()
    env_binary = env_dir / executable
    setting_binary = setting_dir / executable
    env_binary.write_bytes(b"")
    setting_binary.write_bytes(b"")
    monkeypatch.setenv("VCAP_FFMPEG_PATH", str(env_dir))
    monkeypatch.setattr(
        app_settings,
        "load_app_settings",
        lambda *_args, **_kwargs: {"ffmpeg_path": str(setting_dir)},
    )

    media.find_ffmpeg.cache_clear()
    try:
        assert media.find_ffmpeg() == str(env_binary.resolve())
        monkeypatch.delenv("VCAP_FFMPEG_PATH")
        media.find_ffmpeg.cache_clear()
        assert media.find_ffmpeg() == str(setting_binary.resolve())
    finally:
        media.find_ffmpeg.cache_clear()
        media.find_ffprobe.cache_clear()


def test_subtitle_minimum_duration_and_word_wrapping() -> None:
    cues = clamp_segments_to_window(
        [Segment(9.9, 10.0, "alpha beta gamma delta")],
        0.0,
        10.0,
        min_duration_s=0.5,
    )

    assert cues == [Segment(9.5, 10.0, "alpha beta gamma delta")]
    assert "alpha beta\ngamma delta" in to_srt(cues, max_line_chars=11)
    assert "alpha beta\ngamma delta" in to_vtt(cues, max_line_chars=11)
