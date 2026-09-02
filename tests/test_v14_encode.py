from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vcap.core import media, preprocess
from vcap.core.media import MediaError, NVENC_PRESET_MAP, video_encode_args


def test_nvenc_preset_and_quality_argument_mapping() -> None:
    expected = {
        "ultrafast": "p1",
        "superfast": "p2",
        "veryfast": "p3",
        "faster": "p4",
        "fast": "p5",
        "medium": "p5",
        "slow": "p6",
        "slower": "p7",
    }
    assert NVENC_PRESET_MAP == expected
    for preset, mapped in expected.items():
        args = video_encode_args("h264_nvenc", 21, preset)
        assert args == [
            "-c:v",
            "h264_nvenc",
            "-preset",
            mapped,
            "-rc",
            "vbr",
            "-cq",
            "21",
        ]
    assert video_encode_args("libx265", 19, "slow") == [
        "-c:v",
        "libx265",
        "-preset",
        "slow",
        "-crf",
        "19",
    ]


def test_encoder_probe_is_cached_and_missing_nvenc_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return SimpleNamespace(
            returncode=0,
            stdout=" V..... libx264 H.264\n A..... aac AAC\n",
            stderr="",
        )

    media.ffmpeg_encoders.cache_clear()
    monkeypatch.setattr(media, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)
    assert media.resolve_video_encoder("h264_nvenc") == "libx264"
    assert media.resolve_video_encoder("hevc_nvenc") == "libx264"
    assert len(calls) == 1
    assert calls[0][-2:] == ["-hide_banner", "-encoders"]
    media.ffmpeg_encoders.cache_clear()


def test_normalize_retries_encoder_failure_with_libx264(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    target = tmp_path / "normalized.mp4"
    info = SimpleNamespace(
        has_video=True,
        has_audio=True,
        width=64,
        height=64,
        fps=24.0,
        duration=1.0,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48_000,
    )
    commands: list[list[str]] = []

    def fake_ffmpeg(arguments, **_kwargs):
        commands.append(list(arguments))
        if len(commands) == 1:
            raise MediaError("Error while opening encoder for output stream")

    monkeypatch.setattr(preprocess, "probe_media", lambda _path: info)
    monkeypatch.setattr(preprocess, "resolve_video_encoder", lambda _codec: "h264_nvenc")
    monkeypatch.setattr(preprocess, "run_ffmpeg", fake_ffmpeg)
    assert preprocess.normalize_clip_for_model(
        source,
        target,
        target_fps=12,
        max_pixels=4096,
        min_pixels=1024,
        size_multiple=32,
        keep_audio=True,
        audio_sr=48_000,
        encoder="h264_nvenc",
        crf=20,
        preset="slow",
        audio_bitrate="256k",
    ) == target.resolve()
    assert commands[0][commands[0].index("-c:v") + 1] == "h264_nvenc"
    assert commands[1][commands[1].index("-c:v") + 1] == "libx264"
    assert "-cq" in commands[0] and "-crf" in commands[1]
    assert commands[0][commands[0].index("-b:a") + 1] == "256k"
