from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vcap.core.media import find_ffmpeg, find_ffprobe, probe_media
from vcap.core.preprocess import (
    AutoRejectRules,
    FrameSamplingParams,
    analyze_clip_quality,
    ensure_audio_track,
    fits_context,
    normalize_clip_for_model,
    plan_frame_sampling,
    should_reject,
    smart_resize,
    token_budget_estimate,
)


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


@pytest.fixture(scope="module")
def quality_media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not find_ffprobe():
        pytest.skip("ffmpeg/ffprobe are required")
    root = tmp_path_factory.mktemp("quality")
    black = root / "black_silent.mp4"
    visual = root / "visual_no_audio.mp4"
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=black:size=96x64:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(black),
        ]
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(visual),
        ]
    )
    return {"black": black, "visual": visual}


def test_frame_plan_resize_and_token_budget() -> None:
    count, effective, timestamps = plan_frame_sampling(
        10,
        30,
        FrameSamplingParams(strategy="fps", fps=2, max_frames=16, min_frames=4, frame_factor=2),
    )
    assert count == 16
    assert effective == pytest.approx(1.6)
    assert len(timestamps) == 16 and timestamps[-1] < 10
    h, w = smart_resize(1080, 1920, 28, 4 * 28 * 28, 380 * 28 * 28)
    assert h % 28 == w % 28 == 0
    assert 4 * 28 * 28 <= h * w <= 380 * 28 * 28

    budget = token_budget_estimate("qwen3-omni", 20, 320, 640, 10)
    assert budget["audio_tokens"] == 130
    assert budget["video_tokens"] == 2000
    assert fits_context(budget, 4096, 1000)
    assert not fits_context(budget, 3000, 1000)


def test_quality_rejection_and_model_normalization(quality_media: dict[str, Path], tmp_path: Path) -> None:
    quality = analyze_clip_quality(quality_media["black"], sample_frames=4)
    assert quality.has_audio
    assert quality.black_ratio > 0.95
    assert quality.silence_ratio > 0.95
    assert quality.static_score < 0.1
    rejected, reasons = should_reject(
        quality,
        AutoRejectRules(
            max_black_ratio=0.8,
            max_static_score=0.5,
            min_sharpness=1,
            max_silence_ratio=0.8,
        ),
    )
    assert rejected
    assert any("black" in reason for reason in reasons)
    assert any("silent" in reason for reason in reasons)

    normalized = normalize_clip_for_model(
        quality_media["visual"],
        tmp_path / "normalized.mp4",
        target_fps=5,
        max_pixels=12 * 28 * 28,
        min_pixels=4 * 28 * 28,
        size_multiple=28,
        keep_audio=True,
    )
    info = probe_media(normalized)
    assert info.video_codec == "h264"
    assert info.fps == pytest.approx(5, rel=0.01)
    assert info.width and info.width % 28 == 0
    assert info.height and info.height % 28 == 0
    assert info.has_audio and info.audio_sample_rate == 16000 and info.audio_channels == 1

    ensured = ensure_audio_track(quality_media["visual"], tmp_path / "ensured.mp4")
    assert probe_media(ensured).has_audio

