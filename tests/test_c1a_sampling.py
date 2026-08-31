from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vcap.core.media import find_ffmpeg, read_video_frames


@pytest.fixture(scope="module")
def sampling_video() -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is required")
    root = Path(__file__).parents[1] / "temp" / "codex_C1a"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "synthetic_sampling.mp4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x96:rate=12:duration=4",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-g",
        "120",
        "-keyint_min",
        "120",
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return target


def test_uniform_is_exact_seek_sample_with_matching_fps(sampling_video: Path) -> None:
    result = read_video_frames(
        sampling_video,
        sampling="uniform",
        fps=3.0,
        max_frames=6,
        min_frames=2,
        start_s=0.5,
        end_s=3.5,
        size_multiple=8,
    )
    assert len(result.frames) == 6
    assert len(result.timestamps) == 6
    assert result.fps_effective == pytest.approx(2.0)
    assert result.timestamps == sorted(result.timestamps)


def test_keyframe_falls_back_to_uniform_when_trim_has_no_iframes(sampling_video: Path) -> None:
    result = read_video_frames(
        sampling_video,
        sampling="keyframe",
        max_frames=6,
        min_frames=2,
        start_s=0.5,
        end_s=3.5,
        size_multiple=8,
    )
    assert len(result.frames) == 6
    assert result.fps_effective == pytest.approx(2.0)
    assert result.timestamps[0] >= 0.4
    assert result.timestamps[-1] >= 3.3


@pytest.mark.parametrize("strategy", ["fps", "adaptive"])
def test_fps_and_adaptive_are_capped_and_report_real_rate(
    sampling_video: Path,
    strategy: str,
) -> None:
    result = read_video_frames(
        sampling_video,
        sampling=strategy,
        target_fps=3.0,
        max_frames=6,
        min_frames=2,
        start=0.5,
        end=3.5,
        size_multiple=8,
    )
    assert 2 <= len(result.frames) <= 6
    assert result.fps_effective == pytest.approx(len(result.frames) / 3.0)
