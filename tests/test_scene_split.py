from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vcap.core.media import find_ffmpeg, find_ffprobe, probe_media
from vcap.core.scene_split import (
    SceneDetectParams,
    SceneRange,
    cap_scene_lengths,
    detect_scenes,
    enforce_model_limit,
    extract_segment_audio,
    fixed_length_segments,
    merge_short_scenes,
    plan_segments,
    split_video,
)


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


@pytest.fixture(scope="module")
def cut_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not find_ffprobe():
        pytest.skip("ffmpeg/ffprobe are required")
    path = tmp_path_factory.mktemp("scene_split") / "hard_cut.mp4"
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
            "color=c=red:size=160x96:rate=10:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=160x96:rate=10:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2.4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "50",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


def test_detection_planning_and_range_helpers(cut_video: Path) -> None:
    progress: list[tuple[float, str]] = []
    scenes = detect_scenes(
        cut_video,
        SceneDetectParams(threshold=5, min_scene_len_s=0.2, merge_short_scenes=False),
        progress_cb=lambda fraction, message: progress.append((fraction, message)),
    )
    assert len(scenes) >= 2
    assert scenes[0].start_s == pytest.approx(0, abs=0.11)
    assert scenes[-1].end_s == pytest.approx(2.4, abs=0.11)
    assert progress and progress[-1][0] == 1.0

    merged = merge_short_scenes([SceneRange(0, 0.4), SceneRange(0.4, 2)], 1.0)
    assert merged == [SceneRange(0, 2)]
    capped = cap_scene_lengths([SceneRange(0, 5)], 2, overlap_s=0.25)
    assert len(capped) == 3
    assert all(scene.duration_s <= 2.0 + 1e-9 for scene in capped)
    fixed = fixed_length_segments(5, 2, 0.5)
    assert [(item.start_s, item.end_s) for item in fixed] == [
        (0.0, 2.0),
        (1.5, 3.5),
        (3.0, 5.0),
    ]

    limited, warnings = enforce_model_limit([SceneRange(0, 5)], 2, 0, "AVoCaDO")
    assert len(limited) == 3
    assert warnings and "split into 3 clips" in warnings[0]
    plan = plan_segments(
        probe_media(cut_video),
        mode="fixed",
        fixed_chunk_s=1,
        model_max_duration_s=2,
        trim_start_s=0.2,
        trim_end_s=2.2,
    )
    assert len(plan.segments) == 2
    assert plan.source_duration == pytest.approx(2.4, abs=0.1)


def test_copy_and_precise_split_verify_frames(cut_video: Path, tmp_path: Path) -> None:
    segments = [SceneRange(0, 1.2), SceneRange(1.2, 2.4)]
    copied = split_video(cut_video, segments, tmp_path / "copy", mode="copy")
    precise = split_video(cut_video, segments, tmp_path / "precise", mode="precise")
    assert len(copied) == len(precise) == 2
    assert (tmp_path / "copy" / "split_manifest.json").is_file()
    assert (tmp_path / "precise" / "split_manifest.json").is_file()
    for clip in copied + precise:
        assert clip.path.is_file()
        assert clip.expected_frames == 12
        assert clip.actual_frames is not None
        assert abs(clip.actual_frames - clip.expected_frames) <= 1
        assert probe_media(clip.path).has_video
    assert all(clip.actual_frames == clip.expected_frames for clip in precise)

    audio = extract_segment_audio(cut_video, segments[:1], tmp_path / "audio")
    assert len(audio) == 1 and probe_media(audio[0]).has_audio

