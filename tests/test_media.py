from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import vcap.core.media as media_module
from vcap.core.logs import get_log
from vcap.core.media import (
    MediaInfo,
    extract_audio,
    extract_frames_to_files,
    find_ffmpeg,
    find_ffprobe,
    make_thumbnail,
    preview_safe_media,
    probe_media,
    read_audio,
    read_video_frames,
    run_ffmpeg,
    trim_media,
)


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


@pytest.fixture(scope="module")
def media_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not find_ffprobe():
        pytest.skip("ffmpeg/ffprobe are required")
    root = tmp_path_factory.mktemp("tiny_media")
    with_audio = root / "with_audio.mp4"
    no_audio = root / "no_audio.mp4"
    wav = root / "tone.wav"
    image = root / "still.png"
    incompatible = root / "incompatible.mkv"

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
            "testsrc2=size=96x64:rate=10:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1.2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(with_audio),
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
            "testsrc2=size=96x64:rate=10:duration=1.2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "5",
            "-pix_fmt",
            "yuv420p",
            str(no_audio),
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
            "sine=frequency=880:sample_rate=16000:duration=1.0",
            str(wav),
        ]
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(no_audio),
            "-c:v",
            "ffv1",
            str(incompatible),
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
            "testsrc2=size=80x40:rate=1",
            "-frames:v",
            "1",
            "-threads",
            "1",
            str(image),
        ]
    )
    return {
        "video": with_audio,
        "silent": no_audio,
        "wav": wav,
        "image": image,
        "incompatible": incompatible,
    }


def test_probe_video_audio_image_and_bad_file(media_files: dict[str, Path], tmp_path: Path) -> None:
    video = probe_media(media_files["video"])
    assert video.kind == "video"
    assert video.has_video and video.has_audio
    assert video.width == 96 and video.height == 64
    assert video.duration and video.duration > 1
    assert video.fps and video.fps > 0

    silent = probe_media(media_files["silent"])
    assert silent.kind == "video_no_audio" and not silent.has_audio
    audio = probe_media(media_files["wav"])
    assert audio.kind == "audio" and audio.audio_sample_rate == 16000
    image = probe_media(media_files["image"])
    assert image.kind == "image" and (image.width, image.height) == (80, 40)
    bad = tmp_path / "bad.mp4"
    bad.write_text("not media", encoding="utf-8")
    assert probe_media(bad).kind == "unknown"


def test_read_frames_uniform_fps_and_keyframes(media_files: dict[str, Path]) -> None:
    uniform = read_video_frames(
        media_files["video"], num_frames=4, max_pixels=96 * 64, size_multiple=8, sampling="uniform"
    )
    assert uniform.frames.shape[0] == 4
    assert uniform.frames.dtype == np.uint8
    assert uniform.frames.shape[-1] == 3
    assert all(dimension % 8 == 0 for dimension in uniform.resized_size)
    assert len(uniform.timestamps) == 4

    fps = read_video_frames(
        media_files["video"], target_fps=3, max_frames=4, size_multiple=8, sampling="fps"
    )
    assert 2 <= fps.frames.shape[0] <= 4
    keyframes = read_video_frames(media_files["video"], max_frames=4, size_multiple=8, sampling="keyframe")
    assert 1 <= keyframes.frames.shape[0] <= 4


def test_audio_extract_and_read(media_files: dict[str, Path], tmp_path: Path) -> None:
    samples = read_audio(media_files["wav"], 16000)
    assert samples.dtype == np.float32
    assert 15000 <= samples.shape[0] <= 17000
    extracted = extract_audio(media_files["video"], tmp_path / "extracted.wav", start=0.1, end=0.8)
    extracted_samples = read_audio(extracted, 16000)
    assert 9000 <= extracted_samples.shape[0] <= 13000


def test_trim_copy_precise_and_preview(media_files: dict[str, Path], tmp_path: Path) -> None:
    copied = trim_media(media_files["video"], tmp_path / "copy.mp4", 0.1, 0.9, mode="copy")
    precise = trim_media(media_files["video"], tmp_path / "precise.mp4", 0.1, 0.9, mode="precise")
    assert probe_media(copied).has_video
    precise_info = probe_media(precise)
    assert precise_info.has_video and precise_info.duration and precise_info.duration > 0.6

    preview = preview_safe_media(media_files["incompatible"], tmp_path / "previews")
    assert preview.suffix == ".png" and preview.is_file()
    with Image.open(preview) as poster:
        assert poster.width == 960
    assert preview_safe_media(media_files["incompatible"], tmp_path / "previews") == preview


def test_frame_files_thumbnail_and_ffmpeg_progress(media_files: dict[str, Path], tmp_path: Path) -> None:
    files = extract_frames_to_files(
        media_files["video"],
        tmp_path / "frames",
        num_frames=2,
        size_multiple=8,
        image_format="jpg",
    )
    assert [path.name for path in files] == ["frame_000001.jpg", "frame_000002.jpg"]
    assert all(path.is_file() for path in files)
    thumbnail = make_thumbnail(media_files["video"], tmp_path / "thumb.png", width=64)
    assert thumbnail.is_file()
    with Image.open(thumbnail) as image:
        assert image.width == 64

    progress: list[float] = []
    copied = tmp_path / "progress_copy.mkv"
    run_ffmpeg(
        ["-y", "-i", str(media_files["silent"]), "-c", "copy", str(copied)],
        progress.append,
        total_duration=1.2,
    )
    assert copied.is_file()
    assert progress and progress[-1] == 1.0


def test_unicode_voyager_dimensions_are_unambiguous_in_ui() -> None:
    source = Path(
        "F:/SECourses_Video_Captioner_Pro_TEMP/test_media/"
        "vöyager ünicode 日本語 テスト.mp4"
    )
    if not source.is_file():
        pytest.skip("integration media fixture is not present")
    from vcap.ui.components import _media_info_markdown

    info = probe_media(source)
    assert (info.width, info.height) == (960, 720)
    rendered = _media_info_markdown(info)
    assert "960 x 720" in rendered
    assert "980 x 720" not in rendered


def test_preview_safe_media_remuxes_h264_with_stream_copy_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    info = MediaInfo(
        source,
        "video",
        duration=20.0,
        width=1280,
        height=720,
        fps=24.0,
        has_video=True,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
        container="matroska",
    )
    commands: list[list[str]] = []

    def fake_ffmpeg(command: list[str], *_args, **_kwargs) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"remux")

    before = get_log().revision
    monkeypatch.setattr(media_module, "probe_media", lambda _path: info)
    monkeypatch.setattr(media_module, "run_ffmpeg", fake_ffmpeg)

    preview = preview_safe_media(source, tmp_path / "cache")

    assert preview.suffix == ".mp4" and preview.read_bytes() == b"remux"
    assert len(commands) == 1
    assert commands[0].count("copy") == 2
    assert "libx264" not in commands[0] and "aac" not in commands[0]
    lines, _revision = get_log().snapshot(before)
    assert any("Preparing preview" in line for line in lines)


def test_preview_safe_media_uses_poster_for_vp9_without_running_remux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    info = MediaInfo(
        source,
        "video",
        duration=20.0,
        width=1280,
        height=720,
        fps=24.0,
        has_video=True,
        has_audio=True,
        video_codec="vp9",
        audio_codec="opus",
        container="mp4",
    )
    poster = tmp_path / "poster.png"

    def fake_thumbnail(_source: Path, _target: Path, **_kwargs) -> Path:
        poster.write_bytes(b"poster")
        return poster

    monkeypatch.setattr(media_module, "probe_media", lambda _path: info)
    monkeypatch.setattr(media_module, "make_thumbnail", fake_thumbnail)
    monkeypatch.setattr(
        media_module,
        "run_ffmpeg",
        lambda *_args, **_kwargs: pytest.fail("video remux or re-encode was attempted"),
    )

    preview = preview_safe_media(source, tmp_path / "cache")

    assert preview == poster and preview.read_bytes() == b"poster"
