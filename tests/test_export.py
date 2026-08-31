from __future__ import annotations

import re
import tomllib
from pathlib import Path

from vcap.core.export import (
    discover_dataset_folders,
    export_dataset,
    read_flags,
    write_flags,
    write_kohya_musubi_toml,
)


def test_discovery_and_musubi_toml(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    video_dir = root / "2_ohwx"
    image_dir = root / "images"
    video_dir.mkdir(parents=True)
    image_dir.mkdir()
    (video_dir / "clip.mp4").write_bytes(b"video")
    (image_dir / "still.png").write_bytes(b"image")

    folders = discover_dataset_folders(root)
    by_path = {folder.path.name: folder for folder in folders}
    assert by_path["2_ohwx"].num_repeats == 2
    assert by_path["2_ohwx"].name == "ohwx"
    assert by_path["images"].num_repeats == 1

    output = write_kohya_musubi_toml(
        root,
        tmp_path / "dataset.toml",
        kind="video",
        resolution=(960, 544),
        target_frames=[81],
        source_fps=16,
    )
    text = output.read_text(encoding="utf-8")
    assert text.startswith("[[datasets]]")
    assert "video_directory" in text and "/2_ohwx" in text
    assert not re.search(r",\s*]", text)
    config = tomllib.loads(text)
    assert config["datasets"][0]["num_repeats"] == 2
    assert config["datasets"][0]["target_frames"] == [81]
    assert config["general"]["resolution"] == [960, 544]


def test_flags_round_trip_and_approved_export(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    approved = source / "approved.mp4"
    rejected = source / "rejected.mp4"
    approved.write_bytes(b"approved media")
    rejected.write_bytes(b"rejected media")
    (approved.with_suffix(".txt")).write_text("approved caption", encoding="utf-8")
    (rejected.with_suffix(".txt")).write_text("rejected caption", encoding="utf-8")

    flags = {"approved.mp4": "approved", "nested/rejected.mp4": {"approved": False}}
    path = write_flags(source, flags)
    assert path.name == ".vcap_flags.json"
    assert read_flags(source) == flags

    report = export_dataset(
        [
            {"path": approved, "approved": True},
            {"path": rejected, "approved": False},
        ],
        tmp_path / "exported",
        only_approved=True,
    )
    assert report.exported == 1 and report.rejected == 1 and report.skipped == 0
    assert report.media_files[0].read_bytes() == b"approved media"
    assert report.caption_files[0].read_text(encoding="utf-8").strip() == "approved caption"
