from __future__ import annotations

from pathlib import Path

from vcap.core.media import filter_media_paths
from vcap.pipeline.job import InputItem, JobSpec, OutputSpec
from vcap.pipeline.runner import _resolve_inputs


def test_filter_media_paths_combines_kinds_and_semicolon_globs(tmp_path: Path) -> None:
    paths = [
        tmp_path / "clip_01.mp4",
        tmp_path / "clip_02.wav",
        tmp_path / "poster.png",
        tmp_path / "notes.md",
    ]
    assert filter_media_paths(paths, ("video", "audio"), "clip_*.mp4;clip_02.*") == paths[:2]
    assert filter_media_paths(paths, ("text",), "") == [paths[3]]
    assert filter_media_paths(paths, (), "") == []


def test_batch_resolution_filters_after_paired_sidecar_exclusion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("clip.mp4", "clip.txt", "notes.md", "poster.png"):
        (source / name).write_bytes(b"x")
    messages: list[str] = []
    spec = JobSpec.from_settings(
        {
            "model_key": "qwen3_omni_instruct_int4",
            "batch_include_kinds": ["video"],
            "batch_name_filter": "clip_*;clip.mp4",
        },
        [InputItem(source)],
        OutputSpec(kind="batch", outputs_root=tmp_path / "out", recursive=True),
    )
    resolved = _resolve_inputs(spec, messages.append)
    assert [entry.path.name for entry in resolved if entry.path] == ["clip.mp4"]
    assert messages == ["Batch filters skipped 2 file(s)."]
