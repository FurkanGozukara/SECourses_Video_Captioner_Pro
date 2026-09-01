"""Live output-state plumbing: run folder and item artifacts reach the UI during a job."""

from pathlib import Path

from vcap.core.progress import ProgressTracker
from vcap.pipeline.job import ItemResult
from vcap.pipeline.runner import _Emitter, _last_saved_clip
from vcap.ui.tabs.caption_tab import _merge_live_outputs


def test_emitter_payload_carries_run_dir() -> None:
    emitter = _Emitter(None, ProgressTracker(1, ["clip"]))
    assert emitter._payload(0)["run_dir"] is None
    emitter.run_dir = "G:/outputs/0001_run"
    assert emitter._payload(0)["run_dir"] == "G:/outputs/0001_run"


def test_merge_live_outputs_picks_up_run_dir_then_item_artifacts() -> None:
    state: dict = {}
    assert _merge_live_outputs(state, {"run_dir": "R"}, None) is True
    assert _merge_live_outputs(state, {"run_dir": "R"}, None) is False
    assert state == {"run_dir": "R"}
    assert _merge_live_outputs(state, {"run_dir": "R", "outputs": {"txt": "R/a.txt"}}, "running") is False
    assert _merge_live_outputs(
        state, {"run_dir": "R", "outputs": {"txt": "R/a.txt"}, "clip_path": "R/c.mp4"}, "done"
    ) is True
    assert state == {"run_dir": "R", "caption_path": "R/a.txt", "clip_path": "R/c.mp4"}


def test_last_saved_clip_returns_newest_existing_media(tmp_path: Path) -> None:
    kept = tmp_path / "clip_0002.mp4"
    kept.write_bytes(b"x")
    result = ItemResult(0, "video.mp4", "video", "done", segments=[
        {"media_path": str(tmp_path / "missing_0001.mp4")},
        {"media_path": str(kept)},
        {"media_path": str(tmp_path / "deleted_0003.mp4")},
    ])
    assert _last_saved_clip(result) == str(kept)
    assert _last_saved_clip(ItemResult(0, "v", "video", "done")) is None
