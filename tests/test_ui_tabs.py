from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcap.core.registry import SettingsRegistry
from vcap.pipeline.job import JobResult
from vcap.ui.app import build_app
from vcap.ui.tabs.caption_tab import _result_summary
from vcap.ui.tabs.editor_tab import (
    filter_items,
    find_replace_preview,
    paginate_items,
    pagination_math,
    scan_folder,
)
from vcap.ui.tabs.health_tab import render_model_health
from vcap.ui.tabs.recover_tab import build_recovery_diff_table, present_recovery_settings


def test_cancelled_job_summary_reports_cancel_count() -> None:
    result = JobResult(
        items=[],
        counts={"done": 1, "skipped": 0, "failed": 0, "cancelled": 3},
        run_dir="run",
        metadata_path="metadata.json",
        elapsed=2.5,
    )
    label, message, status_class, eta = _result_summary(result)
    assert label == "Cancelled"
    assert message == "Cancelled: 3 cancelled, 1 done, 0 skipped, 0 failed in 2.5s"
    assert status_class == "vc-warn" and eta == "cancelled"


def test_result_summary_reports_nonzero_unsupported_count() -> None:
    result = JobResult(
        items=[],
        counts={"done": 0, "skipped": 3, "failed": 0, "unsupported": 1},
        run_dir="run",
        metadata_path="metadata.json",
        elapsed=0.1,
    )

    label, message, status_class, eta = _result_summary(result)

    assert label == "Complete"
    assert message == "Complete: 0 done, 3 skipped, 0 failed, 1 unsupported in 0.1s"
    assert status_class == "vc-ok" and eta == "done"


def test_editor_scanner_pairs_sidecars_unicode_and_run_layout(tmp_path: Path) -> None:
    (tmp_path / "clip2.mp4").write_bytes(b"media")
    (tmp_path / "clip2.txt").write_text("preferred caption", encoding="utf-8")
    (tmp_path / "clip2.json").write_text('{"text": "json caption"}', encoding="utf-8")
    (tmp_path / "clip10.mp4").write_bytes(b"media")

    unicode_media = tmp_path / "görüntü3.png"
    unicode_media.write_bytes(b"image")
    unicode_media.with_suffix(".txt").write_text("İstanbul çekimi", encoding="utf-8")
    (tmp_path / "orphan.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

    run_dir = tmp_path / "outputs" / "0001_qwen3"
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "clip_0001.mp4").write_bytes(b"clip")
    (run_dir / "caption.txt").write_text("run caption", encoding="utf-8")

    items = scan_folder(tmp_path, recursive=True)
    by_media = {Path(item["media_path"]).name: item for item in items if item.get("media_path")}

    assert by_media["clip2.mp4"]["caption"] == "preferred caption"
    assert [Path(path).suffix for path in by_media["clip2.mp4"]["caption_formats"]] == [".txt", ".json"]
    assert by_media["clip10.mp4"]["status"] == "empty"
    assert by_media["görüntü3.png"]["caption"] == "İstanbul çekimi"
    assert by_media["clip_0001.mp4"]["caption"] == "run caption"
    assert any(Path(item["caption_path"]).name == "orphan.srt" and item["status"] == "no media" for item in items)

    names = [Path(item.get("media_path") or item["caption_path"]).name for item in items]
    assert names.index("clip2.mp4") < names.index("clip10.mp4")


def test_editor_scans_caption_only_run_dirs_without_recursive_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "0001_qwen3"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text('{"items_results": []}', encoding="utf-8")
    (run_dir / "run_log.txt").write_text("not a caption", encoding="utf-8")
    (run_dir / "text_prompt.txt").write_text("editable prompt result", encoding="utf-8")
    (run_dir / "text_prompt.json").write_text(
        '{"text": "editable prompt result"}', encoding="utf-8"
    )

    items = scan_folder(tmp_path, recursive=False)

    assert len(items) == 1
    assert Path(items[0]["caption_path"]).name == "text_prompt.txt"
    assert items[0]["status"] == "no media"
    assert items[0]["media_path"] is None
    assert {Path(path).suffix for path in items[0]["caption_formats"]} == {".txt", ".json"}


def test_editor_filter_logic() -> None:
    items = [
        {"media_path": "one.mp4", "caption": "A bright red car", "chars": 16, "tokens": 4, "flag": "approved", "status": "ok"},
        {"media_path": "two.mp4", "caption": "blue boat", "chars": 9, "tokens": 3, "flag": None, "status": "empty"},
        {"media_path": "three.mp4", "caption": "RED dress in rain", "chars": 17, "tokens": 5, "flag": "rejected", "status": "failed"},
    ]

    assert len(filter_items(items, {"search": "red"})) == 2
    assert filter_items(items, {"search": r"blue\s+boat", "regex": True})[0]["media_path"] == "two.mp4"
    assert filter_items(items, {"min_tokens": 5, "flag": "rejected", "status": "failed"}) == [items[2]]
    assert filter_items(items, {"flag": "unflagged"}) == [items[1]]
    with pytest.raises(Exception):
        filter_items(items, {"search": "[", "regex": True})


def test_editor_pagination_math() -> None:
    assert pagination_math(0, 99, 25) == (1, 1, 0, 0)
    assert pagination_math(214, 3, 50) == (3, 5, 100, 150)
    assert pagination_math(214, 99, 100) == (3, 3, 200, 214)
    page, selected, pages = paginate_items(list(range(11)), 2, 5)
    assert page == [5, 6, 7, 8, 9]
    assert (selected, pages) == (2, 3)


def test_find_replace_preview_counts_files_and_matches() -> None:
    items = [
        {"media_path": "one.mp4", "caption": "cat cat catalog"},
        {"media_path": "two.mp4", "caption": "A Cat waits"},
        {"media_path": "three.mp4", "caption": "dog"},
    ]

    result = find_replace_preview(items, "cat", "fox", whole_words=True)
    assert result["files_changed"] == 2
    assert result["replacement_count"] == 3
    assert result["previews"][0]["new"] == "fox fox catalog"


def test_recover_diff_table_from_metadata_json(tmp_path: Path) -> None:
    registry = SettingsRegistry()
    registry.register("model_key", object(), "default", kind="str", description="Model")
    registry.register("fps", object(), 2.0, kind="float", minimum=0.1, maximum=30.0, description="FPS")
    registry.register("recursive", object(), False, kind="bool", description="Recursive")

    metadata = {
        "_meta": {"format": "secourses_vcap_metadata", "version": 1},
        "app_version": "1.0.0",
        "model_info": {"variant_key": "qwen3_omni_instruct_int8"},
        "settings": {"fps": "4.5", "recursive": "true"},
        "items_results": [],
        "timings": {},
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = build_recovery_diff_table(
        path,
        registry,
        {"model_key": "default", "fps": 4.5, "recursive": False},
    )
    by_key = {row["key"]: row for row in rows}
    assert by_key["model_key"]["saved_value"] == "qwen3_omni_instruct_int8"
    assert by_key["model_key"]["different"] is True
    assert by_key["fps"]["different"] is False
    assert by_key["recursive"]["saved_value"] is True
    assert by_key["recursive"]["different"] is True


def test_recover_model_only_includes_block_swap_budget_keys() -> None:
    registry = SettingsRegistry()
    registry.register(
        "vram_reserve_gb", object(), 2.0, section="runtime", kind="float", description="Reserve"
    )
    registry.register(
        "swap_slots", object(), 2, section="runtime", kind="int", choices=[2, 3], description="Slots"
    )

    values, warnings = present_recovery_settings(
        {"settings": {"vram_reserve_gb": "4.5", "swap_slots": "3"}},
        registry,
        model_prompt_only=True,
        available_gpu_indices=[],
    )

    assert values == {"vram_reserve_gb": 4.5, "swap_slots": 3}
    assert warnings == []


def test_health_formats_worker_block_swap_summary() -> None:
    report = render_model_health(
        {
            "loaded_variant": "qwen3_omni_instruct_int8",
            "block_swap": {
                "mode": "block_swap",
                "layer_count": 48,
                "resident_layers": 39,
                "swapped_layers": 9,
                "slots": 2,
                "pinned_gib": 5.24,
                "expected_peak_gib": 29.5,
                "reserve_gib": 2.0,
            },
        }
    )

    assert "qwen3_omni_instruct_int8" in report
    assert "39/48 resident" in report
    assert "9 swapped" in report
    assert "5.24 GiB pinned" in report


def test_build_app_with_all_tabs_smoke() -> None:
    demo = build_app()
    try:
        assert demo.vcap_context.states["editor_state"] is not None
        assert len(demo.vcap_context.registry.keys()) > 60
        # The header theme button is a browser-only toggle that writes the
        # new mode into the Global Settings radio.
        config = demo.get_config_file()
        button = next(
            item for item in config["components"]
            if item.get("props", {}).get("elem_id") == "vc_toggle_theme"
        )
        assert button["props"]["value"] == "🌗 Light / dark theme"
        toggle = next(
            item for item in config["dependencies"]
            if any(target[0] == button["id"] for target in item["targets"])
        )
        assert toggle["backend_fn"] is False
        assert toggle["outputs"] == [demo.vcap_context.states["theme_component"]._id]
        assert "secourses_theme_mode" in toggle["js"]
    finally:
        demo.vcap_context.pipeline.shutdown()
