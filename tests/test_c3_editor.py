from __future__ import annotations

from pathlib import Path

from vcap.ui.tabs.editor_tab import (
    editor_export_handler,
    editor_filter_handler,
    editor_flag_handler,
    editor_save_handler,
    new_editor_state,
    scan_folder,
)


def _mirrored_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "outputs" / "batch_captions" / "unicöde 日本語"
    root.mkdir(parents=True)
    for index, stem in enumerate(("clip one", "vöyager", "テスト"), start=1):
        (root / f"{stem}.mp4").write_bytes(f"fake-{index}".encode())
        (root / f"{stem}.txt").write_text(f"caption {index}\n", encoding="utf-8")
    return root


def _scanned_state(tmp_path: Path) -> tuple[Path, dict]:
    root = _mirrored_fixture(tmp_path)
    state = new_editor_state(root)
    state["items"] = scan_folder(root, recursive=True)
    state["selected_index"] = 0
    return root, state


def test_filter_errors_always_return_ten_outputs(tmp_path: Path) -> None:
    _, state = _scanned_state(tmp_path)

    invalid_regex = editor_filter_handler(
        state,
        ["[", True, "", "", "", "", "all", "all"],
        preview_cache=tmp_path,
    )
    non_finite = editor_filter_handler(
        state,
        ["", False, float("nan"), "", "", "", "all", "all"],
        preview_cache=tmp_path,
    )

    assert len(invalid_regex) == 10
    assert len(non_finite) == 10
    assert "Invalid filter" in invalid_regex[-1]
    assert "finite" in non_finite[-1]


def test_flag_without_selection_always_returns_eleven_outputs(tmp_path: Path) -> None:
    state = new_editor_state(tmp_path)

    assert len(editor_flag_handler(state, "approved", default_folder=tmp_path)) == 11
    rejected = editor_flag_handler(state, "rejected", default_folder=tmp_path)
    assert len(rejected) == 11
    assert rejected[-1] == "No caption is selected."


def test_unicode_mirrored_scan_and_editor_smoke(tmp_path: Path) -> None:
    root, state = _scanned_state(tmp_path)
    assert len(state["items"]) == 3
    assert all(Path(item["media_path"]).parent == root for item in state["items"])

    saved = editor_save_handler(state, "edited Ω caption")
    assert len(saved) == 5
    saved_state = saved[0]
    first_caption = Path(saved_state["items"][0]["caption_path"])
    assert first_caption.read_text(encoding="utf-8") == "edited Ω caption\n"

    flagged = editor_flag_handler(
        saved_state,
        "approved",
        default_folder=root,
        preview_cache=tmp_path / "previews",
    )
    assert len(flagged) == 11
    flagged_state = flagged[0]
    assert flagged_state["items"][0]["flag"] == "approved"

    export_root = tmp_path / "approved export"
    message = editor_export_handler(flagged_state, export_root, True, ".txt")
    assert message.startswith("Exported 1 approved item(s)")
    assert sorted(path.suffix for path in export_root.iterdir()) == [".mp4", ".txt"]
    assert next(export_root.glob("*.txt")).read_text(encoding="utf-8") == "edited Ω caption\n"
