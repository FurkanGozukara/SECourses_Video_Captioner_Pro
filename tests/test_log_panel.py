"""Newest-first live log merging used by the application log panel."""

from vcap.ui.components import merge_log_newest_first, newest_first


def test_newest_first_reverses_chronological_block() -> None:
    assert newest_first("a\nb\nc") == "c\nb\na"
    assert newest_first("") == ""
    assert newest_first("only") == "only"


def test_merge_prepends_new_lines_latest_on_top() -> None:
    merged = merge_log_newest_first("3\n2\n1", ["4", "5"])
    assert merged.splitlines() == ["5", "4", "3", "2", "1"]


def test_merge_handles_empty_sides_and_trims_oldest_from_bottom() -> None:
    assert merge_log_newest_first("", ["x"]) == "x"
    assert merge_log_newest_first("x", []) == "x"
    merged = merge_log_newest_first("old" * 100, ["new"], limit=10)
    assert merged.startswith("new")
    assert len(merged) == 10
