from __future__ import annotations

import threading
from io import StringIO
from pathlib import Path

from vcap.core.console_progress import clear_all, finalize_progress_line, log, show_progress_line
from vcap.core.logs import clean_status_text, get_log


def test_clean_status_mojibake() -> None:
    assert clean_status_text("Done â†’ next") == "Done -> next"
    assert clean_status_text("value …") == "value ..."


def test_revision_tail_snapshot_and_file_mirror(tmp_path: Path) -> None:
    app_log = get_log()
    start_revision = app_log.revision
    log_path = tmp_path / "run.txt"
    app_log.attach_file(log_path)
    app_log.log("first", scope="Test", console=False)
    app_log.warn("second", console=False)
    app_log.detach_file(log_path)
    lines, revision = app_log.snapshot(start_revision)
    assert revision > start_revision
    assert any("[Test] first" in line for line in lines)
    assert "second" in app_log.tail(1)
    text = log_path.read_text(encoding="utf-8")
    assert "first" in text and "second" in text


def test_thread_safe_logging_and_redirected_progress() -> None:
    app_log = get_log()
    before = app_log.revision
    threads = [
        threading.Thread(target=lambda index=i: app_log.log(f"thread-{index}", console=False))
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines, revision = app_log.snapshot(before)
    assert revision - before == 8
    assert len(lines) == 8

    stream = StringIO()
    show_progress_line("10% working", "job", stream)
    show_progress_line("20% working", "job", stream)
    log("normal", stream)
    finalize_progress_line("job", "100% done", stream)
    clear_all(stream)
    assert "10% working" in stream.getvalue()
    assert "normal" in stream.getvalue()
    assert "100% done" in stream.getvalue()
