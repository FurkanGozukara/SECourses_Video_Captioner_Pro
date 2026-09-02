from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vcap.core import updates
from vcap.core.updates import UpdateStatus, check_for_updates


def _result(returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _fake_git(
    calls: list[tuple[list[str], dict[str, object]]],
    *,
    repository: bool = True,
    fetch_ok: bool = True,
    ahead: int = 0,
    behind: int = 0,
):
    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((list(command), dict(kwargs)))
        arguments = command[1:]
        if arguments == ["rev-parse", "--is-inside-work-tree"]:
            return _result(0 if repository else 128, "true\n" if repository else "", "not a repo")
        if arguments == ["fetch", "--quiet", "origin"]:
            return _result(0 if fetch_ok else 1, stderr="network unavailable")
        if arguments == ["rev-parse", "HEAD"]:
            return _result(stdout="abcdef0123456789\n")
        if arguments[0:3] == ["symbolic-ref", "--quiet", "--short"]:
            return _result(stdout="origin/main\n")
        if arguments == ["rev-parse", "origin/main"]:
            return _result(stdout="1234567890abcdef\n")
        if arguments[0:3] == ["rev-list", "--left-right", "--count"]:
            return _result(stdout=f"{ahead}\t{behind}\n")
        raise AssertionError(arguments)

    return run


def test_update_check_handles_git_missing_not_repo_and_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(updates.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert check_for_updates(tmp_path).message.startswith("Git is not installed")

    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(updates.subprocess, "run", _fake_git(calls, repository=False))
    assert check_for_updates(tmp_path).message == "The app folder is not a Git repository."

    calls.clear()
    monkeypatch.setattr(updates.subprocess, "run", _fake_git(calls, fetch_ok=False))
    status = check_for_updates(tmp_path)
    assert status.ok is False
    assert "offline or origin unavailable" in status.message


@pytest.mark.parametrize(
    ("ahead", "behind", "message_fragment"),
    [
        (0, 3, "Update available: 3 new commit(s) on origin/main."),
        (0, 0, "Up to date (v"),
        (1, 0, "Local checkout is 1 commit(s) ahead of origin/main"),
    ],
)
def test_update_comparisons_use_utf8_timeout_and_expected_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ahead: int,
    behind: int,
    message_fragment: str,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        updates.subprocess,
        "run",
        _fake_git(calls, ahead=ahead, behind=behind),
    )

    status = check_for_updates(tmp_path, timeout_s=7.5)

    assert isinstance(status, UpdateStatus)
    assert status.ok is True
    assert (status.ahead, status.behind) == (ahead, behind)
    assert status.current_commit == "abcdef0123456789"
    assert status.remote_commit == "1234567890abcdef"
    assert status.branch == "origin/main"
    assert message_fragment in status.message
    assert calls[1][0] == ["git", "fetch", "--quiet", "origin"]
    assert all(call[1]["encoding"] == "utf-8" for call in calls)
    assert all(call[1]["errors"] == "replace" for call in calls)
    assert all(call[1]["timeout"] == 7.5 for call in calls)
    assert all(call[1]["text"] is True for call in calls)
