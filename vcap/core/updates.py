"""Non-throwing Git update checks for the desktop settings surface."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from vcap import VERSION

from .paths import normalize_path


@dataclass(frozen=True)
class UpdateStatus:
    """Local and remote Git comparison result."""

    ok: bool
    current_commit: str
    remote_commit: str
    behind: int
    ahead: int
    branch: str
    message: str


def check_for_updates(
    app_dir: str | os.PathLike[str],
    *,
    timeout_s: float = 25.0,
) -> UpdateStatus:
    """Fetch ``origin`` and compare HEAD with its default branch without raising."""

    try:
        directory = normalize_path(app_dir)
    except (OSError, TypeError, ValueError) as exc:
        return UpdateStatus(False, "", "", 0, 0, "", f"Invalid app folder: {exc}")

    try:
        timeout = max(0.1, float(timeout_s))
    except (TypeError, ValueError, OverflowError):
        timeout = 25.0

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(directory),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    try:
        repository = run("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        return UpdateStatus(False, "", "", 0, 0, "", "Git is not installed or is not on PATH.")
    except subprocess.TimeoutExpired:
        return UpdateStatus(False, "", "", 0, 0, "", "Git timed out while checking the app folder.")
    except Exception as exc:
        return UpdateStatus(False, "", "", 0, 0, "", f"Could not run Git: {exc}")
    if repository.returncode != 0 or repository.stdout.strip().casefold() != "true":
        return UpdateStatus(False, "", "", 0, 0, "", "The app folder is not a Git repository.")

    try:
        fetched = run("fetch", "--quiet", "origin")
        if fetched.returncode != 0:
            detail = (fetched.stderr or fetched.stdout or "origin is unavailable").strip()
            return UpdateStatus(
                False,
                "",
                "",
                0,
                0,
                "",
                f"Could not check for updates (offline or origin unavailable): {detail}",
            )
        current_result = run("rev-parse", "HEAD")
        default_result = run(
            "symbolic-ref",
            "--quiet",
            "--short",
            "refs/remotes/origin/HEAD",
        )
        branch = default_result.stdout.strip() if default_result.returncode == 0 else "origin/main"
        if branch.startswith("refs/remotes/"):
            branch = branch.removeprefix("refs/remotes/")
        if not branch.startswith("origin/"):
            branch = "origin/main"
        remote_result = run("rev-parse", branch)
        counts_result = run("rev-list", "--left-right", "--count", f"HEAD...{branch}")
    except subprocess.TimeoutExpired:
        return UpdateStatus(False, "", "", 0, 0, "", "Git timed out while contacting origin.")
    except Exception as exc:
        return UpdateStatus(False, "", "", 0, 0, "", f"Could not compare Git revisions: {exc}")

    if current_result.returncode != 0 or remote_result.returncode != 0:
        detail = (
            remote_result.stderr
            or current_result.stderr
            or "the local or remote revision could not be resolved"
        ).strip()
        return UpdateStatus(False, "", "", 0, 0, branch, f"Could not resolve update revisions: {detail}")
    current = current_result.stdout.strip()
    remote = remote_result.stdout.strip()
    try:
        ahead_text, behind_text = counts_result.stdout.replace("\t", " ").split()[:2]
        ahead = max(0, int(ahead_text))
        behind = max(0, int(behind_text))
    except (TypeError, ValueError, IndexError):
        detail = (counts_result.stderr or counts_result.stdout or "invalid revision count").strip()
        return UpdateStatus(False, current, remote, 0, 0, branch, f"Could not compare Git revisions: {detail}")

    short = current[:7]
    if behind > 0:
        message = (
            f"Update available: {behind} new commit(s) on {branch}. "
            "Run Windows_Install_and_Update.bat (Windows) or `git pull` in the app folder "
            "(Linux), then restart."
        )
    elif ahead > 0:
        message = f"Local checkout is {ahead} commit(s) ahead of {branch} ({short})."
    else:
        message = f"Up to date (v{VERSION}, {short})."
    return UpdateStatus(True, current, remote, behind, ahead, branch, message)


__all__ = ["UpdateStatus", "check_for_updates"]
