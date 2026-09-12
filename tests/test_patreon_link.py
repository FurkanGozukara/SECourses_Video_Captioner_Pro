"""The Gradio Patreon links must open the SECourses Video Captioner Pro release post."""

from __future__ import annotations

import inspect

import pytest

from vcap import PATREON_URL

RELEASE_POST = "https://www.patreon.com/SECourses/posts/secourses-video-168757767"


def test_patreon_url_targets_the_release_post() -> None:
    assert PATREON_URL == RELEASE_POST


def test_gradio_links_use_the_shared_patreon_url() -> None:
    pytest.importorskip("gradio")
    from vcap.ui import app
    from vcap.ui.tabs import changelog_tab

    for module in (app, changelog_tab):
        source = inspect.getsource(module)
        assert "{PATREON_URL}" in source, module.__name__
        assert "patreon.com/SECourses)" not in source, module.__name__
