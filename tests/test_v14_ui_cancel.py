from __future__ import annotations

from vcap.core.subprocess_runner import CancelToken
from vcap.ui.app import build_app
from vcap.ui.tabs.caption_tab import (
    confirm_caption_cancel,
    keep_caption_running,
    request_caption_cancel,
)


def test_caption_cancel_explicit_confirmation_state_machine() -> None:
    assert request_caption_cancel(None) == "inactive"

    token = CancelToken()
    assert request_caption_cancel(token) == "confirm"
    assert token.is_armed()
    assert not token.is_cancelled()

    assert keep_caption_running(token)
    assert not token.is_armed()
    assert not token.is_cancelled()

    assert request_caption_cancel(token) == "confirm"
    assert confirm_caption_cancel(token)
    assert token.is_cancelled()
    assert not token.is_armed()
    assert not confirm_caption_cancel(token)
    assert request_caption_cancel(token) == "inactive"


def test_every_caption_cancel_click_bypasses_the_busy_queue() -> None:
    demo = build_app()
    try:
        config = demo.get_config_file()
        ids = {
            component.get("props", {}).get("elem_id"): component["id"]
            for component in config["components"]
        }
        cancel_ids = {
            ids["vc_caption_cancel"],
            ids["vc_caption_cancel_yes"],
            ids["vc_caption_cancel_keep"],
        }
        dependencies = [
            dependency
            for dependency in config["dependencies"]
            if any(component_id in cancel_ids for component_id, _ in dependency.get("targets", []))
        ]
        assert len(dependencies) == 3
        assert all(dependency.get("queue") is False for dependency in dependencies)
    finally:
        demo.vcap_context.pipeline.shutdown()
