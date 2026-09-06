"""Regression coverage for authoritative inputs and the caption workspace."""

import pytest

from vcap.ui.app import build_app
from vcap.ui.tabs.caption_tab import resolve_caption_inputs_at_start
from vcap.ui.tabs.transcribe_tab import resolve_transcribe_inputs_at_start


@pytest.fixture(scope="module")
def app():
    demo = build_app()
    try:
        yield demo
    finally:
        demo.vcap_context.pipeline_client.shutdown()


def test_previews_cannot_upload_clear_or_write_input_state(app):
    """A rendered/cached preview must never participate in source selection."""
    config = app.get_config_file()
    for handles in (app.vcap_context.caption_handles, app.vcap_context.transcribe_handles):
        media = handles.media
        preview_ids = {media.image._id, media.video._id, media.audio._id}
        for component in config["components"]:
            if component["id"] in preview_ids:
                assert component["props"]["interactive"] is False
        for dependency in config["dependencies"]:
            assert preview_ids.isdisjoint(dependency["inputs"])
            assert all(target not in preview_ids for target, _event in dependency["targets"])


@pytest.mark.parametrize("mode,key,empty", [
    ("upload", "input_files", []),
    ("upload", "input_files", None),
    ("path", "input_path", ""),
    ("path", "input_path", "  "),
    ("folder", "batch_input_folder", ""),
])
@pytest.mark.parametrize("transcribe", [False, True])
def test_cleared_source_never_resurrects_pending_preview(tmp_path, mode, key, empty, transcribe):
    previous = tmp_path / "previous.wav"
    previous.write_bytes(b"previous media")
    resolver = resolve_transcribe_inputs_at_start if transcribe else resolve_caption_inputs_at_start
    if transcribe:
        key = "whisper_" + key
    assert resolver({key: empty}, mode, [str(previous)]) == []


@pytest.mark.parametrize("mode,key", [("upload", "input_files"), ("path", "input_path")])
@pytest.mark.parametrize("transcribe", [False, True])
def test_replaced_source_wins_over_pending_preview(tmp_path, mode, key, transcribe):
    selected = tmp_path / "selected.wav"
    previous = tmp_path / "previous.wav"
    for path in (selected, previous):
        path.write_bytes(b"media")
    resolver = resolve_transcribe_inputs_at_start if transcribe else resolve_caption_inputs_at_start
    if transcribe:
        key = "whisper_" + key
    value = [str(selected)] if mode == "upload" else str(selected)
    assert resolver({key: value}, mode, [str(previous)]) == [str(selected.resolve())]


def test_caption_results_and_actions_share_column_beside_media(app):
    config = app.get_config_file()
    ids = {
        item["props"].get("elem_id"): item["id"]
        for item in config["components"]
    }
    parents = {}

    def visit(node, ancestry=()):
        parents[node["id"]] = ancestry
        for child in node.get("children", []):
            visit(child, (*ancestry, node["id"]))

    visit(config["layout"])
    handles = app.vcap_context.caption_handles
    workspace = ids["vc_caption_workspace"]
    media = ids["vc_caption_media"]
    results = ids["vc_caption_results"]
    assert parents[media][-1] == parents[results][-1] == workspace
    assert media in parents[handles.media.files._id]
    assert media in parents[handles.media.image._id]
    assert results in parents[handles.caption._id]
    assert results in parents[handles.start._id]
    assert workspace not in parents[ids["vc_caption_settings"]]
    entries = {entry.key: entry.component._id for entry in app.vcap_context.registry.entries()}
    for key in ("model_key", "user_prompt", "max_new_tokens"):
        assert ids["vc_caption_settings"] in parents[entries[key]]
