from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from vcap.core.progress import ProgressEvent
from vcap.pipeline.job import ItemResult, JobResult
from vcap.ui.app import build_app
from vcap.ui.tabs.caption_tab import (
    delete_prompt_library_entry,
    failed_item_paths,
    load_prompt_library_entry,
    results_zip_paths,
    render_prompt_preserving_edits,
    retry_failed_inputs,
    run_history_records,
    run_history_rows,
    save_prompt_library_entry,
)
from vcap.ui.tabs.health_tab import (
    delete_model_files_report,
    render_update_status,
    request_model_delete,
)


@dataclass
class _Prompt:
    name: str
    system_prompt: str
    user_prompt: str


class _FakePromptLibrary:
    entries: dict[str, _Prompt] = {}

    def __init__(self, _directory: Path) -> None:
        pass

    def list(self) -> list[_Prompt]:
        return sorted(self.entries.values(), key=lambda entry: entry.name.casefold())

    def save(self, name: str, system: str, user: str) -> _Prompt:
        entry = _Prompt(name, system, user)
        self.entries[name] = entry
        return entry

    def load(self, name: str) -> _Prompt:
        if name not in self.entries:
            raise KeyError(name)
        return self.entries[name]

    def delete(self, name: str) -> bool:
        return self.entries.pop(name, None) is not None


def test_retry_failed_derives_all_failure_source_paths() -> None:
    result = JobResult(
        items=[
            ItemResult(0, "ok.mp4", "video", "done"),
            ItemResult(1, "bad one.mp4", "video", "failed"),
            ItemResult(2, "bad two.wav", "audio", "unsupported"),
            ItemResult(3, "bad three.png", "image", "error"),
            ItemResult(4, "skipped.png", "image", "skipped"),
        ],
        counts={"done": 1, "failed": 1, "unsupported": 1, "error": 1, "skipped": 1},
        run_dir="run",
        metadata_path="metadata.json",
        elapsed=1.0,
    )
    assert failed_item_paths(result) == [
        "bad one.mp4",
        "bad two.wav",
        "bad three.png",
    ]
    state = {
        "failed_paths": failed_item_paths(result),
        "output_kind": "batch",
        "batch_output_folder": "captions",
    }
    assert retry_failed_inputs(state) == (
        ["bad one.mp4", "bad two.wav", "bad three.png"],
        "batch",
        "captions",
    )


def test_caption_and_retry_generators_match_wired_outputs(tmp_path: Path) -> None:
    demo = build_app()
    context = demo.vcap_context
    dependency = next(
        item for item in demo.get_config_file()["dependencies"]
        if item.get("api_name") == "caption"
    )
    run_caption = demo.fns[dependency["id"]].fn
    captured: list[object] = []

    def fake_run_job(spec, sink, _token):
        captured.append(spec)
        sink.on_progress(
            ProgressEvent(
                "Working",
                fraction=0.5,
                item_index=0,
                total_items=len(spec.inputs),
                data={"processed": 0, "remaining": len(spec.inputs)},
            )
        )
        return JobResult(
            items=[
                ItemResult(index, item.path, "video", "failed", message="fake failure")
                for index, item in enumerate(spec.inputs)
            ],
            counts={"done": 0, "failed": len(spec.inputs)},
            run_dir=str(tmp_path / "run"),
            metadata_path=str(tmp_path / "run" / "metadata.json"),
            elapsed=0.01,
        )

    context.pipeline_client.run_job = fake_run_job
    try:
        values = context.registry.dict_to_values(context.registry.defaults())
        normal_updates = list(run_caption(*values, ["bad.mp4"], "upload", "video"))
        assert all(len(update) == len(dependency["outputs"]) for update in normal_updates)
        assert normal_updates[-1][11]["failed_paths"] == ["bad.mp4"]
        assert normal_updates[-1][18]["interactive"] is True

        uploaded = tmp_path / "file component upload.mp4"
        uploaded.write_bytes(b"not decoded by this UI test")
        raw_settings = context.registry.defaults()
        raw_settings["input_files"] = [str(uploaded)]
        raw_values = context.registry.dict_to_values(raw_settings)
        list(run_caption(*raw_values, [], "upload", "video"))
        assert [item.path for item in captured[-1].inputs] == [str(uploaded.resolve())]

        batch_output = tmp_path / "same batch output"
        source_root = tmp_path / "batch source"
        batch_output.mkdir()
        source_root.mkdir()
        retry_state = {
            "failed_paths": ["failed one.mp4", "failed two.wav"],
            "output_kind": "batch",
            "batch_output_folder": str(batch_output),
            "source_root": str(source_root),
            "input_modality": "video",
        }
        retry_updates = list(
            run_caption(*values, ["ignored.mp4"], "upload", "video", retry_state)
        )
        assert all(len(update) == len(dependency["outputs"]) for update in retry_updates)
        assert "Retrying 2 item(s)" in retry_updates[0][1]
        retry_spec = captured[-1]
        assert [item.path for item in retry_spec.inputs] == [
            "failed one.mp4",
            "failed two.wav",
        ]
        assert retry_spec.output.kind == "batch"
        assert retry_spec.output.overwrite is True
        assert Path(retry_spec.output.batch_output_dir) == batch_output
    finally:
        context.pipeline.shutdown()


def test_results_zip_path_uses_run_or_batch_folder(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "0042_qwen3"
    batch = tmp_path / "batch captions"
    run.mkdir(parents=True)
    batch.mkdir()

    source, target = results_zip_paths({"run_dir": str(run)}, tmp_path / "temp")
    assert source == run
    assert target == tmp_path / "temp" / "downloads" / "0042_qwen3.zip"

    source, target = results_zip_paths(
        {"output_kind": "batch", "batch_output_folder": str(batch)},
        tmp_path / "temp",
    )
    assert source == batch
    assert target.name == "batch captions.zip"


def test_run_history_rows_and_hidden_paths_from_fake_summaries(tmp_path: Path) -> None:
    run = tmp_path / "0007_qwen3"
    metadata = run / "metadata.json"
    summary = SimpleNamespace(
        run_dir=str(run),
        name=run.name,
        kind="single",
        model_key="qwen3_omni_instruct_int4",
        created=1_700_000_000.0,
        items=4,
        counts={"done": 3, "failed": 1},
        preview="A caption\nwith spacing",
        metadata_path=str(metadata),
    )
    rows = run_history_rows([summary])
    assert rows[0][:5] == [run.name, "single", "qwen3_omni_instruct_int4", 4, "3 / 1"]
    assert rows[0][6] == "A caption with spacing"
    assert run_history_records([summary]) == [
        {
            "run_dir": str(run),
            "name": run.name,
            "kind": "single",
            "metadata_path": str(metadata),
        }
    ]


def test_prompt_library_handlers_preserve_unicode_and_manual_state(tmp_path: Path) -> None:
    _FakePromptLibrary.entries = {}
    name = "Sahne İstanbul"
    names, selected, status = save_prompt_library_entry(
        tmp_path, name, "system", "user", _FakePromptLibrary
    )
    assert names == [name] and selected == name and "Saved prompt" in status

    system, user, state, status = load_prompt_library_entry(
        tmp_path, name, _FakePromptLibrary
    )
    assert (system, user) == ("system", "user")
    assert state["manual"] is True
    assert "Loaded prompt" in status

    _, system_update, user_update, tracked = render_prompt_preserving_edits(
        "wan22_t2v_dense",
        {},
        system,
        user,
        state,
    )
    assert system_update["__type__"] == "update"
    assert user_update["__type__"] == "update"
    assert tracked["manual"] is True

    names, status = delete_prompt_library_entry(tmp_path, name, _FakePromptLibrary)
    assert names == [] and "Deleted prompt" in status


def test_delete_model_confirm_state_and_fake_delete_report() -> None:
    missing = request_model_delete(
        "qwen3_omni_instruct_int4",
        {"loaded_variant": None},
        lambda _key: 0,
    )
    assert missing["state"] == "blocked"
    assert "No on-disk files" in missing["message"]

    blocked = request_model_delete(
        "qwen3_omni_instruct_int4",
        {"loaded_variant": "qwen3_omni_instruct_int4"},
        lambda _key: 100,
    )
    assert blocked["state"] == "blocked"
    assert "resident" in blocked["message"]

    confirmation = request_model_delete(
        "qwen3_omni_instruct_int4",
        {"loaded_variant": None},
        lambda _key: 2 * 1024**3,
    )
    assert confirmation["state"] == "confirm"
    assert "(2.00 GB)" in confirmation["question"]

    fake_report = SimpleNamespace(
        variant_key="qwen3_omni_instruct_int4",
        folder="models/qwen3",
        files_removed=7,
        bytes_freed=12345,
        errors=[],
    )
    rendered = delete_model_files_report(
        "qwen3_omni_instruct_int4", lambda _key: fake_report
    )
    assert "Removed 7 file(s)" in rendered
    assert "12,345 bytes" in rendered
    assert "vc-ok" in rendered


def test_update_status_colors_up_to_date_behind_and_failed() -> None:
    assert "vc-ok" in render_update_status(
        SimpleNamespace(ok=True, behind=0, message="Up to date.")
    )
    assert "vc-warn" in render_update_status(
        SimpleNamespace(ok=True, behind=3, message="Update available.")
    )
    assert "vc-err" in render_update_status(
        SimpleNamespace(ok=False, behind=0, message="Offline.")
    )


def test_new_action_labels_have_stable_elem_ids() -> None:
    demo = build_app()
    try:
        by_id = {
            item.get("props", {}).get("elem_id"): item.get("props", {}).get("value")
            for item in demo.get_config_file()["components"]
        }
        expected = {
            "vc_preset_delete_yes": "✔ Yes, delete",
            "vc_preset_delete_keep": "✖ Keep preset",
            "vc_copy_caption": "⧉ Copy caption",
            "vc_retry_failed": "🔁 Retry failed",
            "vc_results_zip": "⬇ Results ZIP",
            "vc_run_history_refresh": "🔄 Refresh",
            "vc_run_history_open_folder": "📂 Open folder",
            "vc_run_history_open_editor": "✏️ Open in editor",
            "vc_run_history_recover": "🔁 Recover settings",
            "vc_save_prompt": "💾 Save prompt",
            "vc_load_prompt": "📥 Load prompt",
            "vc_delete_prompt": "🗑 Delete prompt",
            "vc_health_delete_model_files": "🗑 Delete model files",
            "vc_health_delete_model_yes": "✔ Yes, delete",
            "vc_health_delete_model_keep": "✖ Keep files",
            "vc_check_for_updates": "🔎 Check for updates",
        }
        for elem_id, label in expected.items():
            assert by_id[elem_id] == label
        component_labels = {
            item.get("props", {}).get("elem_id"): item.get("props", {}).get("label")
            for item in demo.get_config_file()["components"]
        }
        assert component_labels["vc_batch_zip_upload"] == "…or upload a ZIP archive of media"
        assert component_labels["vc_results_zip_download"] == "Results ZIP download"
        assert component_labels["vc_run_history_table"] == "Recent runs"
    finally:
        demo.vcap_context.pipeline.shutdown()
