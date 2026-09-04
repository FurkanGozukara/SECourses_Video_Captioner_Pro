from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vcap.core.export import export_dataset
from vcap.core.registry import SettingsRegistry
from vcap.models.downloads import format_status_line
from vcap.ui.tabs.dataset_tab import (
    append_deduped_progress_line,
    write_clip_fitness_plan,
)
from vcap.ui.tabs.editor_tab import (
    _selection_payload,
    editor_export_handler,
    filter_items,
    new_editor_state,
    scan_folder,
)
from vcap.ui.tabs.health_tab import selected_model_action_key
from vcap.ui.tabs.recover_tab import present_recovery_settings


def test_run_metadata_resolves_media_and_default_scan_includes_batch_mirror(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run = outputs / "0008_timechat"
    mirror = outputs / "batch_captions"
    run.mkdir(parents=True)
    mirror.mkdir()
    source = tmp_path / "lightning_storm_ünicode.mp4"
    source.write_bytes(b"source media")
    caption = run / "lightning_storm_ünicode.txt"
    caption.write_text("lightning over the city\n", encoding="utf-8")
    (run / "metadata.json").write_text(
        json.dumps({"settings": {"input_files": [str(source)]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (mirror / "vöyager 日本語.mp4").write_bytes(b"mirror media")
    (mirror / "vöyager 日本語.txt").write_text("mirrored caption\n", encoding="utf-8")

    items = scan_folder(outputs, recursive=False)
    run_item = next(item for item in items if Path(item["caption_path"]) == caption)

    assert Path(run_item["media_path"]) == source
    assert Path(run_item["source_media_path"]) == source
    assert any("batch_captions" in item["caption_path"] for item in items)


def test_missing_metadata_source_names_the_missing_path(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    run = outputs / "0009_qwen3"
    run.mkdir(parents=True)
    missing = tmp_path / "gone ünicode.mp4"
    (run / "gone ünicode.txt").write_text("caption\n", encoding="utf-8")
    (run / "metadata.json").write_text(
        json.dumps({"settings": {"input_path": str(missing)}}, ensure_ascii=False),
        encoding="utf-8",
    )

    item = scan_folder(outputs, recursive=False)[0]
    state = new_editor_state(outputs)
    state["items"] = [item]
    state["selected_index"] = 0
    payload = _selection_payload(state, tmp_path / "previews")

    assert item["media_path"] is None
    assert Path(item["source_media_path"]) == missing
    assert payload[3]["value"] == f"Source media not found: {missing}"


def test_zero_numeric_filters_are_unlimited() -> None:
    item = {
        "caption_path": "lightning.txt",
        "caption": "lightning crosses a rainy skyline",
        "chars": 33,
        "tokens": 7,
        "flag": None,
        "status": "no media",
    }

    result = filter_items(
        [item],
        search="lightning",
        min_length=0,
        max_length=0,
        min_tokens=0,
        max_tokens=0,
    )

    assert result == [item]


def test_export_counts_caption_only_and_not_approved_separately(tmp_path: Path) -> None:
    media = tmp_path / "approved.mp4"
    other_media = tmp_path / "not-approved.mp4"
    media.write_bytes(b"approved media")
    other_media.write_bytes(b"other media")
    caption_only = tmp_path / "prompt result.txt"
    caption_only.write_text("prompt caption\n", encoding="utf-8")
    items = [
        {"media_path": media, "caption": "media caption", "flag": "approved"},
        {
            "media_path": None,
            "source_media_path": tmp_path / "missing.mp4",
            "caption_path": caption_only,
            "caption": "prompt caption",
            "flag": "approved",
        },
        {"media_path": other_media, "caption": "not approved", "flag": None},
        {"media_path": None, "caption": "also not approved", "flag": None},
    ]

    excluded = export_dataset(items, tmp_path / "excluded", include_caption_only=False)
    included = export_dataset(items, tmp_path / "included", include_caption_only=True)

    assert (excluded.exported, excluded.no_media, excluded.not_approved, excluded.error_count) == (1, 1, 2, 0)
    assert (included.exported, included.no_media, included.not_approved, included.error_count) == (2, 0, 2, 0)
    assert len(included.media_files) == 1
    assert len(included.caption_files) == 2

    state = new_editor_state(tmp_path)
    state["items"] = items
    message = editor_export_handler(state, tmp_path / "handler", True, ".txt", False)
    assert "Exported 1 approved item(s); no-media 1; not-approved 2; errors 0." in message
    assert "skipped" not in message.casefold()


def _recovery_registry() -> SettingsRegistry:
    registry = SettingsRegistry()
    registry.register("theme_mode", object(), "dark", kind="str")
    registry.register("outputs_dir", object(), "default", kind="str")
    registry.register("temp_dir", object(), "default", kind="str")
    registry.register("models_dir", object(), "default", kind="str")
    registry.register("input_files", object(), [], kind="list")
    registry.register("input_path", object(), "", kind="str")
    registry.register("batch_input_folder", object(), "", kind="str")
    registry.register("batch_output_folder", object(), "", kind="str")
    registry.register("gpu_index", object(), 0, kind="int")
    registry.register("gpu_indices", object(), [], kind="list")
    registry.register("model_key", object(), "timechat_int4", section="model", kind="str")
    return registry


def test_recovery_skips_machine_values_and_paths_unless_opted_in() -> None:
    settings = {
        "theme_mode": "light",
        "outputs_dir": "old outputs",
        "temp_dir": "old temp",
        "models_dir": "old models",
        "input_files": ["clip.mp4"],
        "input_path": "clip.mp4",
        "batch_input_folder": "old input",
        "batch_output_folder": "old output",
        "gpu_index": 3,
        "gpu_indices": [0, 3],
        "model_key": "qwen3_omni_instruct_int4",
    }
    metadata = {"settings": settings}

    default, warnings = present_recovery_settings(
        metadata,
        _recovery_registry(),
        available_gpu_indices=[0],
    )
    with_paths, path_warnings = present_recovery_settings(
        metadata,
        _recovery_registry(),
        restore_paths=True,
        available_gpu_indices=[0],
    )

    assert default == {"gpu_indices": [0], "model_key": "qwen3_omni_instruct_int4"}
    assert not {"theme_mode", "outputs_dir", "temp_dir", "models_dir", "gpu_index"} & with_paths.keys()
    assert set(settings) & with_paths.keys() == {
        "input_files",
        "input_path",
        "batch_input_folder",
        "batch_output_folder",
        "gpu_indices",
        "model_key",
    }
    assert any("Skipped keys:" in warning and "theme_mode" in warning for warning in warnings)
    assert any("Skipped keys:" in warning and "gpu_index" in warning for warning in path_warnings)


def test_fitness_plan_uses_output_folder_and_progress_lines_are_deduped(tmp_path: Path) -> None:
    source = tmp_path / "source ünicode"
    output = tmp_path / "outputs" / "clip_fitness"
    source.mkdir()
    plan = {"source_folder": str(source), "items": []}

    written = write_clip_fitness_plan(plan, output, timestamp="20260831_120000")
    lines: list[str] = []
    assert append_deduped_progress_line(lines, "clip 1.mp4: Splitting clip 1/5") is True
    assert append_deduped_progress_line(lines, "clip 1.mp4: Splitting clip 1/5") is False

    assert written == output / "source ünicode_20260831_120000.json"
    assert written.is_file()
    assert not (source / "clip_fitness_plan.json").exists()
    assert lines == ["clip 1.mp4: Splitting clip 1/5"]
    nested = write_clip_fitness_plan(plan, source / "clip_fitness", timestamp="same-folder")
    assert nested == source / "clip_fitness" / "source ünicode_same-folder.json"


def test_health_protocol_rendering_and_selected_key() -> None:
    rendered = format_status_line(
        'VCAP_STATUS {"key":"timechat_bf16","state":"downloading",'
        '"fraction":0.42,"message":"re-hashing local files"}'
    )

    assert rendered == "timechat_bf16: re-hashing local files (42%)"
    assert "VCAP_STATUS" not in rendered
    assert selected_model_action_key("timechat_bf16") == "timechat_bf16"
    with pytest.raises((KeyError, ValueError)):
        selected_model_action_key("not-a-model")


def test_health_download_and_verify_read_the_dropdown_at_click_time() -> None:
    from vcap.ui.app import build_app

    app = build_app()
    config = app.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    dropdown = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Model action"
    )
    button_dependencies = [
        dependency
        for dependency in config["dependencies"]
        if dependency.get("targets")
        and dependency["targets"][0][0] in components
        and str(components[dependency["targets"][0][0]].get("props", {}).get("value"))
        in {"📥 Download", "🔍 Verify"}
    ]

    assert len(button_dependencies) == 2
    assert all(dependency["inputs"] == [dropdown["id"]] for dependency in button_dependencies)


def test_downloader_verify_falls_back_to_local_presence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    downloader_path = Path(__file__).resolve().parents[2] / "Models_Downloader.py"
    spec = importlib.util.spec_from_file_location("vcap_test_models_downloader", downloader_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    key = "timechat_bf16"
    model_dir = tmp_path / key
    model_dir.mkdir()
    payload = b"local model fixture"
    (model_dir / "model.safetensors").write_bytes(payload)
    monkeypatch.setattr(
        module,
        "_bundled_local_manifest",
        lambda _key: [("model.safetensors", len(payload), "0" * 64)],
    )
    monkeypatch.setattr(
        module,
        "load_remote_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module.DownloadError(
                "remote model folder MonsterMMORPG/Wan_GGUF/timechat_bf16 is not available yet"
            )
        ),
    )

    class DummyDownloader:
        def set_status_key(self, _key: str | None) -> None:
            return None

    assert module.verify_model(key, tmp_path, DummyDownloader()) is True
    output = capsys.readouterr().out
    assert "Remote folder timechat_bf16 is not published yet" in output
    assert "cannot be verified against published digests" in output
