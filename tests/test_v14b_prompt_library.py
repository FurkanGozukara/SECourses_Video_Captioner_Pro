from __future__ import annotations

import json
from pathlib import Path

import pytest

from vcap.core.presets import sanitize_preset_name
from vcap.core.prompt_library import PromptEntry, PromptLibrary


def test_prompt_library_save_load_list_overwrite_delete_and_unicode(tmp_path: Path) -> None:
    directory = tmp_path / "prompts"
    library = PromptLibrary(directory)
    assert not directory.exists()

    first = library.save("My prompt / 日本語", "Sistem: görüntü", "Kullanıcı 日本語", "notlar")
    second = library.save("alpha", "system", "user")

    assert isinstance(first, PromptEntry)
    assert first.name == "My prompt / 日本語"
    assert Path(first.path).name == f"{sanitize_preset_name(first.name)}.json"
    assert json.loads(Path(first.path).read_text(encoding="utf-8"))["user_prompt"] == "Kullanıcı 日本語"
    assert [entry.name for entry in library.list()] == ["alpha", "My prompt / 日本語"]
    assert library.exists("My prompt / 日本語")
    assert library.load("My prompt / 日本語").system_prompt == "Sistem: görüntü"

    overwritten = library.save("My prompt / 日本語", "new system", "new user", "new notes")
    assert overwritten.path == first.path
    assert library.load("My prompt / 日本語").user_prompt == "new user"
    assert library.delete(second.name) is True
    assert library.delete(second.name) is False
    assert not library.exists(second.name)


def test_prompt_library_missing_load_raises_key_error(tmp_path: Path) -> None:
    library = PromptLibrary(tmp_path / "missing")
    with pytest.raises(KeyError):
        library.load("does not exist")
