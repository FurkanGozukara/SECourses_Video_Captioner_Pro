"""Small UTF-8 JSON library for reusable system and user prompts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vcap import PRESETS_DIR

from .paths import normalize_path
from .presets import sanitize_preset_name


@dataclass(frozen=True)
class PromptEntry:
    """One reusable prompt pair."""

    name: str
    path: str
    system_prompt: str
    user_prompt: str
    notes: str
    updated: float


class PromptLibrary:
    """Persist named prompt pairs as independent JSON files."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        self.directory = normalize_path(directory or (PRESETS_DIR / "prompts"))

    def _find_path(self, name: str) -> Path | None:
        safe = sanitize_preset_name(name)
        direct = self.directory / f"{safe}.json"
        try:
            if direct.is_file():
                return direct
            if not self.directory.is_dir():
                return None
            expected = safe.casefold()
            return next(
                (
                    path
                    for path in self.directory.glob("*.json")
                    if path.stem.casefold() == expected and path.is_file()
                ),
                None,
            )
        except OSError:
            return None

    @staticmethod
    def _entry(path: Path, payload: dict[str, Any]) -> PromptEntry:
        try:
            updated = float(payload.get("saved_at"))
        except (TypeError, ValueError):
            updated = float(path.stat().st_mtime)
        return PromptEntry(
            name=str(payload.get("name") or path.stem),
            path=str(path.resolve(strict=False)),
            system_prompt=str(payload.get("system_prompt") or ""),
            user_prompt=str(payload.get("user_prompt") or ""),
            notes=str(payload.get("notes") or ""),
            updated=updated,
        )

    @classmethod
    def _read(cls, path: Path) -> PromptEntry:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Prompt file must contain a JSON object: {path}")
        return cls._entry(path, payload)

    def list(self) -> list[PromptEntry]:
        """Return valid entries sorted by display name."""

        try:
            paths = list(self.directory.glob("*.json")) if self.directory.is_dir() else []
        except OSError:
            return []
        entries: list[PromptEntry] = []
        for path in paths:
            try:
                entries.append(self._read(path))
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sorted(entries, key=lambda entry: (entry.name.casefold(), entry.name))

    def save(
        self,
        name: str,
        system_prompt: str,
        user_prompt: str,
        notes: str = "",
    ) -> PromptEntry:
        """Atomically save or overwrite a named prompt."""

        display_name = str(name).strip() or "default"
        safe = sanitize_preset_name(display_name)
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._find_path(safe) or (self.directory / f"{safe}.json")
        payload = {
            "name": display_name,
            "system_prompt": str(system_prompt),
            "user_prompt": str(user_prompt),
            "notes": str(notes),
            "saved_at": time.time(),
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.directory,
                prefix=f".{safe}.",
                suffix=".json.tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        return self._entry(target, payload)

    def load(self, name: str) -> PromptEntry:
        """Load one prompt or raise ``KeyError`` when its name is absent."""

        path = self._find_path(name)
        if path is None:
            raise KeyError(name)
        return self._read(path)

    def delete(self, name: str) -> bool:
        """Delete one prompt, returning whether a file was removed."""

        path = self._find_path(name)
        if path is None:
            return False
        path.unlink()
        return True

    def exists(self, name: str) -> bool:
        """Return whether a named prompt file exists."""

        return self._find_path(name) is not None


__all__ = ["PromptEntry", "PromptLibrary"]
