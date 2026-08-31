"""Read-only default and atomic user preset storage."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vcap import VERSION

from .paths import natural_sort_key

_FORMAT = "secourses_vcap_preset"
_SCHEMA_VERSION = 1
_LAST_USED = ".last_used_preset.txt"


class PresetError(ValueError):
    """Raised for invalid, protected, or unreadable presets."""


@dataclass(frozen=True)
class PresetEntry:
    """One named preset and its source location."""

    name: str
    path: Path
    is_default: bool


def sanitize_preset_name(name: str) -> str:
    """Return a portable, readable preset stem."""

    safe = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in str(name)
    )
    return safe.strip("._")[:100] or "default"


class PresetStore:
    """Manage protected default presets and writable user presets."""

    def __init__(
        self,
        user_dir: str | os.PathLike[str],
        default_dir: str | os.PathLike[str],
        *,
        default_preset_name: str | None = None,
    ) -> None:
        self.user_dir = Path(user_dir)
        self.default_dir = Path(default_dir)
        self.default_preset_name = (
            str(default_preset_name).strip() if default_preset_name else None
        )
        self.user_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _last_used_path(self) -> Path:
        return self.user_dir / _LAST_USED

    def _entries_in(self, directory: Path, is_default: bool) -> list[PresetEntry]:
        try:
            entries = [
                PresetEntry(path.stem, path, is_default)
                for path in directory.glob("*.json")
                if path.is_file() and not path.name.startswith(".")
            ]
        except OSError:
            return []
        return sorted(entries, key=lambda entry: natural_sort_key(entry.name))

    def list_presets(self) -> list[PresetEntry]:
        """List protected defaults first, followed by non-colliding user presets."""

        defaults = self._entries_in(self.default_dir, True)
        preferred = sanitize_preset_name(self.default_preset_name or "").casefold()
        if preferred:
            defaults.sort(
                key=lambda entry: (
                    sanitize_preset_name(entry.name).casefold() != preferred,
                    natural_sort_key(entry.name),
                )
            )
        protected = {sanitize_preset_name(entry.name).casefold() for entry in defaults}
        users = [
            entry
            for entry in self._entries_in(self.user_dir, False)
            if sanitize_preset_name(entry.name).casefold() not in protected
        ]
        return defaults + users

    def _find(self, name: str) -> PresetEntry | None:
        safe = sanitize_preset_name(name).casefold()
        for entry in self.list_presets():
            if sanitize_preset_name(entry.name).casefold() == safe:
                return entry
        return None

    def _default_names(self) -> set[str]:
        return {
            sanitize_preset_name(entry.name).casefold()
            for entry in self._entries_in(self.default_dir, True)
        }

    @staticmethod
    def _read_payload(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise PresetError(f"Could not load preset '{path.stem}': {exc}") from exc
        if not isinstance(payload, dict):
            raise PresetError("Preset root must be a JSON object")
        meta = payload.get("_meta")
        settings = payload.get("settings")
        if not isinstance(meta, dict) or meta.get("format") != _FORMAT:
            raise PresetError("Unrecognized preset format")
        if meta.get("version") != _SCHEMA_VERSION:
            raise PresetError(f"Unsupported preset version: {meta.get('version')}")
        if not isinstance(settings, dict):
            raise PresetError("Preset settings must be a JSON object")
        return payload

    def load(self, name: str) -> dict[str, Any]:
        """Load a preset's settings and mark it as last used."""

        entry = self._find(name)
        if entry is None:
            raise PresetError(f"Preset not found: {name}")
        payload = self._read_payload(entry.path)
        self.set_last_used(entry.name)
        return deepcopy(payload["settings"])

    def save(
        self,
        name: str,
        settings: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Atomically save a user preset and mark it as last used."""

        if not isinstance(settings, dict):
            raise PresetError("settings must be a dictionary")
        safe = sanitize_preset_name(name)
        if safe.casefold() in self._default_names():
            raise PresetError(f"Preset '{safe}' is a protected default")
        existing_user = next(
            (
                entry
                for entry in self._entries_in(self.user_dir, False)
                if sanitize_preset_name(entry.name).casefold() == safe.casefold()
            ),
            None,
        )
        if existing_user is not None:
            safe = existing_user.name
        target = existing_user.path if existing_user is not None else self.user_dir / f"{safe}.json"
        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        if target.exists():
            try:
                existing = self._read_payload(target)
                created_at = str(existing["_meta"].get("created_at") or now)
            except PresetError:
                pass
        supplied = deepcopy(meta) if isinstance(meta, dict) else {}
        payload_meta = {
            **supplied,
            "format": _FORMAT,
            "version": _SCHEMA_VERSION,
            "app_version": str(supplied.get("app_version") or VERSION),
            "created_at": created_at,
            "modified_at": now,
        }
        payload = {"_meta": payload_meta, "settings": deepcopy(settings)}
        self.user_dir.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=self.user_dir,
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
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        self.set_last_used(safe)
        return safe

    def delete(self, name: str) -> bool:
        """Delete a user preset while refusing protected defaults."""

        safe = sanitize_preset_name(name)
        if safe.casefold() in self._default_names():
            raise PresetError(f"Preset '{safe}' is a protected default")
        entry = next(
            (
                candidate
                for candidate in self._entries_in(self.user_dir, False)
                if sanitize_preset_name(candidate.name).casefold() == safe.casefold()
            ),
            None,
        )
        if entry is None:
            return False
        target = entry.path
        was_last_used = (self.get_last_used() or "").casefold() == entry.name.casefold()
        try:
            target.unlink()
        except OSError as exc:
            raise PresetError(f"Could not delete preset '{safe}': {exc}") from exc
        if was_last_used:
            try:
                self._last_used_path.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def get_last_used(self) -> str | None:
        """Return the valid last-used preset name, if any."""

        try:
            value = self._last_used_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        entry = self._find(value) if value else None
        return entry.name if entry is not None else None

    def startup_preset_name(self) -> str | None:
        """Return last-used, then the configured shipped default, then the first entry."""

        last_used = self.get_last_used()
        if last_used:
            return last_used
        if self.default_preset_name:
            entry = self._find(self.default_preset_name)
            if entry is not None:
                return entry.name
        entries = self.list_presets()
        return entries[0].name if entries else None

    def set_last_used(self, name: str) -> None:
        """Persist the last-used preset marker as UTF-8 text."""

        entry = self._find(name)
        value = entry.name if entry is not None else sanitize_preset_name(name)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._last_used_path.with_suffix(".txt.tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, self._last_used_path)

    def exists(self, name: str) -> bool:
        """Return whether a default or user preset exists."""

        return self._find(name) is not None


def merge_settings(settings: dict[str, Any] | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Fill missing settings and drop unknown keys with Python warnings."""

    current = settings if isinstance(settings, dict) else {}
    unknown = [key for key in current if key not in defaults]
    if unknown:
        warnings.warn(
            f"Dropped unknown preset settings: {', '.join(sorted(map(str, unknown)))}",
            UserWarning,
            stacklevel=2,
        )
    merged = deepcopy(defaults)
    for key in defaults:
        if key in current:
            merged[key] = deepcopy(current[key])
    return merged


__all__ = [
    "PresetEntry",
    "PresetError",
    "PresetStore",
    "merge_settings",
    "sanitize_preset_name",
]
