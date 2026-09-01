"""Ordered settings/component registry with validation and coercion."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class SettingEntry:
    """One registered component and its persistence/validation metadata."""

    key: str
    component: object
    default: Any
    section: str = "general"
    in_preset: bool = True
    in_metadata: bool = True
    kind: str | type | None = None
    choices: tuple[Any, ...] | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    description: str = ""


class SettingsRegistry:
    """Maintain a single ordered source of truth for application settings."""

    def __init__(self) -> None:
        self._entries: list[SettingEntry] = []
        self._by_key: dict[str, SettingEntry] = {}
        self._migrations: list[Callable[[dict[str, Any]], dict[str, Any] | None]] = []

    def add_migration(self, migrate: Callable[[dict[str, Any]], dict[str, Any] | None]) -> None:
        """Register a translation applied to raw settings before coercion.

        Migrations rewrite keys that older presets or run metadata stored under a
        name the UI no longer registers. Each callable receives a copy of the
        mapping and returns the rewritten mapping (or ``None`` to keep it as is).
        """

        if not callable(migrate):
            raise TypeError("migration must be callable")
        self._migrations.append(migrate)

    def register(
        self,
        key: str,
        component: object,
        default: Any,
        *,
        section: str = "general",
        in_preset: bool = True,
        in_metadata: bool = True,
        kind: str | type | None = None,
        choices: Sequence[Any] | None = None,
        minimum: float | int | None = None,
        maximum: float | int | None = None,
        description: str = "",
    ) -> object:
        """Register a component and return it unchanged for inline UI construction."""

        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("Setting key cannot be empty")
        if normalized_key in self._by_key:
            raise KeyError(f"Setting is already registered: {normalized_key}")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum cannot exceed maximum")
        entry = SettingEntry(
            key=normalized_key,
            component=component,
            default=deepcopy(default),
            section=str(section),
            in_preset=bool(in_preset),
            in_metadata=bool(in_metadata),
            kind=kind,
            choices=tuple(choices) if choices is not None else None,
            minimum=minimum,
            maximum=maximum,
            description=str(description),
        )
        self._entries.append(entry)
        self._by_key[normalized_key] = entry
        return component

    def keys(self) -> list[str]:
        """Return registered keys in positional handler order."""

        return [entry.key for entry in self._entries]

    def components(self) -> list[object]:
        """Return registered component objects in handler order."""

        return [entry.component for entry in self._entries]

    def defaults(self) -> dict[str, Any]:
        """Return a deep copy of every current default."""

        return {entry.key: deepcopy(entry.default) for entry in self._entries}

    def entries(self) -> list[SettingEntry]:
        """Return the immutable entry records in registration order."""

        return list(self._entries)

    def values_to_dict(self, values: Sequence[Any]) -> dict[str, Any]:
        """Map positional handler values to setting keys."""

        if len(values) != len(self._entries):
            raise ValueError(f"Expected {len(self._entries)} values, got {len(values)}")
        return {entry.key: value for entry, value in zip(self._entries, values)}

    def dict_to_values(self, d: dict[str, Any]) -> list[Any]:
        """Map settings back to positional values, filling current defaults."""

        source = d if isinstance(d, dict) else {}
        return [deepcopy(source.get(entry.key, entry.default)) for entry in self._entries]

    @staticmethod
    def _cast(value: Any, kind: str | type | None) -> Any:
        if kind is None:
            return value
        normalized = kind.__name__.casefold() if isinstance(kind, type) else str(kind).casefold()
        if normalized in {"bool", "boolean"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            text = str(value).strip().casefold()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"not a boolean: {value}")
        if normalized in {"int", "integer"}:
            if isinstance(value, bool):
                raise ValueError("booleans are not integers here")
            numeric = float(value)
            if not numeric.is_integer():
                raise ValueError(f"not an integer: {value}")
            return int(numeric)
        if normalized in {"float", "number"}:
            if isinstance(value, bool):
                raise ValueError("booleans are not numbers here")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"not a finite number: {value}")
            return numeric
        if normalized in {"str", "string"}:
            return str(value)
        if normalized in {"list", "array", "sequence"}:
            if value is None:
                return []
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return []
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    decoded = [part.strip() for part in text.split(",") if part.strip()]
                value = decoded
            if not isinstance(value, (list, tuple, set)):
                raise ValueError(f"not a list: {value}")
            return list(value)
        raise ValueError(f"unsupported setting kind: {kind}")

    @staticmethod
    def _coerce_choice_item(value: Any, allowed: tuple[Any, ...]) -> Any:
        for choice in allowed:
            if type(value) is type(choice) and value == choice:
                return choice
        for choice in allowed:
            try:
                if isinstance(choice, bool):
                    continue
                if isinstance(choice, int):
                    numeric = float(value)
                    if numeric.is_integer() and int(numeric) == choice:
                        return choice
                elif isinstance(choice, float) and float(value) == choice:
                    return choice
                elif isinstance(choice, str) and str(value) == choice:
                    return choice
            except (TypeError, ValueError):
                continue
        raise ValueError(f"{value!r} is not an allowed choice")

    def coerce(self, d: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
        """Cast, clamp, validate, and default a settings mapping."""

        source = dict(d) if isinstance(d, dict) else {}
        for migrate in self._migrations:
            migrated = migrate(dict(source))
            if isinstance(migrated, dict):
                source = migrated
        result: dict[str, Any] = {}
        warnings: list[str] = []
        for key in source:
            if key not in self._by_key:
                warnings.append(f"Dropped unknown setting '{key}'.")
        for entry in self._entries:
            raw = source.get(entry.key, deepcopy(entry.default))
            try:
                value = self._cast(raw, entry.kind)
            except (TypeError, ValueError) as exc:
                value = deepcopy(entry.default)
                warnings.append(f"{entry.key}: {exc}; using default {entry.default!r}.")
            allowed_choices = None
            if entry.choices is not None:
                allowed_choices = tuple(
                    choice[1]
                    if isinstance(choice, (tuple, list)) and len(choice) >= 2
                    else choice
                    for choice in entry.choices
                )
            if allowed_choices is not None and isinstance(value, list):
                filtered: list[Any] = []
                for item in value:
                    try:
                        selected = self._coerce_choice_item(item, allowed_choices)
                    except ValueError:
                        warnings.append(f"{entry.key}: dropped invalid list choice {item!r}.")
                        continue
                    if selected not in filtered:
                        filtered.append(selected)
                value = filtered
            elif allowed_choices is not None and value not in allowed_choices:
                warnings.append(f"{entry.key}: {value!r} is not an allowed choice; using default.")
                value = deepcopy(entry.default)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if entry.minimum is not None and value < entry.minimum:
                    warnings.append(f"{entry.key}: clamped to minimum {entry.minimum}.")
                    value = math.ceil(entry.minimum) if isinstance(value, int) else float(entry.minimum)
                if entry.maximum is not None and value > entry.maximum:
                    warnings.append(f"{entry.key}: clamped to maximum {entry.maximum}.")
                    value = math.floor(entry.maximum) if isinstance(value, int) else float(entry.maximum)
            result[entry.key] = value
        return result, warnings

    def preset_subset(self, d: dict[str, Any]) -> dict[str, Any]:
        """Return only settings registered for preset persistence."""

        return {
            entry.key: deepcopy(d.get(entry.key, entry.default))
            for entry in self._entries
            if entry.in_preset
        }

    def metadata_subset(self, d: dict[str, Any]) -> dict[str, Any]:
        """Return only settings registered for metadata persistence."""

        return {
            entry.key: deepcopy(d.get(entry.key, entry.default))
            for entry in self._entries
            if entry.in_metadata
        }


__all__ = ["SettingEntry", "SettingsRegistry"]
