"""Schema-driven missing slot computation."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, Field

from neuronium_agent.schemas.tool_schema_registry import (
    extract_required_json_pointers,
    schema_fragment_at_pointer,
)


class MissingSlot(BaseModel):
    """A required value that is missing in current state."""

    path: str
    expected_schema: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    critical: bool = True


def _schema_allows_null(schema_fragment: Mapping[str, Any]) -> bool:
    schema_type = schema_fragment.get("type")
    if isinstance(schema_type, list):
        return "null" in schema_type
    if schema_type == "null":
        return True
    any_of = schema_fragment.get("anyOf")
    if isinstance(any_of, list):
        return any(
            isinstance(branch, Mapping) and branch.get("type") == "null"
            for branch in any_of
        )
    return False


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    return [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    ]


def _state_value_at_pointer(
    state: Mapping[str, Any],
    pointer: str,
) -> tuple[bool, Any]:
    current: Any = state
    for token in _pointer_tokens(pointer):
        if isinstance(current, Mapping):
            if token not in current:
                return False, None
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-":
                return False, None
            if not token.isdigit():
                return False, None
            idx = int(token)
            if idx < 0 or idx >= len(current):
                return False, None
            current = current[idx]
            continue
        return False, None
    return True, current


def _has_missing_ancestor(path: str, missing_paths: set[str]) -> bool:
    parts = path.split("/")
    for idx in range(2, len(parts)):
        parent = "/".join(parts[:idx])
        if parent in missing_paths:
            return True
    return False


def compute_missing_slots(
    state: Mapping[str, Any],
    input_schema: Mapping[str, Any],
    *,
    base_pointer: str = "",
) -> list[MissingSlot]:
    """Compute required missing slots from state+input schema."""

    required_paths = extract_required_json_pointers(
        input_schema,
        base_pointer=base_pointer,
    )
    missing: dict[str, MissingSlot] = {}
    missing_paths: set[str] = set()

    for path in required_paths:
        if _has_missing_ancestor(path, missing_paths):
            continue
        exists, value = _state_value_at_pointer(state, path)
        expected_schema = schema_fragment_at_pointer(input_schema, path)
        if not exists:
            slot = MissingSlot(
                path=path,
                expected_schema=expected_schema,
                reason=f"Required value is missing at path '{path}'.",
                critical=True,
            )
            missing[path] = slot
            missing_paths.add(path)
            continue
        if value is None and not _schema_allows_null(expected_schema):
            missing[path] = MissingSlot(
                path=path,
                expected_schema=expected_schema,
                reason=f"Required value is null at path '{path}'.",
                critical=True,
            )
            missing_paths.add(path)

    return [missing[k] for k in sorted(missing)]


def slot_path_to_legacy_field(path: str) -> str:
    """Best-effort temporary bridge from JSON Pointer path to legacy field."""

    if path.startswith("/inputs/"):
        return path.removeprefix("/inputs/")
    return path.lstrip("/")
