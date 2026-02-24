"""State patch primitives (RFC6902 subset for phase 1)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, model_validator


class StatePatchError(ValueError):
    """Patch application failed."""


class PatchOperation(BaseModel):
    """Single patch operation."""

    op: Literal["add", "replace", "remove"]
    path: str
    value: Any | None = None

    @model_validator(mode="after")
    def _validate_by_op(self) -> PatchOperation:
        if not self.path.startswith("/"):
            raise ValueError("Patch path must start with '/'")
        if self.op in {"add", "replace"} and self.value is None:
            raise ValueError(f"Operation '{self.op}' requires 'value'")
        return self


def _decode_pointer(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise StatePatchError(f"Patch path must start with '/': {pointer!r}")
    return [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    ]


def _resolve_parent(container: Any, tokens: list[str]) -> tuple[Any, str]:
    if not tokens:
        raise StatePatchError("Root path patching is not supported in phase 1")
    parent = container
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                raise StatePatchError(f"Path not found: {'/'.join(tokens[:-1])}")
            parent = parent[token]
            continue
        if isinstance(parent, list):
            if not token.isdigit():
                raise StatePatchError(f"Expected list index, got {token!r}")
            idx = int(token)
            if idx < 0 or idx >= len(parent):
                raise StatePatchError(f"List index out of range: {idx}")
            parent = parent[idx]
            continue
        raise StatePatchError("Cannot traverse non-container value")
    return parent, tokens[-1]


def _apply_add(parent: Any, key: str, value: Any) -> None:
    if isinstance(parent, dict):
        parent[key] = value
        return
    if isinstance(parent, list):
        if key == "-":
            parent.append(value)
            return
        if not key.isdigit():
            raise StatePatchError(f"Expected list index for add, got {key!r}")
        idx = int(key)
        if idx < 0 or idx > len(parent):
            raise StatePatchError(f"Add index out of range: {idx}")
        parent.insert(idx, value)
        return
    raise StatePatchError("Cannot add into non-container value")


def _apply_replace(parent: Any, key: str, value: Any) -> None:
    if isinstance(parent, dict):
        if key not in parent:
            raise StatePatchError(f"Replace target is missing: {key!r}")
        parent[key] = value
        return
    if isinstance(parent, list):
        if not key.isdigit():
            raise StatePatchError(f"Expected list index for replace, got {key!r}")
        idx = int(key)
        if idx < 0 or idx >= len(parent):
            raise StatePatchError(f"Replace index out of range: {idx}")
        parent[idx] = value
        return
    raise StatePatchError("Cannot replace in non-container value")


def _apply_remove(parent: Any, key: str) -> None:
    if isinstance(parent, dict):
        if key not in parent:
            raise StatePatchError(f"Remove target is missing: {key!r}")
        del parent[key]
        return
    if isinstance(parent, list):
        if not key.isdigit():
            raise StatePatchError(f"Expected list index for remove, got {key!r}")
        idx = int(key)
        if idx < 0 or idx >= len(parent):
            raise StatePatchError(f"Remove index out of range: {idx}")
        parent.pop(idx)
        return
    raise StatePatchError("Cannot remove from non-container value")


def apply_patch(
    state: Mapping[str, Any],
    patch_ops: Sequence[PatchOperation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply RFC6902 subset (add/replace/remove) to state copy."""

    result = deepcopy(dict(state))

    for idx, raw_op in enumerate(patch_ops):
        try:
            op = (
                raw_op
                if isinstance(raw_op, PatchOperation)
                else PatchOperation.model_validate(raw_op)
            )
        except Exception as exc:  # pragma: no cover - delegated validation
            raise StatePatchError(f"Invalid patch operation at index {idx}: {exc}") from exc

        tokens = _decode_pointer(op.path)
        parent, key = _resolve_parent(result, tokens)

        if op.op == "add":
            _apply_add(parent, key, deepcopy(op.value))
            continue
        if op.op == "replace":
            _apply_replace(parent, key, deepcopy(op.value))
            continue
        if op.op == "remove":
            _apply_remove(parent, key)
            continue

        raise StatePatchError(f"Unsupported operation at index {idx}: {op.op!r}")

    return result
