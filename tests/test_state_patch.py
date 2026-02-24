from __future__ import annotations

import pytest

from neuronium_agent.planning.state_patch import PatchOperation
from neuronium_agent.planning.state_patch import StatePatchError
from neuronium_agent.planning.state_patch import apply_patch


def test_apply_patch_add_replace_remove_for_dict_paths() -> None:
    state = {"inputs": {"path": "/tmp/a.md"}, "flags": {"enabled": False}}
    patched = apply_patch(
        state,
        [
            PatchOperation(op="add", path="/inputs/encoding", value="utf-8"),
            PatchOperation(op="replace", path="/flags/enabled", value=True),
            PatchOperation(op="remove", path="/inputs/path"),
        ],
    )

    assert patched == {"inputs": {"encoding": "utf-8"}, "flags": {"enabled": True}}
    assert state == {"inputs": {"path": "/tmp/a.md"}, "flags": {"enabled": False}}


def test_apply_patch_supports_list_append_and_replace() -> None:
    state = {"urls": ["https://a.example"]}
    patched = apply_patch(
        state,
        [
            {"op": "add", "path": "/urls/-", "value": "https://b.example"},
            {"op": "replace", "path": "/urls/0", "value": "https://c.example"},
        ],
    )
    assert patched["urls"] == ["https://c.example", "https://b.example"]


def test_apply_patch_raises_deterministic_error_for_missing_path() -> None:
    with pytest.raises(StatePatchError, match="Replace target is missing"):
        apply_patch(
            {"inputs": {}},
            [PatchOperation(op="replace", path="/inputs/path", value="/tmp/x.md")],
        )


def test_apply_patch_supports_escaped_json_pointer_tokens() -> None:
    patched = apply_patch(
        {"inputs": {}},
        [{"op": "add", "path": "/inputs/a~1b", "value": "ok"}],
    )
    assert patched == {"inputs": {"a/b": "ok"}}
