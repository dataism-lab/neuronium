from __future__ import annotations

from neuronium_agent.planning.missing_slots import compute_missing_slots
from neuronium_agent.planning.missing_slots import slot_path_to_legacy_field


def test_compute_missing_slots_from_nested_schema() -> None:
    schema = {
        "type": "object",
        "required": ["inputs"],
        "properties": {
            "inputs": {
                "type": "object",
                "required": ["path", "overwrite"],
                "properties": {
                    "path": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
            },
        },
    }

    slots = compute_missing_slots(state={}, input_schema=schema)
    assert [s.path for s in slots] == ["/inputs"]
    assert slots[0].expected_schema.get("type") == "object"


def test_compute_missing_slots_treats_nullable_required_as_present() -> None:
    schema = {
        "type": "object",
        "required": ["payload"],
        "properties": {
            "payload": {
                "type": "object",
                "required": ["name", "note"],
                "properties": {
                    "name": {"type": "string"},
                    "note": {"type": ["string", "null"]},
                },
            }
        },
    }
    state = {"payload": {"name": "n", "note": None}}

    slots = compute_missing_slots(state=state, input_schema=schema)
    assert slots == []


def test_slot_path_to_legacy_field_bridge() -> None:
    assert slot_path_to_legacy_field("/inputs/output_text") == "output_text"
    assert slot_path_to_legacy_field("/tool_args/path") == "tool_args/path"
