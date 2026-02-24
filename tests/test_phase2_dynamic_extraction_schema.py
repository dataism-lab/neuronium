from __future__ import annotations

import os

import pytest

from neuronium_agent.planning.extraction_contract import extraction_envelope_json_schema
from neuronium_agent.planning.htn_recursive_backend import HtnRecursivePlannerBackend
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec, PlannerRequest


def _make_request(*, metadata: dict, tools: list[str]) -> PlannerRequest:
    return PlannerRequest(
        objective="test objective",
        constraints=[],
        metadata=metadata,
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="phase2test001",
        spec=DynamicPlannerSpec(
            backend_name="htn_recursive_v0",
            backend_version="0",
            allowed_tool_names=tools,
        ),
        operator_catalog_hash="hash-phase2",
    )


def test_dynamic_extraction_schema_uses_tool_input_schema() -> None:
    schema = extraction_envelope_json_schema(
        input_schema={
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
            },
        }
    )

    inputs = schema["properties"]["inputs"]
    assert inputs["required"] == ["url"]
    assert set(inputs["properties"].keys()) == {"timeout_seconds", "url"}
    field_schema = schema["properties"]["missing_fields"]["items"]["properties"]["field"]
    assert field_schema["enum"] == ["url"]


def test_phase2_feature_flag_can_be_scoped_by_metadata() -> None:
    backend = HtnRecursivePlannerBackend()
    request = _make_request(
        metadata={
            "dynamic_extraction_schema": True,
            "dynamic_extraction_schema_runbooks": ["super_agent_v0"],
            "dynamic_extraction_schema_stages": ["super_agent_v0:stage1"],
        },
        tools=["web.fetch_html"],
    )
    assert backend._is_dynamic_extraction_schema_enabled(request) is True

    blocked = _make_request(
        metadata={
            "dynamic_extraction_schema": True,
            "dynamic_extraction_schema_runbooks": ["other_runbook"],
        },
        tools=["web.fetch_html"],
    )
    assert backend._is_dynamic_extraction_schema_enabled(blocked) is False


def test_phase2_dynamic_schema_adds_slots_for_new_allowed_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HtnRecursivePlannerBackend()
    monkeypatch.setenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA", "1")
    monkeypatch.setenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA_RUNBOOKS", "super_agent_v0")
    request = _make_request(
        metadata={},
        tools=["web.fetch_html", "fs.glob"],
    )

    assert backend._is_dynamic_extraction_schema_enabled(request) is True
    dynamic_input_schema = backend._build_dynamic_extraction_input_schema(request)
    assert dynamic_input_schema is not None
    assert "url" in dynamic_input_schema["required"]
    assert "root" in dynamic_input_schema["required"]
    assert "pattern" in dynamic_input_schema["required"]
    assert "url" in dynamic_input_schema["properties"]
    assert "root" in dynamic_input_schema["properties"]
    assert "pattern" in dynamic_input_schema["properties"]

    monkeypatch.delenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA", raising=False)
    monkeypatch.delenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA_RUNBOOKS", raising=False)
    monkeypatch.delenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA_STAGES", raising=False)
    assert os.environ.get("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA") is None
