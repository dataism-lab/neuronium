from __future__ import annotations

import os

import pytest

from neuronium_agent.planning.extraction_contract import (
    ExtractedIntent,
    ExtractionEnvelope,
    extraction_envelope_json_schema,
)
from neuronium_agent.planning.htn_recursive_backend import HtnRecursivePlannerBackend
from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.operator_contracts import OperatorContract
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec, PlannerRequest
from neuronium_agent.planning.tool_schema_registry import ToolSchemaRegistry


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


def test_phase2_validation_uses_dynamic_schema_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HtnRecursivePlannerBackend()
    monkeypatch.setenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA", "1")
    request = _make_request(
        metadata={},
        tools=["web.fetch_html", "fs.glob"],
    )
    dynamic_input_schema = backend._build_dynamic_extraction_input_schema(request)
    assert dynamic_input_schema is not None

    missing = backend._compute_missing_fields(
        request=request,
        envelope=ExtractionEnvelope(intent=ExtractedIntent(task_type="generic_task")),
        metadata={},
        dynamic_input_schema=dynamic_input_schema,
    )
    fields = {m.field for m in missing}
    assert {"url", "root", "pattern"} <= fields


def test_phase4_validation_uses_schema_driven_baseline_when_dynamic_schema_absent() -> None:
    backend = HtnRecursivePlannerBackend()
    request = _make_request(
        metadata={},
        tools=["web.fetch_html", "fs.glob"],
    )

    missing = backend._compute_missing_fields(
        request=request,
        envelope=ExtractionEnvelope(intent=ExtractedIntent(task_type="generic_task")),
        metadata={},
        dynamic_input_schema=None,
    )
    fields = {m.field for m in missing}
    assert "source" in fields
    assert "root" not in fields
    assert "pattern" not in fields


def test_phase4_missing_dedup_is_stable_by_path() -> None:
    backend = HtnRecursivePlannerBackend()
    request = _make_request(
        metadata={},
        tools=["web.fetch_html"],
    )
    missing = backend._compute_missing_fields(
        request=request,
        envelope=ExtractionEnvelope(
            intent=ExtractedIntent(task_type="news_summary"),
            missing_fields=[
                {"field": "url", "reason": "need URL", "critical": True},
                {"field": "inputs.url", "reason": "need URL duplicate", "critical": True},
            ],
        ),
        metadata={},
        dynamic_input_schema=None,
    )
    fields = [m.field for m in missing]
    assert fields.count("url") == 1


def test_phase2_new_catalog_tool_is_reflected_in_extraction_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = HtnRecursivePlannerBackend()
    base_catalog = OperatorCatalog.default()
    new_contract = OperatorContract(
        operator_id="mcp.custom.new_tool",
        node_type="mcp",
        tool_name="custom.new_tool",
        input_schema={
            "type": "object",
            "required": ["api_key"],
            "properties": {
                "api_key": {"type": "string"},
                "mode": {"type": "string"},
            },
        },
        output_schema={"type": "object"},
        deterministic=True,
        replay_required=True,
    )
    patched_catalog = OperatorCatalog(
        by_operator_id={
            **base_catalog.by_operator_id,
            new_contract.operator_id: new_contract,
        },
        by_tool_name={
            **base_catalog.by_tool_name,
            str(new_contract.tool_name): new_contract,
        },
        by_node_type=base_catalog.by_node_type,
    )
    patched_registry = ToolSchemaRegistry(operator_catalog=patched_catalog)
    monkeypatch.setattr(
        ToolSchemaRegistry,
        "from_default_catalog",
        classmethod(lambda cls: patched_registry),
    )
    monkeypatch.setenv("NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA", "1")
    request = _make_request(metadata={}, tools=["custom.new_tool"])

    dynamic_input_schema = backend._build_dynamic_extraction_input_schema(request)
    assert dynamic_input_schema is not None
    assert "api_key" in dynamic_input_schema["required"]

    envelope_schema = extraction_envelope_json_schema(input_schema=dynamic_input_schema)
    assert "api_key" in envelope_schema["properties"]["inputs"]["required"]

    missing = backend._compute_missing_fields(
        request=request,
        envelope=ExtractionEnvelope(intent=ExtractedIntent(task_type="generic_task")),
        metadata={},
        dynamic_input_schema=dynamic_input_schema,
    )
    fields = {m.field for m in missing}
    assert "api_key" in fields
