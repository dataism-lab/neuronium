from __future__ import annotations

import pytest

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode
from neuronium_agent.planning.dynamic_planner import validate_planned_graph
from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.planner_backend import get_planner_backend
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec
from neuronium_agent.planning.planner_contract import PlannerRequest


def test_operator_catalog_allows_known_mcp_tool() -> None:
    catalog = OperatorCatalog.default()
    node = GraphNode(
        node_id="read_000",
        node_type="mcp",
        parameters={"tool_name": "fs.read_text", "tool_args": {"path": "/tmp/x.txt"}},
    )
    catalog.assert_node_allowed(node)

    text_node = GraphNode(
        node_id="extract_entities",
        node_type="mcp",
        parameters={"tool_name": "text.extract_entities", "tool_args": {"text": "hi"}},
    )
    catalog.assert_node_allowed(text_node)

    artifact_node = GraphNode(
        node_id="put_artifact",
        node_type="mcp",
        parameters={
            "tool_name": "artifact.put_json",
            "tool_args": {"artifact_type": "x", "json": {"a": 1}},
        },
    )
    catalog.assert_node_allowed(artifact_node)


def test_operator_catalog_rejects_unknown_mcp_tool() -> None:
    catalog = OperatorCatalog.default()
    node = GraphNode(
        node_id="bad_tool",
        node_type="mcp",
        parameters={"tool_name": "telegram.send_message"},
    )
    with pytest.raises(ValueError, match="no operator contract"):
        catalog.assert_node_allowed(node)


def test_validate_planned_graph_uses_operator_catalog() -> None:
    catalog = OperatorCatalog.default()
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="plan-1", description="bad plan"),
        nodes=[
            GraphNode(
                node_id="n1",
                node_type="mcp",
                parameters={"tool_name": "telegram.send_message"},
            )
        ],
        edges=[],
    )
    spec = DynamicPlannerSpec(
        allowed_node_types=["mcp"],
        allowed_tool_names=[],
    )
    with pytest.raises(ValueError, match="no operator contract"):
        validate_planned_graph(graph, spec=spec, operator_catalog=catalog)


def test_htn_leaf_operator_must_match_allowed_tools() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    spec = DynamicPlannerSpec(
        backend_name="htn_recursive_v0",
        backend_version="0",
        allowed_node_types=["model", "mcp", "aggregate"],
        allowed_tool_names=["fs.read_text"],
    )
    request = PlannerRequest(
        objective="Summarize one webpage",
        constraints=[],
        metadata={"url": "https://example.com"},
        runbook_id="htn_recursive_demo_v0",
        stage_id="htn_recursive_demo_v0:stage1",
        execution_id="htn-op-001",
        spec=spec,
        operator_catalog_hash="hash-op-001",
    )

    def no_execute(
        graph: ActionGraph,
        initial_inputs: dict[str, object],
        suppress: bool,
    ) -> dict[str, NodeOutput]:
        _ = graph, initial_inputs, suppress
        return {}

    result = backend.plan(request=request, execute_graph=no_execute)
    with pytest.raises(ValueError, match="is not allowed"):
        validate_planned_graph(
            result.action_graph,
            spec=spec,
            operator_catalog=OperatorCatalog.default(),
        )
