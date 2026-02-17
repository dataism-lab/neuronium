"""Dynamic planner helpers for runtime ActionGraph generation (v0.3).

This module keeps planner-related logic small and focused:
- build a deterministic planner prompt,
- parse planner output into ``ActionGraph``,
- validate graph safety/integrity constraints before execution.
"""

from __future__ import annotations

import json
from typing import Any

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.planning.dag import ActionGraph
from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec
from neuronium_agent.schemas.registry import SCHEMA_REGISTRY


def action_graph_json_schema() -> dict[str, Any]:
    """Return JSON schema for ActionGraph from the canonical registry."""
    model_cls = SCHEMA_REGISTRY["ActionGraph"]
    return model_cls.model_json_schema()


def build_dynamic_planner_prompt(
    *,
    objective: str,
    constraints: list[str],
    metadata: dict[str, Any],
    runbook_id: str,
    stage_id: str,
    spec: DynamicPlannerSpec,
) -> str:
    """Build a strict planner prompt for runtime graph generation."""
    constraints_block = (
        "\n".join(f"- {c}" for c in constraints)
        if constraints
        else "- (none)"
    )
    allowed_types = ", ".join(spec.allowed_node_types)
    allowed_tools = ", ".join(spec.allowed_tool_names) if spec.allowed_tool_names else "(any)"
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    return (
        "Build an ActionGraph JSON for the provided objective.\n"
        "Return ONLY a JSON object matching the ActionGraph schema.\n"
        "Do not add explanations or markdown.\n\n"
        f"Runbook: {runbook_id}\n"
        f"Stage: {stage_id}\n"
        f"Objective: {objective}\n"
        f"Constraints:\n{constraints_block}\n\n"
        f"Allowed node types: {allowed_types}\n"
        f"Allowed MCP tool names: {allowed_tools}\n\n"
        "Metadata (for planning context):\n"
        f"{metadata_json}\n"
    )


def parse_action_graph_from_model_output(output: NodeOutput) -> ActionGraph:
    """Parse planner model output into ``ActionGraph``."""
    if output.status != "COMPLETED":
        raise ValueError(f"Dynamic planner failed: {output.error or 'unknown error'}")

    payload: dict[str, Any] | None = None
    parsed = output.outputs.get("parsed")
    if isinstance(parsed, dict):
        payload = parsed
    elif isinstance(parsed, str):
        payload = json.loads(parsed)

    if payload is None:
        content = output.outputs.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Dynamic planner returned empty output")
        payload = json.loads(content)

    if not isinstance(payload, dict):
        raise ValueError("Dynamic planner output is not a JSON object")

    return ActionGraph.model_validate(payload)


def validate_planned_graph(
    graph: ActionGraph,
    *,
    spec: DynamicPlannerSpec,
    operator_catalog: OperatorCatalog | None = None,
) -> ActionGraph:
    """Validate runtime graph against planner safety/integrity constraints."""
    if not graph.metadata.plan_id:
        raise ValueError("Dynamic plan must include non-empty metadata.plan_id")

    node_ids = [n.node_id for n in graph.nodes]
    uniq_ids = set(node_ids)
    if len(uniq_ids) != len(node_ids):
        raise ValueError("Dynamic plan contains duplicate node_id values")

    allowed_types = set(spec.allowed_node_types)
    for n in graph.nodes:
        if n.node_type not in allowed_types:
            raise ValueError(
                f"Dynamic plan node_type '{n.node_type}' is not allowed"
            )

        if n.node_type == "mcp" and spec.allowed_tool_names:
            tool_name = str(n.parameters.get("tool_name", ""))
            if tool_name not in set(spec.allowed_tool_names):
                raise ValueError(
                    f"Dynamic plan MCP tool '{tool_name}' is not allowed"
                )
        if operator_catalog is not None:
            operator_catalog.assert_node_allowed(n)

    for e in graph.edges:
        if e.source not in uniq_ids:
            raise ValueError(f"Dynamic plan edge source not found: {e.source}")
        if e.target not in uniq_ids:
            raise ValueError(f"Dynamic plan edge target not found: {e.target}")

    # Ensures acyclic DAG and deterministic ordering.
    graph.topological_order()
    return graph
