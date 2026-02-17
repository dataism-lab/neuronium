"""Planner contracts for dynamic ActionGraph generation.

This module isolates planner-specific request/result types from runbook stage
contracts so that multiple planner backends can be introduced safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuronium_agent.planning.dag import ActionGraph


@dataclass(frozen=True)
class DynamicPlannerSpec:
    """Declarative configuration for dynamic planner stage."""

    planner_node_id: str = "plan_graph"
    planner_system_prompt: str = (
        "You are a strict planning model. "
        "Return ONLY valid JSON that conforms to the provided ActionGraph schema."
    )
    allowed_node_types: list[str] = field(default_factory=lambda: [
        "model",
        "mcp",
        "code",
        "decision",
        "aggregate",
    ])
    allowed_tool_names: list[str] = field(default_factory=list)
    backend_options: dict[str, Any] = field(default_factory=dict)
    backend_name: str = "legacy_dynamic_v1"
    backend_version: str = "1"


@dataclass(frozen=True)
class PlannerRequest:
    """Input envelope consumed by planner backends."""

    objective: str
    constraints: list[str]
    metadata: dict[str, Any]
    runbook_id: str
    stage_id: str
    execution_id: str
    spec: DynamicPlannerSpec
    operator_catalog_hash: str | None = None
    allowed_capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerDecisionTrace:
    """Structured trace for planner reasoning path."""

    subgoals: list[str] = field(default_factory=list)
    selected_methods: list[str] = field(default_factory=list)
    justification_keys: list[str] = field(default_factory=list)
    decomposition_steps: list[dict[str, Any]] = field(default_factory=list)
    method_expansion_path: list[str] = field(default_factory=list)
    leaf_operators: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerResult:
    """Planner output envelope passed back to orchestrator."""

    action_graph: ActionGraph
    backend_name: str
    backend_version: str
    operator_catalog_hash: str | None = None
    decision_trace: PlannerDecisionTrace | None = None


@dataclass(frozen=True)
class PlannerEscalation:
    """Planner outcome requiring user clarification before execution."""

    reason: str
    backend_name: str
    backend_version: str
    clarification_request_artifact_id: str
    missing_fields: list[dict[str, Any]] = field(default_factory=list)
    evidence_artifact_ids: list[str] = field(default_factory=list)
    operator_catalog_hash: str | None = None
    decision_trace: PlannerDecisionTrace | None = None


PlannerOutcome = PlannerResult | PlannerEscalation
