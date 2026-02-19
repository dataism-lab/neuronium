"""Planner backend abstraction for runtime ActionGraph generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode
from neuronium_agent.planning.dynamic_planner import (
    action_graph_json_schema,
    build_dynamic_planner_prompt,
    parse_action_graph_from_model_output,
    validate_planned_graph,
)
from neuronium_agent.planning.htn_recursive_backend import HtnRecursivePlannerBackend
from neuronium_agent.planning.planner_contract import PlannerOutcome, PlannerRequest, PlannerResult


ExecutePlannerGraphFn = Callable[
    [ActionGraph, dict[str, object], bool],
    dict[str, NodeOutput],
]


class PlannerBackend(Protocol):
    """Contract implemented by all planner backends."""

    @property
    def backend_name(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    def plan(
        self,
        *,
        request: PlannerRequest,
        execute_graph: ExecutePlannerGraphFn,
    ) -> PlannerOutcome: ...


@dataclass(frozen=True)
class LegacyDynamicPlannerBackend:
    """Current single-step planner backend, wrapped behind a stable interface."""

    @property
    def backend_name(self) -> str:
        return "legacy_dynamic_v1"

    @property
    def backend_version(self) -> str:
        return "1"

    def plan(
        self,
        *,
        request: PlannerRequest,
        execute_graph: ExecutePlannerGraphFn,
    ) -> PlannerResult:
        planner_prompt = build_dynamic_planner_prompt(
            objective=request.objective,
            constraints=request.constraints,
            metadata=request.metadata,
            runbook_id=request.runbook_id,
            stage_id=request.stage_id,
            spec=request.spec,
        )

        planner_graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"planner-{request.runbook_id}-{request.execution_id[:12]}",
                description=f"Dynamic planner stage for {request.stage_id}",
            ),
            nodes=[
                GraphNode(
                    node_id=request.spec.planner_node_id,
                    node_type="model",
                    label="Plan runtime ActionGraph",
                    parameters={
                        "system_prompt": request.spec.planner_system_prompt,
                        "json_schema": action_graph_json_schema(),
                    },
                    priority=0,
                )
            ],
            edges=[],
        )

        planner_results = execute_graph(
            planner_graph,
            {"prompt": planner_prompt},
            True,
        )
        planner_output = planner_results.get(request.spec.planner_node_id)
        if planner_output is None:
            raise ValueError(
                "Dynamic planner execution produced no output for planner node"
            )

        planned_graph = parse_action_graph_from_model_output(planner_output)
        validated = validate_planned_graph(
            planned_graph,
            spec=request.spec,
            operator_catalog=None,
        )
        return PlannerResult(
            action_graph=validated,
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            operator_catalog_hash=request.operator_catalog_hash,
        )


def get_planner_backend(name: str) -> PlannerBackend:
    """Return planner backend implementation by name."""
    if name == "legacy_dynamic_v1":
        return LegacyDynamicPlannerBackend()
    if name == "htn_recursive_v0":
        return HtnRecursivePlannerBackend()
    raise ValueError(f"Unknown planner backend: {name}")
