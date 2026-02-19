"""Autofix demo runbook: two-stage generate → critic → (on fail) fix → critic.

Stage 1: plan_iter1 (generate → execute → critic).
Stage 2: plan_iter2_fix (fix → execute_fix → critic_fix), graph and
initial_inputs built at runtime from stage 1 results and verdict.

Internal module; not part of PUBLIC_API_SPEC.
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.planning.autofix_helpers import (
    build_added_constraints,
    build_fix_context,
)
from neuronium_agent.planning.dag import ActionGraph
from neuronium_agent.planning.htn import HTNPlanner
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)
from neuronium_agent.verification.demo_critic import DemoCriticVerdict


def _iter2_graph_builder(context: dict[str, Any]) -> tuple[ActionGraph, dict[str, Any]]:
    """Build iter2 graph and fix_context from previous stage results."""
    objective = context["objective"]
    constraints = list(context["constraints"])
    prev_stage_results = context.get("prev_stage_results") or {}
    prev_stage_verdict = context.get("prev_stage_verdict")
    if prev_stage_verdict is None:
        prev_stage_verdict = DemoCriticVerdict(
            verdict="UNCERTAIN",
            confidence=0.0,
            evidence=[],
            gaps=["resume: no verdict from previous stage"],
        )
    fix_context = build_fix_context(prev_stage_results, prev_stage_verdict)
    added_constraints = build_added_constraints(prev_stage_results, prev_stage_verdict)
    planner = HTNPlanner()
    graph = planner.plan_iter2_fix(
        objective,
        constraints + added_constraints,
        fix_context=fix_context,
    )
    return (graph, fix_context)


class AutofixDemoRunbook(Runbook):
    """Two-stage autofix demo: iter1 (generate→execute→critic), iter2 (fix→execute_fix→critic_fix)."""

    @property
    def runbook_id(self) -> str:
        return "autofix_demo"

    @property
    def description(self) -> str:
        return (
            "Fixed 2-iteration autofix demo: generate code → critic; "
            "on fail, fix pipeline → critic_fix."
        )

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        planner = HTNPlanner()
        graph1 = planner.plan_iter1(objective, constraints)
        return [
            ActionGraphStage(
                stage_id="autofix_demo:iter1",
                graph=graph1,
                success_gate=StageSuccessGate(
                    required_completed_nodes=["execute"],
                    critic_node_id="critic",
                ),
                exit_run_on_success=True,
                proceed_to_next_stage_on_fail=True,
            ),
            ActionGraphStage(
                stage_id="autofix_demo:iter2",
                graph=None,
                graph_builder=_iter2_graph_builder,
                success_gate=StageSuccessGate(
                    required_completed_nodes=["execute_fix"],
                    critic_node_id="critic_fix",
                ),
            ),
        ]
