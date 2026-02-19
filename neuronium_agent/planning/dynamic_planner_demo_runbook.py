"""Dynamic planner demo runbook (v0.3, T0.1).

This runbook demonstrates the infrastructure path where COMMIT does not
use a hardcoded execution DAG. Instead, a planner ModelNode returns the
runtime ``ActionGraph`` which is then validated and executed.
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.planning.dag import ActionGraph, GraphMetadata
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)


_DYNAMIC_PLANNER_SYSTEM_PROMPT = (
    "You are a strict DAG planner.\n"
    "Build an ActionGraph for a docs-report style flow.\n"
    "Use node IDs: read_000, merge_docs, draft_report, critic_report.\n"
    "Use node types only from the allowed list.\n"
    "For mcp node use tool_name='fs.read_text' for docs flow.\n"
    "Return only valid JSON matching the ActionGraph schema."
)


class DynamicPlannerDemoV1Runbook(Runbook):
    """Single-stage runbook using dynamic planner at COMMIT."""

    @property
    def runbook_id(self) -> str:
        return "dynamic_planner_demo_v1"

    @property
    def description(self) -> str:
        return (
            "Demo runbook: planner ModelNode generates ActionGraph at runtime, "
            "then orchestrator executes the generated DAG."
        )

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        # Placeholder stage graph: COMMIT will replace this with planned graph.
        stage_graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"plan-dynamic-demo-placeholder-{execution_id[:12]}",
                description=f"Dynamic planner placeholder for: {objective}",
            ),
            nodes=[],
            edges=[],
        )

        return [
            ActionGraphStage(
                stage_id="dynamic_planner_demo_v1:stage1",
                graph=stage_graph,
                initial_inputs_override={"runbook_id": "dynamic_planner_demo_v1"},
                success_gate=StageSuccessGate(
                    required_completed_nodes=["draft_report"],
                    critic_node_id="critic_report",
                ),
                dynamic_planner=DynamicPlannerSpec(
                    planner_node_id="plan_graph",
                    planner_system_prompt=_DYNAMIC_PLANNER_SYSTEM_PROMPT,
                    allowed_node_types=[
                        "model",
                        "mcp",
                        "code",
                        "decision",
                        "aggregate",
                    ],
                    allowed_tool_names=[
                        "fs.read_text",
                        "fs.write_text",
                        "web.fetch_html",
                        "web.extract_article",
                    ],
                ),
            ),
        ]
