"""HTN recursive planner demo runbook (`htn_recursive_v0`)."""

from __future__ import annotations

from typing import Any

from neuronium_agent.planning.dag import ActionGraph, GraphMetadata
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)


_HTN_PLANNER_SYSTEM_PROMPT = (
    "You are an HTN-lite planning assistant. "
    "When asked to select a decomposition method, return strict JSON."
)


class HtnRecursiveDemoV0Runbook(Runbook):
    """Single-stage runbook that uses `htn_recursive_v0` backend."""

    @property
    def runbook_id(self) -> str:
        return "htn_recursive_demo_v0"

    @property
    def description(self) -> str:
        return (
            "Demo runbook: COMMIT performs HTN-lite recursive decomposition "
            "and executes the generated runtime ActionGraph."
        )

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        stage_graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"plan-htn-recursive-placeholder-{execution_id[:12]}",
                description=f"HTN recursive planner placeholder for: {objective}",
            ),
            nodes=[],
            edges=[],
        )

        return [
            ActionGraphStage(
                stage_id="htn_recursive_demo_v0:stage1",
                graph=stage_graph,
                initial_inputs_override={"runbook_id": "htn_recursive_demo_v0"},
                success_gate=StageSuccessGate(
                    required_completed_nodes=["draft_report"],
                    critic_node_id="critic_report",
                ),
                dynamic_planner=DynamicPlannerSpec(
                    planner_node_id="plan_graph_htn",
                    planner_system_prompt=_HTN_PLANNER_SYSTEM_PROMPT,
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
                    backend_name="htn_recursive_v0",
                    backend_version="0",
                    backend_options={
                        "max_depth": 4,
                        "max_frontier": 64,
                        "max_total_nodes": 64,
                        "model_assisted_method_selection": False,
                        "model_assisted_max_calls": 2,
                    },
                ),
            )
        ]
