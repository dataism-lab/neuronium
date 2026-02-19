"""Universal supervised runbook for super-agent flow."""

from __future__ import annotations

from typing import Any

from neuronium_agent.planning.dag import ActionGraph, GraphMetadata
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)


_SUPER_AGENT_PLANNER_SYSTEM_PROMPT = (
    "You are a super-agent planner using HTN-lite decomposition and strict JSON contracts."
)


class SuperAgentV0Runbook(Runbook):
    """Single-stage dynamic runbook intended as universal default."""

    @property
    def runbook_id(self) -> str:
        return "super_agent_v0"

    @property
    def description(self) -> str:
        return (
            "Universal super-agent runbook with extraction, clarification, "
            "HTN planning, execution and critic verification."
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
                plan_id=f"plan-super-agent-placeholder-{execution_id[:12]}",
                description=f"Super-agent placeholder for: {objective}",
            ),
            nodes=[],
            edges=[],
        )

        return [
            ActionGraphStage(
                stage_id="super_agent_v0:stage1",
                graph=stage_graph,
                initial_inputs_override={"runbook_id": "super_agent_v0"},
                success_gate=StageSuccessGate(
                    required_completed_nodes=["draft_report"],
                    critic_node_id="critic_report",
                ),
                dynamic_planner=DynamicPlannerSpec(
                    planner_node_id="plan_graph_super",
                    planner_system_prompt=_SUPER_AGENT_PLANNER_SYSTEM_PROMPT,
                    allowed_node_types=["model", "mcp", "code", "decision", "aggregate"],
                    allowed_tool_names=[
                        "fs.read_text",
                        "fs.write_text",
                        "fs.glob",
                        "web.fetch_html",
                        "web.extract_article",
                        "text.extract_entities",
                        "artifact.put_json",
                        "export.write_text",
                    ],
                    backend_name="htn_recursive_v0",
                    backend_version="0",
                    backend_options={
                        "max_depth": 4,
                        "max_frontier": 64,
                        "max_total_nodes": 64,
                        "model_assisted_method_selection": False,
                        "model_assisted_max_calls": 2,
                        "planner_node_prefix": "super_method_select",
                    },
                ),
            ),
        ]
