"""HTN planner — deterministic DAG templates for the autofix demo loop.

Provides two explicit template functions:

- ``plan_iter1()``  — generate → execute → critic
- ``plan_iter2_fix()`` — fix → execute_fix → critic_fix

No generalized HTN decomposition.  Node IDs are fixed and unique across
iterations so that ``ReplayProvider`` can distinguish replay_data per node.
"""

from __future__ import annotations

import uuid
from typing import Any

from neuronium_agent.planning.dag import (
    ActionGraph,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)
from neuronium_agent.verification.demo_critic import (
    CRITIC_SYSTEM_PROMPT,
    critic_json_schema,
    DemoCriticVerdict,
)


# -- Shared constants --------------------------------------------------------

_CODE_GEN_SYSTEM_PROMPT = (
    "You are a Python code generator.  "
    "Given the user's objective, produce ONLY valid Python code "
    "that accomplishes the task.  Output ONLY the code, no explanation."
)

_CODE_FIX_SYSTEM_PROMPT = (
    "You are a code-fixer.  Produce corrected Python code only.\n\n"
    "Rules:\n"
    "- The fix MUST be strictly targeted to the identified error.\n"
    "- Make the MINIMAL change required — do not refactor, rename "
    "unrelated symbols, reorganize code, add new features, or change "
    "logic outside the smallest necessary scope.\n"
    "- Preserve formatting and structure unless directly required to "
    "fix the error.\n"
    "- Output ONLY the corrected code, no explanation."
)

_CRITIC_JSON_SCHEMA = critic_json_schema()


class HTNPlanner:
    """Hierarchical Task Network planner (demo-only deterministic templates).

    Emits fixed DAG structures — no generalized decomposition.
    """

    def plan(
        self,
        objective: str,
        constraints: list[str] | None = None,
        *,
        plan_id: str | None = None,
    ) -> ActionGraph:
        """Backward-compatible entry point — delegates to :meth:`plan_iter1`."""
        return self.plan_iter1(objective, constraints, plan_id=plan_id)

    # ------------------------------------------------------------------
    # Iteration 1: generate → execute → critic
    # ------------------------------------------------------------------

    def plan_iter1(
        self,
        objective: str,
        constraints: list[str] | None = None,
        *,
        plan_id: str | None = None,
    ) -> ActionGraph:
        """Return a fixed 3-node DAG for the first iteration.

        Nodes: ``generate`` (model) → ``execute`` (code) → ``critic`` (model).
        The critic also receives inputs from ``generate`` so it can see
        both the code and the execution result.
        """
        pid = plan_id or f"plan-iter1-{uuid.uuid4().hex[:12]}"

        generate_node = GraphNode(
            node_id="generate",
            node_type="model",
            label="Generate Python code from objective",
            parameters={"system_prompt": _CODE_GEN_SYSTEM_PROMPT},
            priority=0,
        )

        execute_node = GraphNode(
            node_id="execute",
            node_type="code",
            label="Execute generated Python code",
            parameters={},
            priority=1,
        )

        critic_node = GraphNode(
            node_id="critic",
            node_type="model",
            label="LLM critic — evaluate execution result",
            parameters={
                "system_prompt": CRITIC_SYSTEM_PROMPT,
                "json_schema": _CRITIC_JSON_SCHEMA,
            },
            priority=2,
        )

        edges = [
            GraphEdge(source="generate", target="execute",
                      edge_type="data", label="generated_code"),
            GraphEdge(source="generate", target="critic",
                      edge_type="data", label="code_for_review"),
            GraphEdge(source="execute", target="critic",
                      edge_type="data", label="exec_result"),
        ]

        return ActionGraph(
            metadata=GraphMetadata(
                plan_id=pid,
                description=f"Iter1 plan for: {objective}",
            ),
            nodes=[generate_node, execute_node, critic_node],
            edges=edges,
        )

    # ------------------------------------------------------------------
    # Iteration 2: fix → execute_fix → critic_fix
    # ------------------------------------------------------------------

    def plan_iter2_fix(
        self,
        objective: str,
        constraints: list[str] | None = None,
        fix_context: dict[str, Any] | None = None,
        *,
        plan_id: str | None = None,
    ) -> ActionGraph:
        """Return a fixed 3-node fix-pipeline DAG for the second iteration.

        Nodes: ``fix`` (model) → ``execute_fix`` (code) → ``critic_fix`` (model).
        ``fix_context`` is passed via ``initial_inputs`` by the orchestrator
        so the fix node sees previous code, errors and gaps.
        """
        pid = plan_id or f"plan-iter2-fix-{uuid.uuid4().hex[:12]}"

        fix_node = GraphNode(
            node_id="fix",
            node_type="model",
            label="Fix code based on previous error",
            parameters={"system_prompt": _CODE_FIX_SYSTEM_PROMPT},
            priority=0,
        )

        execute_fix_node = GraphNode(
            node_id="execute_fix",
            node_type="code",
            label="Execute fixed Python code",
            parameters={},
            priority=1,
        )

        critic_fix_node = GraphNode(
            node_id="critic_fix",
            node_type="model",
            label="LLM critic — evaluate fixed execution result",
            parameters={
                "system_prompt": CRITIC_SYSTEM_PROMPT,
                "json_schema": _CRITIC_JSON_SCHEMA,
            },
            priority=2,
        )

        edges = [
            GraphEdge(source="fix", target="execute_fix",
                      edge_type="data", label="fixed_code"),
            GraphEdge(source="fix", target="critic_fix",
                      edge_type="data", label="code_for_review"),
            GraphEdge(source="execute_fix", target="critic_fix",
                      edge_type="data", label="exec_result"),
        ]

        return ActionGraph(
            metadata=GraphMetadata(
                plan_id=pid,
                description=f"Iter2 fix-pipeline for: {objective}",
            ),
            nodes=[fix_node, execute_fix_node, critic_fix_node],
            edges=edges,
        )
