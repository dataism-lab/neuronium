"""DecisionNode — conditional branching in the DAG (IBS §5.2, spec §4.2.3).

v1: evaluates a simple condition expression against inputs and
selects the appropriate branch. The output value under the branch key
must match ``ConditionalBranch.branch_label`` in the graph so the
executor can prune unselected branches (B3).
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput

# Contract: output key for branch selection; value must match ConditionalBranch.branch_label
BRANCH_OUTPUT_KEY = "branch"


class DecisionNode(BaseNode):
    """Route execution along conditional edges.

    Output contract (B3): ``outputs[BRANCH_OUTPUT_KEY]`` is the selected branch
    value (e.g. ``"true"``, ``"false"``). It must match one of
    ``ConditionalBranch.branch_label`` in the graph so the executor
    can exclude nodes in unselected branches.
    """

    node_type: str = "decision"

    def execute(self, node_input: NodeInput) -> NodeOutput:
        condition = node_input.parameters.get(
            "condition",
            self.parameters.get("condition", "true"),
        )
        # v1: simple Python-safe eval of condition against inputs
        try:
            result = bool(eval(condition, {"__builtins__": {}}, node_input.inputs))
        except Exception:
            result = True

        branch_value = "true" if result else "false"
        return NodeOutput(
            outputs={
                BRANCH_OUTPUT_KEY: branch_value,
                "condition": condition,
            },
            status="COMPLETED",
        )
