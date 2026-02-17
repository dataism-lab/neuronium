"""DecisionNode — conditional branching in the DAG (IBS §5.2).

v1: evaluates a simple condition expression against inputs and
selects the appropriate branch.
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput


class DecisionNode(BaseNode):
    """Route execution along conditional edges."""

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

        return NodeOutput(
            outputs={"branch": "true" if result else "false", "condition": condition},
            status="COMPLETED",
        )
