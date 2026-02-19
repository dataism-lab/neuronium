"""AggregateNode — merge outputs from multiple upstream nodes (IBS §5.2).

v1: collects all upstream outputs into a single dict.
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput


class AggregateNode(BaseNode):
    """Merge upstream node outputs into a unified result."""

    node_type: str = "aggregate"

    def execute(self, node_input: NodeInput) -> NodeOutput:
        # All upstream outputs are passed via ``inputs``
        merged: dict[str, Any] = {}
        for key, value in node_input.inputs.items():
            merged[key] = value

        return NodeOutput(outputs=merged, status="COMPLETED")
