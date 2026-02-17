"""Tests for initial inputs propagation in DAGExecutor."""

from __future__ import annotations

from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode


class EchoPromptNode(BaseNode):
    """Test node that echoes the resolved prompt input."""

    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(
            outputs={"prompt": node_input.inputs.get("prompt")},
            status="COMPLETED",
        )


def test_initial_objective_reaches_root_model_prompt() -> None:
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description="root model"),
        nodes=[
            GraphNode(node_id="generate", node_type="model", label="gen", priority=0),
        ],
        edges=[],
    )
    registry = {"generate": EchoPromptNode(node_id="generate")}
    executor = DAGExecutor(registry, execution_id="e1", trace_id="t1", random_seed=0)

    results = executor.execute(graph, initial_inputs={"objective": "X"})
    assert results["generate"].outputs["prompt"] == "X"
