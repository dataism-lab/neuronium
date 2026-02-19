"""Tests for DAGExecutor node-level retry (B1 Part 1)."""

from __future__ import annotations

from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode


class FlakyTimeoutNode(BaseNode):
    """Returns FAILED with timeout N times, then COMPLETED (for retry tests)."""

    def __init__(self, node_id: str, responses: list[NodeOutput]) -> None:
        super().__init__(node_id)
        self._responses = list(responses)

    def execute(self, node_input: NodeInput) -> NodeOutput:
        if self._responses:
            return self._responses.pop(0)
        return NodeOutput(status="COMPLETED", outputs={"content": "ok"})


def test_node_retry_transient_then_success() -> None:
    """Node fails twice with timeout (TRANSIENT), then succeeds; retries occur and result is COMPLETED."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description="retry test"),
        nodes=[GraphNode(node_id="task", node_type="model", label="task", priority=0)],
        edges=[],
    )
    responses = [
        NodeOutput(status="FAILED", error="Request timed out after 60s"),
        NodeOutput(status="FAILED", error="Request timed out after 60s"),
        NodeOutput(status="COMPLETED", outputs={"content": "done"}),
    ]
    registry = {"task": FlakyTimeoutNode("task", responses)}

    events: list[tuple[str, dict]] = []

    def trace_cb(kind: str, payload: dict) -> None:
        events.append((kind, payload))

    executor = DAGExecutor(
        registry,
        execution_id="e1",
        trace_id="t1",
        random_seed=0,
        max_node_retries=3,
        retry_backoff_base_seconds=0.01,
        trace_callback=trace_cb,
    )

    results = executor.execute(graph, initial_inputs={})

    assert results["task"].status == "COMPLETED"
    assert results["task"].outputs.get("content") == "done"

    node_retries = [e for e in events if e[0] == "node_retry"]
    assert len(node_retries) >= 2, "Expected at least 2 node_retry events (after 1st and 2nd failure)"
    assert all(e[1]["node_id"] == "task" for e in node_retries)
