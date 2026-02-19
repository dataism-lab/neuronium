"""Executor interrupt_check and ExecutionOutcome tests (PAUSE_CONTROL Phase 1)."""

from __future__ import annotations

from neuronium_agent.execution import DAGExecutor, ExecutionOutcome
from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphEdge, GraphMetadata, GraphNode
from neuronium_agent.types import InterruptRequest


class SimpleNode(BaseNode):
    """Node that returns its node_id in outputs."""

    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(
            outputs={"value": self.node_id},
            status="COMPLETED",
        )


def test_execute_without_interrupt_check_returns_dict() -> None:
    """Without interrupt_check, execute() returns dict[str, NodeOutput] as before."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description=""),
        nodes=[
            GraphNode(node_id="a", node_type="model", label="A", priority=0),
            GraphNode(node_id="b", node_type="model", label="B", priority=0),
        ],
        edges=[GraphEdge(source="a", target="b")],
    )
    registry = {
        "a": SimpleNode(node_id="a"),
        "b": SimpleNode(node_id="b"),
    }
    executor = DAGExecutor(
        registry,
        execution_id="e1",
        trace_id="t1",
        random_seed=0,
        max_parallel=1,
    )
    result = executor.execute(graph, initial_inputs={})
    assert isinstance(result, dict)
    assert not isinstance(result, ExecutionOutcome)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"].outputs["value"] == "a"
    assert result["b"].outputs["value"] == "b"


def test_execute_with_interrupt_returns_outcome_with_partial_results() -> None:
    """With interrupt_check returning pause after first batch, returns ExecutionOutcome."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description=""),
        nodes=[
            GraphNode(node_id="n1", node_type="model", label="N1", priority=0),
            GraphNode(node_id="n2", node_type="model", label="N2", priority=0),
        ],
        edges=[GraphEdge(source="n1", target="n2")],
    )
    registry = {
        "n1": SimpleNode(node_id="n1"),
        "n2": SimpleNode(node_id="n2"),
    }
    call_count = 0

    def interrupt_after_first_batch() -> InterruptRequest | None:
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            return InterruptRequest(command="pause")
        return None

    executor = DAGExecutor(
        registry,
        execution_id="e1",
        trace_id="t1",
        random_seed=0,
        max_parallel=1,
        interrupt_check=interrupt_after_first_batch,
    )
    result = executor.execute(graph, initial_inputs={})
    assert isinstance(result, ExecutionOutcome)
    assert result.results.keys() == {"n1"}
    assert result.results["n1"].outputs["value"] == "n1"
    assert result.pending == ["n2"]
    assert result.interrupted is not None
    assert result.interrupted.command == "pause"


def test_execute_with_interrupt_check_no_trigger_returns_outcome() -> None:
    """With interrupt_check that never returns request, returns ExecutionOutcome with empty pending."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description=""),
        nodes=[GraphNode(node_id="only", node_type="model", label="O", priority=0)],
        edges=[],
    )
    registry = {"only": SimpleNode(node_id="only")}
    executor = DAGExecutor(
        registry,
        execution_id="e1",
        trace_id="t1",
        random_seed=0,
        interrupt_check=lambda: None,
    )
    result = executor.execute(graph, initial_inputs={})
    assert isinstance(result, ExecutionOutcome)
    assert result.results.keys() == {"only"}
    assert result.pending == []
    assert result.interrupted is None


def test_execute_with_initial_results_runs_only_pending_nodes() -> None:
    """With initial_results (resume at exact pause point), only pending nodes are executed."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description=""),
        nodes=[
            GraphNode(node_id="a", node_type="model", label="A", priority=0),
            GraphNode(node_id="b", node_type="model", label="B", priority=0),
        ],
        edges=[GraphEdge(source="a", target="b")],
    )
    registry = {
        "a": SimpleNode(node_id="a"),
        "b": SimpleNode(node_id="b"),
    }
    # Pre-fill result for "a" as if we resumed from checkpoint
    initial_results = {
        "a": NodeOutput(outputs={"value": "a"}, status="COMPLETED"),
    }
    executor = DAGExecutor(
        registry,
        execution_id="e1",
        trace_id="t1",
        random_seed=0,
        max_parallel=2,
    )
    result = executor.execute(graph, initial_inputs={}, initial_results=initial_results)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"].outputs["value"] == "a"
    assert result["b"].outputs["value"] == "b"
