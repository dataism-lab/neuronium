"""Tests for DAGExecutor conditional branches (B3).

Runtime branch selection: when a decision node completes, only nodes
in the selected branch are executed; nodes in other branches are skipped.
"""

from __future__ import annotations

from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import BaseNode, NodeContext, NodeInput, NodeOutput
from neuronium_agent.nodes.decision_node import BRANCH_OUTPUT_KEY, DecisionNode
from neuronium_agent.planning.dag import (
    ActionGraph,
    ConditionalBranch,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)


class StubSourceNode(BaseNode):
    """Returns a single key 'flag' (bool) for decision condition."""

    def __init__(self, node_id: str, flag: bool) -> None:
        super().__init__(node_id)
        self._flag = flag

    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(
            outputs={"flag": self._flag},
            status="COMPLETED",
        )


class StubLeafNode(BaseNode):
    """Leaf node that returns a fixed label (to assert which branch ran)."""

    def __init__(self, node_id: str, label: str) -> None:
        super().__init__(node_id)
        self._label = label

    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(
            outputs={"branch_label": self._label},
            status="COMPLETED",
        )


def test_conditional_branch_true_selected_only_node_a_runs() -> None:
    """When decision returns 'true', only nodes in branch 'true' execute."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description="conditional"),
        nodes=[
            GraphNode(node_id="source", node_type="model", priority=0),
            GraphNode(node_id="decision", node_type="decision", priority=0),
            GraphNode(node_id="node_a", node_type="code", priority=0),
            GraphNode(node_id="node_b", node_type="code", priority=0),
        ],
        edges=[
            GraphEdge(source="source", target="decision"),
            GraphEdge(source="decision", target="node_a"),
            GraphEdge(source="decision", target="node_b"),
        ],
        conditional_branches=[
            ConditionalBranch(
                decision_node_id="decision",
                branch_label="true",
                target_node_ids=["node_a"],
            ),
            ConditionalBranch(
                decision_node_id="decision",
                branch_label="false",
                target_node_ids=["node_b"],
            ),
        ],
    )
    registry = {
        "source": StubSourceNode("source", flag=True),
        "decision": DecisionNode(node_id="decision", parameters={"condition": "flag"}),
        "node_a": StubLeafNode("node_a", "A"),
        "node_b": StubLeafNode("node_b", "B"),
    }

    events: list[tuple[str, dict]] = []

    def trace_cb(kind: str, payload: dict) -> None:
        events.append((kind, payload))

    executor = DAGExecutor(
        registry,
        execution_id="e1",
        trace_id="t1",
        random_seed=0,
        trace_callback=trace_cb,
    )
    results = executor.execute(graph, initial_inputs={})

    assert "source" in results and results["source"].status == "COMPLETED"
    assert "decision" in results and results["decision"].status == "COMPLETED"
    assert results["decision"].outputs.get(BRANCH_OUTPUT_KEY) == "true"
    assert "node_a" in results and results["node_a"].status == "COMPLETED"
    assert results["node_a"].outputs.get("branch_label") == "A"
    assert "node_b" not in results

    branch_events = [e for e in events if e[0] == "decision_branch_selected"]
    assert len(branch_events) == 1
    assert branch_events[0][1]["node_id"] == "decision"
    assert branch_events[0][1]["branch_value"] == "true"


def test_conditional_branch_false_selected_only_node_b_runs() -> None:
    """When decision returns 'false', only nodes in branch 'false' execute."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p2", description="conditional"),
        nodes=[
            GraphNode(node_id="source", node_type="model", priority=0),
            GraphNode(node_id="decision", node_type="decision", priority=0),
            GraphNode(node_id="node_a", node_type="code", priority=0),
            GraphNode(node_id="node_b", node_type="code", priority=0),
        ],
        edges=[
            GraphEdge(source="source", target="decision"),
            GraphEdge(source="decision", target="node_a"),
            GraphEdge(source="decision", target="node_b"),
        ],
        conditional_branches=[
            ConditionalBranch(
                decision_node_id="decision",
                branch_label="true",
                target_node_ids=["node_a"],
            ),
            ConditionalBranch(
                decision_node_id="decision",
                branch_label="false",
                target_node_ids=["node_b"],
            ),
        ],
    )
    registry = {
        "source": StubSourceNode("source", flag=False),
        "decision": DecisionNode(node_id="decision", parameters={"condition": "flag"}),
        "node_a": StubLeafNode("node_a", "A"),
        "node_b": StubLeafNode("node_b", "B"),
    }

    executor = DAGExecutor(registry, execution_id="e2", trace_id="t2", random_seed=0)
    results = executor.execute(graph, initial_inputs={})

    assert "source" in results and results["decision"].outputs.get(BRANCH_OUTPUT_KEY) == "false"
    assert "node_a" not in results
    assert "node_b" in results and results["node_b"].outputs.get("branch_label") == "B"


def test_no_conditional_branches_all_nodes_run() -> None:
    """Graph without conditional_branches: all nodes execute (backward compatible)."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p3", description="no branches"),
        nodes=[
            GraphNode(node_id="source", node_type="model", priority=0),
            GraphNode(node_id="decision", node_type="decision", priority=0),
            GraphNode(node_id="node_a", node_type="code", priority=0),
        ],
        edges=[
            GraphEdge(source="source", target="decision"),
            GraphEdge(source="decision", target="node_a"),
        ],
        conditional_branches=[],
    )
    registry = {
        "source": StubSourceNode("source", flag=True),
        "decision": DecisionNode(node_id="decision", parameters={"condition": "flag"}),
        "node_a": StubLeafNode("node_a", "A"),
    }

    executor = DAGExecutor(registry, execution_id="e3", trace_id="t3", random_seed=0)
    results = executor.execute(graph, initial_inputs={})

    assert len(results) == 3
    assert "node_a" in results and results["node_a"].status == "COMPLETED"


def test_single_conditional_branch_unselected_skips_target() -> None:
    """Only one branch defined: when decision selects other value, that branch's nodes are skipped."""
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p4", description="single branch"),
        nodes=[
            GraphNode(node_id="source", node_type="model", priority=0),
            GraphNode(node_id="decision", node_type="decision", priority=0),
            GraphNode(node_id="node_a", node_type="code", priority=0),
            GraphNode(node_id="node_b", node_type="code", priority=0),
        ],
        edges=[
            GraphEdge(source="source", target="decision"),
            GraphEdge(source="decision", target="node_a"),
            GraphEdge(source="decision", target="node_b"),
        ],
        conditional_branches=[
            ConditionalBranch(
                decision_node_id="decision",
                branch_label="true",
                target_node_ids=["node_a"],
            ),
        ],
    )
    registry = {
        "source": StubSourceNode("source", flag=False),
        "decision": DecisionNode(node_id="decision", parameters={"condition": "flag"}),
        "node_a": StubLeafNode("node_a", "A"),
        "node_b": StubLeafNode("node_b", "B"),
    }
    executor = DAGExecutor(registry, execution_id="e4", trace_id="t4", random_seed=0)
    results = executor.execute(graph, initial_inputs={})

    assert results["decision"].outputs.get(BRANCH_OUTPUT_KEY) == "false"
    assert "node_a" not in results
    assert "node_b" in results


def test_decision_node_output_contract() -> None:
    """DecisionNode output contains BRANCH_OUTPUT_KEY and value matches branch_label."""
    ctx = NodeContext(execution_id="e", trace_id="t", retry_count=0, random_seed=0)
    node = DecisionNode(node_id="d", parameters={"condition": "x"})
    out_true = node.execute(NodeInput(inputs={"x": True}, parameters={}, context=ctx))
    out_false = node.execute(NodeInput(inputs={"x": False}, parameters={}, context=ctx))

    assert BRANCH_OUTPUT_KEY in out_true.outputs
    assert out_true.outputs[BRANCH_OUTPUT_KEY] == "true"
    assert out_false.outputs[BRANCH_OUTPUT_KEY] == "false"
    assert "condition" in out_true.outputs
