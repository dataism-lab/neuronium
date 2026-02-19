"""Tests for rollback scope computation (B1 Part 2 §3.4.1)."""

from __future__ import annotations

from neuronium_agent.planning.dag import (
    ActionGraph,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)
from neuronium_agent.recovery.scope import compute_rollback_scope


def _graph_chain() -> ActionGraph:
    """Linear chain: a -> b -> c."""
    return ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description="Chain"),
        nodes=[
            GraphNode(node_id="a", node_type="model", priority=0),
            GraphNode(node_id="b", node_type="code", priority=0),
            GraphNode(node_id="c", node_type="model", priority=0),
        ],
        edges=[
            GraphEdge(source="a", target="b"),
            GraphEdge(source="b", target="c"),
        ],
    )


def _graph_diamond() -> ActionGraph:
    """Diamond: a -> b, a -> c, b -> d, c -> d."""
    return ActionGraph(
        metadata=GraphMetadata(plan_id="p2", description="Diamond"),
        nodes=[
            GraphNode(node_id="a", node_type="model", priority=0),
            GraphNode(node_id="b", node_type="code", priority=0),
            GraphNode(node_id="c", node_type="code", priority=0),
            GraphNode(node_id="d", node_type="model", priority=0),
        ],
        edges=[
            GraphEdge(source="a", target="b"),
            GraphEdge(source="a", target="c"),
            GraphEdge(source="b", target="d"),
            GraphEdge(source="c", target="d"),
        ],
    )


class TestComputeRollbackScope:
    """compute_rollback_scope returns correct node_ids and preservation per failure type."""

    def test_node_execution_chain_failed_mid(self) -> None:
        graph = _graph_chain()
        scope = compute_rollback_scope(
            "node_execution",
            graph,
            failed_node_ids={"b"},
            completed_node_ids={"a", "b", "c"},
        )
        assert scope.scope_type.value == "NODE"
        assert scope.node_ids == {"b", "c"}
        assert scope.preservation_node_ids == {"a"}

    def test_node_execution_chain_failed_source(self) -> None:
        graph = _graph_chain()
        scope = compute_rollback_scope(
            "node_execution",
            graph,
            failed_node_ids={"a"},
            completed_node_ids={"a"},
        )
        assert scope.node_ids == {"a", "b", "c"}
        assert scope.preservation_node_ids == set()

    def test_critic_rejection_with_gate_nodes(self) -> None:
        graph = _graph_diamond()
        scope = compute_rollback_scope(
            "critic_rejection",
            graph,
            failed_node_ids={"d"},
            critic_failed=True,
            completed_node_ids={"a", "b", "c", "d"},
            gate_required_node_ids={"d"},
        )
        assert scope.scope_type.value == "SUBGRAPH"
        assert scope.node_ids == {"d"}
        assert scope.preservation_node_ids == {"a", "b", "c"}

    def test_plan_invalidation_full_graph(self) -> None:
        graph = _graph_chain()
        scope = compute_rollback_scope(
            "plan_invalidation",
            graph,
            failed_node_ids=set(),
        )
        assert scope.scope_type.value == "INTENTION"
        assert scope.node_ids == {"a", "b", "c"}
        assert scope.preservation_node_ids == set()

    def test_constraint_violation_full_graph(self) -> None:
        graph = _graph_chain()
        scope = compute_rollback_scope(
            "constraint_violation",
            graph,
            failed_node_ids=set(),
        )
        assert scope.scope_type.value == "INTENTION"
        assert scope.node_ids == {"a", "b", "c"}
