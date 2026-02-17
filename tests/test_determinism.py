"""Determinism tests — same inputs must produce the same trace (IBS §3).

These tests use replay-recorded responses to ensure full determinism
without external service calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuronium_agent._canonical import canonical_json
from neuronium_agent.config import AppConfig, StorageConfig, ProjectConfig
from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import NodeContext, NodeInput, NodeOutput
from neuronium_agent.nodes.model_node import ModelNode
from neuronium_agent.nodes.code_node import CodeNode
from neuronium_agent.planning.dag import (
    ActionGraph,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)


def _make_graph() -> ActionGraph:
    """Create a deterministic test graph: ModelNode → CodeNode."""
    from datetime import datetime, timezone

    fixed_ts = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return ActionGraph(
        metadata=GraphMetadata(
            plan_id="test-plan",
            description="determinism test",
            created_at=fixed_ts,
        ),
        nodes=[
            GraphNode(node_id="gen", node_type="model", label="generate", priority=0),
            GraphNode(node_id="run", node_type="code", label="run code", priority=1),
        ],
        edges=[
            GraphEdge(source="gen", target="run", edge_type="data"),
        ],
    )


def _make_registry_with_replay() -> dict[str, ModelNode | CodeNode]:
    """Create nodes with pre-recorded responses."""
    model = ModelNode(node_id="gen")
    model.set_replay_responses([
        {
            "outputs": {"content": "print('hello world')"},
            "quality_signals": {"tokens_used": 10},
        }
    ])

    code = CodeNode(node_id="run")
    code.set_replay_responses([
        {
            "outputs": {"stdout": "hello world\n", "exit_code": 0},
            "quality_signals": {"latency_ms": 50.0},
            "status": "COMPLETED",
        }
    ])

    return {"gen": model, "run": code}


class TestDeterministicExecution:
    """Same inputs + same recorded responses → identical output."""

    def test_same_outputs_on_replay(self) -> None:
        graph = _make_graph()

        # Run 1
        reg1 = _make_registry_with_replay()
        events1: list[tuple[str, dict]] = []
        exec1 = DAGExecutor(
            reg1,
            execution_id="exec-1",
            trace_id="trace-1",
            random_seed=42,
            trace_callback=lambda k, p: events1.append((k, p)),
        )
        results1 = exec1.execute(graph)

        # Run 2 (identical setup)
        reg2 = _make_registry_with_replay()
        events2: list[tuple[str, dict]] = []
        exec2 = DAGExecutor(
            reg2,
            execution_id="exec-1",
            trace_id="trace-1",
            random_seed=42,
            trace_callback=lambda k, p: events2.append((k, p)),
        )
        results2 = exec2.execute(graph)

        # Outputs must be identical
        for nid in ("gen", "run"):
            r1 = results1[nid]
            r2 = results2[nid]
            assert r1.outputs == r2.outputs, f"Outputs differ for {nid}"
            assert r1.status == r2.status, f"Status differs for {nid}"

    def test_events_match(self) -> None:
        """Trace events (kinds and outputs) must match across runs."""
        graph = _make_graph()

        events_a: list[tuple[str, dict]] = []
        events_b: list[tuple[str, dict]] = []

        for events_list in (events_a, events_b):
            reg = _make_registry_with_replay()
            ex = DAGExecutor(
                reg,
                execution_id="exec-det",
                trace_id="trace-det",
                random_seed=0,
                trace_callback=lambda k, p, el=events_list: el.append((k, p)),
            )
            ex.execute(graph)

        # Same number of events
        assert len(events_a) == len(events_b)

        # Same kinds in same order
        kinds_a = [k for k, _ in events_a]
        kinds_b = [k for k, _ in events_b]
        assert kinds_a == kinds_b

        # Same output data
        for (ka, pa), (kb, pb) in zip(events_a, events_b):
            assert pa.get("node_id") == pb.get("node_id")
            assert pa.get("status") == pb.get("status")

    def test_topological_order_deterministic(self) -> None:
        """Topological order must be the same every time."""
        graph = _make_graph()
        order1 = graph.topological_order()
        order2 = graph.topological_order()
        assert order1 == order2 == ["gen", "run"]

    def test_canonical_json_deterministic(self) -> None:
        """Canonical JSON of the same graph → same string."""
        g1 = _make_graph()
        g2 = _make_graph()
        s1 = canonical_json(g1.model_dump(mode="json"))
        s2 = canonical_json(g2.model_dump(mode="json"))
        assert s1 == s2
