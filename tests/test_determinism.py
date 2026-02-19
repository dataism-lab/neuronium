"""Determinism tests — same inputs must produce the same trace (IBS §3).

These tests use replay-recorded responses to ensure full determinism
without external service calls.

B11: determinism contract (get_determinism_contract), strict mode, determinism_audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuronium_agent._canonical import canonical_json
from neuronium_agent.config import AppConfig, DeterminismConfig, ProjectConfig, StorageConfig
from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import NodeContext, NodeInput, NodeOutput
from neuronium_agent.nodes.model_node import ModelNode
from neuronium_agent.nodes.code_node import CodeNode
from neuronium_agent.nodes.mcp_node import McpToolNode
from neuronium_agent.nodes.determinism import DeterminismContract
from neuronium_agent.planning.dag import (
    ActionGraph,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)
from neuronium_agent.errors import ConfigError
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


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


class TestDeterminismContractB11:
    """B11: get_determinism_contract() and declared_non_deterministic."""

    def test_model_node_uses_seed(self) -> None:
        node = ModelNode(node_id="m1")
        c = node.get_determinism_contract()
        assert c.uses_seed is True
        assert c.declared_non_deterministic is False

    def test_code_node_uses_seed(self) -> None:
        node = CodeNode(node_id="c1")
        c = node.get_determinism_contract()
        assert c.uses_seed is True
        assert c.declared_non_deterministic is False

    def test_mcp_node_default_deterministic(self) -> None:
        node = McpToolNode(node_id="mcp1")
        c = node.get_determinism_contract()
        assert c.uses_seed is False
        assert c.declared_non_deterministic is False

    def test_mcp_node_declared_non_deterministic(self) -> None:
        node = McpToolNode(node_id="mcp2", deterministic=False)
        c = node.get_determinism_contract()
        assert c.uses_seed is False
        assert c.declared_non_deterministic is True


class TestDeterminismStrictModeB11:
    """B11: strict mode rejects declared non-deterministic nodes unless in allowlist."""

    def test_strict_rejects_mcp_non_deterministic(
        self, tmp_path: Path
    ) -> None:
        from neuronium_agent.core.orchestrator import Orchestrator
        from neuronium_agent.config import DeterminismConfig

        config = AppConfig(
            project=ProjectConfig(name="t", data_dir=str(tmp_path / ".n")),
            storage=StorageConfig(
                fs_cas_root=str(tmp_path / "blobs"),
                sqlite_path=str(tmp_path / "idx.sqlite3"),
            ),
            determinism=DeterminismConfig(strict=True),
        )
        blob = FsCasStore(config.storage.fs_cas_root)
        idx = SqliteIndexStore(config.storage.sqlite_path)
        orch = Orchestrator(config, blob, idx)

        graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id="p",
                description="strict test",
                created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            ),
            nodes=[
                GraphNode(
                    node_id="mcp_nd",
                    node_type="mcp",
                    label="tool",
                    priority=0,
                    parameters={"deterministic": False},
                ),
            ],
            edges=[],
        )
        with pytest.raises(ConfigError, match="declared non-deterministic"):
            orch._build_node_registry(graph)

    def test_strict_allows_mcp_non_deterministic_when_in_allowlist(
        self, tmp_path: Path
    ) -> None:
        from neuronium_agent.core.orchestrator import Orchestrator

        config = AppConfig(
            project=ProjectConfig(name="t", data_dir=str(tmp_path / ".n")),
            storage=StorageConfig(
                fs_cas_root=str(tmp_path / "blobs"),
                sqlite_path=str(tmp_path / "idx.sqlite3"),
            ),
            determinism=DeterminismConfig(
                strict=True,
                mcp_allow_non_deterministic_tool_ids=["mcp_nd"],
            ),
        )
        blob = FsCasStore(config.storage.fs_cas_root)
        idx = SqliteIndexStore(config.storage.sqlite_path)
        orch = Orchestrator(config, blob, idx)

        graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id="p",
                description="allowlist test",
                created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            ),
            nodes=[
                GraphNode(
                    node_id="mcp_nd",
                    node_type="mcp",
                    label="tool",
                    priority=0,
                    parameters={"deterministic": False},
                ),
            ],
            edges=[],
        )
        registry = orch._build_node_registry(graph)
        assert "mcp_nd" in registry
        assert registry["mcp_nd"].get_determinism_contract().declared_non_deterministic is True
