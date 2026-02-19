"""Integration tests for B1 Part 1: recovery (RETRY_STAGE / ESCALATE) in runbook path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, RecoveryConfig, StorageConfig
from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)
from neuronium_agent.planning.runbook_registry import register_runbook
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.types import RunRequest


class FailingThenSucceedingNode(BaseNode):
    """Node that returns FAILED on first execute(), COMPLETED on second.

    Uses a shared mutable list so that when the stage is retried (new registry),
    the same list is used and the second call returns COMPLETED.
    """

    def __init__(self, node_id: str, responses: list[NodeOutput]) -> None:
        super().__init__(node_id)
        self._responses = responses  # shared list, mutated by pop(0)

    def execute(self, node_input: NodeInput) -> NodeOutput:
        if self._responses:
            return self._responses.pop(0)
        return NodeOutput(status="COMPLETED", outputs={"content": "ok"})


class AlwaysFailingNode(BaseNode):
    """Node that always returns FAILED with PERSISTENT error."""

    def __init__(self, node_id: str, error: str = "Invalid parameter") -> None:
        super().__init__(node_id)
        self._error = error

    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(status="FAILED", error=self._error)


class RecoveryTestRunbook(Runbook):
    """Single-stage runbook with one required node for recovery tests."""

    def __init__(self, runbook_id: str = "recovery_test_v1") -> None:
        self._runbook_id = runbook_id

    @property
    def runbook_id(self) -> str:
        return self._runbook_id

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        plan_id = f"plan-recovery-test-{execution_id[:8]}"
        graph = ActionGraph(
            metadata=GraphMetadata(plan_id=plan_id, description="Recovery test"),
            nodes=[
                GraphNode(node_id="task", node_type="model", label="Task", priority=0),
            ],
            edges=[],
        )
        return [
            ActionGraphStage(
                stage_id="recovery_test:stage1",
                graph=graph,
                initial_inputs_override={},
                success_gate=StageSuccessGate(
                    required_completed_nodes=["task"],
                    critic_node_id=None,
                ),
            ),
        ]


def _make_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_config: RecoveryConfig | None = None,
) -> AgentRunner:
    monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-fake-key")
    config = AppConfig(
        project=ProjectConfig(name="test", data_dir=str(tmp_path / ".n")),
        storage=StorageConfig(
            fs_cas_root=str(tmp_path / "blobs"),
            sqlite_path=str(tmp_path / "index.sqlite3"),
        ),
        recovery=recovery_config or RecoveryConfig(
            max_node_retries=2,
            max_stage_retries=2,
        ),
    )
    blob = FsCasStore(config.storage.fs_cas_root)
    idx = SqliteIndexStore(config.storage.sqlite_path)
    return AgentRunner(config, blob, idx)


class TestRecoveryRetryStage:
    """Gate fail with TRANSIENT → RETRY_STAGE → re-execute → stage passes."""

    def test_recovery_decision_retry_stage_then_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        register_runbook(RecoveryTestRunbook("recovery_test_retry_v1"))
        # max_node_retries=1 so we get at most 2 attempts; both stay TRANSIENT
        # (retry_count 0,1 < threshold 2), then we return FAILED with TRANSIENT.
        # Stage sees TRANSIENT → RETRY_STAGE. Second run: node returns COMPLETED.
        runner = _make_runner(
            tmp_path, monkeypatch,
            recovery_config=RecoveryConfig(max_node_retries=1, max_stage_retries=2),
        )
        orig_build = runner._orchestrator._build_node_registry
        shared_responses = [
            NodeOutput(status="FAILED", error="Request timed out after 60s"),
            NodeOutput(status="FAILED", error="Request timed out after 60s"),
            NodeOutput(status="COMPLETED", outputs={"content": "# Done"}),
        ]

        def patched_build(
            graph: ActionGraph, *, stage_default_model_id=None, **kwargs
        ):
            registry = orig_build(
                graph, stage_default_model_id=stage_default_model_id, **kwargs
            )
            if "task" in registry:
                registry["task"] = FailingThenSucceedingNode(
                    "task", shared_responses
                )
            return registry

        runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]

        handle = runner.start(
            RunRequest(
                objective="Recovery test",
                metadata={"runbook_id": "recovery_test_retry_v1"},
            )
        )

        events = list(runner.get_trace_events(handle.trace_id))
        recovery_events = [e for e in events if e.get("kind") == "recovery_decision"]
        assert len(recovery_events) >= 1
        assert recovery_events[0]["payload"]["action"] == "RETRY_STAGE"

        stage_ends = [e for e in events if e.get("kind") == "stage_end"]
        assert len(stage_ends) >= 1
        assert stage_ends[-1]["payload"]["success"] is True

        status = runner.get_status(handle)
        assert status.state == "COMPLETED"


class TestRecoveryEscalation:
    """Gate fail with PERSISTENT → ESCALATE → RunState.PAUSED."""

    def test_recovery_escalate_paused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        register_runbook(RecoveryTestRunbook("recovery_test_escalate_v1"))
        runner = _make_runner(tmp_path, monkeypatch)
        orig_build = runner._orchestrator._build_node_registry

        def patched_build(
            graph: ActionGraph, *, stage_default_model_id=None, **kwargs
        ):
            registry = orig_build(
                graph, stage_default_model_id=stage_default_model_id, **kwargs
            )
            if "task" in registry:
                registry["task"] = AlwaysFailingNode(
                    "task", error="Invalid parameter: foo"
                )
            return registry

        runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]

        handle = runner.start(
            RunRequest(
                objective="Recovery escalate test",
                metadata={"runbook_id": "recovery_test_escalate_v1"},
            )
        )

        status = runner.get_status(handle)
        assert status.state == "PAUSED"

        events = list(runner.get_trace_events(handle.trace_id))
        recovery_events = [e for e in events if e.get("kind") == "recovery_decision"]
        assert len(recovery_events) >= 1
        payload = recovery_events[0]["payload"]
        assert payload["action"] == "ESCALATE"
        # B1 Part 2: recovery_decision includes rollback scope for audit
        assert "rollback_scope_type" in payload
        assert "rollback_node_ids" in payload
        assert "task" in payload["rollback_node_ids"]
