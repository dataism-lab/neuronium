"""Phase B acceptance: checkpoint/resume + declarative meta-control.

Tests verify:
  1. ``build_checkpoint_payload`` / ``load_state_from_checkpoint`` round-trip.
  2. Phase-boundary checkpoints are recorded during a run.
  3. Declarative ``apply_control`` commands (pause, continue, stop, revise, escalate).
  4. ``resume_run`` restores from phase-boundary checkpoint and continues.
  5. Resume invariant: only valid phase-boundary labels are accepted.
  6. Deterministic transition coverage for control state machine.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, StorageConfig, ProjectConfig
from neuronium_agent.core.state import AgentState, Intention, IntentionPhase, RunState
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.planning.htn import HTNPlanner
from neuronium_agent.trace.checkpoints import (
    MID_EXECUTE_REQUIRED_KEYS,
    PHASE_BOUNDARIES,
    CheckpointError,
    build_checkpoint_payload,
    get_latest_phase_boundary_checkpoint,
    load_state_from_checkpoint,
)
from neuronium_agent.trace.recorder import TraceRecorder
from neuronium_agent.types import ControlCommand, RunRequest


# ---------------------------------------------------------------------------
# Replay data for deterministic runs
# ---------------------------------------------------------------------------

_ITER1_PASS: dict[str, list[dict]] = {
    "generate": [{
        "outputs": {"content": "print('hello')"},
        "quality_signals": {"tokens_used": 5},
    }],
    "execute": [{
        "outputs": {"stdout": "hello\n", "exit_code": 0},
        "quality_signals": {"latency_ms": 10.0},
        "status": "COMPLETED",
    }],
    "critic": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 1.0,
                "evidence": ["exit_code=0"],
                "gaps": [],
            }),
        },
        "quality_signals": {"tokens_used": 10},
    }],
}

_ITER1_FAIL: dict[str, list[dict]] = {
    "generate": [{
        "outputs": {"content": "print('oops')"},
        "quality_signals": {"tokens_used": 5},
    }],
    "execute": [{
        "outputs": {"stdout": "", "stderr": "SyntaxError", "exit_code": 1},
        "quality_signals": {"latency_ms": 10.0},
        "status": "FAILED",
    }],
    "critic": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "FAIL",
                "confidence": 0.8,
                "evidence": [],
                "gaps": ["execution failed"],
            }),
        },
        "quality_signals": {"tokens_used": 10},
    }],
}

_ITER2_PASS: dict[str, list[dict]] = {
    "generate_fix": [{
        "outputs": {"content": "print('fixed')"},
        "quality_signals": {"tokens_used": 5},
    }],
    "execute_fix": [{
        "outputs": {"stdout": "fixed\n", "exit_code": 0},
        "quality_signals": {"latency_ms": 10.0},
        "status": "COMPLETED",
    }],
    "critic_fix": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 1.0,
                "evidence": ["exit_code=0"],
                "gaps": [],
            }),
        },
        "quality_signals": {"tokens_used": 10},
    }],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _state() -> AgentState:
    """Minimal AgentState for unit tests."""
    return AgentState(
        trace_id="test-trace-001",
        execution_id="test-exec-001",
        run_state=RunState.RUNNING,
        intention=Intention(
            intention_id="int-001",
            objective="Test objective",
            phase=IntentionPhase.EXECUTE,
        ),
    )


def _make_runner(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_map: dict[str, list[dict]],
) -> AgentRunner:
    """Build an AgentRunner with monkey-patched replay for all nodes."""
    monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-fake-key")

    config = AppConfig(
        project=ProjectConfig(name="test", data_dir=str(tmp_dir / ".n")),
        storage=StorageConfig(
            fs_cas_root=str(tmp_dir / "blobs"),
            sqlite_path=str(tmp_dir / "index.sqlite3"),
        ),
    )
    blob = FsCasStore(config.storage.fs_cas_root)
    idx = SqliteIndexStore(config.storage.sqlite_path)
    runner = AgentRunner(config, blob, idx)

    orig_build = runner._orchestrator._build_node_registry

    def patched_build(graph, *, stage_default_model_id=None, **kwargs):
        registry = orig_build(graph, stage_default_model_id=stage_default_model_id, **kwargs)
        for nid, node in registry.items():
            if hasattr(node, "set_replay_responses") and nid in replay_map:
                node.set_replay_responses(replay_map[nid])
        return registry

    runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]
    return runner


# ===================================================================
# 1. Checkpoint payload round-trip
# ===================================================================


class TestCheckpointPayload:
    """Unit tests for ``build_checkpoint_payload`` / ``load_state_from_checkpoint``."""

    def test_roundtrip(self, _state: AgentState) -> None:
        payload = build_checkpoint_payload(
            _state, iteration=1, phase_boundary="after_execute_iter1",
        )
        restored_state, ctx = load_state_from_checkpoint(payload)

        assert restored_state.trace_id == _state.trace_id
        assert restored_state.execution_id == _state.execution_id
        assert restored_state.run_state == _state.run_state
        assert ctx["iteration"] == 1
        assert ctx["phase_boundary"] == "after_execute_iter1"

    def test_extra_context_preserved(self, _state: AgentState) -> None:
        payload = build_checkpoint_payload(
            _state,
            iteration=1,
            phase_boundary="after_adapt_iter1",
            extra={"fix_context": {"key": "value"}, "added_constraints": ["c1"]},
        )
        _, ctx = load_state_from_checkpoint(payload)

        assert ctx["fix_context"] == {"key": "value"}
        assert ctx["added_constraints"] == ["c1"]

    def test_runbook_id_and_metadata_preserved(self, _state: AgentState) -> None:
        """A2: runbook_id and metadata in extra are preserved in resume_context."""
        payload = build_checkpoint_payload(
            _state,
            iteration=1,
            phase_boundary="after_execute_iter1",
            extra={
                "runbook_id": "docs_report_v1",
                "metadata": {"doc_paths": ["/tmp/a.md"], "runbook_id": "docs_report_v1"},
            },
        )
        _, ctx = load_state_from_checkpoint(payload)
        assert ctx["runbook_id"] == "docs_report_v1"
        assert ctx["metadata"] == {"doc_paths": ["/tmp/a.md"], "runbook_id": "docs_report_v1"}

    def test_invalid_boundary_raises(self, _state: AgentState) -> None:
        payload = build_checkpoint_payload(
            _state, iteration=1, phase_boundary="invalid_boundary",
        )
        with pytest.raises(CheckpointError, match="not valid for resume"):
            load_state_from_checkpoint(payload)

    def test_missing_agent_state_raises(self) -> None:
        with pytest.raises(CheckpointError, match="agent_state"):
            load_state_from_checkpoint({"resume_context": {"phase_boundary": "final"}})

    def test_paused_mid_execute_missing_required_key_raises(self, _state: AgentState) -> None:
        """Phase 3: paused_mid_execute requires all MID_EXECUTE_REQUIRED_KEYS."""
        base_extra = {
            "stage_index": 0,
            "iteration": 1,
            "plan_id": "plan-1",
            "runbook_id": "autofix_demo",
            "completed_node_results": {},
            "pending_node_ids": [],
        }
        # iteration is always added by build_checkpoint_payload(iteration=...); test the rest
        keys_from_extra = MID_EXECUTE_REQUIRED_KEYS - {"iteration"}
        for key in keys_from_extra:
            extra = {k: v for k, v in base_extra.items() if k != key}
            payload = build_checkpoint_payload(
                _state,
                iteration=1,
                phase_boundary="paused_mid_execute",
                extra=extra,
            )
            with pytest.raises(CheckpointError, match="missing required keys"):
                load_state_from_checkpoint(payload)

    def test_paused_mid_execute_completed_node_results_not_dict_raises(
        self, _state: AgentState
    ) -> None:
        payload = build_checkpoint_payload(
            _state,
            iteration=1,
            phase_boundary="paused_mid_execute",
            extra={
                "stage_index": 0,
                "iteration": 1,
                "plan_id": "p",
                "runbook_id": "r",
                "completed_node_results": "not-a-dict",
                "pending_node_ids": [],
            },
        )
        with pytest.raises(CheckpointError, match="completed_node_results.*must be a dict"):
            load_state_from_checkpoint(payload)

    def test_paused_mid_execute_pending_node_ids_not_list_raises(
        self, _state: AgentState
    ) -> None:
        payload = build_checkpoint_payload(
            _state,
            iteration=1,
            phase_boundary="paused_mid_execute",
            extra={
                "stage_index": 0,
                "iteration": 1,
                "plan_id": "p",
                "runbook_id": "r",
                "completed_node_results": {},
                "pending_node_ids": "not-a-list",
            },
        )
        with pytest.raises(CheckpointError, match="pending_node_ids.*must be a list"):
            load_state_from_checkpoint(payload)

    def test_paused_mid_execute_valid_restores_all_context(self, _state: AgentState) -> None:
        """Phase 3: valid paused_mid_execute checkpoint restores full resume_context."""
        extra = {
            "stage_index": 1,
            "iteration": 2,
            "plan_id": "plan-42",
            "runbook_id": "autofix_demo",
            "completed_node_results": {
                "n1": {"outputs": {"x": 1}, "status": "COMPLETED", "quality_signals": {}},
            },
            "pending_node_ids": ["n2", "n3"],
            "metadata": {"key": "value"},
            "planned_graph": {"metadata": {"plan_id": "plan-42"}, "nodes": [], "edges": []},
        }
        payload = build_checkpoint_payload(
            _state,
            iteration=2,
            phase_boundary="paused_mid_execute",
            extra=extra,
        )
        state, ctx = load_state_from_checkpoint(payload)
        assert ctx["phase_boundary"] == "paused_mid_execute"
        assert ctx["stage_index"] == 1
        assert ctx["iteration"] == 2
        assert ctx["plan_id"] == "plan-42"
        assert ctx["runbook_id"] == "autofix_demo"
        assert ctx["completed_node_results"] == extra["completed_node_results"]
        assert ctx["pending_node_ids"] == ["n2", "n3"]
        assert ctx["metadata"] == {"key": "value"}
        assert ctx["planned_graph"] == extra["planned_graph"]

    def test_all_phase_boundaries_accepted(self, _state: AgentState) -> None:
        for boundary in PHASE_BOUNDARIES:
            extra: dict | None = None
            if boundary == "paused_mid_execute":
                extra = {
                    "stage_index": 0,
                    "iteration": 1,
                    "plan_id": "plan-1",
                    "runbook_id": "autofix_demo",
                    "completed_node_results": {"n1": {"outputs": {}, "status": "COMPLETED"}},
                    "pending_node_ids": ["n2"],
                }
            payload = build_checkpoint_payload(
                _state,
                iteration=1,
                phase_boundary=boundary,
                extra=extra,
            )
            state, ctx = load_state_from_checkpoint(payload)
            assert ctx["phase_boundary"] == boundary


# ===================================================================
# 2. Phase-boundary checkpoints recorded during run
# ===================================================================


class TestPhaseBoundaryCheckpoints:
    """Verify that the orchestrator writes checkpoints at phase boundaries."""

    def test_successful_run_has_checkpoints(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        handle = runner.start(RunRequest(objective="CP test"))

        events = list(runner._index.get_trace_events(handle.trace_id))
        cp_events = [e for e in events if e["kind"] == "checkpoint"]

        assert len(cp_events) >= 1, "At least 1 checkpoint expected"

        boundaries = [
            e["payload"].get("resume_context", {}).get("phase_boundary")
            for e in cp_events
        ]
        # Final checkpoint must be present
        assert "final" in boundaries

    def test_successful_run_has_determinism_audit_event(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """B11: trace contains determinism_audit event (nodes_using_seed, seed_value)."""
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        handle = runner.start(RunRequest(objective="Determinism audit test"))

        events = list(runner._index.get_trace_events(handle.trace_id))
        audit_events = [e for e in events if e.get("kind") == "determinism_audit"]

        assert len(audit_events) >= 1, "At least one determinism_audit event expected"
        payload = audit_events[0].get("payload", {})
        assert "nodes_using_seed" in payload
        assert "seed_value" in payload
        assert payload.get("seed_present") is True

    def test_checkpoints_have_valid_boundaries(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        handle = runner.start(RunRequest(objective="Boundary test"))

        events = list(runner._index.get_trace_events(handle.trace_id))
        cp_events = [e for e in events if e["kind"] == "checkpoint"]

        for cp in cp_events:
            boundary = cp["payload"].get("resume_context", {}).get("phase_boundary")
            assert boundary in PHASE_BOUNDARIES, (
                "Unexpected boundary: " + str(boundary)
            )

    def test_checkpoints_contain_runbook_id_and_metadata(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A2: every phase-boundary checkpoint includes runbook_id and metadata."""
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        custom_metadata = {"runbook_id": "autofix_demo", "custom_key": "custom_value"}
        handle = runner.start(RunRequest(
            objective="A2 checkpoint metadata test",
            metadata=custom_metadata,
        ))

        events = list(runner._index.get_trace_events(handle.trace_id))
        cp_events = [e for e in events if e["kind"] == "checkpoint"]
        assert len(cp_events) >= 1

        for cp in cp_events:
            rc = cp["payload"].get("resume_context", {})
            assert "runbook_id" in rc, "Checkpoint must have runbook_id in resume_context"
            assert rc["runbook_id"] == "autofix_demo"
            assert "metadata" in rc, "Checkpoint must have metadata in resume_context"
            assert isinstance(rc["metadata"], dict)
            assert rc["metadata"].get("custom_key") == "custom_value"


# ===================================================================
# 3. Declarative meta-control
# ===================================================================


class TestDeclarativeControl:
    """Control commands must be purely declarative: state + checkpoint + trace."""

    @pytest.fixture()
    def running_runner(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[AgentRunner, str]:
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        handle = runner.start(RunRequest(objective="Control test"))
        return runner, handle.trace_id

    def test_pause_transitions_to_paused(
        self, running_runner: tuple[AgentRunner, str],
    ) -> None:
        runner, trace_id = running_runner
        # Run is COMPLETED — cannot pause. Start a fresh run that we can control.
        # Instead, test that pause on a RUNNING state works via the orchestrator directly.
        state = AgentState(
            trace_id="ctrl-test-pause",
            execution_id="exec-001",
            run_state=RunState.RUNNING,
            intention=Intention(
                intention_id="int-001",
                objective="test",
                phase=IntentionPhase.EXECUTE,
            ),
        )
        orch = runner._orchestrator
        orch._states["ctrl-test-pause"] = state
        recorder = TraceRecorder("ctrl-test-pause", runner._index)
        orch._recorders["ctrl-test-pause"] = recorder

        # Persist run so index_store knows about it
        runner._index.upsert_run(
            trace_id="ctrl-test-pause",
            execution_id="exec-001",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cmd = ControlCommand(type="pause", payload={})
        status = orch.apply_control("ctrl-test-pause", cmd)
        assert status.state == "PAUSED"

    def test_continue_transitions_to_running(
        self, running_runner: tuple[AgentRunner, str],
    ) -> None:
        runner, _ = running_runner
        orch = runner._orchestrator

        state = AgentState(
            trace_id="ctrl-test-cont",
            execution_id="exec-002",
            run_state=RunState.PAUSED,
            intention=Intention(
                intention_id="int-002",
                objective="test",
                phase=IntentionPhase.EXECUTE,
            ),
        )
        orch._states["ctrl-test-cont"] = state
        recorder = TraceRecorder("ctrl-test-cont", runner._index)
        orch._recorders["ctrl-test-cont"] = recorder

        runner._index.upsert_run(
            trace_id="ctrl-test-cont",
            execution_id="exec-002",
            state="PAUSED",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cmd = ControlCommand(type="continue", payload={})
        status = orch.apply_control("ctrl-test-cont", cmd)
        assert status.state == "RUNNING"

    def test_stop_transitions_to_cancelled(
        self, running_runner: tuple[AgentRunner, str],
    ) -> None:
        runner, _ = running_runner
        orch = runner._orchestrator

        state = AgentState(
            trace_id="ctrl-test-stop",
            execution_id="exec-003",
            run_state=RunState.RUNNING,
            intention=Intention(
                intention_id="int-003",
                objective="test",
                phase=IntentionPhase.EXECUTE,
            ),
        )
        orch._states["ctrl-test-stop"] = state
        recorder = TraceRecorder("ctrl-test-stop", runner._index)
        orch._recorders["ctrl-test-stop"] = recorder

        runner._index.upsert_run(
            trace_id="ctrl-test-stop",
            execution_id="exec-003",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cmd = ControlCommand(type="stop", payload={})
        status = orch.apply_control("ctrl-test-stop", cmd)
        assert status.state == "CANCELLED"

    def test_revise_adds_constraints(
        self, running_runner: tuple[AgentRunner, str],
    ) -> None:
        runner, _ = running_runner
        orch = runner._orchestrator

        state = AgentState(
            trace_id="ctrl-test-revise",
            execution_id="exec-004",
            run_state=RunState.RUNNING,
            intention=Intention(
                intention_id="int-004",
                objective="test",
                constraints=["existing"],
                phase=IntentionPhase.EXECUTE,
            ),
        )
        orch._states["ctrl-test-revise"] = state
        recorder = TraceRecorder("ctrl-test-revise", runner._index)
        orch._recorders["ctrl-test-revise"] = recorder

        runner._index.upsert_run(
            trace_id="ctrl-test-revise",
            execution_id="exec-004",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cmd = ControlCommand(
            type="revise",
            payload={"constraints_add": ["new_constraint"]},
        )
        orch.apply_control("ctrl-test-revise", cmd)

        assert "new_constraint" in state.intention.constraints  # type: ignore[union-attr]

    def test_escalate_transitions_to_paused(
        self, running_runner: tuple[AgentRunner, str],
    ) -> None:
        runner, _ = running_runner
        orch = runner._orchestrator

        state = AgentState(
            trace_id="ctrl-test-esc",
            execution_id="exec-005",
            run_state=RunState.RUNNING,
            intention=Intention(
                intention_id="int-005",
                objective="test",
                phase=IntentionPhase.CONTROL,
            ),
        )
        orch._states["ctrl-test-esc"] = state
        recorder = TraceRecorder("ctrl-test-esc", runner._index)
        orch._recorders["ctrl-test-esc"] = recorder

        runner._index.upsert_run(
            trace_id="ctrl-test-esc",
            execution_id="exec-005",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cmd = ControlCommand(type="escalate", payload={})
        status = orch.apply_control("ctrl-test-esc", cmd)
        assert status.state == "PAUSED"

    def test_control_writes_checkpoint(
        self, running_runner: tuple[AgentRunner, str],
    ) -> None:
        """Every control command must write a checkpoint trace event."""
        runner, _ = running_runner
        orch = runner._orchestrator

        state = AgentState(
            trace_id="ctrl-test-cp",
            execution_id="exec-006",
            run_state=RunState.RUNNING,
            intention=Intention(
                intention_id="int-006",
                objective="test",
                phase=IntentionPhase.EXECUTE,
            ),
        )
        orch._states["ctrl-test-cp"] = state
        recorder = TraceRecorder("ctrl-test-cp", runner._index)
        orch._recorders["ctrl-test-cp"] = recorder

        runner._index.upsert_run(
            trace_id="ctrl-test-cp",
            execution_id="exec-006",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        cmd = ControlCommand(type="pause", payload={})
        orch.apply_control("ctrl-test-cp", cmd)

        events = list(runner._index.get_trace_events("ctrl-test-cp"))
        cp_events = [e for e in events if e["kind"] == "checkpoint"]
        assert len(cp_events) >= 1


# ===================================================================
# 4. Deterministic transition coverage
# ===================================================================


class TestStateTransitions:
    """Deterministic coverage of the RunState transition table."""

    _VALID_TRANSITIONS = {
        (RunState.PENDING, RunState.RUNNING),
        (RunState.PENDING, RunState.FAILED),
        (RunState.PENDING, RunState.CANCELLED),
        (RunState.RUNNING, RunState.PAUSED),
        (RunState.RUNNING, RunState.COMPLETED),
        (RunState.RUNNING, RunState.FAILED),
        (RunState.RUNNING, RunState.CANCELLED),
        (RunState.PAUSED, RunState.RUNNING),
        (RunState.PAUSED, RunState.CANCELLED),
    }

    _TERMINAL_STATES = {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}

    def test_all_valid_transitions_succeed(self) -> None:
        for from_state, to_state in self._VALID_TRANSITIONS:
            state = AgentState(
                trace_id="tx",
                execution_id="ex",
                run_state=from_state,
            )
            state.transition_to(to_state, "test transition")
            assert state.run_state == to_state

    def test_terminal_states_reject_all_transitions(self) -> None:
        for terminal in self._TERMINAL_STATES:
            state = AgentState(
                trace_id="tx",
                execution_id="ex",
                run_state=terminal,
            )
            for target in RunState:
                if target == terminal:
                    continue
                with pytest.raises(ValueError, match="Cannot transition"):
                    state.transition_to(target)

    def test_invalid_paused_to_completed_rejected(self) -> None:
        state = AgentState(trace_id="tx", execution_id="ex", run_state=RunState.PAUSED)
        with pytest.raises(ValueError):
            state.transition_to(RunState.COMPLETED)


# ===================================================================
# 5. Resume invariant
# ===================================================================


class TestResumeInvariant:
    """Only phase-boundary checkpoints with valid labels are accepted."""

    def test_get_latest_skips_invalid_labels(self, index_store) -> None:  # type: ignore[no-untyped-def]
        recorder = TraceRecorder("inv-test", index_store)

        # Write a checkpoint with an invalid boundary
        recorder.record_checkpoint({
            "agent_state": AgentState(
                trace_id="inv-test",
                execution_id="ex",
            ).model_dump(mode="json"),
            "resume_context": {
                "iteration": 1,
                "phase_boundary": "mid_node_hack",
            },
        })

        result = get_latest_phase_boundary_checkpoint(index_store, "inv-test")
        assert result is None, "Invalid boundary must be skipped"

    def test_get_latest_returns_valid_boundary(self, index_store) -> None:  # type: ignore[no-untyped-def]
        # Persist the run first
        index_store.upsert_run(
            trace_id="inv-test-2",
            execution_id="ex",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        recorder = TraceRecorder("inv-test-2", index_store)

        # Write invalid then valid
        recorder.record_checkpoint({
            "agent_state": AgentState(
                trace_id="inv-test-2",
                execution_id="ex",
            ).model_dump(mode="json"),
            "resume_context": {"iteration": 1, "phase_boundary": "bad_label"},
        })
        recorder.record_checkpoint({
            "agent_state": AgentState(
                trace_id="inv-test-2",
                execution_id="ex",
            ).model_dump(mode="json"),
            "resume_context": {"iteration": 1, "phase_boundary": "after_execute_iter1"},
        })

        result = get_latest_phase_boundary_checkpoint(index_store, "inv-test-2")
        assert result is not None
        assert result["resume_context"]["phase_boundary"] == "after_execute_iter1"


# ===================================================================
# 6. Resume from checkpoint (integration)
# ===================================================================


class TestResumeFromCheckpoint:
    """Integration: start run → inject pause checkpoint → resume."""

    def test_resume_requires_running_state(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resume must fail if run is not RUNNING."""
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)

        # Write a run + paused checkpoint manually
        runner._index.upsert_run(
            trace_id="resume-test-1",
            execution_id="ex",
            state="PAUSED",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        recorder = TraceRecorder("resume-test-1", runner._index)
        cp = build_checkpoint_payload(
            AgentState(
                trace_id="resume-test-1",
                execution_id="ex",
                run_state=RunState.PAUSED,
            ),
            iteration=1,
            phase_boundary="paused",
        )
        recorder.record_checkpoint(cp)

        with pytest.raises(CheckpointError, match="expected RUNNING"):
            runner.resume_run("resume-test-1")

    def test_resume_requires_checkpoint(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resume must fail if no checkpoint exists."""
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)

        runner._index.upsert_run(
            trace_id="resume-test-2",
            execution_id="ex",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        with pytest.raises(CheckpointError, match="No phase-boundary checkpoint"):
            runner.resume_run("resume-test-2")

    def test_resume_backward_compatible_without_runbook_id_and_metadata(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A2: resume from old checkpoint (no runbook_id/metadata in resume_context) still works."""
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        trace_id = "resume-old-format"
        runner._index.upsert_run(
            trace_id=trace_id,
            execution_id="exec-old",
            state="RUNNING",
            objective="Old format resume",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        # Decision so _infer_runbook_id finds runbook_id
        runner._index.append_trace_event(
            trace_id,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": "decision",
                "payload": {"description": "Runbook selected", "runbook_id": "autofix_demo"},
            },
        )
        # Old-format checkpoint: no runbook_id, no metadata in resume_context.
        # Use iteration=2, after_control_iter2 so resume skips to end without running nodes.
        # Stage 2 has graph_builder so we must include planned_graph for resume to succeed.
        planner = HTNPlanner()
        iter2_graph = planner.plan_iter2_fix(
            "Old format resume", [], fix_context={"fix_context": "minimal"},
        )
        state = AgentState(
            trace_id=trace_id,
            execution_id="exec-old",
            run_state=RunState.RUNNING,
            intention=Intention(
                intention_id="int-old",
                objective="Old format resume",
                phase=IntentionPhase.EXECUTE,
                plan_id=iter2_graph.metadata.plan_id,
            ),
        )
        cp_payload = build_checkpoint_payload(
            state,
            iteration=2,
            phase_boundary="after_control_iter2",
            extra={
                "stage_id": "autofix_demo:iter2",
                "stage_index": 1,
                "planned_graph": iter2_graph.model_dump(mode="json"),
                "gate_snapshot": {
                    "required_nodes_ok": True,
                    "critic_verdict": "PASS",
                    "critic_confidence": 0.9,
                    "critic_evidence": ["resume backward compat"],
                    "critic_gaps": [],
                },
            },
        )
        runner._index.append_trace_event(
            trace_id,
            {"ts": datetime.now(timezone.utc).isoformat(), "kind": "checkpoint", "payload": cp_payload},
        )

        handle = runner.resume_run(trace_id)
        status = runner.get_status(handle)
        # Resume should succeed using _infer_runbook_id and _infer_runbook_metadata
        assert status.state == "COMPLETED"
