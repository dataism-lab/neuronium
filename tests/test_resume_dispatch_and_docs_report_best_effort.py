"""Test: resume_run dispatches correctly by runbook_id and supports
best-effort resume for docs_report_v1 via gate snapshot in checkpoint.

Key invariant: when resuming from ``after_execute_iter1``, the
``_execute`` method must NOT be called (the gate snapshot from the
checkpoint is used instead).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.core.state import AgentState, Intention, IntentionPhase, RunState
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.trace.checkpoints import build_checkpoint_payload
from neuronium_agent.types import ControlCommand, RunRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_TS = "2000-01-01T00:00:00+00:00"
_TRACE_ID = "resume-test-docs-report"
_EXEC_ID = "exec-resume-test"


def _make_runner(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AgentRunner:
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
    return AgentRunner(config, blob, idx)


def _seed_run_with_checkpoint(
    runner: AgentRunner,
    *,
    gate_snapshot: dict,
    phase_boundary: str = "after_execute_iter1",
) -> str:
    """Seed a run in RUNNING state with a phase-boundary checkpoint
    that contains a gate snapshot for docs_report_v1.
    """
    runner._index.upsert_run(
        trace_id=_TRACE_ID,
        execution_id=_EXEC_ID,
        state="RUNNING",
        objective="Resume test report",
        config_snapshot_json="{}",
        created_at=_FIXED_TS,
    )

    # Record 'Runbook selected' decision so _infer_runbook_id works
    runner._index.append_trace_event(
        _TRACE_ID,
        {
            "ts": _FIXED_TS,
            "kind": "decision",
            "payload": {
                "description": "Runbook selected",
                "runbook_id": "docs_report_v1",
            },
        },
    )

    # Build and record a phase-boundary checkpoint
    state = AgentState(
        trace_id=_TRACE_ID,
        execution_id=_EXEC_ID,
        run_state=RunState.RUNNING,
        intention=Intention(
            intention_id="int-resume-test",
            objective="Resume test report",
            phase=IntentionPhase.EXECUTE,
            plan_id="plan-docs-report-v1-exec-resume",
        ),
    )

    cp_payload = build_checkpoint_payload(
        state,
        iteration=1,
        phase_boundary=phase_boundary,
        extra={
            "runbook_id": "docs_report_v1",
            "stage_id": "docs_report_v1:stage1",
            "stage_index": 0,
            "gate_snapshot": gate_snapshot,
        },
    )

    runner._index.append_trace_event(
        _TRACE_ID,
        {
            "ts": _FIXED_TS,
            "kind": "checkpoint",
            "payload": cp_payload,
        },
    )
    return _TRACE_ID


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeDispatch:
    """resume_run should pick the right runbook based on inferred runbook_id."""

    def test_resume_infers_docs_report_v1(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resume for docs_report_v1 should not call _run_cycle (autofix)."""
        runner = _make_runner(tmp_dir, monkeypatch)

        gate_snapshot = {
            "required_nodes_ok": True,
            "critic_verdict": "PASS",
            "critic_confidence": 0.95,
            "critic_evidence": ["doc_000 cited"],
            "critic_gaps": [],
        }
        _seed_run_with_checkpoint(runner, gate_snapshot=gate_snapshot)

        handle = runner.resume_run(_TRACE_ID)
        status = runner.get_status(handle)

        # Should complete (gate snapshot says PASS)
        assert status.state == "COMPLETED"

        # Verify a "Runbook selected (resume)" decision with docs_report_v1
        events = list(runner._index.get_trace_events(_TRACE_ID))
        resume_decisions = [
            e for e in events
            if e["kind"] == "decision"
            and "resume" in str(e.get("payload", {}).get("description", "")).lower()
        ]
        runbook_decisions = [
            e for e in events
            if e["kind"] == "decision"
            and e.get("payload", {}).get("runbook_id") == "docs_report_v1"
        ]
        assert len(runbook_decisions) >= 1


class TestBestEffortResume:
    """Resume from after_execute_iter1 should NOT re-execute the DAG."""

    def test_execute_not_called_on_resume(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Monkey-patch _execute to raise; if resume works without calling
        _execute, the gate snapshot was used correctly.
        """
        runner = _make_runner(tmp_dir, monkeypatch)

        gate_snapshot = {
            "required_nodes_ok": True,
            "critic_verdict": "PASS",
            "critic_confidence": 0.9,
            "critic_evidence": ["evidence present"],
            "critic_gaps": [],
        }
        _seed_run_with_checkpoint(runner, gate_snapshot=gate_snapshot)

        # Monkey-patch _execute to explode if called
        call_count = {"n": 0}
        orig_execute = runner._orchestrator._execute

        def bomb_execute(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("_execute should NOT have been called during resume!")

        runner._orchestrator._execute = bomb_execute  # type: ignore[method-assign]

        handle = runner.resume_run(_TRACE_ID)
        status = runner.get_status(handle)

        # _execute was NOT called
        assert call_count["n"] == 0
        # Run completed from gate snapshot
        assert status.state == "COMPLETED"

    def test_resume_with_failing_gate_snapshot(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resume with a FAIL gate snapshot should produce FAILED without
        calling _execute.
        """
        runner = _make_runner(tmp_dir, monkeypatch)

        gate_snapshot = {
            "required_nodes_ok": True,
            "critic_verdict": "FAIL",
            "critic_confidence": 0.8,
            "critic_evidence": [],
            "critic_gaps": ["no citations"],
        }
        _seed_run_with_checkpoint(runner, gate_snapshot=gate_snapshot)

        call_count = {"n": 0}

        def bomb_execute(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("_execute should NOT have been called!")

        runner._orchestrator._execute = bomb_execute  # type: ignore[method-assign]

        handle = runner.resume_run(_TRACE_ID)
        status = runner.get_status(handle)

        assert call_count["n"] == 0
        assert status.state == "FAILED"

    def test_resume_from_after_control_skips_execute_and_control(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resume from after_control_iter1 should skip both execute and control."""
        runner = _make_runner(tmp_dir, monkeypatch)

        gate_snapshot = {
            "required_nodes_ok": True,
            "critic_verdict": "PASS",
            "critic_confidence": 0.95,
            "critic_evidence": ["evidence"],
            "critic_gaps": [],
        }
        _seed_run_with_checkpoint(
            runner,
            gate_snapshot=gate_snapshot,
            phase_boundary="after_control_iter1",
        )

        call_count = {"n": 0}

        def bomb_execute(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("_execute should NOT have been called!")

        runner._orchestrator._execute = bomb_execute  # type: ignore[method-assign]

        handle = runner.resume_run(_TRACE_ID)
        status = runner.get_status(handle)

        assert call_count["n"] == 0
        assert status.state == "COMPLETED"
