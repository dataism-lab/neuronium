"""Phase 7 (PAUSE_CONTROL): full control cycle integration tests.

Tests:
  7.1 — start → pause after first node → PAUSED, checkpoint paused_mid_execute.
  7.2 — PAUSED → continue → resume_run → COMPLETED.
  7.3 — stop graceful vs stop immediate (checkpoint content difference).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.trace.checkpoints import (
    get_latest_phase_boundary_checkpoint,
)
from neuronium_agent.types import ControlCommand, InterruptRequest, RunRequest


# ---------------------------------------------------------------------------
# Synchronisation helper — eliminates race between _run_runbook clearing
# stale interrupts (pop) and the test injecting a new one.
# ---------------------------------------------------------------------------

class _SyncInterruptDict(dict):
    """Blocks the worker on its first ``pop()`` until :meth:`arm` injects
    the desired :class:`InterruptRequest` and releases the worker."""

    def __init__(self) -> None:
        super().__init__()
        self._pop_event = threading.Event()
        self._continue_event = threading.Event()
        self._first_pop_done = False

    def pop(self, *args, **kwargs):  # type: ignore[override]
        result = super().pop(*args, **kwargs)
        if not self._first_pop_done:
            self._first_pop_done = True
            self._pop_event.set()
            self._continue_event.wait(timeout=10.0)
        return result

    def arm(self, trace_id: str, request: InterruptRequest) -> None:
        """Block until first ``pop()``, inject *request*, unblock worker."""
        assert self._pop_event.wait(timeout=5.0), (
            "_run_runbook should have called pop()"
        )
        self[trace_id] = request
        self._continue_event.set()


# ---------------------------------------------------------------------------
# Replay data (autofix_demo iter1: generate → execute → critic)
# ---------------------------------------------------------------------------

_REPLAY_MAP: dict[str, list[dict]] = {
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


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_runner(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_map: dict[str, list[dict]] | None = None,
) -> AgentRunner:
    """Build AgentRunner with replay for deterministic runs."""
    monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-fake-key")
    if replay_map is None:
        replay_map = _REPLAY_MAP

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


def _start_and_interrupt(
    runner: AgentRunner,
    objective: str,
    request: InterruptRequest,
):
    """Start a run and deterministically inject *request* mid-execution.

    Uses :class:`_SyncInterruptDict` to guarantee the interrupt is set
    after ``_run_runbook`` clears stale data but before the executor's
    ``interrupt_check`` runs.
    """
    sync = _SyncInterruptDict()
    runner._orchestrator._interrupt_requests = sync  # type: ignore[assignment]

    handle_holder: list = []

    def worker() -> None:
        handle = runner.start(
            RunRequest(objective=objective),
            on_handle_ready=lambda h: handle_holder.append(h),
        )
        handle_holder.append(("done", handle))

    t = threading.Thread(target=worker)
    t.start()

    for _ in range(500):
        if handle_holder:
            break
        time.sleep(0.01)
    assert handle_holder, "on_handle_ready was not called"
    handle = handle_holder[0]

    sync.arm(handle.trace_id, request)

    t.join(timeout=30.0)
    assert not t.is_alive(), "Worker thread should have finished"
    return handle


# ---------------------------------------------------------------------------
# 7.1 — Mid-execution pause
# ---------------------------------------------------------------------------

def test_start_then_pause_mid_execution_leaves_run_paused_with_checkpoint(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.1: Deterministic interrupt after first batch → PAUSED + checkpoint."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle = _start_and_interrupt(
        runner, "Full control 7.1",
        InterruptRequest(command="pause", mode="graceful"),
    )

    assert runner.get_status(handle).state == "PAUSED"

    events = list(runner._index.get_trace_events(handle.trace_id))
    cp_events = [e for e in events if e.get("kind") == "checkpoint"]
    boundaries = [
        e.get("payload", {}).get("resume_context", {}).get("phase_boundary")
        for e in cp_events
    ]
    assert "paused_mid_execute" in boundaries, (
        f"Trace should contain paused_mid_execute checkpoint; got {boundaries}"
    )
    mid_cp = next(
        (
            e["payload"]
            for e in cp_events
            if e.get("payload", {})
            .get("resume_context", {})
            .get("phase_boundary")
            == "paused_mid_execute"
        ),
        None,
    )
    assert mid_cp is not None
    assert "completed_node_results" in mid_cp.get("resume_context", {})
    assert "pending_node_ids" in mid_cp.get("resume_context", {})


# ---------------------------------------------------------------------------
# 7.2 — PAUSED → continue → resume_run → COMPLETED
# ---------------------------------------------------------------------------

def test_paused_continue_resume_run_completes(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.2: Full cycle — pause → continue → resume_run → COMPLETED."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle = _start_and_interrupt(
        runner, "Full control 7.2",
        InterruptRequest(command="pause", mode="graceful"),
    )

    assert runner.get_status(handle).state == "PAUSED"

    runner.control(handle, ControlCommand(type="continue", payload={}))
    resume_handle = runner.resume_run(handle.trace_id)
    assert resume_handle.trace_id == handle.trace_id

    status = runner.get_status(handle)
    assert status.state == "COMPLETED", f"Expected COMPLETED, got {status.state}"


# ---------------------------------------------------------------------------
# 7.3 — Stop graceful vs stop immediate
# ---------------------------------------------------------------------------

def test_stop_graceful_writes_full_mid_execute_checkpoint(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.3 (graceful): checkpoint with paused_mid_execute and full context."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle = _start_and_interrupt(
        runner, "Stop test graceful",
        InterruptRequest(command="stop", mode="graceful"),
    )

    assert runner.get_status(handle).state == "CANCELLED"

    cp = get_latest_phase_boundary_checkpoint(runner._index, handle.trace_id)
    assert cp is not None
    rc = cp.get("resume_context", {})
    assert rc.get("phase_boundary") == "paused_mid_execute"
    assert "completed_node_results" in rc
    assert "pending_node_ids" in rc
    assert "planned_graph" in rc


def test_stop_immediate_writes_minimal_checkpoint(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.3 (immediate): checkpoint cancelled_mid_execute without full resume context."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle = _start_and_interrupt(
        runner, "Stop test immediate",
        InterruptRequest(command="stop", mode="immediate"),
    )

    assert runner.get_status(handle).state == "CANCELLED"

    cp = get_latest_phase_boundary_checkpoint(runner._index, handle.trace_id)
    assert cp is not None
    rc = cp.get("resume_context", {})
    assert rc.get("phase_boundary") == "cancelled_mid_execute"
    assert rc.get("completed_node_results") is None or rc.get("completed_node_results") == {}
    assert rc.get("pending_node_ids") is None or rc.get("pending_node_ids") == []
