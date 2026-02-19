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
# Fixtures
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


# ---------------------------------------------------------------------------
# 7.1 — Mid-execution pause: start → pause after first node → PAUSED + checkpoint
# ---------------------------------------------------------------------------

def test_start_then_pause_mid_execution_leaves_run_paused_with_checkpoint(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.1: Set interrupt from main thread as soon as we have handle; executor sees it after first batch."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle_holder: list = []

    def run_start() -> None:
        handle = runner.start(
            RunRequest(objective="Full control 7.1"),
            on_handle_ready=lambda h: handle_holder.append(h),
        )
        handle_holder.append(("done", handle))

    thread = threading.Thread(target=run_start)
    thread.start()
    # Set interrupt as soon as we have handle (_run_runbook clears it at start, so we set after that)
    for _ in range(500):
        if handle_holder:
            break
        time.sleep(0.01)
    assert handle_holder, "on_handle_ready should have been called"
    handle = handle_holder[0]
    runner._orchestrator._interrupt_requests[handle.trace_id] = InterruptRequest(command="pause", mode="graceful")

    thread.join(timeout=30.0)
    assert not thread.is_alive(), "Worker thread should have finished"

    assert hasattr(handle, "trace_id")
    status = runner.get_status(handle)
    assert status.state == "PAUSED", f"Expected PAUSED, got {status.state}"

    # Mid-execute pause writes paused_mid_execute then _record_control_decision adds "paused"; either is valid
    events = list(runner._index.get_trace_events(handle.trace_id))
    cp_events = [e for e in events if e.get("kind") == "checkpoint"]
    boundaries = [e.get("payload", {}).get("resume_context", {}).get("phase_boundary") for e in cp_events]
    assert "paused_mid_execute" in boundaries, (
        f"Trace should contain paused_mid_execute checkpoint for resume; got {boundaries}"
    )
    latest = get_latest_phase_boundary_checkpoint(runner._index, handle.trace_id)
    assert latest is not None
    rc = latest.get("resume_context", {})
    # resumed context for exact pause point comes from paused_mid_execute (may not be latest due to "paused" record)
    mid_cp = next(
        (e["payload"] for e in cp_events if e.get("payload", {}).get("resume_context", {}).get("phase_boundary") == "paused_mid_execute"),
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
    """7.2: Full cycle start → set interrupt from main thread → continue → resume_run → COMPLETED."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle_holder: list = []

    def run_start() -> None:
        handle = runner.start(
            RunRequest(objective="Full control 7.2"),
            on_handle_ready=lambda h: handle_holder.append(h),
        )
        handle_holder.append(("done", handle))

    thread = threading.Thread(target=run_start)
    thread.start()
    for _ in range(500):
        if handle_holder:
            break
        time.sleep(0.01)
    assert handle_holder
    handle = handle_holder[0]
    runner._orchestrator._interrupt_requests[handle.trace_id] = InterruptRequest(command="pause", mode="graceful")

    thread.join(timeout=30.0)
    assert not thread.is_alive()

    assert runner.get_status(handle).state == "PAUSED"

    runner.control(handle, ControlCommand(type="continue", payload={}))
    resume_handle = runner.resume_run(handle.trace_id)
    assert resume_handle.trace_id == handle.trace_id

    status = runner.get_status(handle)
    assert status.state == "COMPLETED", f"Expected COMPLETED, got {status.state}"


# ---------------------------------------------------------------------------
# 7.3 — Stop graceful vs stop immediate (checkpoint difference)
# ---------------------------------------------------------------------------

def _run_until_stop(
    runner: AgentRunner,
    handle_holder: list,
    stop_payload: dict,
) -> None:
    def run_start() -> None:
        handle = runner.start(
            RunRequest(objective="Stop test"),
            on_handle_ready=lambda h: handle_holder.append(h),
        )
        handle_holder.append(("done", handle))

    thread = threading.Thread(target=run_start)
    thread.start()
    for _ in range(500):
        if handle_holder:
            break
        time.sleep(0.01)
    assert handle_holder
    handle = handle_holder[0]
    runner._orchestrator._interrupt_requests[handle.trace_id] = InterruptRequest(
        command="stop",
        mode=stop_payload.get("mode", "graceful"),
        export_path=stop_payload.get("export_path"),
    )
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert runner.get_status(handle).state == "CANCELLED"


def test_stop_graceful_writes_full_mid_execute_checkpoint(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.3 (graceful): Stop with mode=graceful leaves checkpoint with phase_boundary paused_mid_execute and full context."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle_holder: list = []
    _run_until_stop(runner, handle_holder, {"mode": "graceful"})
    handle = handle_holder[0]

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
    """7.3 (immediate): Stop with mode=immediate leaves checkpoint cancelled_mid_execute without full resume context."""
    runner = _make_runner(tmp_dir, monkeypatch)
    handle_holder: list = []
    _run_until_stop(runner, handle_holder, {"mode": "immediate"})
    handle = handle_holder[0]

    cp = get_latest_phase_boundary_checkpoint(runner._index, handle.trace_id)
    assert cp is not None
    rc = cp.get("resume_context", {})
    assert rc.get("phase_boundary") == "cancelled_mid_execute"
    # Minimal checkpoint: no completed_node_results / pending_node_ids for resume
    assert rc.get("completed_node_results") is None or rc.get("completed_node_results") == {}
    assert rc.get("pending_node_ids") is None or rc.get("pending_node_ids") == []
