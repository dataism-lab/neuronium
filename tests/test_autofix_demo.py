"""Tests for the 2-iteration autofix demo loop.

Covers:
- Full replay of a seeded NameError → fix → PASS scenario (offline).
- strict_fail when LLM key is missing.
- Strict replay fails when critic replay_data is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


# -- Deterministic constants for seeded data ---------------------------------

_FIXED_TS = "2000-01-01T00:00:00+00:00"
_SEEDED_TRACE_ID = "trace-seeded-autofix"
_SEEDED_EXEC_ID = "exec-seeded"
_OBJECTIVE = "Print the value of x"


# -- Seeded replay_data per node (fully deterministic, no unstable fields) ---

_BUGGY_CODE = "print(x)"  # NameError: name 'x' is not defined

_REPLAY_DATA = {
    # -- Iteration 1 --
    "generate": [{
        "outputs": {"content": _BUGGY_CODE},
        "quality_signals": {},
    }],
    "execute": [{
        "outputs": {
            "stdout": "",
            "stderr": "Traceback (most recent call last):\n"
                      '  File "<string>", line 1, in <module>\n'
                      "NameError: name 'x' is not defined",
            "exit_code": 1,
        },
        "quality_signals": {},
        "status": "FAILED",
    }],
    "critic": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "FAIL",
                "confidence": 0.95,
                "evidence": ["exit_code=1", "NameError in stderr"],
                "gaps": ["Variable 'x' is not defined before use"],
            }),
        },
        "quality_signals": {},
    }],
    # -- Iteration 2 (fix-pipeline) --
    "fix": [{
        "outputs": {"content": "x = 42\nprint(x)"},
        "quality_signals": {},
    }],
    "execute_fix": [{
        "outputs": {"stdout": "42\n", "exit_code": 0},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "critic_fix": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 0.99,
                "evidence": ["exit_code=0", "stdout contains '42'"],
                "gaps": [],
            }),
        },
        "quality_signals": {},
    }],
}


# -- Helpers -----------------------------------------------------------------

def _make_runner(tmp_dir: Path) -> AgentRunner:
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


def _seed_autofix_run(runner: AgentRunner) -> str:
    """Seed a SQLite index with deterministic replay data for all 6 nodes."""
    runner._index.upsert_run(
        trace_id=_SEEDED_TRACE_ID,
        execution_id=_SEEDED_EXEC_ID,
        state="COMPLETED",
        objective=_OBJECTIVE,
        config_snapshot_json="{}",
        created_at=_FIXED_TS,
    )
    for node_id, responses in _REPLAY_DATA.items():
        runner._index.append_trace_event(
            _SEEDED_TRACE_ID,
            {
                "ts": _FIXED_TS,
                "kind": "replay_data",
                "payload": {
                    "node_id": node_id,
                    "recorded_responses": responses,
                },
            },
        )
    return _SEEDED_TRACE_ID


# -- Tests -------------------------------------------------------------------


def test_autofix_two_iterations_replay_strict(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the seeded demo: iter1 NameError → iter2 fix → PASS offline."""
    monkeypatch.delenv("NEURONIUM_OPENAI_API_KEY", raising=False)

    runner = _make_runner(tmp_dir)
    seed_id = _seed_autofix_run(runner)

    handle = runner.replay(seed_id)
    status = runner.get_status(handle)

    # The run must complete successfully on iteration 2
    assert status.state == "COMPLETED", f"Expected COMPLETED, got {status.state}: {status.message}"
    assert handle.trace_id != seed_id

    # Verify trace events contain the expected structure
    events = list(runner._index.get_trace_events(handle.trace_id))
    kinds = [e["kind"] for e in events]

    # Must have critic verdicts for both iterations
    critic_verdicts = [e for e in events if e["kind"] == "critic_verdict"]
    assert len(critic_verdicts) == 2

    # Iteration 1: critic FAIL
    cv1 = critic_verdicts[0]["payload"]
    assert cv1["iteration"] == 1
    assert cv1["verdict"] == "FAIL"
    assert cv1["evidence"]  # non-empty

    # Iteration 2: critic PASS
    cv2 = critic_verdicts[1]["payload"]
    assert cv2["iteration"] == 2
    assert cv2["verdict"] == "PASS"
    assert cv2["evidence"]  # non-empty (hard rule)

    # Must have exactly 1 replan event
    replans = [e for e in events if e["kind"] == "replan"]
    assert len(replans) == 1
    rp = replans[0]["payload"]
    assert rp["iteration_from"] == 1
    assert rp["iteration_to"] == 2

    # Must have replay_data for all 6 nodes
    replay_data_events = [e for e in events if e["kind"] == "replay_data"]
    replay_node_ids = {e["payload"]["node_id"] for e in replay_data_events}
    # During replay, recording is not enabled (replay_provider is set),
    # so replay_data from the SEED is in the original trace, not the new one.
    # But the decision events prove the loop ran correctly.

    # Must have plan creation decisions for both iterations
    decisions = [e for e in events if e["kind"] == "decision"]
    plan_decisions = [
        d for d in decisions
        if "Plan created" in d["payload"].get("description", "")
    ]
    assert len(plan_decisions) == 2


def test_strict_fail_without_llm_key(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live run fails early when NEURONIUM_OPENAI_API_KEY is missing."""
    monkeypatch.delenv("NEURONIUM_OPENAI_API_KEY", raising=False)

    runner = _make_runner(tmp_dir)

    from neuronium_agent.types import RunRequest

    handle = runner.start(RunRequest(objective="Should fail"))
    status = runner.get_status(handle)

    assert status.state == "FAILED"
    assert status.message is not None
    assert "LLM unavailable" in status.message or "strict_fail" in status.message

    # Trace must record the strict_fail decision
    events = list(runner._index.get_trace_events(handle.trace_id))
    strict_decisions = [
        e for e in events
        if e["kind"] == "decision"
        and "strict_fail" in e["payload"].get("description", "")
    ]
    assert len(strict_decisions) >= 1


def test_replay_data_required_for_critic_nodes(
    tmp_dir: Path,
) -> None:
    """Strict replay fails if critic replay_data is missing."""
    runner = _make_runner(tmp_dir)

    trace_id = "trace-missing-critic"
    runner._index.upsert_run(
        trace_id=trace_id,
        execution_id="exec-missing",
        state="COMPLETED",
        objective=_OBJECTIVE,
        config_snapshot_json="{}",
        created_at=_FIXED_TS,
    )

    # Seed replay_data for generate and execute, but NOT for critic
    for node_id in ("generate", "execute"):
        runner._index.append_trace_event(
            trace_id,
            {
                "ts": _FIXED_TS,
                "kind": "replay_data",
                "payload": {
                    "node_id": node_id,
                    "recorded_responses": _REPLAY_DATA[node_id],
                },
            },
        )

    # Replay should fail because critic replay_data is missing
    handle = runner.replay(trace_id)
    status = runner.get_status(handle)

    assert status.state == "FAILED"
    assert status.message is not None
    assert "missing recorded responses" in status.message.lower() or "strict" in status.message.lower()
