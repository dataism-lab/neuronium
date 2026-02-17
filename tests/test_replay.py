"""Replay tests for strict offline execution.

Updated for the 2-iteration autofix loop (3-node DAGs with critic).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


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


def _seed_original_run_with_replay_data(runner: AgentRunner, trace_id: str) -> None:
    """Seed a run that succeeds on iteration 1 (generate+execute+critic)."""
    now = datetime.now(timezone.utc).isoformat()
    runner._index.upsert_run(
        trace_id=trace_id,
        execution_id="exec-original",
        state="COMPLETED",
        objective="Print hello",
        config_snapshot_json="{}",
        created_at=now,
    )
    runner._index.append_trace_event(
        trace_id,
        {
            "ts": now,
            "kind": "replay_data",
            "payload": {
                "node_id": "generate",
                "recorded_responses": [
                    {
                        "outputs": {"content": "print('hello from replay')"},
                        "quality_signals": {"tokens_used": 7},
                        "status": "COMPLETED",
                    }
                ],
            },
        },
    )
    runner._index.append_trace_event(
        trace_id,
        {
            "ts": now,
            "kind": "replay_data",
            "payload": {
                "node_id": "execute",
                "recorded_responses": [
                    {
                        "outputs": {"stdout": "hello from replay\n", "exit_code": 0},
                        "quality_signals": {"latency_ms": 9.0},
                        "status": "COMPLETED",
                    }
                ],
            },
        },
    )
    runner._index.append_trace_event(
        trace_id,
        {
            "ts": now,
            "kind": "replay_data",
            "payload": {
                "node_id": "critic",
                "recorded_responses": [
                    {
                        "outputs": {
                            "content": json.dumps({
                                "verdict": "PASS",
                                "confidence": 1.0,
                                "evidence": ["exit_code=0", "stdout correct"],
                                "gaps": [],
                            }),
                        },
                        "quality_signals": {"tokens_used": 10},
                        "status": "COMPLETED",
                    }
                ],
            },
        },
    )


def test_replay_runs_offline_with_new_trace_id(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEURONIUM_OPENAI_API_KEY", raising=False)
    runner = _make_runner(tmp_dir)
    original_trace_id = "trace-original"
    _seed_original_run_with_replay_data(runner, original_trace_id)

    handle = runner.replay(original_trace_id)
    status = runner.get_status(handle)

    assert handle.trace_id != original_trace_id
    assert status.state == "COMPLETED"

    replay_events = list(runner._index.get_trace_events(handle.trace_id))
    assert any(
        ev["kind"] == "decision"
        and ev["payload"].get("replay_of_trace_id") == original_trace_id
        for ev in replay_events
    )


def test_replay_strict_fails_when_recorded_responses_missing(
    tmp_dir: Path,
) -> None:
    runner = _make_runner(tmp_dir)
    original_trace_id = "trace-missing"
    now = datetime.now(timezone.utc).isoformat()
    runner._index.upsert_run(
        trace_id=original_trace_id,
        execution_id="exec-original",
        state="COMPLETED",
        objective="Print hello",
        config_snapshot_json="{}",
        created_at=now,
    )
    # Seed only one node; "execute" and "critic" remain missing.
    runner._index.append_trace_event(
        original_trace_id,
        {
            "ts": now,
            "kind": "replay_data",
            "payload": {
                "node_id": "generate",
                "recorded_responses": [
                    {
                        "outputs": {"content": "print('hello')"},
                        "quality_signals": {"tokens_used": 5},
                        "status": "COMPLETED",
                    }
                ],
            },
        },
    )

    handle = runner.replay(original_trace_id)
    status = runner.get_status(handle)
    assert status.state == "FAILED"
    assert status.message is not None
    assert "missing recorded responses" in status.message


def test_replay_strict_ignores_node_end_fallback(
    tmp_dir: Path,
) -> None:
    """node_end events must NOT substitute for replay_data in strict mode.

    Even when node_end outputs exist for every node, strict replay must
    fail if replay_data is missing for at least one replay-capable node.
    """
    runner = _make_runner(tmp_dir)
    original_trace_id = "trace-fallback-trap"
    now = datetime.now(timezone.utc).isoformat()
    runner._index.upsert_run(
        trace_id=original_trace_id,
        execution_id="exec-original",
        state="COMPLETED",
        objective="Print hello",
        config_snapshot_json="{}",
        created_at=now,
    )
    # replay_data only for "generate" — "execute" and "critic" have NO replay_data.
    runner._index.append_trace_event(
        original_trace_id,
        {
            "ts": now,
            "kind": "replay_data",
            "payload": {
                "node_id": "generate",
                "recorded_responses": [
                    {
                        "outputs": {"content": "print('hello')"},
                        "quality_signals": {"tokens_used": 5},
                        "status": "COMPLETED",
                    }
                ],
            },
        },
    )
    # node_end exists for ALL nodes (as would happen in a real trace).
    for nid in ("generate", "execute", "critic"):
        runner._index.append_trace_event(
            original_trace_id,
            {
                "ts": now,
                "kind": "node_end",
                "payload": {
                    "node_id": nid,
                    "status": "COMPLETED",
                    "outputs": {"stdout": "..."},
                },
            },
        )

    handle = runner.replay(original_trace_id)
    status = runner.get_status(handle)
    # Strict replay must NOT silently use node_end as a substitute.
    assert status.state == "FAILED"
    assert status.message is not None
    assert "missing recorded responses" in status.message
