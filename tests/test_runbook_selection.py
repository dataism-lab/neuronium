"""Test: RunRequest(metadata={"runbook_id":"docs_report_v1"}) selects the
correct runbook and records a 'Runbook selected' trace decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.types import RunRequest


# ---------------------------------------------------------------------------
# Replay data for docs_report_v1 (single stage: read_000, merge, draft, critic)
# ---------------------------------------------------------------------------

_DOCS_REPORT_REPLAY: dict[str, list[dict]] = {
    "read_000": [{
        "outputs": {"doc_000": "# Test Doc\nHello world", "doc_000__path": "/tmp/a.md"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "merge_docs": [{
        "outputs": {"doc_000": "# Test Doc\nHello world"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "draft_report": [{
        "outputs": {"content": "# Report\nBased on [doc_000].\n## Action items\n- None"},
        "quality_signals": {"tokens_used": 20},
    }],
    "critic_report": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 0.95,
                "evidence": ["references doc_000"],
                "gaps": [],
            }),
        },
        "quality_signals": {"tokens_used": 10},
    }],
}


def _make_runner(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_map: dict[str, list[dict]],
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
    runner = AgentRunner(config, blob, idx)

    orig_build = runner._orchestrator._build_node_registry

    def patched_build(graph):
        registry = orig_build(graph)
        for nid, node in registry.items():
            if hasattr(node, "set_replay_responses") and nid in replay_map:
                node.set_replay_responses(replay_map[nid])
        return registry

    runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]
    return runner


def test_runbook_selected_event_for_docs_report(
    tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running with runbook_id=docs_report_v1 records a matching trace decision."""
    runner = _make_runner(tmp_dir, monkeypatch, _DOCS_REPORT_REPLAY)

    handle = runner.start(RunRequest(
        objective="Test report",
        metadata={"runbook_id": "docs_report_v1", "doc_paths": ["/tmp/a.md"]},
    ))

    events = list(runner._index.get_trace_events(handle.trace_id))
    decisions = [
        e for e in events
        if e["kind"] == "decision"
        and e.get("payload", {}).get("description") == "Runbook selected"
    ]
    assert len(decisions) >= 1
    assert decisions[0]["payload"]["runbook_id"] == "docs_report_v1"


def test_docs_report_runbook_completes(
    tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """docs_report_v1 should complete when critic returns PASS with evidence."""
    runner = _make_runner(tmp_dir, monkeypatch, _DOCS_REPORT_REPLAY)

    handle = runner.start(RunRequest(
        objective="Test report",
        metadata={"runbook_id": "docs_report_v1", "doc_paths": ["/tmp/a.md"]},
    ))

    status = runner.get_status(handle)
    assert status.state == "COMPLETED"


def test_unknown_runbook_fails(
    tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered runbook_id should produce a FAILED run."""
    runner = _make_runner(tmp_dir, monkeypatch, {})

    handle = runner.start(RunRequest(
        objective="Test",
        metadata={"runbook_id": "nonexistent_runbook"},
    ))

    status = runner.get_status(handle)
    assert status.state == "FAILED"
