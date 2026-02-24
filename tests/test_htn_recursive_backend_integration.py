from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.types import RunRequest


_REPLAY_MAP: dict[str, list[dict]] = {
    "persist_user_request": [{
        "outputs": {"artifact_id": "sha256:req-integration-001"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "extract_entities": [{
        "outputs": {"urls": [], "file_paths": [], "basenames": []},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "htn_method_select_extract_envelope": [{
        "outputs": {
            "parsed": {
                "intent": {"task_type": "docs_summary", "confidence": 0.9},
                "inputs": {"doc_paths": ["/tmp/a.md"]},
                "missing_fields": [],
                "extras": {},
            },
        },
        "quality_signals": {"tokens_used": 8},
        "status": "COMPLETED",
    }],
    "read_000": [{
        "outputs": {
            "doc_000": "# Test doc\nHTN backend",
            "doc_000__path": "/tmp/a.md",
        },
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "merge_docs": [{
        "outputs": {"doc_000": "# Test doc\nHTN backend"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "draft_report": [{
        "outputs": {"content": "# HTN Report [doc_000]\n## Action items\n- OK"},
        "quality_signals": {"tokens_used": 20},
    }],
    "critic_report": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 0.95,
                "evidence": ["doc_000 cited"],
                "gaps": [],
            }),
        },
        "quality_signals": {"tokens_used": 10},
    }],
}


def _make_runner(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_map: dict[str, list[dict]] | None = None,
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

    def patched_build(graph, *, stage_default_model_id=None, **kwargs):
        registry = orig_build(graph, stage_default_model_id=stage_default_model_id, **kwargs)
        active_replay_map = replay_map if replay_map is not None else _REPLAY_MAP
        for nid, node in registry.items():
            if hasattr(node, "set_replay_responses") and nid in active_replay_map:
                node.set_replay_responses(active_replay_map[nid])
        return registry

    runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]
    return runner


def test_htn_recursive_backend_stage_runs_with_dynamic_commit(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(tmp_dir, monkeypatch)

    handle = runner.start(RunRequest(
        objective="Create a short report from one local document.",
        metadata={
            "runbook_id": "htn_recursive_demo_v0",
            "doc_paths": ["/tmp/a.md"],
        },
    ))
    status = runner.get_status(handle)
    assert status.state == "COMPLETED"

    events = list(runner._index.get_trace_events(handle.trace_id))
    dynamic_decisions = [
        e for e in events
        if e["kind"] == "decision"
        and e.get("payload", {}).get("description") == "Plan created (dynamic)"
    ]
    assert dynamic_decisions
    payload = dynamic_decisions[0]["payload"]
    assert payload["planner_backend"] == "htn_recursive_v0"
    assert payload["planner_backend_version"] == "0"
    assert set(payload["nodes"]) >= {"read_000", "merge_docs", "draft_report", "critic_report"}
    trace_payload = payload.get("planner_decision_trace", {})
    assert trace_payload
    assert trace_payload.get("notes", {}).get("context_kind") == "docs"


def test_htn_recursive_backend_stage_pauses_on_schema_driven_missing_source(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_map = dict(_REPLAY_MAP)
    replay_map["htn_method_select_extract_envelope"] = [{
        "outputs": {
            "parsed": {
                "intent": {"task_type": "generic_task", "confidence": 0.9},
                "inputs": {},
                "missing_fields": [],
                "extras": {},
            },
        },
        "quality_signals": {"tokens_used": 8},
        "status": "COMPLETED",
    }]
    replay_map["htn_method_select_clarification_questions"] = [{
        "outputs": {"parsed": {"questions": []}},
        "quality_signals": {"tokens_used": 5},
        "status": "COMPLETED",
    }]
    replay_map["persist_clarification_request"] = [{
        "outputs": {"artifact_id": "sha256:clarify-integration-001"},
        "quality_signals": {},
        "status": "COMPLETED",
    }]
    runner = _make_runner(tmp_dir, monkeypatch, replay_map=replay_map)

    handle = runner.start(RunRequest(
        objective="Do something without source",
        metadata={"runbook_id": "htn_recursive_demo_v0"},
        mode="supervised",
    ))
    status = runner.get_status(handle)
    assert status.state == "PAUSED"
    pause_context = runner.get_latest_pause_context(handle.trace_id)
    assert pause_context is not None
    assert str(pause_context.get("clarification_request_artifact_id", "")).strip()
