from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.types import ControlCommand, RunRequest


_REPLAY_MAP: dict[str, list[dict]] = {
    "super_method_select_clarification_questions": [
        {
            "outputs": {
                "content": json.dumps({
                    "questions": [
                        {
                            "key": "url",
                            "prompt": "Укажи URL новости или статьи.",
                            "expected_type": "url",
                            "required": True,
                            "examples": [],
                        },
                    ],
                }),
            },
            "quality_signals": {"tokens_used": 8},
            "status": "COMPLETED",
        },
    ],
    "super_method_select_extract_envelope": [
        # 1st call: before clarification — missing url so planner escalates
        {
            "outputs": {
                "parsed": {
                    "intent": {"task_type": "news_summary", "confidence": 0.9},
                    "inputs": {},
                    "missing_fields": [
                        {"field": "url", "reason": "URL is required", "critical": True},
                    ],
                    "extras": {},
                },
            },
            "quality_signals": {"tokens_used": 12},
            "status": "COMPLETED",
        },
        # 2nd and later calls: after resume (user provided url) — no missing fields
        {
            "outputs": {
                "parsed": {
                    "intent": {"task_type": "news_summary", "confidence": 0.9},
                    "inputs": {"url": "https://example.com/news/1"},
                    "missing_fields": [],
                    "extras": {},
                },
            },
            "quality_signals": {"tokens_used": 12},
            "status": "COMPLETED",
        },
        {
            "outputs": {
                "parsed": {
                    "intent": {"task_type": "news_summary", "confidence": 0.9},
                    "inputs": {"url": "https://example.com/news/1"},
                    "missing_fields": [],
                    "extras": {},
                },
            },
            "quality_signals": {"tokens_used": 12},
            "status": "COMPLETED",
        },
    ],
    "fetch_html": [{
        "outputs": {"html": "<html><body>News</body></html>", "status_code": 200},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "extract_article": [{
        "outputs": {"title": "News", "text": "Main article text"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "draft_report": [{
        "outputs": {"content": "# Summary\n- key point"},
        "quality_signals": {"tokens_used": 20},
        "status": "COMPLETED",
    }],
    "critic_report": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 0.9,
                "evidence": ["source_ref"],
                "gaps": [],
            }),
        },
        "quality_signals": {"tokens_used": 10},
        "status": "COMPLETED",
    }],
}


def _make_runner(tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AgentRunner:
    monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-fake-key")
    config = AppConfig(
        project=ProjectConfig(name="test", data_dir=str(tmp_dir / ".n")),
        storage=StorageConfig(
            fs_cas_root=str(tmp_dir / "blobs"),
            sqlite_path=str(tmp_dir / "index.sqlite3"),
        ),
    )
    runner = AgentRunner(
        config,
        FsCasStore(config.storage.fs_cas_root),
        SqliteIndexStore(config.storage.sqlite_path),
    )
    original_build = runner._orchestrator._build_node_registry

    def patched_build(graph, *, stage_default_model_id=None, **kwargs):
        registry = original_build(graph, stage_default_model_id=stage_default_model_id, **kwargs)
        for nid, node in registry.items():
            if hasattr(node, "set_replay_responses") and nid in _REPLAY_MAP:
                node.set_replay_responses(_REPLAY_MAP[nid])
        return registry

    runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]
    return runner


def test_supervised_clarification_pause_revise_resume_flow(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(tmp_dir, monkeypatch)

    handle = runner.start(RunRequest(
        objective="Сделай сводку новости",
        mode="supervised",
        metadata={"runbook_id": "super_agent_v0"},
    ))
    paused = runner.get_status(handle)
    assert paused.state == "PAUSED"

    pause_context = runner.get_latest_pause_context(handle.trace_id)
    assert pause_context is not None
    request_artifact_id = str(
        pause_context.get("clarification_request_artifact_id", "")
    ).strip()
    assert request_artifact_id

    clarification_request = runner.read_artifact_json(request_artifact_id)
    questions = clarification_request.get("questions", [])
    assert isinstance(questions, list)
    assert questions

    revise_status = runner.control(
        handle,
        ControlCommand(
            type="revise",
            payload={
                "clarification_request_artifact_id": request_artifact_id,
                "answers": {"url": "https://example.com/news/1"},
            },
        ),
    )
    assert revise_status.state == "PAUSED"
    continue_status = runner.control(
        handle,
        ControlCommand(type="continue", payload={}),
    )
    assert continue_status.state == "RUNNING"

    resumed_handle = runner.resume_run(handle.trace_id)
    final_status = runner.get_status(resumed_handle)
    assert final_status.state == "COMPLETED"

    events = list(runner._index.get_trace_events(handle.trace_id))
    assert any(
        e["kind"] == "decision"
        and e.get("payload", {}).get("description") == "Escalation requested"
        for e in events
    )

    revise_events = [
        e for e in events
        if e["kind"] == "decision"
        and e.get("payload", {}).get("description") == "control_command: revise"
    ]
    assert revise_events
    control_payload = revise_events[-1]["payload"]["payload"]
    response_artifact_id = str(
        control_payload.get("clarification_response_artifact_id", "")
    ).strip()
    assert response_artifact_id

    rows = runner._index._fetchall(  # type: ignore[attr-defined]
        "SELECT parent_artifact_id, child_artifact_id, kind FROM lineage_edges WHERE child_artifact_id=?",
        (response_artifact_id,),
    )
    assert rows
    assert rows[0]["parent_artifact_id"] == request_artifact_id
    assert rows[0]["kind"] == "clarification"


def test_replay_preserves_paused_clarification_context(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(tmp_dir, monkeypatch)
    handle = runner.start(RunRequest(
        objective="Сделай сводку новости",
        mode="supervised",
        metadata={"runbook_id": "super_agent_v0"},
    ))
    assert runner.get_status(handle).state == "PAUSED"
    original_pause = runner.get_latest_pause_context(handle.trace_id)
    assert original_pause is not None
    original_aid = str(
        original_pause.get("clarification_request_artifact_id", "")
    ).strip()
    assert original_aid

    replay_handle = runner.replay(handle.trace_id)
    replay_status = runner.get_status(replay_handle)
    assert replay_status.state == "PAUSED"
    replay_pause = runner.get_latest_pause_context(replay_handle.trace_id)
    assert replay_pause is not None
    replay_aid = str(
        replay_pause.get("clarification_request_artifact_id", "")
    ).strip()
    assert replay_aid == original_aid
