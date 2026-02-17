"""Test: stage_start / stage_end trace events are recorded for runbook runs."""

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
# Replay data (same minimal set as test_runbook_selection)
# ---------------------------------------------------------------------------

_DOCS_REPORT_REPLAY: dict[str, list[dict]] = {
    "read_000": [{
        "outputs": {"doc_000": "content", "doc_000__path": "/tmp/a.md"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "merge_docs": [{
        "outputs": {"doc_000": "content"},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "draft_report": [{
        "outputs": {"content": "# Report [doc_000]\n## Action items\n- OK"},
        "quality_signals": {"tokens_used": 15},
    }],
    "critic_report": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 0.9,
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
            if hasattr(node, "set_replay_responses") and nid in _DOCS_REPORT_REPLAY:
                node.set_replay_responses(_DOCS_REPORT_REPLAY[nid])
        return registry

    runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]
    return runner


class TestStageEvents:
    """Verify stage_start / stage_end events in trace."""

    def test_stage_start_present(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch)
        handle = runner.start(RunRequest(
            objective="Stage event test",
            metadata={"runbook_id": "docs_report_v1", "doc_paths": ["/tmp/a.md"]},
        ))

        events = list(runner._index.get_trace_events(handle.trace_id))
        stage_starts = [e for e in events if e["kind"] == "stage_start"]

        assert len(stage_starts) >= 1
        payload = stage_starts[0]["payload"]
        assert payload["runbook_id"] == "docs_report_v1"
        assert payload["stage_id"] == "docs_report_v1:stage1"
        assert payload["stage_index"] == 0

    def test_stage_end_present(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch)
        handle = runner.start(RunRequest(
            objective="Stage event test",
            metadata={"runbook_id": "docs_report_v1", "doc_paths": ["/tmp/a.md"]},
        ))

        events = list(runner._index.get_trace_events(handle.trace_id))
        stage_ends = [e for e in events if e["kind"] == "stage_end"]

        assert len(stage_ends) >= 1
        payload = stage_ends[0]["payload"]
        assert payload["runbook_id"] == "docs_report_v1"
        assert payload["stage_id"] == "docs_report_v1:stage1"
        assert payload["success"] is True
        assert payload["reason"] == "quality gate passed"

    def test_critic_verdict_present(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch)
        handle = runner.start(RunRequest(
            objective="Critic verdict test",
            metadata={"runbook_id": "docs_report_v1", "doc_paths": ["/tmp/a.md"]},
        ))

        events = list(runner._index.get_trace_events(handle.trace_id))
        verdicts = [e for e in events if e["kind"] == "critic_verdict"]

        assert len(verdicts) >= 1
        payload = verdicts[0]["payload"]
        assert payload["critic_node_id"] == "critic_report"
        assert payload["verdict"] == "PASS"

    def test_stage_end_failure_on_critic_fail(
        self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When critic returns FAIL, stage_end records success=False."""
        monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-fake-key")

        fail_replay = dict(_DOCS_REPORT_REPLAY)
        fail_replay["critic_report"] = [{
            "outputs": {
                "content": json.dumps({
                    "verdict": "FAIL",
                    "confidence": 0.8,
                    "evidence": [],
                    "gaps": ["missing citations"],
                }),
            },
            "quality_signals": {"tokens_used": 10},
        }]

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
                if hasattr(node, "set_replay_responses") and nid in fail_replay:
                    node.set_replay_responses(fail_replay[nid])
            return registry

        runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]

        handle = runner.start(RunRequest(
            objective="Fail test",
            metadata={"runbook_id": "docs_report_v1", "doc_paths": ["/tmp/a.md"]},
        ))

        status = runner.get_status(handle)
        assert status.state == "FAILED"

        events = list(runner._index.get_trace_events(handle.trace_id))
        stage_ends = [e for e in events if e["kind"] == "stage_end"]
        assert len(stage_ends) >= 1
        assert stage_ends[0]["payload"]["success"] is False
