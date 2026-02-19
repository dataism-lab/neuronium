"""End-to-end API tests — vertical slice through the system.

Uses replay-recorded responses (no external calls needed).
Updated for the 2-iteration autofix loop (3-node DAGs with critic).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner, create_runner
from neuronium_agent.config import AppConfig, StorageConfig, ProjectConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.types import RunRequest, RunStatus


# Deterministic replay responses that make the run succeed on iteration 1:
# generate → valid code, execute → success, critic → PASS with evidence.
_ITER1_RESPONSES: dict[str, list[dict]] = {
    "generate": [{
        "outputs": {"content": "print('hello from test')"},
        "quality_signals": {"tokens_used": 5},
    }],
    "execute": [{
        "outputs": {"stdout": "hello from test\n", "exit_code": 0},
        "quality_signals": {"latency_ms": 10.0},
        "status": "COMPLETED",
    }],
    "critic": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 1.0,
                "evidence": ["exit_code=0", "stdout matches expected output"],
                "gaps": [],
            }),
        },
        "quality_signals": {"tokens_used": 10},
    }],
}


class TestAgentRunnerWithReplay:
    """Test AgentRunner using pre-recorded ModelNode responses.

    This validates the full vertical slice:
    run → plan_iter1 → DAG → execute → critic → trace events → export.
    """

    @pytest.fixture()
    def runner(self, tmp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AgentRunner:
        # Provide a fake API key so the LLM-availability preflight check passes.
        # Actual LLM calls are never made — nodes use monkey-patched replay data.
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

        # Monkey-patch the orchestrator to inject replay into nodes
        orig_build = runner._orchestrator._build_node_registry

        def patched_build(graph, *, stage_default_model_id=None, **kwargs):
            registry = orig_build(graph, stage_default_model_id=stage_default_model_id, **kwargs)
            for nid, node in registry.items():
                if hasattr(node, "set_replay_responses") and nid in _ITER1_RESPONSES:
                    node.set_replay_responses(_ITER1_RESPONSES[nid])
            return registry

        runner._orchestrator._build_node_registry = patched_build
        return runner

    def test_run_completes(self, runner: AgentRunner) -> None:
        req = RunRequest(objective="Print hello")
        handle = runner.start(req)
        status = runner.get_status(handle)

        assert status.state == "COMPLETED"
        assert handle.trace_id

    def test_trace_events_recorded(self, runner: AgentRunner) -> None:
        req = RunRequest(objective="Print hello")
        handle = runner.start(req)

        events = list(runner._index.get_trace_events(handle.trace_id))
        assert len(events) > 0

        kinds = {e["kind"] for e in events}
        assert "decision" in kinds
        assert "node_start" in kinds
        assert "node_end" in kinds
        assert "critic_verdict" in kinds

    def test_export_trace_jsonl(
        self, runner: AgentRunner, tmp_dir: Path
    ) -> None:
        req = RunRequest(objective="Export test")
        handle = runner.start(req)

        export_path = tmp_dir / "trace.jsonl"
        runner.export_trace(handle, "jsonl", str(export_path))

        assert export_path.exists()
        lines = export_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) > 0

    def test_export_trace_json(
        self, runner: AgentRunner, tmp_dir: Path
    ) -> None:
        req = RunRequest(objective="Export JSON test")
        handle = runner.start(req)

        export_path = tmp_dir / "trace.json"
        runner.export_trace(handle, "json", str(export_path))

        import json as json_mod
        data = json_mod.loads(export_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_artifacts_persisted(self, runner: AgentRunner) -> None:
        req = RunRequest(objective="Artifact test")
        handle = runner.start(req)

        # There should be at least one artifact in the blob store
        # (we can check via the index store)
        events = list(runner._index.get_trace_events(handle.trace_id))
        assert any(e["kind"] == "node_end" for e in events)

    def test_get_trace_events_public_api(self, runner: AgentRunner) -> None:
        req = RunRequest(objective="Trace events method test")
        handle = runner.start(req)
        events = runner.get_trace_events(handle.trace_id)
        assert isinstance(events, list)
        assert any(e.get("kind") == "decision" for e in events)

    def test_get_latest_rendered_artifact_path(self, runner: AgentRunner) -> None:
        req = RunRequest(objective="Rendered path method test")
        handle = runner.start(req)
        rendered_path = runner.get_latest_rendered_artifact_path(handle.trace_id)
        assert rendered_path is not None
        assert Path(rendered_path).exists()
