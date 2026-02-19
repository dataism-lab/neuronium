"""B2 Part 1: Verdict-driven local fix — unit and integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, RecoveryConfig, StorageConfig
from neuronium_agent.execution.executor import _build_default_prompt, _build_fix_prompt
from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphEdge, GraphMetadata, GraphNode
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)
from neuronium_agent.planning.runbook_registry import register_runbook
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.types import RunRequest
from neuronium_agent.verification.demo_critic import (
    DemoCriticVerdict,
    parse_critic_verdict,
    critic_json_schema,
)


class TestDemoCriticVerdictSuggestions:
    """DemoCriticVerdict supports optional suggestions (B2 §7.2.2)."""

    def test_suggestions_default_empty(self) -> None:
        v = DemoCriticVerdict(verdict="FAIL", confidence=0.8, evidence=[], gaps=["g1"])
        assert v.suggestions == []

    def test_suggestions_parsed_from_dict(self) -> None:
        v = DemoCriticVerdict(
            verdict="FAIL",
            confidence=0.8,
            evidence=[],
            gaps=["g1"],
            suggestions=[{"action": "add validation", "expected_improvement": "correctness"}],
        )
        assert len(v.suggestions) == 1
        assert v.suggestions[0]["action"] == "add validation"

    def test_parse_old_json_without_suggestions(self) -> None:
        raw = '{"verdict": "FAIL", "confidence": 0.5, "evidence": [], "gaps": ["missing check"]}'
        v = parse_critic_verdict(raw)
        assert v.verdict == "FAIL"
        assert v.gaps == ["missing check"]
        assert v.suggestions == []

    def test_critic_json_schema_suggestions_strict(self) -> None:
        schema = critic_json_schema()
        assert "suggestions" in schema.get("properties", {})
        required = schema.get("required", [])
        assert "suggestions" in required
        items = schema["properties"]["suggestions"]["items"]
        assert items["additionalProperties"] is False


class TestExecutorVerdictFixPrompt:
    """Executor injects verdict_fix (gaps/suggestions) into prompts (B2)."""

    def test_build_default_prompt_includes_verdict_fix(self) -> None:
        inputs = {
            "objective": "Do X",
            "verdict_fix": {
                "gaps": ["Gap one", "Gap two"],
                "suggestions": [
                    {"action": "Add check", "expected_improvement": "Fewer errors"},
                ],
            },
        }
        prompt = _build_default_prompt(inputs)
        assert "PREVIOUS ATTEMPT FEEDBACK" in prompt
        assert "Gap one" in prompt
        assert "Gap two" in prompt
        assert "Add check" in prompt
        assert "expected" in prompt or "Fewer errors" in prompt
        assert "verdict_fix" not in prompt  # excluded from CONTEXT

    def test_build_default_prompt_verdict_fix_excluded_from_context(self) -> None:
        inputs = {
            "objective": "Do X",
            "verdict_fix": {"gaps": ["g1"], "suggestions": []},
            "other_key": "other_val",
        }
        prompt = _build_default_prompt(inputs)
        assert "PREVIOUS ATTEMPT FEEDBACK" in prompt
        assert "other_key" in prompt
        assert "other_val" in prompt

    def test_build_fix_prompt_uses_verdict_fix_gaps(self) -> None:
        inputs = {
            "objective": "Fix code",
            "previous_code": "x=1",
            "verdict_fix": {"gaps": ["Use type hints"], "suggestions": []},
        }
        prompt = _build_fix_prompt(inputs)
        assert "Use type hints" in prompt
        assert "GAPS IDENTIFIED" in prompt


# --- Integration: verdict local fix path --------------------------------------


class CriticFailThenPassNode(BaseNode):
    """Critic node: first call returns FAIL with gaps, second call returns PASS with evidence.

    Uses the same list reference (no copy) so that when the stage is retried and a new
    registry is built, the second node gets the remaining response (PASS).
    """

    def __init__(self, node_id: str, responses: list[NodeOutput]) -> None:
        super().__init__(node_id)
        self._responses = responses  # shared list: first run pops FAIL, second run gets PASS

    def execute(self, node_input: NodeInput) -> NodeOutput:
        if self._responses:
            return self._responses.pop(0)
        return NodeOutput(
            status="COMPLETED",
            outputs={
                "content": json.dumps({
                    "verdict": "PASS",
                    "confidence": 1.0,
                    "evidence": ["ok"],
                    "gaps": [],
                    "suggestions": [],
                }),
                "parsed": {
                    "verdict": "PASS",
                    "confidence": 1.0,
                    "evidence": ["ok"],
                    "gaps": [],
                    "suggestions": [],
                },
            },
        )


class VerdictFixTestRunbook(Runbook):
    """Single stage: one model node + one critic node; gate requires both and critic PASS."""

    @property
    def runbook_id(self) -> str:
        return "verdict_fix_test_v1"

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        plan_id = f"plan-verdict-fix-{execution_id[:8]}"
        graph = ActionGraph(
            metadata=GraphMetadata(plan_id=plan_id, description="Verdict fix test"),
            nodes=[
                GraphNode(node_id="task", node_type="model", label="Task", priority=0),
                GraphNode(node_id="critic", node_type="model", label="Critic", priority=1),
            ],
            edges=[GraphEdge(source="task", target="critic", edge_type="data")],
        )
        return [
            ActionGraphStage(
                stage_id="verdict_fix_test:stage1",
                graph=graph,
                initial_inputs_override={},
                success_gate=StageSuccessGate(
                    required_completed_nodes=["task"],
                    critic_node_id="critic",
                ),
            ),
        ]


def _make_runner_b2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_verdict_fix_attempts: int = 1,
) -> AgentRunner:
    monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-fake-key")
    config = AppConfig(
        project=ProjectConfig(name="test", data_dir=str(tmp_path / ".n")),
        storage=StorageConfig(
            fs_cas_root=str(tmp_path / "blobs"),
            sqlite_path=str(tmp_path / "index.sqlite3"),
        ),
        recovery=RecoveryConfig(
            max_node_retries=1,
            max_stage_retries=2,
            max_verdict_fix_attempts=max_verdict_fix_attempts,
        ),
    )
    blob = FsCasStore(config.storage.fs_cas_root)
    idx = SqliteIndexStore(config.storage.sqlite_path)
    return AgentRunner(config, blob, idx)


class TestVerdictLocalFixIntegration:
    """B2 Part 1: On critic FAIL with gaps, verdict_local_fix_retry is taken then stage can pass."""

    def test_verdict_local_fix_retry_then_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        register_runbook(VerdictFixTestRunbook())
        runner = _make_runner_b2(tmp_path, monkeypatch, max_verdict_fix_attempts=1)

        fail_verdict = {
            "verdict": "FAIL",
            "confidence": 0.8,
            "evidence": [],
            "gaps": ["Improve the output quality"],
            "suggestions": [{"action": "Add more detail", "expected_improvement": "Clarity"}],
        }
        pass_verdict = {
            "verdict": "PASS",
            "confidence": 1.0,
            "evidence": ["Meets criteria"],
            "gaps": [],
            "suggestions": [],
        }

        critic_responses = [
            NodeOutput(
                status="COMPLETED",
                outputs={
                    "content": json.dumps(fail_verdict),
                    "parsed": fail_verdict,
                },
            ),
            NodeOutput(
                status="COMPLETED",
                outputs={
                    "content": json.dumps(pass_verdict),
                    "parsed": pass_verdict,
                },
            ),
        ]

        orig_build = runner._orchestrator._build_node_registry

        def patched_build(
            graph: ActionGraph, *, stage_default_model_id=None, **kwargs
        ) -> dict[str, BaseNode]:
            registry = orig_build(
                graph, stage_default_model_id=stage_default_model_id, **kwargs
            )
            if "task" in registry:
                registry["task"] = FailingThenSucceedingTaskNode("task")
            if "critic" in registry:
                registry["critic"] = CriticFailThenPassNode("critic", critic_responses)
            return registry

        runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]

        handle = runner.start(
            RunRequest(
                objective="Verdict fix test",
                metadata={"runbook_id": "verdict_fix_test_v1"},
            ),
        )

        events = list(runner.get_trace_events(handle.trace_id))
        verdict_retry_events = [e for e in events if e.get("kind") == "verdict_local_fix_retry"]
        assert len(verdict_retry_events) >= 1, "Expected at least one verdict_local_fix_retry"
        payload = verdict_retry_events[0]["payload"]
        assert payload.get("stage_id") == "verdict_fix_test:stage1"
        assert "verdict_fix" in payload
        assert payload["verdict_fix"].get("gaps") == ["Improve the output quality"]

        stage_ends = [e for e in events if e.get("kind") == "stage_end"]
        assert len(stage_ends) >= 1
        assert stage_ends[-1]["payload"]["success"] is True

        status = runner.get_status(handle)
        assert status.state == "COMPLETED"


class FailingThenSucceedingTaskNode(BaseNode):
    """Task node: first call COMPLETED, second call COMPLETED (so stage can pass both runs)."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self._call_count = 0

    def execute(self, node_input: NodeInput) -> NodeOutput:
        self._call_count += 1
        return NodeOutput(
            status="COMPLETED",
            outputs={"content": f"result-{self._call_count}"},
        )
