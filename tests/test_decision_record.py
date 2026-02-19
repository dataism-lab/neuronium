"""B4: Formal Decision Record (§10.1.1) — unit and integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuronium_agent.api import AgentRunner
from neuronium_agent.config import AppConfig, ProjectConfig, StorageConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.trace.decision_record import (
    DecisionAuthority,
    DecisionRecord,
    DecisionType,
    OptionConsidered,
    OutcomeCorrelation,
    SelectedOption,
)
from neuronium_agent.trace.recorder import TraceRecorder
from neuronium_agent.types import RunRequest


# ---------------------------------------------------------------------------
# Replay seed for deterministic run (minimal pass, same shape as test_checkpoints)
# ---------------------------------------------------------------------------

_ITER1_PASS: dict[str, list[dict]] = {
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


def _make_runner(
    tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    replay_map: dict[str, list[dict]],
) -> AgentRunner:
    """Build an AgentRunner with monkey-patched replay (same pattern as test_checkpoints_and_control)."""
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
        for nid, node in registry.items():
            if hasattr(node, "set_replay_responses") and nid in replay_map:
                node.set_replay_responses(replay_map[nid])
        return registry

    runner._orchestrator._build_node_registry = patched_build  # type: ignore[method-assign]
    return runner


# ===================================================================
# Unit: DecisionRecord model
# ===================================================================


class TestDecisionRecordModel:
    """DecisionRecord serialization and from_legacy per §10.1.1."""

    def test_to_payload_has_required_keys(self) -> None:
        rec = DecisionRecord(
            decision_type=DecisionType.PLANNING,
            selected_option=SelectedOption(
                option_id="runbook_1",
                selection_rationale="Runbook selected",
                decision_authority=DecisionAuthority.COMPONENT,
            ),
        )
        payload = rec.to_payload()
        dr = payload["decisionRecord"]
        assert "id" in dr
        assert "timestamp" in dr
        assert dr["decisionType"] == "planning"
        assert dr["selectedOption"]["optionId"] == "runbook_1"
        assert dr["selectedOption"]["selectionRationale"] == "Runbook selected"
        assert dr["selectedOption"]["decisionAuthority"] == "component"

    def test_to_payload_options_considered(self) -> None:
        rec = DecisionRecord(
            decision_type=DecisionType.ADAPTATION,
            options_considered=[
                OptionConsidered(option_id="RETRY", description="Retry"),
                OptionConsidered(option_id="ESCALATE", description="Escalate"),
            ],
            selected_option=SelectedOption(
                option_id="RETRY",
                selection_rationale="Retry chosen",
                decision_authority=DecisionAuthority.COMPONENT,
            ),
        )
        payload = rec.to_payload()
        opts = payload["decisionRecord"].get("optionsConsidered", [])
        assert len(opts) == 2
        assert opts[0]["optionId"] == "RETRY"
        assert opts[1]["optionId"] == "ESCALATE"

    def test_to_payload_outcome_correlation(self) -> None:
        rec = DecisionRecord(
            decision_type=DecisionType.ESCALATION,
            selected_option=SelectedOption(
                option_id="ESCALATE",
                selection_rationale="Escalated",
                decision_authority=DecisionAuthority.COMPONENT,
            ),
            outcome_correlation=OutcomeCorrelation(
                actual_outcome="escalation",
                quality_assessment="failure",
            ),
        )
        payload = rec.to_payload()
        oc = payload["decisionRecord"].get("outcomeCorrelation")
        assert oc is not None
        assert oc["actualOutcome"] == "escalation"
        assert oc["qualityAssessment"] == "failure"

    def test_from_legacy_builds_minimal_record(self) -> None:
        rec = DecisionRecord.from_legacy(
            "Runbook selected",
            {"runbook_id": "autofix_demo"},
            decision_type=DecisionType.PLANNING,
        )
        assert rec.decision_type == DecisionType.PLANNING
        assert rec.selected_option.option_id == "autofix_demo"
        assert rec.selected_option.selection_rationale == "Runbook selected"

    def test_from_legacy_fallback_option_id(self) -> None:
        rec = DecisionRecord.from_legacy("Unknown event", {}, decision_type=DecisionType.CONTROL)
        assert rec.selected_option.option_id == "legacy"


# ===================================================================
# Unit: TraceRecorder produces decisionRecord in payload
# ===================================================================


class TestRecorderDecisionPayload:
    """TraceRecorder.record_decision writes decisionRecord per §10.1.1."""

    def test_legacy_call_produces_decision_record(self, tmp_path: Path) -> None:
        store = SqliteIndexStore(str(tmp_path / "idx.db"))
        rec = TraceRecorder("trace-1", store)
        rec.record_decision("Runbook selected", {"runbook_id": "docs_report"})

        events = rec.load_events()
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["description"] == "Runbook selected"
        assert payload["runbook_id"] == "docs_report"
        assert "decisionRecord" in payload
        dr = payload["decisionRecord"]
        assert dr["decisionType"] in ("planning", "control")
        assert "selectedOption" in dr
        assert dr["selectedOption"]["selectionRationale"] == "Runbook selected"
        assert "traceId" in dr
        assert dr["traceId"] == "trace-1"

    def test_record_param_produces_full_structure(self, tmp_path: Path) -> None:
        store = SqliteIndexStore(str(tmp_path / "idx.db"))
        rec = TraceRecorder("trace-2", store)
        record = DecisionRecord(
            decision_type=DecisionType.META_CONTROL,
            selected_option=SelectedOption(
                option_id="continue",
                selection_rationale="control_command: continue",
                decision_authority=DecisionAuthority.USER,
            ),
        )
        rec.record_decision("control_command: continue", {"command": "continue"}, record=record)

        events = rec.load_events()
        assert len(events) == 1
        dr = events[0]["payload"]["decisionRecord"]
        assert dr["decisionType"] == "meta-control"
        assert dr["selectedOption"]["optionId"] == "continue"
        assert dr["selectedOption"]["decisionAuthority"] == "user"


# ===================================================================
# Integration: run produces decision events with decisionRecord
# ===================================================================


class TestDecisionRecordInTrace:
    """End-to-end: trace contains decision events with formal decisionRecord."""

    def test_successful_run_has_decision_events_with_decision_record(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        handle = runner.start(RunRequest(objective="B4 decision record test"))

        events = list(runner._index.get_trace_events(handle.trace_id))
        decision_events = [e for e in events if e.get("kind") == "decision"]

        assert len(decision_events) >= 1, "At least one decision event expected"

        with_record = [e for e in decision_events if "decisionRecord" in e.get("payload", {})]
        assert len(with_record) == len(decision_events), (
            "Every decision event must have decisionRecord in payload"
        )

        for ev in decision_events:
            dr = ev["payload"]["decisionRecord"]
            assert "decisionType" in dr, "decisionRecord must have decisionType"
            assert "selectedOption" in dr, "decisionRecord must have selectedOption"
            assert "optionId" in dr["selectedOption"]
            assert "selectionRationale" in dr["selectedOption"]

    def test_runbook_selected_has_planning_type(
        self,
        tmp_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner = _make_runner(tmp_dir, monkeypatch, _ITER1_PASS)
        handle = runner.start(RunRequest(
            objective="B4 planning decision",
            metadata={"runbook_id": "autofix_demo"},
        ))

        events = list(runner._index.get_trace_events(handle.trace_id))
        decisions = [e for e in events if e.get("kind") == "decision"]

        runbook_decisions = [
            e for e in decisions
            if e["payload"].get("description", "").startswith("Runbook selected")
        ]
        assert len(runbook_decisions) >= 1
        dr = runbook_decisions[0]["payload"]["decisionRecord"]
        assert dr["decisionType"] == "planning"
        assert dr["selectedOption"]["optionId"] == "autofix_demo"
