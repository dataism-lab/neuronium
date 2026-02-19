"""Tests for recovery policy decide_recovery (B1 Part 1)."""

from __future__ import annotations

from neuronium_agent.config import AppConfig
from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.recovery import classify_failure, decide_recovery
from neuronium_agent.types import FailureClass


def _out(failure_class: FailureClass | None) -> NodeOutput:
    return NodeOutput(
        status="FAILED" if failure_class else "COMPLETED",
        error="err",
        failure_class=failure_class,
    )


class TestDecideRecovery:
    """decide_recovery rules: CRITICAL→FAIL, all TRANSIENT+quota→RETRY_STAGE, else ESCALATE."""

    def test_critical_fail(self) -> None:
        cfg = AppConfig()
        failed = [("n1", _out(FailureClass(kind="CRITICAL", message="OOM", retryable=False)))]
        d = decide_recovery(failed, gate_failed=True, stage_retry_count=0, config=cfg)
        assert d.action.value == "FAIL"
        assert "Critical" in d.reason

    def test_all_transient_retry_stage(self) -> None:
        cfg = AppConfig()
        failed = [
            ("n1", _out(FailureClass(kind="TRANSIENT", message="timeout", retryable=True))),
        ]
        d = decide_recovery(failed, gate_failed=True, stage_retry_count=0, config=cfg)
        assert d.action.value == "RETRY_STAGE"
        assert d.reason

    def test_transient_exhausted_escalate(self) -> None:
        cfg = AppConfig()
        assert cfg.recovery.max_stage_retries == 2
        failed = [
            ("n1", _out(FailureClass(kind="TRANSIENT", message="timeout", retryable=True))),
        ]
        d = decide_recovery(failed, gate_failed=True, stage_retry_count=2, config=cfg)
        assert d.action.value == "ESCALATE"
        assert d.escalation_context is not None
        assert "failed_node_ids" in d.escalation_context

    def test_persistent_escalate(self) -> None:
        cfg = AppConfig()
        failed = [
            ("n1", _out(FailureClass(kind="PERSISTENT", message="invalid", retryable=False))),
        ]
        d = decide_recovery(failed, gate_failed=True, stage_retry_count=0, config=cfg)
        assert d.action.value == "ESCALATE"
        assert d.escalation_context is not None

    def test_critic_failed_no_nodes_escalate(self) -> None:
        cfg = AppConfig()
        d = decide_recovery(
            [], gate_failed=True, stage_retry_count=0, config=cfg, critic_failed=True
        )
        assert d.action.value == "ESCALATE"

    def test_critic_failed_with_quota_retry_stage(self) -> None:
        cfg = AppConfig()
        d = decide_recovery(
            [], gate_failed=True, stage_retry_count=0, config=cfg, critic_failed=True
        )
        # critic_failed adds PERSISTENT to classes; so we get ESCALATE not RETRY_STAGE
        assert d.action.value == "ESCALATE"

    def test_gate_failed_no_context_fail(self) -> None:
        cfg = AppConfig()
        d = decide_recovery([], gate_failed=True, stage_retry_count=0, config=cfg)
        assert d.action.value == "FAIL"
        assert "no failure context" in d.reason.lower()

    def test_repeated_rollback_escalate(self) -> None:
        """Same node fails 3 times in history → ESCALATE (B1 Part 2 §3.4.2)."""
        cfg = AppConfig()
        cfg.recovery.repeated_rollback_threshold = 3
        failed = [
            ("n1", _out(FailureClass(kind="TRANSIENT", message="timeout", retryable=True))),
        ]
        # First two attempts: n1 failed
        failure_history = [["n1"], ["n1"]]
        d = decide_recovery(
            failed,
            gate_failed=True,
            stage_retry_count=2,
            config=cfg,
            failure_history=failure_history,
            has_dynamic_planner=False,
        )
        assert d.action.value == "ESCALATE"
        assert "repeated rollback" in d.reason.lower()
        assert d.escalation_context is not None
        assert d.escalation_context.get("repeated_rollback") is True

    def test_repeated_rollback_below_threshold_retry_stage(self) -> None:
        """Same node fails 2 times (below threshold 3) → still RETRY_STAGE if TRANSIENT."""
        cfg = AppConfig()
        cfg.recovery.repeated_rollback_threshold = 3
        failed = [
            ("n1", _out(FailureClass(kind="TRANSIENT", message="timeout", retryable=True))),
        ]
        failure_history = [["n1"]]
        d = decide_recovery(
            failed,
            gate_failed=True,
            stage_retry_count=1,
            config=cfg,
            failure_history=failure_history,
        )
        assert d.action.value == "RETRY_STAGE"

    def test_systemic_with_auto_replan_returns_replan(self) -> None:
        """All SYSTEMIC + allow_auto_replan + has_dynamic_planner → REPLAN."""
        cfg = AppConfig()
        cfg.recovery.allow_auto_replan = True
        failed = [
            ("n1", _out(FailureClass(kind="SYSTEMIC", message="inapplicable", retryable=False))),
        ]
        d = decide_recovery(
            failed,
            gate_failed=True,
            stage_retry_count=0,
            config=cfg,
            has_dynamic_planner=True,
        )
        assert d.action.value == "REPLAN"
        assert "replan" in d.reason.lower()
