"""Recovery policy: decide RETRY_STAGE / ESCALATE / FAIL / REPLAN (B1 Part 1 + Part 2, §3.4).

After a stage gate failure, chooses whether to retry the stage, escalate to user
(PAUSED), fail the run, or auto-replan. Resume after ESCALATE is via control(continue)
loading the phase-boundary checkpoint and continuing from the same stage.
"""

from __future__ import annotations

from collections import Counter

from neuronium_agent.config import AppConfig
from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.recovery.models import (
    RecoveryAction,
    RecoveryDecision,
    RollbackScope,
)


def decide_recovery(
    failed_nodes: list[tuple[str, NodeOutput]],
    gate_failed: bool,
    stage_retry_count: int,
    config: AppConfig,
    *,
    critic_failed: bool = False,
    failure_history: list[list[str]] | None = None,
    rollback_scope: RollbackScope | None = None,
    has_dynamic_planner: bool = False,
) -> RecoveryDecision:
    """Decide recovery action after stage gate failure.

    Rules (B1 Part 1 + Part 2):
    - Any CRITICAL → FAIL.
    - Repeated rollback (same node fails >= repeated_rollback_threshold times) → ESCALATE or REPLAN.
    - Else if all TRANSIENT and stage_retry_count < max_stage_retries → RETRY_STAGE.
    - Else if any PERSISTENT or stage_retry_count exhausted → ESCALATE (or REPLAN if SYSTEMIC + allow_auto_replan).
    - Else → FAIL.
    """
    max_stage = config.recovery.max_stage_retries
    current_ids = [nid for nid, _ in failed_nodes]

    # Build list of failure classes (including virtual "critic failed" as PERSISTENT)
    classes: list[str] = []
    for _nid, out in failed_nodes:
        if out.failure_class:
            classes.append(out.failure_class.kind)
    if critic_failed and not failed_nodes:
        classes.append("PERSISTENT")

    if "CRITICAL" in classes:
        return RecoveryDecision(
            action=RecoveryAction.FAIL,
            reason="Critical failure; cannot retry",
            rollback_scope=rollback_scope,
        )

    # B1 Part 2: repeated rollback (§3.4.2)
    all_attempts: list[list[str]] = list(failure_history) if failure_history else []
    all_attempts.append(current_ids)
    counts: Counter[str] = Counter()
    for attempt in all_attempts:
        for nid in attempt:
            counts[nid] += 1
    threshold = config.recovery.repeated_rollback_threshold
    if any(c >= threshold for c in counts.values()):
        if config.recovery.allow_auto_replan and has_dynamic_planner:
            return RecoveryDecision(
                action=RecoveryAction.REPLAN,
                reason="Repeated rollback; auto-replan",
                rollback_scope=rollback_scope,
            )
        return RecoveryDecision(
            action=RecoveryAction.ESCALATE,
            reason="Repeated rollback; same node(s) failed repeatedly (§3.4.2)",
            escalation_context={
                "failure_classes": classes,
                "stage_retry_count": stage_retry_count,
                "failed_node_ids": current_ids,
                "repeated_rollback": True,
            },
            rollback_scope=rollback_scope,
        )

    if not classes:
        return RecoveryDecision(
            action=RecoveryAction.FAIL,
            reason="Gate failed with no failure context",
            rollback_scope=rollback_scope,
        )

    all_transient = all(c == "TRANSIENT" for c in classes)
    if all_transient and stage_retry_count < max_stage:
        return RecoveryDecision(
            action=RecoveryAction.RETRY_STAGE,
            reason=f"All failures transient; retrying stage (attempt {stage_retry_count + 1}/{max_stage})",
            rollback_scope=rollback_scope,
        )

    # PERSISTENT, SYSTEMIC, or stage retries exhausted — ESCALATE or REPLAN
    if (
        config.recovery.allow_auto_replan
        and has_dynamic_planner
        and all(c == "SYSTEMIC" for c in classes)
    ):
        return RecoveryDecision(
            action=RecoveryAction.REPLAN,
            reason="Systemic failure; auto-replan",
            rollback_scope=rollback_scope,
        )
    return RecoveryDecision(
        action=RecoveryAction.ESCALATE,
        reason="Persistent or systemic failure, or stage retries exhausted; escalating to user",
        escalation_context={
            "failure_classes": classes,
            "stage_retry_count": stage_retry_count,
            "failed_node_ids": current_ids,
        },
        rollback_scope=rollback_scope,
    )
