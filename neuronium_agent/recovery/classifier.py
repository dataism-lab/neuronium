"""Failure classification (B1 Part 1, §3.4 / IBS §5.5.3).

Classifies node failures for recovery policy: TRANSIENT / PERSISTENT / SYSTEMIC / CRITICAL.
"""

from __future__ import annotations

from neuronium_agent.types import FailureClass


def classify_failure(
    node_id: str,
    node_type: str,
    status: str,
    error: str | None,
    retry_count: int,
    *,
    retry_count_upgrade_threshold: int = 2,
) -> FailureClass:
    """Classify a node failure for recovery policy.

    Heuristics per spec §5.5.3:
    - TRANSIENT: likely success on retry (timeout, rate limit, temporary unavailability).
    - PERSISTENT: consistent failure (invalid params, missing deps, logic errors).
    - SYSTEMIC: fundamental approach flaw (method inapplicable, constraint unsatisfiability).
    - CRITICAL: safety violation or resource exhaustion.

    When retry_count >= retry_count_upgrade_threshold, we upgrade to PERSISTENT to stop retrying.
    """
    error_lower = (error or "").lower()

    # CRITICAL: safety / resource exhaustion
    if any(
        phrase in error_lower
        for phrase in (
            "permission denied",
            "access denied",
            "security",
            "oom",
            "out of memory",
            "memory limit",
            "killed",
            "sigkill",
        )
    ):
        return FailureClass(
            kind="CRITICAL",
            message=error or "Critical failure",
            retryable=False,
        )

    # Too many retries already → treat as PERSISTENT
    if retry_count >= retry_count_upgrade_threshold:
        return FailureClass(
            kind="PERSISTENT",
            message=error or f"Failed after {retry_count} retries",
            retryable=False,
        )

    # TRANSIENT: network/timeouts/rate limits
    if any(
        phrase in error_lower
        for phrase in (
            "timeout",
            "timed out",
            "rate limit",
            "rate_limit",
            "too many requests",
            "connection refused",
            "connection reset",
            "temporary",
            "unavailable",
            "503",
            "502",
            "504",
        )
    ):
        return FailureClass(
            kind="TRANSIENT",
            message=error or "Transient failure",
            retryable=True,
        )

    # PERSISTENT: auth, invalid input, logic
    if any(
        phrase in error_lower
        for phrase in (
            "invalid",
            "validation",
            "authentication",
            "unauthorized",
            "forbidden",
            "not found",
            "no such file",
            "syntax error",
            "typeerror",
            "valueerror",
            "keyerror",
            "no implementation registered",
        )
    ):
        return FailureClass(
            kind="PERSISTENT",
            message=error or "Persistent failure",
            retryable=False,
        )

    # SYSTEMIC: structural / approach
    if any(
        phrase in error_lower
        for phrase in (
            "constraint",
            "unsatisfiable",
            "inapplicable",
            "deadlock",
            "cycle",
        )
    ):
        return FailureClass(
            kind="SYSTEMIC",
            message=error or "Systemic failure",
            retryable=False,
        )

    # Default: unknown → PERSISTENT (do not retry indefinitely)
    return FailureClass(
        kind="PERSISTENT",
        message=error or "Unclassified failure",
        retryable=False,
    )
