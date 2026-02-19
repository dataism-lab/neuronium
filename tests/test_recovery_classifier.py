"""Tests for failure classification (B1 Part 1)."""

from __future__ import annotations

import pytest

from neuronium_agent.recovery.classifier import classify_failure


class TestClassifyFailure:
    """classify_failure heuristics per spec §5.5.3."""

    def test_timeout_transient(self) -> None:
        fc = classify_failure(
            "n1", "model", "FAILED", "Request timed out after 60s", 0
        )
        assert fc.kind == "TRANSIENT"
        assert fc.retryable is True

    def test_rate_limit_transient(self) -> None:
        fc = classify_failure(
            "n1", "mcp", "FAILED", "Rate limit exceeded", 0
        )
        assert fc.kind == "TRANSIENT"
        assert fc.retryable is True

    def test_permission_denied_persistent(self) -> None:
        fc = classify_failure(
            "n1", "code", "FAILED", "Permission denied", 0
        )
        assert fc.kind == "CRITICAL"
        assert fc.retryable is False

    def test_high_retry_count_upgrade_to_persistent(self) -> None:
        fc = classify_failure(
            "n1", "model", "FAILED", "Request timed out", 2
        )
        assert fc.kind == "PERSISTENT"
        assert fc.retryable is False

    def test_invalid_parameter_persistent(self) -> None:
        fc = classify_failure(
            "n1", "model", "FAILED", "Invalid parameter: foo", 0
        )
        assert fc.kind == "PERSISTENT"
        assert fc.retryable is False

    def test_no_implementation_registered_persistent(self) -> None:
        fc = classify_failure(
            "n1", "mcp", "FAILED", "No implementation registered for node x", 0
        )
        assert fc.kind == "PERSISTENT"
        assert fc.retryable is False

    def test_oom_critical(self) -> None:
        fc = classify_failure(
            "n1", "code", "FAILED", "Container killed: OOM", 0
        )
        assert fc.kind == "CRITICAL"
        assert fc.retryable is False

    def test_unknown_default_persistent(self) -> None:
        fc = classify_failure(
            "n1", "model", "FAILED", "Some unknown error", 0
        )
        assert fc.kind == "PERSISTENT"
        assert fc.retryable is False

    def test_empty_error_persistent(self) -> None:
        fc = classify_failure("n1", "code", "FAILED", None, 0)
        assert fc.kind == "PERSISTENT"
        assert fc.message
