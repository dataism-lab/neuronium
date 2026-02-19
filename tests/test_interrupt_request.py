"""InterruptRequest contract tests (PAUSE_CONTROL_IMPLEMENTATION_PLAN §0.1).

Type exists and is used in tests — acceptance for Phase 0.1.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuronium_agent.types import InterruptRequest


class TestInterruptRequest:
    """InterruptRequest: construction, defaults, validation."""

    def test_pause_default_graceful(self) -> None:
        req = InterruptRequest(command="pause")
        assert req.command == "pause"
        assert req.mode == "graceful"
        assert req.export_path is None

    def test_stop_explicit_graceful(self) -> None:
        req = InterruptRequest(command="stop", mode="graceful")
        assert req.command == "stop"
        assert req.mode == "graceful"

    def test_stop_immediate(self) -> None:
        req = InterruptRequest(command="stop", mode="immediate")
        assert req.command == "stop"
        assert req.mode == "immediate"

    def test_export_path_optional(self) -> None:
        req = InterruptRequest(
            command="stop",
            mode="graceful",
            export_path="/tmp/trace.zip",
        )
        assert req.export_path == "/tmp/trace.zip"

    def test_invalid_command_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InterruptRequest(command="continue")  # type: ignore[arg-type]

    def test_import_from_package(self) -> None:
        """InterruptRequest is re-exported from neuronium_agent (public API)."""
        from neuronium_agent import InterruptRequest as IR

        req = IR(command="pause")
        assert req.command == "pause"
