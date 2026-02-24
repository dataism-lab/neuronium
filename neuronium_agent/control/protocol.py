"""Control Protocol (IBS §11).

Handles user commands: continue, pause, revise, replan, stop.
v1: basic command handling.  NL feedback → control signals is future work.
"""

from __future__ import annotations

import logging
from typing import Any

from neuronium_agent.types import ControlCommand

logger = logging.getLogger(__name__)


class ControlProtocol:
    """Process user control commands (IBS §11.1).

    In v1, commands are applied directly to the run state.
    Future: NL feedback parsing, clarification requests.
    """

    def classify_command(self, command: ControlCommand) -> dict[str, Any]:
        """Classify a control command into an internal signal.

        Returns a dict with:
        - ``action``: the resolved action (same as command.type in v1)
        - ``confidence``: how confident the classification is (1.0 in v1)
        """
        return {
            "action": command.type,
            "confidence": 1.0,
            "payload": command.payload,
        }

    def classify_feedback_text(self, feedback_text: str) -> dict[str, Any]:
        """Classify NL feedback into a control intent envelope (IBS §9.2)."""
        text = str(feedback_text).strip()
        if not text:
            return {"action": "clarify", "confidence": 0.0, "payload": {}}
        return {
            "action": "revise",
            "confidence": 0.5,
            "payload": {"answer_text": text},
        }

    def needs_clarification(self, command: ControlCommand) -> bool:
        """Whether the command is ambiguous and needs clarification.

        v1: always returns False (direct command mapping).
        """
        return False
