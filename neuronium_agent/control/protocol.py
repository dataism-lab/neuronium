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

    def needs_clarification(self, command: ControlCommand) -> bool:
        """Whether the command is ambiguous and needs clarification.

        v1: always returns False (direct command mapping).
        """
        return False
