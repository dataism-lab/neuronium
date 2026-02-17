"""Agent state model and intention lifecycle (IBS §1.1, ROADMAP Stage 2).

Cognitive Core cycle: Commit → Execute → Control → Adapt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IntentionPhase(str, Enum):
    """Phase within the Commit/Execute/Control/Adapt cycle."""

    COMMIT = "COMMIT"
    EXECUTE = "EXECUTE"
    CONTROL = "CONTROL"
    ADAPT = "ADAPT"
    DONE = "DONE"
    FAILED = "FAILED"


class RunState(str, Enum):
    """Top-level run state (PUBLIC_API_SPEC §2.1)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Intention(BaseModel):
    """An agent intention (objective + plan ref)."""

    intention_id: str
    objective: str
    constraints: list[str] = Field(default_factory=list)
    plan_id: str | None = None
    phase: IntentionPhase = IntentionPhase.COMMIT


class AgentState(BaseModel):
    """Full snapshot of agent state at a point in time."""

    trace_id: str
    execution_id: str
    run_state: RunState = RunState.PENDING
    intention: Intention | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    progress: float = 0.0
    current_node_ref: str | None = None
    message: str | None = None

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition_to(self, new_state: RunState, message: str | None = None) -> None:
        """Transition the run to a new state (with basic validation)."""
        valid_transitions: dict[RunState, set[RunState]] = {
            RunState.PENDING: {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED},
            RunState.RUNNING: {
                RunState.PAUSED,
                RunState.COMPLETED,
                RunState.FAILED,
                RunState.CANCELLED,
            },
            RunState.PAUSED: {RunState.RUNNING, RunState.CANCELLED},
            RunState.COMPLETED: set(),
            RunState.FAILED: set(),
            RunState.CANCELLED: set(),
        }
        allowed = valid_transitions.get(self.run_state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Cannot transition from {self.run_state} to {new_state}"
            )
        self.run_state = new_state
        if message:
            self.message = message
