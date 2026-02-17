"""Public DTO types (PUBLIC_API_SPEC §2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Input for starting an agent run."""

    objective: str
    constraints: list[str] = Field(default_factory=list)
    mode: Literal["batch", "supervised"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunHandle(BaseModel):
    """Opaque handle returned after starting a run."""

    trace_id: str
    execution_id: str
    created_at: datetime


class RunStatus(BaseModel):
    """Current status of a run."""

    state: Literal[
        "PENDING", "RUNNING", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"
    ]
    progress: float | None = None
    current_node_ref: str | None = None
    message: str | None = None


class ControlCommand(BaseModel):
    """User control command (Control Protocol §11)."""

    type: Literal["continue", "pause", "revise", "replan", "stop", "escalate"]
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

TraceExportFormat = Literal["jsonl", "json", "zip"]


# ---------------------------------------------------------------------------
# Internal-but-shared types (used across modules)
# ---------------------------------------------------------------------------

class NodeStatus(BaseModel):
    """Status of a single node execution (IBS §6.2)."""

    status: Literal[
        "PENDING", "READY", "RUNNING", "COMPLETED",
        "FAILED", "TIMEOUT", "RETRYING", "CANCELLED",
    ] = "PENDING"


class FailureClass(BaseModel):
    """Failure classification (IBS §6.3)."""

    kind: Literal["TRANSIENT", "PERSISTENT", "SYSTEMIC", "CRITICAL"]
    message: str = ""
    retryable: bool = False
