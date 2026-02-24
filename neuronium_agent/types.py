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
    mode: Literal["batch", "supervised", "interactive"] | None = None
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
    # revise payload (phase 3): prefers `patch` (RFC6902 subset ops),
    # keeps legacy `answers` temporarily for backward compatibility bridge.
    payload: dict[str, Any] = Field(default_factory=dict)


class InterruptRequest(BaseModel):
    """Internal contract for pause/stop (PAUSE_CONTROL_IMPLEMENTATION_PLAN §0.1).

    Used by orchestrator and executor to agree on interrupt semantics.
    For pause, mode is effectively always graceful per spec §9.1.2.
    For stop, mode selects graceful (wait for checkpoints) vs immediate (abort).
    v1: immediate does not cancel in-flight nodes; executor exits after current
    batch (same as graceful). Difference is checkpoint size only.
    """

    command: Literal["pause", "stop"]
    mode: Literal["graceful", "immediate"] = "graceful"
    export_path: str | None = None


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
