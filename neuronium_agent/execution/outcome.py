"""Execution outcome type for interruptible DAG execution (PAUSE_CONTROL Phase 1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.types import InterruptRequest


class ExecutionOutcome(BaseModel):
    """Result of DAG execution when interrupt_check is enabled.

    When execution completes normally, ``pending`` is empty and ``interrupted`` is None.
    When execution is interrupted (pause/stop), ``interrupted`` is set and ``pending``
    holds the node ids that were not yet executed; ``results`` contains partial outputs.
    """

    results: dict[str, NodeOutput] = Field(
        default_factory=dict,
        description="Completed node outputs (partial if interrupted).",
    )
    pending: list[str] = Field(
        default_factory=list,
        description="Node ids not executed by the time of exit (empty on normal completion).",
    )
    interrupted: InterruptRequest | None = Field(
        default=None,
        description="Set when execution was stopped by interrupt_check (pause/stop).",
    )
