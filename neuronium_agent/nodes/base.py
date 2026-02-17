"""Unified node contract (IBS §6).

Every node implementation inherits from :class:`BaseNode` and implements
:meth:`execute`.  Inputs / outputs follow the typed I/O contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# I/O contracts
# ---------------------------------------------------------------------------

class NodeContext(BaseModel):
    """Execution context injected into every node."""

    execution_id: str
    trace_id: str
    retry_count: int = 0
    random_seed: int = 0


class NodeInput(BaseModel):
    """Unified node input (IBS §6.1)."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: NodeContext


class QualitySignals(BaseModel):
    """Optional quality / confidence signals produced by a node."""

    confidence: float | None = None
    tokens_used: int | None = None
    latency_ms: float | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class NodeOutput(BaseModel):
    """Unified node output (IBS §6.1)."""

    outputs: dict[str, Any] = Field(default_factory=dict)
    quality_signals: QualitySignals = Field(default_factory=QualitySignals)
    status: Literal[
        "COMPLETED", "FAILED", "TIMEOUT"
    ] = "COMPLETED"
    error: str | None = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseNode(ABC):
    """Abstract base for all DAG node types."""

    node_type: str = "base"

    def __init__(self, node_id: str, parameters: dict[str, Any] | None = None) -> None:
        self.node_id = node_id
        self.parameters = parameters or {}

    @abstractmethod
    def execute(self, node_input: NodeInput) -> NodeOutput:
        """Execute the node logic and return output."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.node_id!r}>"
