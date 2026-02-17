"""Public error hierarchy (PUBLIC_API_SPEC §5).

All public errors inherit from ``NeuroniumError``.
They are deterministically serialisable into trace events.
"""

from __future__ import annotations

from typing import Any


class NeuroniumError(Exception):
    """Base exception for the neuronium-agent library."""

    def __init__(
        self,
        message: str = "",
        *,
        node_ref: str | None = None,
        trace_id: str | None = None,
        classification: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.node_ref = node_ref
        self.trace_id = trace_id
        self.classification = classification
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialise for trace / audit logging."""
        return {
            "type": type(self).__name__,
            "message": str(self),
            "node_ref": self.node_ref,
            "trace_id": self.trace_id,
            "classification": self.classification,
            "details": self.details,
        }


class ConfigError(NeuroniumError):
    """Invalid or missing configuration."""


class ValidationError(NeuroniumError):
    """Data validation failure (pydantic / JSON-Schema boundary)."""


class StorageError(NeuroniumError):
    """Blob or index store error."""


class McpError(NeuroniumError):
    """MCP server communication or policy error."""


class SandboxError(NeuroniumError):
    """CodeNode Docker sandbox error."""


class ReplayError(NeuroniumError):
    """Trace replay mismatch or missing data."""
