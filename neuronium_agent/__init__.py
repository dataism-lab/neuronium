"""neuronium_agent — Commitment-aware AI Super Agent library."""

__version__ = "0.1.0"

# Public re-exports (lazy, to avoid heavy imports at package level)
from neuronium_agent.types import (
    ControlCommand,
    InterruptRequest,
    RunHandle,
    RunRequest,
    RunStatus,
    TraceExportFormat,
)
from neuronium_agent.errors import (
    ConfigError,
    McpError,
    NeuroniumError,
    ReplayError,
    SandboxError,
    StorageError,
    ValidationError,
)

__all__ = [
    # DTO
    "RunRequest",
    "RunHandle",
    "RunStatus",
    "ControlCommand",
    "InterruptRequest",
    "TraceExportFormat",
    # Errors
    "NeuroniumError",
    "ConfigError",
    "ValidationError",
    "StorageError",
    "McpError",
    "SandboxError",
    "ReplayError",
]
