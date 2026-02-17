"""Schema registry — canonical list of Pydantic models whose JSON Schemas
are exported as Stage 1 deliverables.

Each entry maps a short stable name to its Pydantic model class.
The name is used as the filename (``<name>.schema.json``).
"""

from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

# -- Domain models -----------------------------------------------------------
from neuronium_agent.core.state import AgentState, Intention
from neuronium_agent.planning.dag import (
    ActionGraph,
    ConditionalBranch,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)

# -- Node contracts ----------------------------------------------------------
from neuronium_agent.nodes.base import (
    NodeContext,
    NodeInput,
    NodeOutput,
    QualitySignals,
)

# -- Memory ------------------------------------------------------------------
from neuronium_agent.memory.models import (
    ChunkLocator,
    EvidenceRef,
    MemoryIngestRequest,
    MemoryIngestResult,
    MemoryQuery,
    MemoryQueryConstraints,
    MemoryQueryStats,
    MemoryResult,
    RetrievedChunk,
)

# -- Verification ------------------------------------------------------------
from neuronium_agent.verification.critic import (
    CriticInput,
    CriticVerdict,
    Evidence,
)
from neuronium_agent.verification.demo_critic import DemoCriticVerdict

# -- Public API types --------------------------------------------------------
from neuronium_agent.types import (
    ControlCommand,
    FailureClass,
    NodeStatus,
    RunHandle,
    RunRequest,
    RunStatus,
)


SCHEMA_REGISTRY: dict[str, Type[BaseModel]] = {
    # Core state
    "AgentState": AgentState,
    "Intention": Intention,
    # Planning / DAG
    "ActionGraph": ActionGraph,
    "GraphNode": GraphNode,
    "GraphEdge": GraphEdge,
    "GraphMetadata": GraphMetadata,
    "ConditionalBranch": ConditionalBranch,
    # Node I/O
    "NodeContext": NodeContext,
    "NodeInput": NodeInput,
    "NodeOutput": NodeOutput,
    "QualitySignals": QualitySignals,
    # Memory
    "ChunkLocator": ChunkLocator,
    "EvidenceRef": EvidenceRef,
    "MemoryIngestRequest": MemoryIngestRequest,
    "MemoryIngestResult": MemoryIngestResult,
    "MemoryQuery": MemoryQuery,
    "MemoryQueryConstraints": MemoryQueryConstraints,
    "MemoryQueryStats": MemoryQueryStats,
    "MemoryResult": MemoryResult,
    "RetrievedChunk": RetrievedChunk,
    # Verification
    "CriticInput": CriticInput,
    "CriticVerdict": CriticVerdict,
    "Evidence": Evidence,
    "DemoCriticVerdict": DemoCriticVerdict,
    # Public API
    "RunRequest": RunRequest,
    "RunHandle": RunHandle,
    "RunStatus": RunStatus,
    "ControlCommand": ControlCommand,
    "NodeStatus": NodeStatus,
    "FailureClass": FailureClass,
}
"""Stable mapping of schema name → Pydantic model.

Ordering is alphabetical by convention; the export function sorts
by key to ensure determinism regardless of dict insertion order.
"""
