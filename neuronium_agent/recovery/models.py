"""Recovery policy models (B1 Part 1 + B1 Part 2)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RecoveryAction(str, Enum):
    """Action to take after stage gate failure."""

    RETRY_STAGE = "RETRY_STAGE"
    ESCALATE = "ESCALATE"
    FAIL = "FAIL"
    REPLAN = "REPLAN"


class RollbackScopeType(str, Enum):
    """Type of rollback scope per spec §3.4.1."""

    NODE = "NODE"  # Affected node + transitive dependents
    SUBGRAPH = "SUBGRAPH"  # Rejected output + dependent subgraph (e.g. critic)
    CONSTRAINT_SCOPE = "CONSTRAINT_SCOPE"  # All nodes since constraint binding
    INTENTION = "INTENTION"  # Entire Action Graph


class RollbackScope(BaseModel):
    """Rollback scope and preservation set per spec §3.4.1."""

    scope_type: RollbackScopeType
    node_ids: set[str] = Field(default_factory=set)
    preservation_node_ids: set[str] = Field(default_factory=set)
    preservation_artifact_ids: list[str] = Field(default_factory=list)


class RecoveryDecision(BaseModel):
    """Result of recovery policy evaluation."""

    action: RecoveryAction
    reason: str = ""
    escalation_context: dict[str, Any] | None = Field(default=None)
    rollback_scope: RollbackScope | None = Field(default=None)
