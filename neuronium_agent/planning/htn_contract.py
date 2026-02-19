"""HTN-lite planning contracts for recursive decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HtnSubgoal:
    """A decomposition unit in HTN planning."""

    subgoal_id: str
    title: str
    depth: int
    parent_subgoal_id: str | None = None
    kind: str = "generic"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HtnMethodChoice:
    """Selected decomposition method for a specific subgoal."""

    subgoal_id: str
    method_id: str
    rationale_key: str
    produced_subgoal_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HtnLeafOperator:
    """Leaf operator that will become a DAG node."""

    subgoal_id: str
    node_id: str
    node_type: str
    tool_name: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HtnDecompositionStep:
    """Single decomposition step for trace/audit."""

    step_index: int
    subgoal_id: str
    depth: int
    action: str
    details: dict[str, Any] = field(default_factory=dict)
