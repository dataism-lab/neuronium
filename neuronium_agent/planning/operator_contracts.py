"""Operator contracts used by planner validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatorContract:
    """Machine-readable operator contract for planning and validation."""

    operator_id: str
    node_type: str
    tool_name: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False
    replay_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "node_type": self.node_type,
            "tool_name": self.tool_name,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "policy": self.policy,
            "deterministic": self.deterministic,
            "replay_required": self.replay_required,
        }
