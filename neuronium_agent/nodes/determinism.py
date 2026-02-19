"""Determinism contract for DAG nodes (Spec §1.2.1, §6.1.2).

Nodes declare whether they use a random seed and whether they are
declared non-deterministic (rejected in strict mode at registry build).
"""

from __future__ import annotations

from pydantic import BaseModel


class DeterminismContract(BaseModel):
    """Declared determinism behaviour of a node."""

    uses_seed: bool = False
    """True if the node uses NodeContext.random_seed for reproducible behaviour."""

    declared_non_deterministic: bool = False
    """True if the node is explicitly declared non-deterministic (e.g. external tool).
    When config.determinism.strict is True, such nodes are rejected at registry build."""
