"""Rollback scope computation per spec §3.4.1 (B1 Part 2)."""

from __future__ import annotations

from typing import Literal

from neuronium_agent.planning.dag import ActionGraph
from neuronium_agent.recovery.models import RollbackScope, RollbackScopeType


FailureType = Literal[
    "node_execution",
    "critic_rejection",
    "constraint_violation",
    "plan_invalidation",
]


def _transitive_dependents(graph: ActionGraph, seed_node_ids: set[str]) -> set[str]:
    """Return seed_node_ids plus all nodes that depend on them (downstream)."""
    adj = graph.adjacency()
    result = set(seed_node_ids)
    queue = list(seed_node_ids)
    while queue:
        nid = queue.pop()
        for succ in adj.get(nid, []):
            if succ not in result:
                result.add(succ)
                queue.append(succ)
    return result


def compute_rollback_scope(
    failure_type: FailureType,
    graph: ActionGraph,
    failed_node_ids: set[str],
    *,
    critic_failed: bool = False,
    completed_node_ids: set[str] | None = None,
    gate_required_node_ids: set[str] | None = None,
) -> RollbackScope:
    """Compute rollback scope and preservation set per spec §3.4.1.

    - Node execution error: affected nodes + transitive dependents; preservation = other completed.
    - Critic rejection: rejected nodes (under gate/critic) + dependents; preservation = rest.
    - Constraint violation / Plan invalidation: entire graph (INTENTION), preservation empty.
    """
    all_node_ids = set(graph.node_ids())
    completed = completed_node_ids or set()

    if failure_type in ("constraint_violation", "plan_invalidation"):
        return RollbackScope(
            scope_type=RollbackScopeType.INTENTION,
            node_ids=all_node_ids,
            preservation_node_ids=set(),
        )

    if failure_type == "critic_rejection" and critic_failed and gate_required_node_ids:
        rejected = failed_node_ids or gate_required_node_ids
        rollback_ids = _transitive_dependents(graph, rejected)
        preservation_ids = completed - rollback_ids
        return RollbackScope(
            scope_type=RollbackScopeType.SUBGRAPH,
            node_ids=rollback_ids,
            preservation_node_ids=preservation_ids,
        )

    # node_execution (or critic_rejection without gate_required_node_ids)
    rollback_ids = _transitive_dependents(graph, failed_node_ids)
    preservation_ids = completed - rollback_ids
    return RollbackScope(
        scope_type=RollbackScopeType.NODE,
        node_ids=rollback_ids,
        preservation_node_ids=preservation_ids,
    )
