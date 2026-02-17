"""Action Graph (DAG) model (IBS §5).

Serialisable DAG structure: metadata, typed nodes, typed edges,
and optional conditional branches.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A single node in the Action Graph."""

    node_id: str
    node_type: Literal["model", "mcp", "code", "decision", "aggregate"]
    label: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Priority for deterministic tie-breaking (lower = higher priority)
    priority: int = 0


class GraphEdge(BaseModel):
    """Directed edge in the Action Graph."""

    source: str  # node_id
    target: str  # node_id
    edge_type: Literal["data", "control", "resource", "conditional"] = "data"
    label: str = ""
    condition: str | None = None  # for conditional edges


class ConditionalBranch(BaseModel):
    """Named conditional branch originating from a decision node."""

    decision_node_id: str
    branch_label: str
    target_node_ids: list[str]


class GraphMetadata(BaseModel):
    """Metadata attached to an Action Graph."""

    plan_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    description: str = ""
    version: int = 1


class ActionGraph(BaseModel):
    """The complete Action Graph (DAG) plan (IBS §5.1)."""

    metadata: GraphMetadata
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    conditional_branches: list[ConditionalBranch] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def node_map(self) -> dict[str, GraphNode]:
        return {n.node_id: n for n in self.nodes}

    def adjacency(self) -> dict[str, list[str]]:
        """Return adjacency list: node_id → [successor node_ids]."""
        adj: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            adj.setdefault(e.source, []).append(e.target)
        return adj

    def predecessors(self) -> dict[str, list[str]]:
        """Return reverse adjacency: node_id → [predecessor node_ids]."""
        preds: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        for e in self.edges:
            preds.setdefault(e.target, []).append(e.source)
        return preds

    def topological_order(self) -> list[str]:
        """Kahn's algorithm.  Ties broken by (priority, node_id)."""
        in_deg: dict[str, int] = {n.node_id: 0 for n in self.nodes}
        adj = self.adjacency()
        for e in self.edges:
            in_deg[e.target] = in_deg.get(e.target, 0) + 1

        nmap = self.node_map()
        # seed: nodes with in-degree 0, sorted deterministically
        queue = sorted(
            [nid for nid, d in in_deg.items() if d == 0],
            key=lambda nid: (nmap[nid].priority, nid),
        )
        order: list[str] = []
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for succ in adj.get(node_id, []):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)
                    queue.sort(key=lambda nid: (nmap[nid].priority, nid))

        if len(order) != len(self.nodes):
            raise ValueError("ActionGraph contains a cycle")
        return order
