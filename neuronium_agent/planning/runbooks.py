"""Runbook planning templates (v0.2).

This module provides deterministic ActionGraph templates for practical
"business automation" style runs, without relying on generalized HTN.

It also contains concrete :class:`Runbook` implementations registered
via :mod:`neuronium_agent.planning.runbook_registry`.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

from neuronium_agent.planning.dag import (
    ActionGraph,
    GraphEdge,
    GraphMetadata,
    GraphNode,
)
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)
from neuronium_agent.verification.demo_critic import critic_json_schema
from neuronium_agent.verification.business_critic import BUSINESS_CRITIC_SYSTEM_PROMPT


_DOC_REPORT_SYSTEM_PROMPT = (
    "You are an operations analyst.\n"
    "Task: produce a clear, structured business-ready report based strictly on the provided context.\n"
    "\n"
    "Rules:\n"
    "- Use ONLY the provided context (documents) — do not invent facts.\n"
    "- When you state an important claim, cite evidence by referencing the document keys "
    "(e.g. [doc_000], [doc_001]).\n"
    "- Output in Markdown.\n"
    "- Include a short 'Action items' section.\n"
)


def plan_docs_report_v1(
    *,
    objective: str,
    constraints: list[str] | None,
    doc_paths: list[str],
    plan_id: str,
) -> ActionGraph:
    """Build a deterministic DAG that reads local docs and drafts a report.

    Graph:
      read_doc_* (mcp local tools) -> merge (aggregate) -> draft (model) -> critic (model)
                                     /------------------------------------/
    """
    constraints = constraints or []
    # Deterministic: sort doc paths, then index them.
    sorted_paths = sorted(doc_paths)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for i, p in enumerate(sorted_paths):
        doc_key = f"doc_{i:03d}"
        node_id = f"read_{i:03d}"
        nodes.append(
            GraphNode(
                node_id=node_id,
                node_type="mcp",
                label=f"Read document {doc_key}",
                parameters={
                    "tool_name": "fs.read_text",
                    "tool_args": {"path": p, "out_key": doc_key},
                },
                priority=0,
            )
        )

    merge_node = GraphNode(
        node_id="merge_docs",
        node_type="aggregate",
        label="Merge documents",
        parameters={},
        priority=1,
    )
    nodes.append(merge_node)

    draft_node = GraphNode(
        node_id="draft_report",
        node_type="model",
        label="Draft business report from documents",
        parameters={"system_prompt": _DOC_REPORT_SYSTEM_PROMPT},
        priority=2,
    )
    nodes.append(draft_node)

    critic_node = GraphNode(
        node_id="critic_report",
        node_type="model",
        label="Critic — verify report quality and evidence",
        parameters={
            "system_prompt": BUSINESS_CRITIC_SYSTEM_PROMPT,
            "json_schema": critic_json_schema(),
        },
        priority=3,
    )
    nodes.append(critic_node)

    # Edges: all reads -> merge
    for i in range(len(sorted_paths)):
        edges.append(
            GraphEdge(
                source=f"read_{i:03d}",
                target="merge_docs",
                edge_type="data",
                label="doc_text",
            )
        )

    # merge -> draft, merge -> critic, draft -> critic
    edges.extend(
        [
            GraphEdge(source="merge_docs", target="draft_report", edge_type="data"),
            GraphEdge(source="merge_docs", target="critic_report", edge_type="data"),
            GraphEdge(source="draft_report", target="critic_report", edge_type="data"),
        ]
    )

    return ActionGraph(
        metadata=GraphMetadata(
            plan_id=plan_id,
            description=f"Docs report v1 for: {objective}",
        ),
        nodes=nodes,
        edges=edges,
        # NOTE: meta_extra is not part of GraphMetadata contract; keep in trace events instead.
        # We avoid extending the DAG model here to preserve Stage 1 schemas.
    )


# ---------------------------------------------------------------------------
# Runbook implementation (registered via runbook_registry)
# ---------------------------------------------------------------------------

_DEFAULT_DOC_PATHS = [
    # Preferred new locations (docs/ structure).
    "docs/architecture/Super_Agent_presentation.md",
    "docs/roadmap/ROADMAP.md",
    "docs/roadmap/ROADMAP_STATUS.md",
    "docs/architecture/Implementation_Binding_Spec.md",
    "docs/architecture/STORAGE_SCHEMA_SPEC.md",
]


def _prefer_existing_paths(paths: list[str]) -> list[str]:
    """Best-effort resolve doc paths across repo reorganisations.

    If a preferred docs/* path does not exist yet, fall back to the basename
    in the repository root (legacy layout).
    """
    resolved: list[str] = []
    for p in paths:
        if Path(p).exists():
            resolved.append(p)
            continue
        legacy = Path(p).name
        if Path(legacy).exists():
            resolved.append(legacy)
        else:
            # Keep original (will fail loudly if executed).
            resolved.append(p)
    return resolved


class DocsReportV1Runbook(Runbook):
    """Single-stage runbook: read local docs -> draft report -> critic."""

    @property
    def runbook_id(self) -> str:
        return "docs_report_v1"

    @property
    def description(self) -> str:
        return "Read local documents and produce a business-ready report with critic verification."

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        doc_paths: list[str] = metadata.get("doc_paths") or []  # type: ignore[assignment]
        if not isinstance(doc_paths, list) or not doc_paths:
            doc_paths = _prefer_existing_paths(list(_DEFAULT_DOC_PATHS))

        plan_id = f"plan-docs-report-v1-{execution_id[:12]}"

        graph = plan_docs_report_v1(
            objective=objective,
            constraints=constraints,
            doc_paths=doc_paths,
            plan_id=plan_id,
        )

        return [
            ActionGraphStage(
                stage_id="docs_report_v1:stage1",
                graph=graph,
                initial_inputs_override={"runbook_id": "docs_report_v1"},
                success_gate=StageSuccessGate(
                    required_completed_nodes=["draft_report"],
                    critic_node_id="critic_report",
                ),
            ),
        ]

