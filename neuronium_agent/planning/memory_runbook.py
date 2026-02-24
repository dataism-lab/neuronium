"""Hybrid-memory report runbook — demo case C (Stage 5).

Two-stage runbook demonstrating "memory as a system component":

* **Stage 1 (Ingest + Retrieve)**: ingest internal_docs and user_docs into
  ``memory_chunks``, then run ``memory.query`` against the objective.
* **Stage 2 (Synthesise + Verify)**: a ModelNode drafts a report from the
  retrieved context, and a critic verifies evidence-linked citations.

Source separation is demonstrated via ``source_kind`` / ``visibility``
metadata on chunks and ``EvidenceRef`` in the query result.
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
from neuronium_agent.verification.memory_critic import (
    MEMORY_BUSINESS_CRITIC_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INTERNAL_DOC_PATHS: list[str] = [
    # Preferred new locations (docs/ structure).
    "docs/architecture/Super_Agent_presentation.md",
    "docs/roadmap/ROADMAP_FULL_IMPLEMENTATION_FROM_CURRENT_STATE.md",
    "docs/architecture/Implementation_Binding_Spec.md",
    "docs/architecture/STORAGE_SCHEMA_SPEC.md",
]


def _prefer_existing_paths(paths: list[str]) -> list[str]:
    """Best-effort resolve doc paths across repo reorganisations.

    Missing files are tolerated by ``memory.ingest_files`` (warnings), but we
    still try to point to the most likely existing locations.
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
            resolved.append(p)
    return resolved

_MEMORY_DRAFT_SYSTEM_PROMPT = (
    "You are an operations analyst.\n"
    "Task: produce a clear, structured business-ready report based "
    "strictly on the RETRIEVED MEMORY CHUNKS provided.\n"
    "\n"
    "Rules:\n"
    "- Use ONLY the provided memory chunks — do not invent facts.\n"
    "- When you state an important claim, cite evidence by referencing "
    "the chunk keys (e.g. [mem_000], [mem_001]).\n"
    "- Clearly mark whether a source is 'internal' or 'user' when relevant.\n"
    "- Output in Markdown.\n"
    "- Include a short 'Action items' section.\n"
)


# ---------------------------------------------------------------------------
# DAG builders (deterministic)
# ---------------------------------------------------------------------------

def _build_ingest_retrieve_graph(
    *,
    objective: str,
    internal_doc_paths: list[str],
    user_doc_paths: list[str],
    plan_id: str,
) -> ActionGraph:
    """Stage 1: ingest files into memory + query.

    Graph::

        ingest_internal (mcp) ─┐
                                ├─ query (mcp)
        ingest_user (mcp) ─────┘
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # -- ingest internal docs ------------------------------------------------
    nodes.append(
        GraphNode(
            node_id="ingest_internal",
            node_type="mcp",
            label="Ingest internal documents into memory",
            parameters={
                "tool_name": "memory.ingest_files",
                "tool_args": {
                    "paths": sorted(internal_doc_paths),
                    "source_kind": "internal_docs",
                    "visibility": "audit_only",
                },
            },
            priority=0,
        )
    )

    # -- ingest user docs (may be empty) ------------------------------------
    nodes.append(
        GraphNode(
            node_id="ingest_user",
            node_type="mcp",
            label="Ingest user documents into memory",
            parameters={
                "tool_name": "memory.ingest_files",
                "tool_args": {
                    "paths": sorted(user_doc_paths),
                    "source_kind": "user_docs",
                    "visibility": "user",
                },
            },
            priority=0,
        )
    )

    # -- query ---------------------------------------------------------------
    nodes.append(
        GraphNode(
            node_id="memory_query",
            node_type="mcp",
            label="Retrieve relevant chunks from memory",
            parameters={
                "tool_name": "memory.query",
                "tool_args": {
                    "query": objective,
                    "mode": "hybrid",
                    "top_k": 10,
                },
            },
            priority=1,
        )
    )

    edges.extend([
        GraphEdge(
            source="ingest_internal",
            target="memory_query",
            edge_type="control",
            label="ingestion_done",
        ),
        GraphEdge(
            source="ingest_user",
            target="memory_query",
            edge_type="control",
            label="ingestion_done",
        ),
    ])

    return ActionGraph(
        metadata=GraphMetadata(
            plan_id=plan_id,
            description=f"Memory ingest+retrieve for: {objective}",
        ),
        nodes=nodes,
        edges=edges,
    )


def _build_synthesise_verify_graph(
    *,
    objective: str,
    plan_id: str,
) -> ActionGraph:
    """Stage 2: draft report from memory context + critic.

    Graph::

        draft_report (model) ─── critic_report (model)
    """
    nodes: list[GraphNode] = [
        GraphNode(
            node_id="draft_report",
            node_type="model",
            label="Draft report from retrieved memory chunks",
            parameters={"system_prompt": _MEMORY_DRAFT_SYSTEM_PROMPT},
            priority=0,
        ),
        GraphNode(
            node_id="critic_report",
            node_type="model",
            label="Critic — verify report quality and evidence citations",
            parameters={
                "system_prompt": MEMORY_BUSINESS_CRITIC_SYSTEM_PROMPT,
                "json_schema": critic_json_schema(),
            },
            priority=1,
        ),
    ]
    edges = [
        GraphEdge(
            source="draft_report",
            target="critic_report",
            edge_type="data",
            label="report_text",
        ),
    ]
    return ActionGraph(
        metadata=GraphMetadata(
            plan_id=plan_id,
            description=f"Synthesise+verify for: {objective}",
        ),
        nodes=nodes,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Runbook
# ---------------------------------------------------------------------------

class HybridMemoryReportV1Runbook(Runbook):
    """Two-stage runbook: ingest+retrieve → synthesise+verify (demo case C)."""

    @property
    def runbook_id(self) -> str:
        return "hybrid_memory_report_v1"

    @property
    def description(self) -> str:
        return (
            "Ingest internal and user documents into memory, retrieve "
            "relevant chunks, draft a business report with evidence-linked "
            "citations, and verify with a memory-aware critic."
        )

    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        internal_doc_paths: list[str] = _prefer_existing_paths(list(
            metadata.get("internal_doc_paths") or _DEFAULT_INTERNAL_DOC_PATHS
        ))
        user_doc_paths: list[str] = list(
            metadata.get("user_doc_paths") or []
        )

        eid_short = execution_id[:12]

        # Stage 1: Ingest + Retrieve
        stage1_graph = _build_ingest_retrieve_graph(
            objective=objective,
            internal_doc_paths=internal_doc_paths,
            user_doc_paths=user_doc_paths,
            plan_id=f"plan-mem-ingest-{eid_short}",
        )

        # Stage 2: Synthesise + Verify
        stage2_graph = _build_synthesise_verify_graph(
            objective=objective,
            plan_id=f"plan-mem-synth-{eid_short}",
        )

        return [
            ActionGraphStage(
                stage_id="hybrid_memory_report_v1:ingest_retrieve",
                graph=stage1_graph,
                initial_inputs_override={
                    "runbook_id": "hybrid_memory_report_v1",
                    "objective": objective,
                },
                success_gate=StageSuccessGate(
                    required_completed_nodes=["memory_query"],
                ),
            ),
            ActionGraphStage(
                stage_id="hybrid_memory_report_v1:synthesise_verify",
                graph=stage2_graph,
                initial_inputs_override={
                    "runbook_id": "hybrid_memory_report_v1",
                    "objective": objective,
                },
                success_gate=StageSuccessGate(
                    required_completed_nodes=["draft_report"],
                    critic_node_id="critic_report",
                ),
            ),
        ]
