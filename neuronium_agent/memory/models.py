"""Memory subsystem contracts — GraphRAG-lite (ROADMAP Stage 5).

Pydantic models for the Unified Query Interface, evidence references,
and ingestion DTOs.  These form the typed contract between the memory
tools (``memory.ingest_files``, ``memory.query``) and the rest of the
system (runbooks, critics, trace).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence / provenance
# ---------------------------------------------------------------------------

class ChunkLocator(BaseModel):
    """Points to a specific span inside a source document."""

    source_uri: str = ""
    start_char: int | None = None
    end_char: int | None = None
    start_line: int | None = None
    end_line: int | None = None


class EvidenceRef(BaseModel):
    """A normalised reference to a retrieved memory chunk with provenance.

    ``quote_hash`` is sha-256 of the normalised quote text (lower-cased,
    whitespace-collapsed) and can be used to verify that a citation was
    not fabricated.
    """

    chunk_id: str
    source_artifact_id: str = ""
    source_kind: Literal["internal_docs", "user_docs", "tool_output"] = "user_docs"
    visibility: Literal["user", "internal", "audit_only"] = "user"
    locator: ChunkLocator = Field(default_factory=ChunkLocator)
    quote: str = ""
    quote_hash: str = ""
    relevance_score: float = 0.0
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Retrieved chunk (what comes back from storage)
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A single chunk returned by a memory query."""

    chunk_id: str
    text: str
    source_artifact_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


# ---------------------------------------------------------------------------
# Unified Query Interface
# ---------------------------------------------------------------------------

class MemoryQueryConstraints(BaseModel):
    """Filters narrowing down a memory query."""

    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    source_filter: list[str] = Field(default_factory=list)
    source_kind_filter: list[Literal[
        "internal_docs", "user_docs", "tool_output"
    ]] = Field(default_factory=list)
    visibility_filter: list[Literal[
        "user", "internal", "audit_only"
    ]] = Field(default_factory=list)
    confidence_threshold: float = 0.0


class MemoryQuery(BaseModel):
    """Input contract for ``memory.query`` (IBS §8.1.2).

    ``require_exact_mode`` — when *True* the system must return a
    deterministic error instead of falling back to a different mode.
    """

    query: str
    mode: Literal["structured", "hybrid", "semantic", "iterative"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)
    constraints: MemoryQueryConstraints = Field(
        default_factory=MemoryQueryConstraints,
    )
    require_exact_mode: bool = False


class MemoryQueryStats(BaseModel):
    """Lightweight statistics attached to every query result."""

    total_chunks_scanned: int = 0
    retrieval_time_ms: float = 0.0


class MemoryResult(BaseModel):
    """Output contract for ``memory.query``.

    ``effective_mode`` shows which mode was *actually* used (may differ
    from the requested mode when a fallback occurs).
    ``warnings`` contains human-readable messages about any fallback or
    degradation that took place.
    """

    effective_mode: Literal[
        "structured", "hybrid", "semantic", "iterative"
    ] = "hybrid"
    warnings: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    stats: MemoryQueryStats = Field(default_factory=MemoryQueryStats)


# ---------------------------------------------------------------------------
# Ingestion DTOs
# ---------------------------------------------------------------------------

class MemoryIngestRequest(BaseModel):
    """Input contract for ``memory.ingest_files``."""

    paths: list[str]
    source_kind: Literal["internal_docs", "user_docs", "tool_output"] = "user_docs"
    visibility: Literal["user", "internal", "audit_only"] = "user"
    chunk_max_chars: int = Field(default=2000, ge=100, le=50_000)
    chunk_overlap_chars: int = Field(default=200, ge=0)


class MemoryIngestResult(BaseModel):
    """Output contract for ``memory.ingest_files``."""

    ingested_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    source_artifact_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
