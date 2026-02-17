"""Memory tools — ``memory.ingest_files`` and ``memory.query`` (Stage 5).

These are invoked via the local-tool dispatch in
:func:`~neuronium_agent.tools.local_tools.invoke_local_tool` when a
:class:`ToolRuntime` is available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuronium_agent.tools.local_tools import (
    ToolExecutionError,
    _ensure_allowed_path,
    _normalize_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunking helpers (deterministic)
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    *,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[tuple[int, int]]:
    """Return ``(start_char, end_char)`` spans for fixed-size chunks.

    Deterministic: same text + params → same spans.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        start = end - overlap
    return spans


def _quote_hash(text: str) -> str:
    """SHA-256 of normalised quote (lower, collapsed whitespace)."""
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# memory.ingest_files
# ---------------------------------------------------------------------------

def invoke_memory_ingest_files(
    args: dict[str, Any],
    *,
    policy: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """Ingest local files into memory_chunks with provenance metadata."""
    from neuronium_agent._canonical import artifact_id, canonical_bytes

    paths_raw: list[str] = list(args.get("paths", []))
    source_kind: str = args.get("source_kind", "user_docs")
    visibility: str = args.get("visibility", "user")
    chunk_max: int = int(args.get("chunk_max_chars", 2000))
    chunk_overlap: int = int(args.get("chunk_overlap_chars", 200))

    if not paths_raw:
        raise ToolExecutionError("memory.ingest_files: missing 'paths'")

    memory_store = getattr(runtime, "memory_store", None)
    blob_store = getattr(runtime, "blob_store", None)
    index_store = getattr(runtime, "index_store", None)
    if memory_store is None:
        raise ToolExecutionError(
            "memory.ingest_files: MemoryStore not available in ToolRuntime"
        )

    # Deterministic: sort paths.
    sorted_paths = sorted(paths_raw)

    roots_allowlist = list(policy.get("fs_roots_allowlist", []))
    max_bytes = int(policy.get("fs_max_read_bytes", 1_000_000))

    all_chunk_ids: list[str] = []
    source_artifact_ids: list[str] = []
    warnings: list[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for file_path_str in sorted_paths:
        p = _normalize_path(file_path_str)
        _ensure_allowed_path(p, roots_allowlist=roots_allowlist)

        if not p.exists():
            warnings.append(f"File not found, skipped: {file_path_str}")
            continue

        data = p.read_bytes()
        if len(data) > max_bytes:
            warnings.append(
                f"File too large ({len(data)} bytes), skipped: {file_path_str}"
            )
            continue

        text = data.decode("utf-8", errors="replace")

        # Create artifact snapshot for the source document.
        content_bytes = canonical_bytes({"text": text, "path": str(p)})
        ctx = {
            "timestamp": now_iso,
            "node_ref": "memory.ingest_files",
            "input_artifact_ids": [],
        }
        aid = artifact_id(content_bytes, ctx)

        if blob_store is not None:
            try:
                blob_store.put(aid, content_bytes, "application/json")
            except Exception:
                pass  # idempotent
        if index_store is not None:
            try:
                index_store.record_artifact_metadata(
                    artifact_id=aid,
                    artifact_type="memory_source_document",
                    created_at=now_iso,
                    produced_by_node_ref="memory.ingest_files",
                    inputs_json=json.dumps({"path": str(p)}, sort_keys=True),
                    quality_signals_json="{}",
                    blob_key=aid,
                    media_type="application/json",
                    size_bytes=len(content_bytes),
                )
            except Exception:
                pass  # idempotent

        source_artifact_ids.append(aid)

        # Chunk the document.
        spans = _chunk_text(text, max_chars=chunk_max, overlap=chunk_overlap)
        for idx, (sc, ec) in enumerate(spans):
            chunk_text = text[sc:ec]
            # Deterministic chunk_id: hash of (source_artifact_id + index).
            raw = f"{aid}:{idx}".encode("utf-8")
            chunk_id = f"mc_{hashlib.sha256(raw).hexdigest()[:24]}"

            meta = {
                "source_kind": source_kind,
                "visibility": visibility,
                "source_uri": str(p),
                "source_id": aid,
                "start_char": sc,
                "end_char": ec,
            }

            memory_store.upsert_chunk(
                chunk_id=chunk_id,
                source_artifact_id=aid,
                text=chunk_text,
                metadata_json=json.dumps(meta, sort_keys=True),
                created_at=now_iso,
            )
            all_chunk_ids.append(chunk_id)

    return {
        "ingested_count": len(source_artifact_ids),
        "chunk_ids": all_chunk_ids,
        "source_artifact_ids": source_artifact_ids,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# memory.query
# ---------------------------------------------------------------------------

def invoke_memory_query(
    args: dict[str, Any],
    *,
    policy: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """Execute a memory query and return results with evidence refs."""
    memory_store = getattr(runtime, "memory_store", None)
    config = getattr(runtime, "config", None)
    if memory_store is None:
        raise ToolExecutionError(
            "memory.query: MemoryStore not available in ToolRuntime"
        )

    query_text: str = str(args.get("query", ""))
    mode: str = str(args.get("mode", "hybrid"))
    top_k: int = int(args.get("top_k", 5))
    require_exact: bool = bool(args.get("require_exact_mode", False))

    constraints = args.get("constraints", {}) or {}
    source_kind_filter: list[str] = list(constraints.get("source_kind_filter", []))
    visibility_filter: list[str] = list(constraints.get("visibility_filter", []))
    confidence_threshold: float = float(constraints.get("confidence_threshold", 0.0))

    # Determine effective mode.
    effective_mode = mode
    warnings: list[str] = []

    semantic_enabled = False
    if config is not None:
        semantic_enabled = getattr(
            getattr(config, "memory", None), "semantic_search", None
        ) is not None and getattr(
            getattr(config, "memory", None).semantic_search, "enabled", False
        )

    if mode == "semantic" and not semantic_enabled:
        if require_exact:
            raise ToolExecutionError(
                "memory.query: mode='semantic' requested with "
                "require_exact_mode=True, but semantic backend is disabled. "
                "DependencyMissing: semantic_search.enabled=false"
            )
        effective_mode = "hybrid"
        warnings.append(
            "SEMANTIC_BACKEND_DISABLED_FALLBACK_TO_HYBRID: "
            "semantic_search.enabled=false, falling back to hybrid mode."
        )

    # Build metadata filters from constraints.
    meta_filters: dict[str, Any] = {}
    if source_kind_filter:
        # We'll filter in-memory after retrieval for list-type filters.
        pass
    if visibility_filter:
        pass

    t0 = time.monotonic()

    # Retrieve chunks.
    if effective_mode in ("structured", "hybrid"):
        raw_results = memory_store.search_keyword_topk(
            query_text,
            top_k=top_k * 3,  # over-fetch to allow filtering
            metadata_filters=meta_filters,
        )
    else:
        # Iterative / other: same as hybrid for now.
        raw_results = memory_store.search_keyword_topk(
            query_text,
            top_k=top_k * 3,
            metadata_filters=meta_filters,
        )

    total_scanned = memory_store.count_chunks(metadata_filters=meta_filters)

    # Apply source_kind / visibility filters in Python for determinism.
    filtered: list[dict[str, Any]] = []
    for row in raw_results:
        meta = _parse_metadata(row.get("metadata_json", "{}"))
        if source_kind_filter and meta.get("source_kind") not in source_kind_filter:
            continue
        if visibility_filter and meta.get("visibility") not in visibility_filter:
            continue
        filtered.append(row)

    # Trim to top_k.
    filtered = filtered[:top_k]

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Build response.
    retrieved_chunks: list[dict[str, Any]] = []
    evidence_refs: list[dict[str, Any]] = []

    for row in filtered:
        meta = _parse_metadata(row.get("metadata_json", "{}"))
        score = float(row.get("_score", 0.0))

        retrieved_chunks.append({
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "source_artifact_id": row.get("source_artifact_id", ""),
            "metadata": meta,
            "score": score,
        })

        # Build EvidenceRef.
        quote = row["text"][:200] if len(row["text"]) > 200 else row["text"]
        evidence_refs.append({
            "chunk_id": row["chunk_id"],
            "source_artifact_id": row.get("source_artifact_id", ""),
            "source_kind": meta.get("source_kind", "user_docs"),
            "visibility": meta.get("visibility", "user"),
            "locator": {
                "source_uri": meta.get("source_uri", ""),
                "start_char": meta.get("start_char"),
                "end_char": meta.get("end_char"),
            },
            "quote": quote,
            "quote_hash": _quote_hash(quote),
            "relevance_score": score,
            "confidence": min(score / max(top_k, 1), 1.0),
        })

    return {
        "effective_mode": effective_mode,
        "warnings": warnings,
        "retrieved_chunks": retrieved_chunks,
        "evidence_refs": evidence_refs,
        "stats": {
            "total_chunks_scanned": total_scanned,
            "retrieval_time_ms": round(elapsed_ms, 2),
        },
    }


def _parse_metadata(raw: str) -> dict[str, Any]:
    """Safely parse metadata_json."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}
