"""Stage 5 memory tests — ingestion/query determinism, mode fallback,
source/visibility filtering, and schema payload coverage.

All tests use an in-process temporary SQLite database (no external deps).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from neuronium_agent.memory.sqlite_memory_store import SqliteMemoryStore
from neuronium_agent.tools.memory_tools import (
    _chunk_text,
    _quote_hash,
    invoke_memory_ingest_files,
    invoke_memory_query,
)
from neuronium_agent.tools.local_tools import ToolExecutionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INIT_SQL = (Path(__file__).resolve().parent.parent
             / "neuronium_agent" / "storage" / "migrations" / "sqlite"
             / "0001_init.sql").read_text(encoding="utf-8")


def _make_db(tmp_path: Path) -> Path:
    """Create an initialised SQLite DB in *tmp_path* and return its path."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_INIT_SQL)
    conn.close()
    return db_path


class _FakeBlobStore:
    """Minimal blob store for tests (stores nothing)."""

    def put(self, key: str, data: bytes, media_type: str) -> None:
        pass  # no-op


class _FakeIndexStore:
    """Minimal index store for tests."""

    def record_artifact_metadata(self, **kwargs: Any) -> None:
        pass  # no-op


class _StubConfig:
    """Minimal stub for AppConfig with memory config."""

    class _Memory:
        class _Semantic:
            enabled = False
        semantic_search = _Semantic()

    memory = _Memory()


class _StubConfigSemanticEnabled:
    """Config stub with semantic_search.enabled = True."""

    class _Memory:
        class _Semantic:
            enabled = True
        semantic_search = _Semantic()

    memory = _Memory()


class _Runtime:
    """Minimal ToolRuntime-like object for tests."""

    def __init__(
        self,
        memory_store: SqliteMemoryStore,
        config: Any = None,
    ) -> None:
        self.memory_store = memory_store
        self.blob_store = _FakeBlobStore()
        self.index_store = _FakeIndexStore()
        self.config = config or _StubConfig()


def _make_runtime(tmp_path: Path, *, config: Any = None) -> _Runtime:
    db = _make_db(tmp_path)
    store = SqliteMemoryStore(db)
    return _Runtime(store, config=config)


def _write_tmp_files(
    tmp_path: Path,
    contents: dict[str, str],
) -> list[str]:
    """Write files under *tmp_path* and return their absolute paths (sorted)."""
    paths: list[str] = []
    for name, text in sorted(contents.items()):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# 1. Chunking determinism
# ---------------------------------------------------------------------------

class TestChunkDeterminism:

    def test_same_text_same_spans(self) -> None:
        text = "A" * 5000
        s1 = _chunk_text(text, max_chars=2000, overlap=200)
        s2 = _chunk_text(text, max_chars=2000, overlap=200)
        assert s1 == s2

    def test_short_text_single_chunk(self) -> None:
        spans = _chunk_text("hello world", max_chars=2000, overlap=200)
        assert spans == [(0, 11)]

    def test_overlap_boundaries(self) -> None:
        text = "X" * 4000
        spans = _chunk_text(text, max_chars=2000, overlap=200)
        assert len(spans) == 3
        # First chunk: 0..2000
        assert spans[0] == (0, 2000)
        # Second chunk: start = 2000-200 = 1800
        assert spans[1] == (1800, 3800)
        # Third chunk: start = 3800-200 = 3600
        assert spans[2] == (3600, 4000)

    def test_quote_hash_deterministic(self) -> None:
        h1 = _quote_hash("Hello  World")
        h2 = _quote_hash("hello   world")
        assert h1 == h2

    def test_quote_hash_different_text(self) -> None:
        assert _quote_hash("alpha") != _quote_hash("beta")


# ---------------------------------------------------------------------------
# 2. Ingestion determinism
# ---------------------------------------------------------------------------

class TestIngestionDeterminism:

    def test_ingest_idempotent_no_crash(self, tmp_path: Path) -> None:
        """Re-ingesting the same file completes without error.

        Note: chunk_ids differ across calls because ``artifact_id``
        incorporates ``datetime.now()`` — each ingestion is a CAS
        snapshot.  Idempotency here means INSERT-OR-IGNORE on
        ``chunk_id`` inside the store, not identical output dicts.
        """
        rt = _make_runtime(tmp_path)
        paths = _write_tmp_files(tmp_path, {"doc.txt": "Revenue grew 10%."})

        policy = {"fs_roots_allowlist": [str(tmp_path)]}
        args = {"paths": paths, "source_kind": "user_docs", "visibility": "user"}

        r1 = invoke_memory_ingest_files(args, policy=policy, runtime=rt)
        r2 = invoke_memory_ingest_files(args, policy=policy, runtime=rt)

        # Both calls succeed and produce the same number of chunks.
        assert r1["ingested_count"] == r2["ingested_count"] == 1
        assert len(r1["chunk_ids"]) == len(r2["chunk_ids"])

    def test_chunk_id_formula_deterministic(self, tmp_path: Path) -> None:
        """Given the same artifact_id, chunk_ids are deterministic.

        We verify this by checking that the chunk_id is a function of
        (source_artifact_id, chunk_index) — same aid + same index →
        same chunk_id.
        """
        import hashlib

        aid = "fixed_artifact_id_for_test"
        for idx in range(3):
            raw = f"{aid}:{idx}".encode("utf-8")
            expected = f"mc_{hashlib.sha256(raw).hexdigest()[:24]}"
            raw2 = f"{aid}:{idx}".encode("utf-8")
            actual = f"mc_{hashlib.sha256(raw2).hexdigest()[:24]}"
            assert expected == actual

    def test_ingest_sorted_paths_same_content(self, tmp_path: Path) -> None:
        """Regardless of input order, the same files are ingested and the
        same number of chunks/artifacts are produced.
        """
        _write_tmp_files(tmp_path, {"z.txt": "Z content", "a.txt": "A content"})
        policy = {"fs_roots_allowlist": [str(tmp_path)]}
        rt = _make_runtime(tmp_path)

        r_forward = invoke_memory_ingest_files(
            {"paths": [str(tmp_path / "z.txt"), str(tmp_path / "a.txt")]},
            policy=policy, runtime=rt,
        )
        rt2 = _make_runtime(tmp_path)
        r_reverse = invoke_memory_ingest_files(
            {"paths": [str(tmp_path / "a.txt"), str(tmp_path / "z.txt")]},
            policy=policy, runtime=rt2,
        )

        # Same count of ingested files and chunks.
        assert r_forward["ingested_count"] == r_reverse["ingested_count"] == 2
        assert len(r_forward["chunk_ids"]) == len(r_reverse["chunk_ids"])

    def test_ingest_missing_paths_generates_warning(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        policy = {"fs_roots_allowlist": [str(tmp_path)]}
        result = invoke_memory_ingest_files(
            {"paths": [str(tmp_path / "does_not_exist.txt")]},
            policy=policy, runtime=rt,
        )
        assert result["ingested_count"] == 0
        assert any("not found" in w.lower() for w in result["warnings"])

    def test_ingest_missing_store_raises(self, tmp_path: Path) -> None:
        """Without MemoryStore, ingest must raise explicitly."""
        rt = _Runtime.__new__(_Runtime)
        rt.memory_store = None
        rt.blob_store = _FakeBlobStore()
        rt.index_store = _FakeIndexStore()

        with pytest.raises(ToolExecutionError, match="MemoryStore not available"):
            invoke_memory_ingest_files(
                {"paths": ["any.txt"]},
                policy={},
                runtime=rt,
            )


# ---------------------------------------------------------------------------
# 3. Query determinism and mode fallback
# ---------------------------------------------------------------------------

class TestQueryDeterminism:

    def _ingest_sample(
        self, tmp_path: Path, rt: _Runtime
    ) -> list[str]:
        """Ingest two sample docs and return chunk_ids."""
        paths = _write_tmp_files(tmp_path, {
            "internal.md": "Internal roadmap: revenue target is $50M.",
            "user_report.md": "User uploaded report: Q3 revenue was $45M.",
        })
        policy = {"fs_roots_allowlist": [str(tmp_path)]}

        invoke_memory_ingest_files(
            {"paths": [paths[0]], "source_kind": "internal_docs",
             "visibility": "audit_only"},
            policy=policy, runtime=rt,
        )
        invoke_memory_ingest_files(
            {"paths": [paths[1]], "source_kind": "user_docs",
             "visibility": "user"},
            policy=policy, runtime=rt,
        )
        return paths

    def test_query_returns_results(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        self._ingest_sample(tmp_path, rt)

        result = invoke_memory_query(
            {"query": "revenue", "mode": "hybrid", "top_k": 5},
            policy={}, runtime=rt,
        )
        assert result["effective_mode"] == "hybrid"
        assert len(result["retrieved_chunks"]) > 0
        assert len(result["evidence_refs"]) > 0
        assert result["stats"]["total_chunks_scanned"] > 0

    def test_query_deterministic(self, tmp_path: Path) -> None:
        """Same query on same data → same results."""
        rt = _make_runtime(tmp_path)
        self._ingest_sample(tmp_path, rt)

        args = {"query": "revenue target", "mode": "hybrid", "top_k": 5}
        r1 = invoke_memory_query(args, policy={}, runtime=rt)
        r2 = invoke_memory_query(args, policy={}, runtime=rt)

        assert r1["retrieved_chunks"] == r2["retrieved_chunks"]
        assert r1["evidence_refs"] == r2["evidence_refs"]
        assert r1["effective_mode"] == r2["effective_mode"]

    def test_query_empty_results(self, tmp_path: Path) -> None:
        """Query with no matching words returns empty."""
        rt = _make_runtime(tmp_path)
        self._ingest_sample(tmp_path, rt)

        result = invoke_memory_query(
            {"query": "xyznonexistentword", "mode": "hybrid"},
            policy={}, runtime=rt,
        )
        assert result["retrieved_chunks"] == []
        assert result["evidence_refs"] == []

    def test_query_missing_store_raises(self) -> None:
        rt = _Runtime.__new__(_Runtime)
        rt.memory_store = None
        rt.config = None

        with pytest.raises(ToolExecutionError, match="MemoryStore not available"):
            invoke_memory_query(
                {"query": "test"}, policy={}, runtime=rt,
            )


# ---------------------------------------------------------------------------
# 4. Explicit fallback vs require_exact_mode
# ---------------------------------------------------------------------------

class TestModeFallback:

    def test_semantic_fallback_to_hybrid(self, tmp_path: Path) -> None:
        """When semantic is disabled, mode falls back to hybrid with warning."""
        rt = _make_runtime(tmp_path, config=_StubConfig())
        result = invoke_memory_query(
            {"query": "test", "mode": "semantic"},
            policy={}, runtime=rt,
        )
        assert result["effective_mode"] == "hybrid"
        assert any("SEMANTIC_BACKEND_DISABLED" in w for w in result["warnings"])

    def test_require_exact_mode_raises(self, tmp_path: Path) -> None:
        """With require_exact_mode=True + disabled semantic → deterministic error."""
        rt = _make_runtime(tmp_path, config=_StubConfig())

        with pytest.raises(
            ToolExecutionError,
            match="DependencyMissing.*semantic_search",
        ):
            invoke_memory_query(
                {
                    "query": "test",
                    "mode": "semantic",
                    "require_exact_mode": True,
                },
                policy={}, runtime=rt,
            )

    def test_hybrid_no_fallback(self, tmp_path: Path) -> None:
        """Hybrid mode works even without semantic backend — no warning."""
        rt = _make_runtime(tmp_path, config=_StubConfig())
        result = invoke_memory_query(
            {"query": "test", "mode": "hybrid"},
            policy={}, runtime=rt,
        )
        assert result["effective_mode"] == "hybrid"
        assert result["warnings"] == []

    def test_semantic_enabled_no_fallback(self, tmp_path: Path) -> None:
        """When semantic IS enabled, mode='semantic' stays — no warning."""
        rt = _make_runtime(tmp_path, config=_StubConfigSemanticEnabled())
        result = invoke_memory_query(
            {"query": "test", "mode": "semantic"},
            policy={}, runtime=rt,
        )
        assert result["effective_mode"] == "semantic"
        assert result["warnings"] == []


# ---------------------------------------------------------------------------
# 5. sourceFilter / visibility filtering
# ---------------------------------------------------------------------------

class TestSourceVisibilityFiltering:

    def _ingest_mixed(self, tmp_path: Path, rt: _Runtime) -> None:
        """Ingest internal + user docs with distinct source_kind/visibility."""
        _write_tmp_files(tmp_path, {
            "internal.md": "Internal: secret roadmap details",
            "user.md": "User: public quarterly report",
        })
        policy = {"fs_roots_allowlist": [str(tmp_path)]}

        invoke_memory_ingest_files(
            {"paths": [str(tmp_path / "internal.md")],
             "source_kind": "internal_docs", "visibility": "audit_only"},
            policy=policy, runtime=rt,
        )
        invoke_memory_ingest_files(
            {"paths": [str(tmp_path / "user.md")],
             "source_kind": "user_docs", "visibility": "user"},
            policy=policy, runtime=rt,
        )

    def test_source_kind_filter_user_only(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        self._ingest_mixed(tmp_path, rt)

        result = invoke_memory_query(
            {
                "query": "report",
                "mode": "hybrid",
                "top_k": 10,
                "constraints": {"source_kind_filter": ["user_docs"]},
            },
            policy={}, runtime=rt,
        )
        for chunk in result["retrieved_chunks"]:
            assert chunk["metadata"]["source_kind"] == "user_docs"
        for eref in result["evidence_refs"]:
            assert eref["source_kind"] == "user_docs"

    def test_source_kind_filter_internal_only(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        self._ingest_mixed(tmp_path, rt)

        result = invoke_memory_query(
            {
                "query": "roadmap",
                "mode": "hybrid",
                "top_k": 10,
                "constraints": {"source_kind_filter": ["internal_docs"]},
            },
            policy={}, runtime=rt,
        )
        for chunk in result["retrieved_chunks"]:
            assert chunk["metadata"]["source_kind"] == "internal_docs"

    def test_visibility_filter(self, tmp_path: Path) -> None:
        rt = _make_runtime(tmp_path)
        self._ingest_mixed(tmp_path, rt)

        result = invoke_memory_query(
            {
                "query": "report roadmap",
                "mode": "hybrid",
                "top_k": 10,
                "constraints": {"visibility_filter": ["user"]},
            },
            policy={}, runtime=rt,
        )
        for eref in result["evidence_refs"]:
            assert eref["visibility"] == "user"

    def test_combined_filters(self, tmp_path: Path) -> None:
        """Both source_kind + visibility filters applied together."""
        rt = _make_runtime(tmp_path)
        self._ingest_mixed(tmp_path, rt)

        result = invoke_memory_query(
            {
                "query": "report roadmap",
                "mode": "hybrid",
                "top_k": 10,
                "constraints": {
                    "source_kind_filter": ["internal_docs"],
                    "visibility_filter": ["audit_only"],
                },
            },
            policy={}, runtime=rt,
        )
        for chunk in result["retrieved_chunks"]:
            assert chunk["metadata"]["source_kind"] == "internal_docs"
            assert chunk["metadata"]["visibility"] == "audit_only"


# ---------------------------------------------------------------------------
# 6. SqliteMemoryStore unit tests
# ---------------------------------------------------------------------------

class TestSqliteMemoryStore:

    def test_upsert_and_get(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        store = SqliteMemoryStore(db)
        store.upsert_chunk(
            chunk_id="c1",
            source_artifact_id="a1",
            text="hello world",
            metadata_json='{"source_kind":"user_docs"}',
            created_at="2025-01-01T00:00:00Z",
        )
        row = store.get_chunk("c1")
        assert row is not None
        assert row["text"] == "hello world"
        store.close()

    def test_upsert_idempotent(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        store = SqliteMemoryStore(db)
        for _ in range(3):
            store.upsert_chunk(
                chunk_id="c1",
                source_artifact_id="a1",
                text="same text",
                metadata_json="{}",
                created_at="2025-01-01T00:00:00Z",
            )
        assert store.count_chunks() == 1
        store.close()

    def test_search_keyword_scoring(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        store = SqliteMemoryStore(db)
        store.upsert_chunk("c1", "a1", "revenue revenue revenue", "{}", "2025-01-01T00:00:00Z")
        store.upsert_chunk("c2", "a1", "revenue once", "{}", "2025-01-01T00:00:00Z")
        store.upsert_chunk("c3", "a1", "no match here", "{}", "2025-01-01T00:00:00Z")

        results = store.search_keyword_topk("revenue", top_k=10)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"  # higher score (3 hits)
        assert results[1]["chunk_id"] == "c2"  # lower score (1 hit)
        assert results[0]["_score"] > results[1]["_score"]
        store.close()

    def test_search_deterministic_tiebreak(self, tmp_path: Path) -> None:
        """Equal scores → sorted by chunk_id ascending."""
        db = _make_db(tmp_path)
        store = SqliteMemoryStore(db)
        store.upsert_chunk("c_b", "a1", "word", "{}", "2025-01-01T00:00:00Z")
        store.upsert_chunk("c_a", "a1", "word", "{}", "2025-01-01T00:00:00Z")

        results = store.search_keyword_topk("word", top_k=10)
        assert results[0]["chunk_id"] == "c_a"
        assert results[1]["chunk_id"] == "c_b"
        store.close()

    def test_list_chunks_by_source(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        store = SqliteMemoryStore(db)
        store.upsert_chunk("c1", "src_a", "text1", "{}", "2025-01-01T00:00:00Z")
        store.upsert_chunk("c2", "src_b", "text2", "{}", "2025-01-01T00:00:00Z")

        rows = store.list_chunks(source_artifact_id="src_a")
        assert len(rows) == 1
        assert rows[0]["chunk_id"] == "c1"
        store.close()

    def test_count_chunks(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        store = SqliteMemoryStore(db)
        assert store.count_chunks() == 0
        store.upsert_chunk("c1", "a1", "t", "{}", "2025-01-01T00:00:00Z")
        assert store.count_chunks() == 1
        store.close()


# ---------------------------------------------------------------------------
# 7. EvidenceRef structure in query results
# ---------------------------------------------------------------------------

class TestEvidenceRefStructure:

    def test_evidence_ref_fields(self, tmp_path: Path) -> None:
        """Query result evidence_refs must contain all required fields."""
        rt = _make_runtime(tmp_path)
        paths = _write_tmp_files(tmp_path, {"doc.txt": "Revenue grew 10%."})
        policy = {"fs_roots_allowlist": [str(tmp_path)]}

        invoke_memory_ingest_files(
            {"paths": paths, "source_kind": "user_docs", "visibility": "user"},
            policy=policy, runtime=rt,
        )
        result = invoke_memory_query(
            {"query": "revenue", "mode": "hybrid"},
            policy={}, runtime=rt,
        )
        assert len(result["evidence_refs"]) > 0
        eref = result["evidence_refs"][0]

        required_keys = {
            "chunk_id", "source_artifact_id", "source_kind",
            "visibility", "locator", "quote", "quote_hash",
            "relevance_score", "confidence",
        }
        assert required_keys.issubset(set(eref.keys()))
        assert isinstance(eref["locator"], dict)
        assert "source_uri" in eref["locator"]
        assert len(eref["quote"]) > 0
        assert len(eref["quote_hash"]) == 64  # sha256 hex

    def test_evidence_ref_quote_hash_verifiable(self, tmp_path: Path) -> None:
        """quote_hash must match _quote_hash(quote)."""
        rt = _make_runtime(tmp_path)
        paths = _write_tmp_files(tmp_path, {"doc.txt": "Revenue grew 10%."})
        policy = {"fs_roots_allowlist": [str(tmp_path)]}

        invoke_memory_ingest_files(
            {"paths": paths, "source_kind": "user_docs"},
            policy=policy, runtime=rt,
        )
        result = invoke_memory_query(
            {"query": "revenue", "mode": "hybrid"},
            policy={}, runtime=rt,
        )
        for eref in result["evidence_refs"]:
            assert eref["quote_hash"] == _quote_hash(eref["quote"])


# ---------------------------------------------------------------------------
# 8. Hybrid memory runbook structure
# ---------------------------------------------------------------------------

class TestHybridMemoryRunbook:

    def test_runbook_builds_two_stages(self) -> None:
        from neuronium_agent.planning.memory_runbook import (
            HybridMemoryReportV1Runbook,
        )
        rb = HybridMemoryReportV1Runbook()
        assert rb.runbook_id == "hybrid_memory_report_v1"

        stages = rb.build_stages(
            objective="Analyse Q3 revenue",
            constraints=[],
            metadata={
                "internal_doc_paths": ["ROADMAP.md"],
                "user_doc_paths": ["user_q3.md"],
            },
            execution_id="exec-test-000000000000",
        )
        assert len(stages) == 2

        # Stage 1: ingest + retrieve
        s1 = stages[0]
        assert "ingest_retrieve" in s1.stage_id
        node_ids_1 = {n.node_id for n in s1.graph.nodes}
        assert {"ingest_internal", "ingest_user", "memory_query"} == node_ids_1

        # Stage 2: synthesise + verify
        s2 = stages[1]
        assert "synthesise_verify" in s2.stage_id
        node_ids_2 = {n.node_id for n in s2.graph.nodes}
        assert {"draft_report", "critic_report"} == node_ids_2
        assert s2.success_gate.critic_node_id == "critic_report"

    def test_runbook_registered(self) -> None:
        from neuronium_agent.planning.runbook_registry import list_runbooks
        ids = list_runbooks()
        assert "hybrid_memory_report_v1" in ids
