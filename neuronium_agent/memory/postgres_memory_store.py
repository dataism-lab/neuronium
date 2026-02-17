"""Postgres-backed MemoryStore (optional — extras ``[postgres]``, Stage 5).

Mirrors :class:`SqliteMemoryStore` using ``psycopg`` (v3).
"""

from __future__ import annotations

import json
from typing import Any

from neuronium_agent.errors import StorageError
from neuronium_agent.memory.store import MemoryStore

try:
    import psycopg  # type: ignore[import-untyped]
except ImportError:
    psycopg = None  # type: ignore[assignment]


class PostgresMemoryStore(MemoryStore):
    """Postgres memory-chunk store (production adapter)."""

    def __init__(self, dsn: str, *, schema: str = "neuronium_agent") -> None:
        if psycopg is None:
            raise StorageError(
                "psycopg not installed. Run: pip install neuronium-agent[postgres]"
            )
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._schema = schema
        self._conn.execute(f"SET search_path TO {self._schema}")

    # -- write ---------------------------------------------------------------

    def upsert_chunk(
        self,
        chunk_id: str,
        source_artifact_id: str,
        text: str,
        metadata_json: str,
        created_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_chunks
                (chunk_id, source_artifact_id, text, metadata_json, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (chunk_id, source_artifact_id, text, metadata_json, created_at),
        )

    # -- read ----------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM memory_chunks WHERE chunk_id=%s", (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        cols = [d.name for d in (self._conn.execute(
            "SELECT * FROM memory_chunks LIMIT 0"
        ).description or [])]
        return dict(zip(cols, row))

    def list_chunks(
        self,
        *,
        source_artifact_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        order_by: str = "chunk_id",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if source_artifact_id is not None:
            clauses.append("source_artifact_id = %s")
            params.append(source_artifact_id)

        if metadata_filters:
            for key, value in sorted(metadata_filters.items()):
                clauses.append(
                    "metadata_json::jsonb ->> %s = %s"
                )
                params.append(key)
                params.append(
                    json.dumps(value) if not isinstance(value, str) else value,
                )

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        safe_order = order_by if order_by in ("chunk_id", "created_at") else "chunk_id"

        rows = self._conn.execute(
            f"SELECT * FROM memory_chunks{where} ORDER BY {safe_order} LIMIT %s",
            tuple(params) + (limit,),
        ).fetchall()

        if not rows:
            return []
        cols = [d.name for d in (self._conn.execute(
            "SELECT * FROM memory_chunks LIMIT 0"
        ).description or [])]
        return [dict(zip(cols, r)) for r in rows]

    def search_keyword_topk(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword search using Postgres ILIKE with deterministic scoring."""
        words = [w.strip().lower() for w in query.split() if w.strip()]
        if not words:
            return []

        # Build scoring expression: count matches per word.
        # Fallback: fetch all, score in Python (same as SQLite for consistency).
        all_chunks = self.list_chunks(
            metadata_filters=metadata_filters,
            limit=10_000,
        )

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in all_chunks:
            text_lower = row["text"].lower()
            score = sum(text_lower.count(w) for w in words)
            if score > 0:
                scored.append((score, row["chunk_id"], row))

        scored.sort(key=lambda t: (-t[0], t[1]))
        results: list[dict[str, Any]] = []
        for score_val, _cid, row in scored[:top_k]:
            row["_score"] = float(score_val)
            results.append(row)
        return results

    def count_chunks(
        self,
        *,
        metadata_filters: dict[str, Any] | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if metadata_filters:
            for key, value in sorted(metadata_filters.items()):
                clauses.append("metadata_json::jsonb ->> %s = %s")
                params.append(key)
                params.append(
                    json.dumps(value) if not isinstance(value, str) else value,
                )
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) AS cnt FROM memory_chunks{where}",
            tuple(params),
        ).fetchone()
        return int(row[0]) if row else 0

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
