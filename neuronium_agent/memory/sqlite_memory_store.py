"""SQLite-backed MemoryStore (default OSS, Stage 5).

Operates on the same database file as :class:`SqliteIndexStore`.
Tables ``memory_chunks`` / ``memory_embeddings`` are created by the
shared migration ``0001_init.sql``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from neuronium_agent.memory.store import MemoryStore


class SqliteMemoryStore(MemoryStore):
    """SQLite memory-chunk store (shares the index DB by default)."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()

    # -- helpers -------------------------------------------------------------

    def _execute_commit(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchone()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchall()

    # -- write ---------------------------------------------------------------

    def upsert_chunk(
        self,
        chunk_id: str,
        source_artifact_id: str,
        text: str,
        metadata_json: str,
        created_at: str,
    ) -> None:
        try:
            self._execute_commit(
                """
                INSERT INTO memory_chunks
                    (chunk_id, source_artifact_id, text, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, source_artifact_id, text, metadata_json, created_at),
            )
        except sqlite3.IntegrityError:
            pass  # idempotent

    # -- read ----------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM memory_chunks WHERE chunk_id=?", (chunk_id,)
        )
        return dict(row) if row else None

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
            clauses.append("source_artifact_id = ?")
            params.append(source_artifact_id)

        if metadata_filters:
            for key, value in sorted(metadata_filters.items()):
                clauses.append(
                    "json_extract(metadata_json, ?) = ?"
                )
                params.append(f"$.{key}")
                params.append(json.dumps(value) if not isinstance(value, str) else value)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        # Deterministic ordering: only allow safe column names.
        safe_order = order_by if order_by in ("chunk_id", "created_at") else "chunk_id"

        rows = self._fetchall(
            f"SELECT * FROM memory_chunks{where} ORDER BY {safe_order} LIMIT ?",
            tuple(params) + (limit,),
        )
        return [dict(r) for r in rows]

    def search_keyword_topk(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Simple keyword search using SQLite LIKE with deterministic scoring.

        Scoring: number of non-overlapping occurrences of each query word
        (case-insensitive) in the chunk text.  Ties broken by ``chunk_id``.
        """
        words = [w.strip().lower() for w in query.split() if w.strip()]
        if not words:
            return []

        clauses: list[str] = []
        params: list[Any] = []

        if metadata_filters:
            for key, value in sorted(metadata_filters.items()):
                clauses.append("json_extract(metadata_json, ?) = ?")
                params.append(f"$.{key}")
                params.append(
                    json.dumps(value) if not isinstance(value, str) else value,
                )

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._fetchall(
            f"SELECT * FROM memory_chunks{where} ORDER BY chunk_id",
            tuple(params),
        )

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in rows:
            row_dict = dict(row)
            text_lower = row_dict["text"].lower()
            score = sum(text_lower.count(w) for w in words)
            if score > 0:
                scored.append((score, row_dict["chunk_id"], row_dict))

        # Descending score, ascending chunk_id for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))
        results: list[dict[str, Any]] = []
        for score_val, _cid, row_dict in scored[:top_k]:
            row_dict["_score"] = float(score_val)
            results.append(row_dict)
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
                clauses.append("json_extract(metadata_json, ?) = ?")
                params.append(f"$.{key}")
                params.append(
                    json.dumps(value) if not isinstance(value, str) else value,
                )
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        row = self._fetchone(
            f"SELECT COUNT(*) AS cnt FROM memory_chunks{where}",
            tuple(params),
        )
        return int(row["cnt"]) if row else 0

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()
