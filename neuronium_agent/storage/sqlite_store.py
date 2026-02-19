"""SQLite index store (STORAGE_SCHEMA §2)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from neuronium_agent.errors import StorageError
from neuronium_agent.storage.index_store import IndexStore
from neuronium_agent.storage.migrator import apply_migrations


class SqliteIndexStore(IndexStore):
    """SQLite-backed metadata / lineage / trace index (default OSS)."""

    def __init__(self, db_path: str | Path, *, auto_migrate: bool = True) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Used from DAGExecutor worker threads via TraceRecorder.
        # We serialize DB access with a lock.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()

        if auto_migrate:
            apply_migrations(self._conn, dialect="sqlite")

    def _execute_commit(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a statement and commit atomically under the lock."""
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        """Execute a query and fetchone atomically under the lock."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchone()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute a query and fetchall atomically under the lock."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchall()

    # -- runs ---------------------------------------------------------------

    def upsert_run(
        self,
        trace_id: str,
        execution_id: str,
        state: str,
        objective: str,
        config_snapshot_json: str,
        created_at: str,
    ) -> None:
        self._execute_commit(
            """
            INSERT INTO runs (trace_id, execution_id, state, objective,
                              config_snapshot_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET state=excluded.state
            """,
            (trace_id, execution_id, state, objective,
             config_snapshot_json, created_at),
        )

    def get_run(self, trace_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM runs WHERE trace_id=?", (trace_id,)
        )
        return dict(row) if row else None

    def update_run_state(self, trace_id: str, state: str) -> None:
        self._execute_commit(
            "UPDATE runs SET state=? WHERE trace_id=?", (state, trace_id)
        )

    # -- artifacts -----------------------------------------------------------

    def record_artifact_metadata(
        self,
        artifact_id: str,
        artifact_type: str,
        created_at: str,
        produced_by_node_ref: str,
        inputs_json: str,
        quality_signals_json: str,
        blob_key: str,
        media_type: str,
        size_bytes: int,
    ) -> None:
        try:
            self._execute_commit(
                """
                INSERT INTO artifacts
                    (artifact_id, artifact_type, created_at,
                     produced_by_node_ref, inputs_json,
                     quality_signals_json, blob_key,
                     media_type, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id, artifact_type, created_at,
                    produced_by_node_ref, inputs_json,
                    quality_signals_json, blob_key,
                    media_type, size_bytes,
                ),
            )
        except sqlite3.IntegrityError:
            pass  # idempotent — already exists

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)
        )
        return dict(row) if row else None

    def mark_artifacts_deprecated(
        self, artifact_ids: list[str], reason: str = "rollback"
    ) -> None:
        if not artifact_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in artifact_ids)
        with self._lock:
            self._conn.execute(
                f"UPDATE artifacts SET deprecated_at=? WHERE artifact_id IN ({placeholders})",
                (now, *artifact_ids),
            )
            self._conn.commit()

    # -- lineage -------------------------------------------------------------

    def record_lineage_edge(
        self, parent_id: str, child_id: str, kind: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._execute_commit(
                """
                INSERT INTO lineage_edges
                    (parent_artifact_id, child_artifact_id, kind, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (parent_id, child_id, kind, now),
            )
        except sqlite3.IntegrityError:
            pass

    # -- trace events --------------------------------------------------------

    def append_trace_event(
        self, trace_id: str, event: dict[str, Any]
    ) -> None:
        ts = event.get("ts", datetime.now(timezone.utc).isoformat())
        self._execute_commit(
            """
            INSERT INTO trace_events
                (trace_id, ts, span_id, parent_span_id, kind, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                ts,
                event.get("span_id"),
                event.get("parent_span_id"),
                event.get("kind", "unknown"),
                json.dumps(event.get("payload", {}), sort_keys=True),
            ),
        )

    def get_trace_events(
        self, trace_id: str
    ) -> Iterable[dict[str, Any]]:
        rows = self._fetchall(
            """
            SELECT event_id, trace_id, ts, span_id, parent_span_id,
                   kind, payload_json
            FROM trace_events
            WHERE trace_id=?
            ORDER BY event_id
            """,
            (trace_id,),
        )
        return [
            {
                "event_id": r["event_id"],
                "trace_id": r["trace_id"],
                "ts": r["ts"],
                "span_id": r["span_id"],
                "parent_span_id": r["parent_span_id"],
                "kind": r["kind"],
                "payload": json.loads(r["payload_json"]),
            }
            for r in rows
        ]

    # -- node executions -----------------------------------------------------

    def upsert_node_execution(
        self,
        node_execution_id: str,
        trace_id: str,
        node_ref: str,
        attempt: int,
        status: str,
        started_at: str | None,
        ended_at: str | None,
        inputs_json: str,
        outputs_json: str | None,
        error_json: str | None,
    ) -> None:
        self._execute_commit(
            """
            INSERT INTO node_executions
                (node_execution_id, trace_id, node_ref, attempt,
                 status, started_at, ended_at,
                 inputs_json, outputs_json, error_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_execution_id)
            DO UPDATE SET
                status=excluded.status,
                ended_at=excluded.ended_at,
                outputs_json=excluded.outputs_json,
                error_json=excluded.error_json
            """,
            (
                node_execution_id, trace_id, node_ref, attempt,
                status, started_at, ended_at,
                inputs_json, outputs_json, error_json,
            ),
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()
