"""Postgres index store adapter (optional — extras ``[postgres]``).

Structurally mirrors :class:`SqliteIndexStore`, uses ``psycopg`` (v3).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

from neuronium_agent.errors import StorageError
from neuronium_agent.storage.index_store import IndexStore

try:
    import psycopg  # type: ignore[import-untyped]
except ImportError:
    psycopg = None  # type: ignore[assignment]


class PostgresIndexStore(IndexStore):
    """Postgres-backed index store (production adapter)."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "neuronium_agent",
        auto_migrate: bool = True,
    ) -> None:
        if psycopg is None:
            raise StorageError(
                "psycopg not installed. Run: pip install neuronium-agent[postgres]"
            )
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._schema = schema

        # Ensure schema exists
        self._conn.execute(
            f"CREATE SCHEMA IF NOT EXISTS {self._schema}"
        )
        self._conn.execute(f"SET search_path TO {self._schema}")

        if auto_migrate:
            self._apply_pg_migrations()

    def _apply_pg_migrations(self) -> None:
        """Apply Postgres migrations (reuse the same pattern as SQLite)."""
        from neuronium_agent.storage.migrator import _migration_dir

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL
            )
            """
        )

        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM schema_version"
        ).fetchone()
        current = row[0] if row else 0

        mig_dir = _migration_dir("postgres")
        if not mig_dir.exists():
            return

        for sql_file in sorted(mig_dir.glob("*.sql")):
            version = int(sql_file.stem.split("_", maxsplit=1)[0])
            if version <= current:
                continue
            sql = sql_file.read_text(encoding="utf-8")
            self._conn.execute(sql)
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) "
                "VALUES (%s, %s)",
                (version, now),
            )

    # -- runs ----------------------------------------------------------------

    def upsert_run(
        self,
        trace_id: str,
        execution_id: str,
        state: str,
        objective: str,
        config_snapshot_json: str,
        created_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (trace_id, execution_id, state, objective,
                              config_snapshot_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(trace_id) DO UPDATE SET state=EXCLUDED.state
            """,
            (trace_id, execution_id, state, objective,
             config_snapshot_json, created_at),
        )

    def get_run(self, trace_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE trace_id=%s", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d.name for d in self._conn.execute(
            "SELECT * FROM runs LIMIT 0"
        ).description or []]
        return dict(zip(cols, row))

    def update_run_state(self, trace_id: str, state: str) -> None:
        self._conn.execute(
            "UPDATE runs SET state=%s WHERE trace_id=%s",
            (state, trace_id),
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
        self._conn.execute(
            """
            INSERT INTO artifacts
                (artifact_id, artifact_type, created_at,
                 produced_by_node_ref, inputs_json,
                 quality_signals_json, blob_key,
                 media_type, size_bytes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(artifact_id) DO NOTHING
            """,
            (
                artifact_id, artifact_type, created_at,
                produced_by_node_ref, inputs_json,
                quality_signals_json, blob_key,
                media_type, size_bytes,
            ),
        )

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id=%s", (artifact_id,)
        ).fetchone()
        return dict(row) if row else None  # type: ignore[arg-type]

    # -- lineage -------------------------------------------------------------

    def record_lineage_edge(
        self, parent_id: str, child_id: str, kind: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO lineage_edges
                (parent_artifact_id, child_artifact_id, kind, created_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (parent_id, child_id, kind, now),
        )

    # -- trace events --------------------------------------------------------

    def append_trace_event(
        self, trace_id: str, event: dict[str, Any]
    ) -> None:
        ts = event.get("ts", datetime.now(timezone.utc).isoformat())
        self._conn.execute(
            """
            INSERT INTO trace_events
                (trace_id, ts, span_id, parent_span_id, kind, payload_json)
            VALUES (%s, %s, %s, %s, %s, %s)
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
        rows = self._conn.execute(
            """
            SELECT event_id, trace_id, ts, span_id, parent_span_id,
                   kind, payload_json
            FROM trace_events
            WHERE trace_id=%s
            ORDER BY event_id
            """,
            (trace_id,),
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                "event_id": r[0],
                "trace_id": r[1],
                "ts": str(r[2]),
                "span_id": r[3],
                "parent_span_id": r[4],
                "kind": r[5],
                "payload": json.loads(r[6]) if isinstance(r[6], str) else r[6],
            })
        return result

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
        self._conn.execute(
            """
            INSERT INTO node_executions
                (node_execution_id, trace_id, node_ref, attempt,
                 status, started_at, ended_at,
                 inputs_json, outputs_json, error_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(node_execution_id)
            DO UPDATE SET
                status=EXCLUDED.status,
                ended_at=EXCLUDED.ended_at,
                outputs_json=EXCLUDED.outputs_json,
                error_json=EXCLUDED.error_json
            """,
            (
                node_execution_id, trace_id, node_ref, attempt,
                status, started_at, ended_at,
                inputs_json, outputs_json, error_json,
            ),
        )

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
