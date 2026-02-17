"""Tests for storage layer (FS CAS + SQLite)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from neuronium_agent._canonical import canonical_json
from neuronium_agent.errors import StorageError
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


class TestFsCasStore:
    """FS CAS blob store tests."""

    def test_put_and_get(self, blob_store: FsCasStore) -> None:
        aid = "sha256:aabbccdd0011223344556677889900aabbccdd0011223344556677889900aabb"
        data = b"hello world"
        blob_store.put(aid, data, "text/plain")
        assert blob_store.get(aid) == data

    def test_exists(self, blob_store: FsCasStore) -> None:
        aid = "sha256:1122334455667788990011223344556677889900aabbccdd0011223344556677"
        assert not blob_store.exists(aid)
        blob_store.put(aid, b"data", "application/octet-stream")
        assert blob_store.exists(aid)

    def test_get_missing_raises(self, blob_store: FsCasStore) -> None:
        with pytest.raises(StorageError):
            blob_store.get("sha256:0000000000000000000000000000000000000000000000000000000000000000")

    def test_put_idempotent(self, blob_store: FsCasStore) -> None:
        aid = "sha256:aaaa000000000000000000000000000000000000000000000000000000000000"
        blob_store.put(aid, b"data", "text/plain")
        blob_store.put(aid, b"data", "text/plain")  # should not raise
        assert blob_store.get(aid) == b"data"


class TestSqliteIndexStore:
    """SQLite index store tests."""

    def test_upsert_and_get_run(self, index_store: SqliteIndexStore) -> None:
        index_store.upsert_run(
            trace_id="t1",
            execution_id="e1",
            state="RUNNING",
            objective="test",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        run = index_store.get_run("t1")
        assert run is not None
        assert run["state"] == "RUNNING"

    def test_update_run_state(self, index_store: SqliteIndexStore) -> None:
        index_store.upsert_run(
            trace_id="t2",
            execution_id="e2",
            state="PENDING",
            objective="test2",
            config_snapshot_json="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        index_store.update_run_state("t2", "COMPLETED")
        run = index_store.get_run("t2")
        assert run["state"] == "COMPLETED"

    def test_append_and_get_trace_events(
        self, index_store: SqliteIndexStore
    ) -> None:
        index_store.append_trace_event("t1", {
            "ts": "2026-01-01T00:00:00Z",
            "kind": "decision",
            "payload": {"msg": "hello"},
        })
        index_store.append_trace_event("t1", {
            "ts": "2026-01-01T00:00:01Z",
            "kind": "node_start",
            "payload": {"node_id": "n1"},
        })
        events = list(index_store.get_trace_events("t1"))
        assert len(events) == 2
        assert events[0]["kind"] == "decision"
        assert events[1]["kind"] == "node_start"

    def test_record_artifact_metadata(
        self, index_store: SqliteIndexStore
    ) -> None:
        index_store.record_artifact_metadata(
            artifact_id="sha256:abc123",
            artifact_type="test_output",
            created_at=datetime.now(timezone.utc).isoformat(),
            produced_by_node_ref="exec:plan/phase/node",
            inputs_json="[]",
            quality_signals_json="{}",
            blob_key="sha256:abc123",
            media_type="application/json",
            size_bytes=42,
        )
        art = index_store.get_artifact("sha256:abc123")
        assert art is not None
        assert art["artifact_type"] == "test_output"

    def test_record_lineage_edge(self, index_store: SqliteIndexStore) -> None:
        index_store.record_lineage_edge("parent1", "child1", "producedFrom")
        # Idempotent — should not raise
        index_store.record_lineage_edge("parent1", "child1", "producedFrom")

    def test_upsert_node_execution(
        self, index_store: SqliteIndexStore
    ) -> None:
        index_store.upsert_node_execution(
            node_execution_id="ne1",
            trace_id="t1",
            node_ref="ref1",
            attempt=1,
            status="RUNNING",
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=None,
            inputs_json="{}",
            outputs_json=None,
            error_json=None,
        )
        # Update to completed
        index_store.upsert_node_execution(
            node_execution_id="ne1",
            trace_id="t1",
            node_ref="ref1",
            attempt=1,
            status="COMPLETED",
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
            inputs_json="{}",
            outputs_json='{"result": "ok"}',
            error_json=None,
        )
