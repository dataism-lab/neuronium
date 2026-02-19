"""Immutability tests — artifacts must not be modifiable after creation (IBS §3.2).

Any "update" must create a *new* artifact with a new ID and lineage edge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuronium_agent._canonical import artifact_id, canonical_bytes
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


class TestBlobImmutability:
    """Blob store must be append-only (create + read, no update/delete)."""

    def test_blob_not_overwritten(self, blob_store: FsCasStore) -> None:
        """Putting the same ID twice does not change the content."""
        aid = "sha256:ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00"
        blob_store.put(aid, b"original", "text/plain")
        blob_store.put(aid, b"modified_attempt", "text/plain")
        # Original content is preserved
        assert blob_store.get(aid) == b"original"

    def test_different_content_different_id(self, blob_store: FsCasStore) -> None:
        """Different content always gets a different artifact ID."""
        ctx = {"ts": "2026-01-01T00:00:00Z", "node_ref": "n"}
        c1 = canonical_bytes({"value": "v1"})
        c2 = canonical_bytes({"value": "v2"})
        id1 = artifact_id(c1, ctx)
        id2 = artifact_id(c2, ctx)
        assert id1 != id2

        blob_store.put(id1, c1, "application/json")
        blob_store.put(id2, c2, "application/json")
        assert blob_store.get(id1) == c1
        assert blob_store.get(id2) == c2


class TestArtifactMetadataImmutability:
    """Index store artifact records must be append-only."""

    def test_duplicate_insert_is_noop(
        self, index_store: SqliteIndexStore
    ) -> None:
        """Inserting the same artifact_id twice does not overwrite."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        index_store.record_artifact_metadata(
            artifact_id="sha256:aaa",
            artifact_type="original_type",
            created_at=now,
            produced_by_node_ref="ref1",
            inputs_json="[]",
            quality_signals_json="{}",
            blob_key="sha256:aaa",
            media_type="application/json",
            size_bytes=10,
        )
        # Attempt to overwrite
        index_store.record_artifact_metadata(
            artifact_id="sha256:aaa",
            artifact_type="modified_type",
            created_at=now,
            produced_by_node_ref="ref2",
            inputs_json="[]",
            quality_signals_json="{}",
            blob_key="sha256:aaa",
            media_type="application/json",
            size_bytes=99,
        )
        art = index_store.get_artifact("sha256:aaa")
        # Original metadata preserved
        assert art["artifact_type"] == "original_type"
        assert art["size_bytes"] == 10


class TestTraceImmutability:
    """Trace events are append-only — cannot be updated or deleted."""

    def test_events_append_only(
        self, index_store: SqliteIndexStore
    ) -> None:
        """Adding events only grows the list, never modifies existing."""
        index_store.append_trace_event("t1", {
            "ts": "2026-01-01T00:00:00Z",
            "kind": "decision",
            "payload": {"step": 1},
        })
        events_before = list(index_store.get_trace_events("t1"))

        index_store.append_trace_event("t1", {
            "ts": "2026-01-01T00:00:01Z",
            "kind": "decision",
            "payload": {"step": 2},
        })
        events_after = list(index_store.get_trace_events("t1"))

        # First event unchanged
        assert events_after[0]["payload"] == events_before[0]["payload"]
        # New event appended
        assert len(events_after) == len(events_before) + 1


class TestLineageImmutability:
    """Lineage edges form an immutable provenance graph."""

    def test_update_creates_new_artifact(
        self,
        blob_store: FsCasStore,
        index_store: SqliteIndexStore,
    ) -> None:
        """Simulating an 'update' must create a new artifact + lineage edge."""
        from datetime import datetime, timezone

        ctx_base = {"ts": "2026-01-01", "node_ref": "n1"}
        content_v1 = canonical_bytes({"version": 1})
        id_v1 = artifact_id(content_v1, ctx_base)

        blob_store.put(id_v1, content_v1, "application/json")
        index_store.record_artifact_metadata(
            artifact_id=id_v1,
            artifact_type="document",
            created_at=datetime.now(timezone.utc).isoformat(),
            produced_by_node_ref="n1",
            inputs_json="[]",
            quality_signals_json="{}",
            blob_key=id_v1,
            media_type="application/json",
            size_bytes=len(content_v1),
        )

        # "Update" → new artifact
        ctx_v2 = {"ts": "2026-01-02", "node_ref": "n2"}
        content_v2 = canonical_bytes({"version": 2})
        id_v2 = artifact_id(content_v2, ctx_v2)

        blob_store.put(id_v2, content_v2, "application/json")
        index_store.record_artifact_metadata(
            artifact_id=id_v2,
            artifact_type="document",
            created_at=datetime.now(timezone.utc).isoformat(),
            produced_by_node_ref="n2",
            inputs_json=f'["{id_v1}"]',
            quality_signals_json="{}",
            blob_key=id_v2,
            media_type="application/json",
            size_bytes=len(content_v2),
        )

        # Lineage edge: v1 → v2
        index_store.record_lineage_edge(id_v1, id_v2, "transformedFrom")

        # Both versions exist independently
        assert blob_store.get(id_v1) == content_v1
        assert blob_store.get(id_v2) == content_v2
        assert id_v1 != id_v2
