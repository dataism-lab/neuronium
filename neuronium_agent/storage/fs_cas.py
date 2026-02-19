"""Filesystem Content-Addressed Store (STORAGE_SCHEMA §1).

Layout:
    <root>/sha256/<p1p2>/<p3p4>/<artifact_id>.blob
    <root>/sha256/<p1p2>/<p3p4>/<artifact_id>.meta.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from neuronium_agent.errors import StorageError
from neuronium_agent.storage.blob_store import BlobStore


def _shard_path(root: Path, artifact_id: str) -> Path:
    """Compute sharded directory + filename for an artifact ID.

    Expects ``sha256:<hex>``.  Uses first 4 hex chars for two-level sharding.
    """
    _, hex_hash = artifact_id.split(":", maxsplit=1)
    p1p2 = hex_hash[:2]
    p3p4 = hex_hash[2:4]
    return root / "sha256" / p1p2 / p3p4


class FsCasStore(BlobStore):
    """Immutable filesystem CAS blob store."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # -- BlobStore interface --------------------------------------------------

    def put(self, artifact_id: str, blob_bytes: bytes, media_type: str) -> None:
        shard = _shard_path(self._root, artifact_id)
        shard.mkdir(parents=True, exist_ok=True)
        blob_path = shard / f"{artifact_id}.blob"
        meta_path = shard / f"{artifact_id}.meta.json"

        if blob_path.exists():
            return  # idempotent

        blob_path.write_bytes(blob_bytes)

        meta = {
            "artifact_id": artifact_id,
            "media_type": media_type,
            "size_bytes": len(blob_bytes),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": artifact_id,
        }
        meta_path.write_text(
            json.dumps(meta, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def get(self, artifact_id: str) -> bytes:
        shard = _shard_path(self._root, artifact_id)
        blob_path = shard / f"{artifact_id}.blob"
        if not blob_path.exists():
            raise StorageError(f"Blob not found: {artifact_id}")
        return blob_path.read_bytes()

    def exists(self, artifact_id: str) -> bool:
        shard = _shard_path(self._root, artifact_id)
        return (shard / f"{artifact_id}.blob").exists()
