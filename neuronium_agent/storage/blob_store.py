"""Abstract Blob Store (content-addressed) — PUBLIC_API_SPEC §4.1."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BlobStore(ABC):
    """Content-addressed immutable blob storage."""

    @abstractmethod
    def put(self, artifact_id: str, blob_bytes: bytes, media_type: str) -> None:
        """Store *blob_bytes* under *artifact_id*. Idempotent."""

    @abstractmethod
    def get(self, artifact_id: str) -> bytes:
        """Retrieve raw bytes. Raises ``StorageError`` if not found."""

    @abstractmethod
    def exists(self, artifact_id: str) -> bool:
        """Check whether the blob exists."""
