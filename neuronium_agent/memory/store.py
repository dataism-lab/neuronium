"""Abstract MemoryStore — internal interface for chunk persistence (Stage 5).

Concrete backends: :class:`SqliteMemoryStore`, :class:`PostgresMemoryStore`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    """Internal interface for memory-chunk storage and retrieval."""

    # -- write ---------------------------------------------------------------

    @abstractmethod
    def upsert_chunk(
        self,
        chunk_id: str,
        source_artifact_id: str,
        text: str,
        metadata_json: str,
        created_at: str,
    ) -> None:
        """Insert or ignore a memory chunk."""

    # -- read ----------------------------------------------------------------

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Return a single chunk row as dict, or *None*."""

    @abstractmethod
    def list_chunks(
        self,
        *,
        source_artifact_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        order_by: str = "chunk_id",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return chunks matching filters, ordered deterministically."""

    @abstractmethod
    def search_keyword_topk(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword search across chunk text.

        Returns rows sorted by relevance score (desc), then ``chunk_id``
        (asc) for deterministic tie-breaking.
        """

    @abstractmethod
    def count_chunks(
        self,
        *,
        metadata_filters: dict[str, Any] | None = None,
    ) -> int:
        """Return total number of chunks matching optional filters."""

    # -- lifecycle -----------------------------------------------------------

    @abstractmethod
    def close(self) -> None: ...
