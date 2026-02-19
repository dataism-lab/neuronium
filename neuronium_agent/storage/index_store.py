"""Abstract Index Store — PUBLIC_API_SPEC §4.1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class IndexStore(ABC):
    """Metadata / lineage / trace index (SQLite or Postgres)."""

    # -- runs ----------------------------------------------------------------
    @abstractmethod
    def upsert_run(
        self,
        trace_id: str,
        execution_id: str,
        state: str,
        objective: str,
        config_snapshot_json: str,
        created_at: str,
    ) -> None: ...

    @abstractmethod
    def get_run(self, trace_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def update_run_state(self, trace_id: str, state: str) -> None: ...

    # -- artifacts -----------------------------------------------------------
    @abstractmethod
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
    ) -> None: ...

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def mark_artifacts_deprecated(
        self, artifact_ids: list[str], reason: str = "rollback"
    ) -> None:
        """Mark artifacts as deprecated for lineage (B1 Part 2 §3.4.1)."""

    # -- lineage -------------------------------------------------------------
    @abstractmethod
    def record_lineage_edge(
        self,
        parent_id: str,
        child_id: str,
        kind: str,
    ) -> None: ...

    # -- trace events --------------------------------------------------------
    @abstractmethod
    def append_trace_event(
        self,
        trace_id: str,
        event: dict[str, Any],
    ) -> None: ...

    @abstractmethod
    def get_trace_events(
        self,
        trace_id: str,
    ) -> Iterable[dict[str, Any]]: ...

    # -- node executions -----------------------------------------------------
    @abstractmethod
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
    ) -> None: ...

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    def close(self) -> None: ...
