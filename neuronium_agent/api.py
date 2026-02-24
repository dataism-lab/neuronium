"""Public API facade (PUBLIC_API_SPEC §3).

``AgentRunner`` is the main entry point for external applications.
``create_runner(config)`` is the recommended factory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from neuronium_agent.config import AppConfig, load_config
from neuronium_agent.core.orchestrator import Orchestrator
from neuronium_agent.errors import ConfigError, NeuroniumError
from neuronium_agent.storage.blob_store import BlobStore
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.index_store import IndexStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.trace.exporter import TraceExporter
from neuronium_agent.types import (
    ControlCommand,
    RunHandle,
    RunRequest,
    RunStatus,
    TraceExportFormat,
)


class AgentRunner:
    """Public facade for running the agent (PUBLIC_API_SPEC §3.1).

    Does not require Postgres / Redis for basic usage.
    """

    def __init__(
        self,
        config: AppConfig,
        blob_store: BlobStore,
        index_store: IndexStore,
        *,
        trace_event_listener: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._config = config
        self._blob = blob_store
        self._index = index_store
        self._orchestrator = Orchestrator(
            config,
            blob_store,
            index_store,
            trace_event_listener=trace_event_listener,
        )
        self._exporter = TraceExporter()

    # -- Mandatory methods (PUBLIC_API_SPEC) --------------------------------

    def start(
        self,
        request: RunRequest,
        *,
        on_handle_ready: Callable[[RunHandle], None] | None = None,
    ) -> RunHandle:
        """Start an agent run and return a handle.

        If on_handle_ready is provided (e.g. for interactive CLI), it is
        called with the handle before execution blocks, so the caller can
        send control commands during the run.
        """
        return self._orchestrator.start(request, on_handle_ready=on_handle_ready)

    def get_status(self, handle: RunHandle) -> RunStatus:
        """Get current status of a run."""
        return self._orchestrator.get_status(handle.trace_id)

    def control(self, handle: RunHandle, command: ControlCommand) -> RunStatus:
        """Send a declarative control command.

        Delegates to :meth:`Orchestrator.apply_control` which performs
        state transition → checkpoint → trace decision without executing
        DAG nodes.
        """
        return self._orchestrator.apply_control(handle.trace_id, command)

    def resume_run(self, trace_id: str) -> RunHandle:
        """Resume a run from its latest phase-boundary checkpoint.

        The run must be in RUNNING state (use ``control continue`` first
        if the run was paused).
        """
        return self._orchestrator.resume_run(trace_id)

    def export_trace(
        self,
        handle: RunHandle,
        format: TraceExportFormat,
        path: str,
    ) -> None:
        """Export trace events to a file."""
        events = list(self._index.get_trace_events(handle.trace_id))
        self._exporter.export(events, path, fmt=format)

    # -- Optional methods (v1 stub) -----------------------------------------

    def replay(self, trace_id: str) -> RunHandle:
        """Replay a run from recorded trace using strict offline mode."""
        from neuronium_agent.trace.replay import ReplayProvider, load_replay_events

        events = load_replay_events(self._index, trace_id)
        provider = ReplayProvider(events)
        return self._orchestrator.replay(trace_id, provider)

    def get_latest_pause_context(self, trace_id: str) -> dict[str, Any] | None:
        """Return latest paused checkpoint resume_context, if present."""
        events = list(self._index.get_trace_events(trace_id))
        for event in reversed(events):
            if event.get("kind") != "checkpoint":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            resume_context = payload.get("resume_context", {})
            if not isinstance(resume_context, dict):
                continue
            if resume_context.get("phase_boundary") == "paused":
                return resume_context
        return None

    def read_artifact_json(self, artifact_id: str) -> dict[str, Any]:
        """Load artifact bytes and decode as UTF-8 JSON object."""
        raw = self._blob.get(artifact_id)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Artifact JSON payload must be an object")
        return data

    def get_trace_events(self, trace_id: str) -> list[dict[str, Any]]:
        """Return all recorded trace events for *trace_id*."""
        return list(self._index.get_trace_events(trace_id))

    def get_latest_rendered_artifact_path(self, trace_id: str) -> str | None:
        """Return latest rendered HTML path for a run, if indexed."""
        index_path = Path(self._config.project.data_dir) / "rendered" / "index.jsonl"
        if not index_path.exists():
            return None
        lines = index_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("trace_id", "")) != trace_id:
                continue
            artifact_path = str(row.get("artifact_path", "")).strip()
            return artifact_path or None
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_runner(
    config: AppConfig | None = None,
    *,
    trace_event_listener: Callable[[dict[str, Any]], None] | None = None,
) -> AgentRunner:
    """Create an :class:`AgentRunner` from config (PUBLIC_API_SPEC §3.2).

    If *config* is ``None``, loads from the default config chain.
    """
    if config is None:
        config = load_config()

    blob_store = _create_blob_store(config)
    index_store = _create_index_store(config)

    return AgentRunner(
        config,
        blob_store,
        index_store,
        trace_event_listener=trace_event_listener,
    )


# ---------------------------------------------------------------------------
# Store factories
# ---------------------------------------------------------------------------

def _create_blob_store(config: AppConfig) -> BlobStore:
    if config.storage.blob_backend == "fs_cas":
        return FsCasStore(config.storage.fs_cas_root)
    raise ConfigError(
        f"Unsupported blob backend: {config.storage.blob_backend}"
    )


def _create_index_store(config: AppConfig) -> IndexStore:
    backend = config.storage.index_backend
    if backend == "sqlite":
        return SqliteIndexStore(
            config.storage.sqlite_path,
            auto_migrate=config.storage.migrations_auto_apply,
        )
    if backend == "postgres":
        dsn = config.storage.postgres_dsn
        if not dsn:
            raise ConfigError(
                "postgres_dsn is required when index_backend='postgres'"
            )
        from neuronium_agent.storage.postgres_store import PostgresIndexStore

        return PostgresIndexStore(
            dsn,
            schema=config.storage.postgres_schema,
            auto_migrate=config.storage.migrations_auto_apply,
        )
    raise ConfigError(f"Unsupported index backend: {backend}")
