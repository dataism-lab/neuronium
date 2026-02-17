"""Public API facade (PUBLIC_API_SPEC §3).

``AgentRunner`` is the main entry point for external applications.
``create_runner(config)`` is the recommended factory.
"""

from __future__ import annotations

from typing import Any

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
    ) -> None:
        self._config = config
        self._blob = blob_store
        self._index = index_store
        self._orchestrator = Orchestrator(config, blob_store, index_store)
        self._exporter = TraceExporter()

    # -- Mandatory methods (PUBLIC_API_SPEC) --------------------------------

    def start(self, request: RunRequest) -> RunHandle:
        """Start an agent run and return a handle."""
        return self._orchestrator.start(request)

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_runner(config: AppConfig | None = None) -> AgentRunner:
    """Create an :class:`AgentRunner` from config (PUBLIC_API_SPEC §3.2).

    If *config* is ``None``, loads from the default config chain.
    """
    if config is None:
        config = load_config()

    blob_store = _create_blob_store(config)
    index_store = _create_index_store(config)

    return AgentRunner(config, blob_store, index_store)


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
