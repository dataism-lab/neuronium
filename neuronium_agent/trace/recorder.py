"""Trace recorder — append-only event log (IBS §3, STORAGE_SCHEMA §2.2.5).

Every decision / node execution / tool call / critic evaluation
is recorded as a trace event with correlation IDs.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from neuronium_agent._canonical import canonical_json
from neuronium_agent.storage.index_store import IndexStore


class TraceRecorder:
    """Append-only trace event recorder.

    Writes events to the :class:`IndexStore` and maintains an in-memory log.
    """

    def __init__(
        self,
        trace_id: str,
        index_store: IndexStore,
    ) -> None:
        self.trace_id = trace_id
        self._store = index_store
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a trace event and persist it."""
        with self._lock:
            event = {
                "trace_id": self.trace_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "span_id": span_id or uuid.uuid4().hex[:16],
                "parent_span_id": parent_span_id,
                "kind": kind,
                "payload": payload,
            }
            self._events.append(event)
            self._store.append_trace_event(self.trace_id, event)
            return event

    def record_node_start(
        self,
        node_id: str,
        node_ref: str,
        node_type: str,
        inputs: dict[str, Any],
    ) -> str:
        """Record node execution start.  Returns span_id."""
        span_id = uuid.uuid4().hex[:16]
        self.record(
            "node_start",
            {
                "node_id": node_id,
                "node_ref": node_ref,
                "node_type": node_type,
                "inputs_summary": _summarise(inputs),
            },
            span_id=span_id,
        )
        return span_id

    def record_node_end(
        self,
        node_id: str,
        node_ref: str,
        status: str,
        outputs: dict[str, Any],
        error: str | None = None,
        span_id: str | None = None,
    ) -> None:
        self.record(
            "node_end",
            {
                "node_id": node_id,
                "node_ref": node_ref,
                "status": status,
                "outputs_summary": _summarise(outputs),
                "error": error,
            },
            span_id=span_id,
        )

    def record_decision(self, description: str, details: dict[str, Any]) -> None:
        self.record("decision", {"description": description, **details})

    def record_checkpoint(self, state_snapshot: dict[str, Any]) -> None:
        self.record("checkpoint", state_snapshot)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return a copy of in-memory events."""
        return list(self._events)

    def load_events(self) -> list[dict[str, Any]]:
        """Load all events from the index store."""
        return list(self._store.get_trace_events(self.trace_id))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(data: dict[str, Any], max_len: int = 200) -> dict[str, str]:
    """Create a short summary of a dict for trace payloads."""
    out: dict[str, str] = {}
    for k, v in data.items():
        s = str(v)
        out[k] = s[:max_len] + "…" if len(s) > max_len else s
    return out
