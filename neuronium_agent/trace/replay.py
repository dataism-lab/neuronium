"""Trace replay — re-execute a run using recorded external responses (IBS §3.3).

For every non-deterministic call (LLM, MCP, CodeNode) the replay engine
substitutes the recorded response instead of calling the external service.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from neuronium_agent.errors import ReplayError
from neuronium_agent.nodes.base import BaseNode
from neuronium_agent.storage.index_store import IndexStore


class ReplayProvider:
    """Provides recorded responses to nodes during replay.

    Parses trace events and separates two response sources:

    - **replay_data** (primary): full recorded request/response pairs written
      by the orchestrator after live execution.  These are the *only* source
      accepted in strict mode.
    - **node_end** (fallback): outputs extracted from ``node_end`` trace
      events.  Lower fidelity (may contain summaries instead of raw outputs).
      Only used when ``strict=False`` (best-effort replay).
    """

    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        # Primary source — high-fidelity recorded responses.
        self._replay_data: dict[str, list[dict[str, Any]]] = {}
        # Fallback source — extracted from node_end events (best-effort).
        self._node_end_fallback: dict[str, list[dict[str, Any]]] = {}
        self._planner_catalog_hashes: list[str] = []

        for ev in events:
            kind = ev.get("kind", "")
            payload = ev.get("payload", {})
            if kind == "replay_data":
                nid = payload.get("node_id", "")
                recs = payload.get("recorded_responses", [])
                if nid and isinstance(recs, list) and recs:
                    self._replay_data.setdefault(nid, []).extend(list(recs))
            elif kind == "node_end":
                nid = payload.get("node_id", "")
                outputs = payload.get(
                    "outputs_summary", payload.get("outputs", {})
                )
                status = payload.get("status", "COMPLETED")
                if nid:
                    self._node_end_fallback.setdefault(nid, []).append({
                        "outputs": outputs,
                        "status": status,
                        "quality_signals": {},
                    })
            elif kind == "decision":
                description = str(payload.get("description", ""))
                if description == "Planner request envelope":
                    catalog_hash = str(payload.get("operator_catalog_hash", "")).strip()
                    if catalog_hash:
                        self._planner_catalog_hashes.append(catalog_hash)

    def inject(
        self,
        node_registry: dict[str, BaseNode],
        *,
        strict: bool = True,
    ) -> dict[str, list[str]]:
        """Inject recorded responses into node instances.

        In **strict** mode only ``replay_data`` is accepted as a valid source.
        If any replay-capable node lacks ``replay_data``, a
        :class:`ReplayError` is raised — no live calls will be attempted.

        In **best-effort** mode (``strict=False``) ``node_end`` outputs are
        used as a fallback, with a warning recorded in the report.

        Returns a report dict with keys:
        - ``injected``: node IDs that received replay responses
        - ``injected_fallback``: node IDs served from ``node_end`` fallback
        - ``missing``: replay-capable node IDs without any responses
        """
        report: dict[str, list[str]] = {
            "injected": [],
            "injected_fallback": [],
            "missing": [],
        }

        for nid, node in node_registry.items():
            if not hasattr(node, "set_replay_responses"):
                continue

            if nid in self._replay_data:
                node.set_replay_responses(self._replay_data[nid])
                report["injected"].append(nid)
            elif not strict and nid in self._node_end_fallback:
                node.set_replay_responses(self._node_end_fallback[nid])
                report["injected_fallback"].append(nid)
            else:
                report["missing"].append(nid)

        if strict and report["missing"]:
            missing = ", ".join(sorted(report["missing"]))
            raise ReplayError(
                f"Strict replay failed: missing recorded responses "
                f"for nodes: {missing}"
            )
        return report

    def latest_operator_catalog_hash(self) -> str | None:
        """Return the latest planner operator-catalog hash from source trace."""
        if not self._planner_catalog_hashes:
            return None
        return self._planner_catalog_hashes[-1]


def load_replay_events(
    index_store: IndexStore,
    trace_id: str,
) -> list[dict[str, Any]]:
    """Load trace events for replay from the index store."""
    events = list(index_store.get_trace_events(trace_id))
    if not events:
        raise ReplayError(f"No trace events found for trace_id={trace_id}")
    return events
