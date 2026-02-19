"""Phase-boundary checkpoint utilities (ROADMAP Stage 2).

Checkpoints are stored as ``trace_events`` with ``kind="checkpoint"``.
Each checkpoint payload has the structure::

    {
        "agent_state": { ... AgentState dump ... },
        "resume_context": {
            "iteration": <int>,
            "phase_boundary": "<label>",
            ... extra context for resume ...
        }
    }

**Resume invariant**: restoration is allowed **only** from a
phase-boundary checkpoint; mid-node restoration is never attempted.
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.core.state import AgentState
from neuronium_agent.errors import NeuroniumError
from neuronium_agent.storage.index_store import IndexStore


class CheckpointError(NeuroniumError):
    """Raised when checkpoint loading or validation fails."""


# -- Phase-boundary labels (v1) ----------------------------------------------

PHASE_BOUNDARIES = frozenset({
    "after_commit_iter1",
    "after_execute_iter1",
    "after_control_iter1",
    "after_adapt_iter1",
    "after_commit_iter2",
    "after_execute_iter2",
    "after_control_iter2",
    "final",
    "paused",
})
"""Valid phase-boundary labels that support resume."""


# -- Reading -----------------------------------------------------------------

def get_latest_phase_boundary_checkpoint(
    index_store: IndexStore,
    trace_id: str,
) -> dict[str, Any] | None:
    """Return the latest phase-boundary checkpoint payload, or ``None``.

    Only checkpoints whose ``resume_context.phase_boundary`` is in
    :data:`PHASE_BOUNDARIES` are considered valid for resume.
    """
    events = list(index_store.get_trace_events(trace_id))
    checkpoints = [
        e for e in events
        if e.get("kind") == "checkpoint"
    ]
    # Walk backwards to find the latest valid phase-boundary checkpoint
    for cp in reversed(checkpoints):
        payload = cp.get("payload", {})
        rc = payload.get("resume_context", {})
        if rc.get("phase_boundary") in PHASE_BOUNDARIES:
            return payload
    return None


def load_state_from_checkpoint(
    checkpoint_payload: dict[str, Any],
) -> tuple[AgentState, dict[str, Any]]:
    """Restore ``AgentState`` and resume context from a checkpoint payload.

    Returns
    -------
    (state, resume_context)
        *state* is a validated :class:`AgentState` instance.
        *resume_context* is the dict with ``iteration``, ``phase_boundary``,
        and any extra data stored at checkpoint time.

    Raises
    ------
    CheckpointError
        If the payload is malformed or the phase boundary is invalid.
    """
    agent_data = checkpoint_payload.get("agent_state")
    if not isinstance(agent_data, dict):
        raise CheckpointError(
            "Checkpoint payload missing 'agent_state' dict"
        )

    resume_ctx = checkpoint_payload.get("resume_context", {})
    if not isinstance(resume_ctx, dict):
        raise CheckpointError(
            "Checkpoint payload 'resume_context' must be a dict"
        )

    boundary = resume_ctx.get("phase_boundary", "")
    if boundary not in PHASE_BOUNDARIES:
        raise CheckpointError(
            "Checkpoint phase_boundary '{}' is not valid for resume. "
            "Valid boundaries: {}".format(boundary, sorted(PHASE_BOUNDARIES))
        )

    state = AgentState.model_validate(agent_data)
    return state, resume_ctx


# -- Writing (helper for orchestrator) ---------------------------------------

def build_checkpoint_payload(
    state: AgentState,
    *,
    iteration: int,
    phase_boundary: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured checkpoint payload.

    This is the canonical format written by ``TraceRecorder.record_checkpoint``.
    """
    resume_ctx: dict[str, Any] = {
        "iteration": iteration,
        "phase_boundary": phase_boundary,
    }
    if extra:
        resume_ctx.update(extra)

    return {
        "agent_state": state.model_dump(mode="json"),
        "resume_context": resume_ctx,
    }
