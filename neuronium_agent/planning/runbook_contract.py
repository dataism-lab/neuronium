"""Runbook contract — internal abstraction for deterministic stage-based plans.

A Runbook is a sequence of ActionGraph stages executed one by one through
the standard COMMIT -> EXECUTE -> CONTROL -> ADAPT cycle.  Each stage
carries a declarative success gate that the orchestrator evaluates after
execution.

This module is **internal** (not part of PUBLIC_API_SPEC).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from neuronium_agent.planning.dag import ActionGraph


# ---------------------------------------------------------------------------
# Success gate (declarative, serialisable for checkpoints)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageSuccessGate:
    """Declarative quality gate evaluated after DAG execution.

    * *required_completed_nodes* — node IDs that must have status COMPLETED.
    * *critic_node_id* — if set, the critic's ``DemoCriticVerdict`` is
      extracted; PASS requires non-empty evidence.
    """

    required_completed_nodes: list[str] = field(default_factory=list)
    critic_node_id: str | None = None

    # -- Serialisation helpers (for checkpoint resume_context) ---------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_completed_nodes": list(self.required_completed_nodes),
            "critic_node_id": self.critic_node_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageSuccessGate:
        return cls(
            required_completed_nodes=list(d.get("required_completed_nodes", [])),
            critic_node_id=d.get("critic_node_id"),
        )


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

@dataclass
class ActionGraphStage:
    """A single stage inside a runbook.

    Attributes
    ----------
    stage_id:
        Stable identifier (e.g. ``"docs_report_v1:stage1"``).
    graph:
        The ActionGraph (DAG) to execute in this stage.
    initial_inputs_override:
        Extra key/value pairs injected as ``initial_inputs`` into the
        DAG executor (merged with objective/constraints).
    success_gate:
        Declarative gate checked after execution.
    """

    stage_id: str
    graph: ActionGraph
    initial_inputs_override: dict[str, Any] = field(default_factory=dict)
    success_gate: StageSuccessGate = field(default_factory=StageSuccessGate)


# ---------------------------------------------------------------------------
# Runbook (abstract base)
# ---------------------------------------------------------------------------

class Runbook(abc.ABC):
    """Abstract base for a deterministic runbook.

    Subclasses implement :meth:`build_stages` which returns an ordered list
    of :class:`ActionGraphStage` instances.
    """

    @property
    @abc.abstractmethod
    def runbook_id(self) -> str:
        """Stable identifier used by CLI ``--runbook`` and ``RunRequest.metadata``."""

    @property
    def description(self) -> str:  # pragma: no cover
        return ""

    @abc.abstractmethod
    def build_stages(
        self,
        *,
        objective: str,
        constraints: list[str],
        metadata: dict[str, Any],
        execution_id: str,
    ) -> list[ActionGraphStage]:
        """Build the ordered list of stages for this runbook.

        Must be **deterministic**: same inputs → same stages (stable
        plan_ids, node_ids, edges).
        """
