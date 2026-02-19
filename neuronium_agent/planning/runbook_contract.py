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
from typing import Any, Callable

from neuronium_agent.planning.dag import ActionGraph
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec

# Result of graph_builder(context): either graph only, or (graph, initial_inputs_override)
GraphBuilderResult = ActionGraph | tuple[ActionGraph, dict[str, Any] | None]
GraphBuilder = Callable[[dict[str, Any]], GraphBuilderResult]


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
        The ActionGraph (DAG) to execute in this stage. Optional when
        graph_builder is provided; then the orchestrator uses the graph
        returned by graph_builder(context).
    initial_inputs_override:
        Extra key/value pairs injected as ``initial_inputs`` into the
        DAG executor (merged with objective/constraints).
    success_gate:
        Declarative gate checked after execution.
    dynamic_planner:
        Optional dynamic planner configuration. If provided, COMMIT runs a
        planner ModelNode that returns the stage graph at runtime.
    graph_builder:
        Optional callable(context) -> ActionGraph | (ActionGraph, dict|None).
        If set, the orchestrator calls it with context (objective, constraints,
        prev_stage_results, prev_stage_verdict, ...) and uses the returned
        graph for this stage. If the return value is a tuple, the second
        element is used as initial_inputs_override for this stage (merged
        with initial_inputs_override).
    exit_run_on_success:
        If True, when this stage's gate passes the run completes (COMPLETED,
        final checkpoint) and no further stages run. Used e.g. for autofix
        iter1: on PASS we are done.
    proceed_to_next_stage_on_fail:
        If True, when this stage's gate fails the orchestrator does not enter
        recovery; it proceeds to the next stage (prev_stage_results/verdict
        set for graph_builder). Used e.g. for autofix iter1: on FAIL we run iter2.
    default_model_id:
        Optional model catalog id. If set, all model nodes in this stage that
        do not have ``parameters["model_id"]`` in the graph use this id for
        resolution (see CONFIG_SPEC §2.6.1).
    """

    stage_id: str
    graph: ActionGraph | None = None
    initial_inputs_override: dict[str, Any] = field(default_factory=dict)
    success_gate: StageSuccessGate = field(default_factory=StageSuccessGate)
    dynamic_planner: DynamicPlannerSpec | None = None
    graph_builder: GraphBuilder | None = None
    exit_run_on_success: bool = False
    proceed_to_next_stage_on_fail: bool = False
    default_model_id: str | None = None


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
