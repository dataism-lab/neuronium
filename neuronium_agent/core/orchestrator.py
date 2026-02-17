"""Orchestrator — the Cognitive Core loop (IBS §1.1).

Supports two execution paths:

1. **Fixed 2-iteration autofix demo** (``runbook_id="autofix_demo"``):
   iter1 → critic → (if fail) → replan → iter2 → critic → done/fail.

2. **Generic N-stage runbook runner** (any registered ``Runbook``):
   each stage goes through COMMIT → EXECUTE → CONTROL → ADAPT with
   quality-gate evaluation.  Resume/skip logic is generalised for an
   arbitrary number of stages.

Phase-boundary checkpoints are recorded between each phase transition
to support resume.  Meta-control commands are **declarative**: they
mutate state + write decision/checkpoint but never execute DAG nodes.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from neuronium_agent._canonical import canonical_json, artifact_id, canonical_bytes
from neuronium_agent.artifacts.local_index import LocalArtifactIndex, LocalIndexEntry
from neuronium_agent.artifacts.renderer import render_run_artifact
from neuronium_agent.artifacts.user_renderer import (
    extract_user_facing_summary,
    render_user_facing_html,
)
from neuronium_agent.config import AppConfig
from neuronium_agent.core.state import (
    AgentState,
    Intention,
    IntentionPhase,
    RunState,
)
from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import BaseNode, NodeOutput
from neuronium_agent.nodes.code_node import CodeNode
from neuronium_agent.nodes.mcp_node import McpToolNode
from neuronium_agent.nodes.model_node import ModelNode
from neuronium_agent.planning.dag import ActionGraph
from neuronium_agent.planning.dynamic_planner import validate_planned_graph
from neuronium_agent.planning.htn import HTNPlanner
from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.planner_backend import get_planner_backend
from neuronium_agent.planning.planner_contract import (
    DynamicPlannerSpec,
    PlannerEscalation,
    PlannerOutcome,
    PlannerRequest,
    PlannerResult,
)
from neuronium_agent.planning.runbook_contract import StageSuccessGate
from neuronium_agent.planning.runbook_registry import get_runbook
from neuronium_agent.storage.blob_store import BlobStore
from neuronium_agent.storage.index_store import IndexStore
from neuronium_agent.trace.checkpoints import (
    build_checkpoint_payload,
    get_latest_phase_boundary_checkpoint,
    load_state_from_checkpoint,
    CheckpointError,
)
from neuronium_agent.trace.recorder import TraceRecorder
from neuronium_agent.trace.replay import ReplayProvider
from neuronium_agent.types import ControlCommand, RunHandle, RunRequest, RunStatus
from neuronium_agent.verification.demo_critic import (
    DemoCriticVerdict,
    parse_critic_verdict,
)

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 2


class Orchestrator:
    """High-level orchestration: autofix demo loop + generic N-stage runbook runner."""

    def __init__(
        self,
        config: AppConfig,
        blob_store: BlobStore,
        index_store: IndexStore,
        *,
        trace_event_listener: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.blob_store = blob_store
        self.index_store = index_store
        self._planner = HTNPlanner()
        self._states: dict[str, AgentState] = {}
        self._recorders: dict[str, TraceRecorder] = {}
        self._memory_store: Any | None = None  # lazy init
        self._operator_catalog = OperatorCatalog.default()
        self._trace_event_listener = trace_event_listener

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(self, request: RunRequest) -> RunHandle:
        """Start a new agent run (Commit phase)."""
        trace_id = uuid.uuid4().hex
        execution_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        state = AgentState(
            trace_id=trace_id,
            execution_id=execution_id,
            created_at=now,
        )

        intention = Intention(
            intention_id=uuid.uuid4().hex,
            objective=request.objective,
            constraints=request.constraints,
        )
        state.intention = intention

        self._states[trace_id] = state

        # Persist run
        config_snap = canonical_json(self.config.model_dump(mode="json"))
        self.index_store.upsert_run(
            trace_id=trace_id,
            execution_id=execution_id,
            state="PENDING",
            objective=request.objective,
            config_snapshot_json=config_snap,
            created_at=now.isoformat(),
        )

        # Create trace recorder
        recorder = TraceRecorder(
            trace_id,
            self.index_store,
            event_listener=self._trace_event_listener,
        )
        self._recorders[trace_id] = recorder

        recorder.record_decision(
            "Intention committed",
            {"objective": request.objective, "constraints": request.constraints},
        )

        runbook_id = (
            (request.metadata or {}).get("runbook_id")  # type: ignore[union-attr]
            or "autofix_demo"
        )
        recorder.record_decision(
            "Runbook selected",
            {"runbook_id": runbook_id},
        )

        handle = RunHandle(
            trace_id=trace_id,
            execution_id=execution_id,
            created_at=now,
        )

        # Run synchronously (batch mode)
        if runbook_id == "autofix_demo":
            self._run_cycle(state, recorder)
        elif get_runbook(runbook_id) is not None:
            self._run_runbook(
                state, recorder,
                runbook_id=runbook_id,
                metadata=dict(request.metadata or {}),
            )
        else:
            state.transition_to(RunState.FAILED, f"Unknown runbook_id={runbook_id!r}")
            self.index_store.update_run_state(state.trace_id, "FAILED")
            recorder.record("error", {"error": f"Unknown runbook_id={runbook_id!r}"})

        return handle

    def replay(
        self,
        original_trace_id: str,
        replay_provider: ReplayProvider,
    ) -> RunHandle:
        """Replay a run from recorded responses using a new trace ID."""
        run = self.index_store.get_run(original_trace_id)
        if run is None:
            from neuronium_agent.errors import ReplayError

            raise ReplayError(f"Run not found for trace_id={original_trace_id}")

        trace_id = uuid.uuid4().hex
        execution_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        state = AgentState(
            trace_id=trace_id,
            execution_id=execution_id,
            created_at=now,
        )
        state.intention = Intention(
            intention_id=uuid.uuid4().hex,
            objective=run.get("objective", ""),
            constraints=[],
        )
        self._states[trace_id] = state

        config_snap = run.get("config_snapshot_json") or canonical_json(
            self.config.model_dump(mode="json")
        )
        self.index_store.upsert_run(
            trace_id=trace_id,
            execution_id=execution_id,
            state="PENDING",
            objective=state.intention.objective,
            config_snapshot_json=config_snap,
            created_at=now.isoformat(),
        )

        recorder = TraceRecorder(
            trace_id,
            self.index_store,
            event_listener=self._trace_event_listener,
        )
        self._recorders[trace_id] = recorder
        recorder.record_decision(
            "Replay started",
            {"replay_of_trace_id": original_trace_id, "strict": True},
        )

        handle = RunHandle(
            trace_id=trace_id,
            execution_id=execution_id,
            created_at=now,
        )
        runbook_id = self._infer_runbook_id(original_trace_id) or "autofix_demo"
        recorder.record_decision(
            "Runbook selected (replay)",
            {"runbook_id": runbook_id, "replay_of_trace_id": original_trace_id},
        )
        if runbook_id == "autofix_demo":
            self._run_cycle(state, recorder, replay_provider=replay_provider)
        elif get_runbook(runbook_id) is not None:
            # Best-effort metadata reconstruction for replay
            replay_metadata = self._infer_runbook_metadata(
                original_trace_id, runbook_id,
            )
            self._run_runbook(
                state, recorder,
                runbook_id=runbook_id,
                metadata=replay_metadata,
                replay_provider=replay_provider,
            )
        else:
            state.transition_to(RunState.FAILED, f"Unknown runbook_id={runbook_id!r} (replay)")
            self.index_store.update_run_state(state.trace_id, "FAILED")
            recorder.record("error", {"error": f"Unknown runbook_id={runbook_id!r} (replay)"})
        return handle

    def get_status(self, trace_id: str) -> RunStatus:
        state = self._states.get(trace_id)
        if state is None:
            # Try loading from store
            run = self.index_store.get_run(trace_id)
            if run is None:
                return RunStatus(state="FAILED", message="Run not found")
            return RunStatus(state=run["state"])

        return RunStatus(
            state=state.run_state.value,
            progress=state.progress,
            current_node_ref=state.current_node_ref,
            message=state.message,
        )

    def get_recorder(self, trace_id: str) -> TraceRecorder | None:
        return self._recorders.get(trace_id)

    # ------------------------------------------------------------------
    # Declarative meta-control (no DAG execution)
    # ------------------------------------------------------------------

    def apply_control(
        self,
        trace_id: str,
        command: ControlCommand,
    ) -> RunStatus:
        """Apply a control command **declaratively**.

        Performs: state transition → checkpoint → trace decision.
        Never executes DAG nodes or triggers orchestration.
        """
        state = self._states.get(trace_id)
        recorder = self._recorders.get(trace_id)

        # If state not in memory, try to restore from checkpoint
        if state is None:
            cp = get_latest_phase_boundary_checkpoint(
                self.index_store, trace_id,
            )
            if cp is None:
                run = self.index_store.get_run(trace_id)
                if run is None:
                    return RunStatus(state="FAILED", message="Run not found")
                return RunStatus(state=run["state"], message="No checkpoint for control")

            state, _resume_ctx = load_state_from_checkpoint(cp)
            self._states[trace_id] = state

        if recorder is None:
            recorder = TraceRecorder(
                trace_id,
                self.index_store,
                event_listener=self._trace_event_listener,
            )
            self._recorders[trace_id] = recorder

        cmd = command.type
        payload = command.payload

        if cmd == "pause":
            state.transition_to(RunState.PAUSED, "User command: pause")
            self.index_store.update_run_state(trace_id, "PAUSED")
            self._record_control_decision(recorder, state, cmd, payload)
            return self.get_status(trace_id)

        if cmd == "continue":
            state.transition_to(RunState.RUNNING, "User command: continue")
            self.index_store.update_run_state(trace_id, "RUNNING")
            self._record_control_decision(recorder, state, cmd, payload)
            return self.get_status(trace_id)

        if cmd == "stop":
            state.transition_to(RunState.CANCELLED, "User command: stop")
            self.index_store.update_run_state(trace_id, "CANCELLED")
            self._record_control_decision(recorder, state, cmd, payload)
            return self.get_status(trace_id)

        if cmd == "replan":
            if state.intention:
                state.intention.phase = IntentionPhase.ADAPT
            self._record_control_decision(recorder, state, cmd, payload)
            return self.get_status(trace_id)

        if cmd == "revise":
            if state.intention:
                state.intention.phase = IntentionPhase.ADAPT
            new_constraints = payload.get("constraints_add", [])
            if state.intention and new_constraints:
                state.intention.constraints.extend(new_constraints)
            answers = payload.get("answers", {})
            if not isinstance(answers, dict):
                answers = {}
            answer_text = payload.get("answer_text")
            if not isinstance(answer_text, str):
                answer_text = None
            request_artifact_id = str(
                payload.get("clarification_request_artifact_id", "")
            ).strip()

            decision_payload = dict(payload)
            if answers or answer_text:
                plan_id = (
                    state.intention.plan_id
                    if state.intention and state.intention.plan_id
                    else "no-plan"
                )
                response_payload = {
                    "request_artifact_id": request_artifact_id,
                    "answers": answers,
                    "answer_text": answer_text,
                }
                response_artifact_id = self._persist_control_artifact(
                    artifact_type="planner.clarification_response",
                    payload=response_payload,
                    produced_by_node_ref=(
                        f"{state.execution_id}:{plan_id}/control/revise"
                    ),
                    parent_artifact_ids=[request_artifact_id] if request_artifact_id else [],
                )
                decision_payload["clarification_response_artifact_id"] = response_artifact_id
                if request_artifact_id:
                    self.index_store.record_lineage_edge(
                        request_artifact_id,
                        response_artifact_id,
                        "clarification",
                    )
            self._record_control_decision(recorder, state, cmd, decision_payload)
            return self.get_status(trace_id)

        if cmd == "escalate":
            state.transition_to(RunState.PAUSED, "User command: escalate")
            self.index_store.update_run_state(trace_id, "PAUSED")
            self._record_control_decision(recorder, state, cmd, payload)
            return self.get_status(trace_id)

        return self.get_status(trace_id)

    def _record_control_decision(
        self,
        recorder: TraceRecorder,
        state: AgentState,
        cmd: str,
        payload: dict[str, Any],
    ) -> None:
        """Write a decision + phase-boundary checkpoint for a control command."""
        recorder.record_decision(
            "control_command: " + cmd,
            {"command": cmd, "payload": payload},
        )
        iteration = payload.get("iteration", 0)
        cp_payload = build_checkpoint_payload(
            state,
            iteration=iteration,
            phase_boundary="paused" if state.run_state == RunState.PAUSED else "after_control_command",
            extra={"control_command": cmd, "control_payload": payload},
        )
        recorder.record_checkpoint(cp_payload)

    # ------------------------------------------------------------------
    # Resume from phase-boundary checkpoint
    # ------------------------------------------------------------------

    def resume_run(self, trace_id: str) -> RunHandle:
        """Resume a run from its latest phase-boundary checkpoint.

        The run must be in ``RUNNING`` state in the store (set by
        ``control continue``).  The checkpoint itself may still show
        ``PAUSED`` — this is expected because ``control continue``
        transitions state *after* the checkpoint was written.
        """
        # 1. Verify the *current* DB state is RUNNING
        run = self.index_store.get_run(trace_id)
        if run is None:
            raise CheckpointError(
                "Run not found for trace_id=" + trace_id
            )
        if run["state"] != "RUNNING":
            raise CheckpointError(
                "Cannot resume: store state is {}, expected RUNNING. "
                "Use 'control continue' first.".format(run["state"])
            )

        # 2. Load last valid phase-boundary checkpoint
        cp = get_latest_phase_boundary_checkpoint(
            self.index_store, trace_id,
        )
        if cp is None:
            raise CheckpointError(
                "No phase-boundary checkpoint found for trace_id=" + trace_id
            )

        state, resume_ctx = load_state_from_checkpoint(cp)
        # Override run_state to RUNNING (we verified it from DB above)
        state.run_state = RunState.RUNNING

        self._states[trace_id] = state
        recorder = TraceRecorder(
            trace_id,
            self.index_store,
            event_listener=self._trace_event_listener,
        )
        self._recorders[trace_id] = recorder

        recorder.record_decision(
            "Resume from checkpoint",
            {"phase_boundary": resume_ctx.get("phase_boundary"),
             "iteration": resume_ctx.get("iteration")},
        )

        # Dispatch based on runbook_id
        runbook_id = (
            resume_ctx.get("runbook_id")
            or self._infer_runbook_id(trace_id)
            or "autofix_demo"
        )
        recorder.record_decision(
            "Runbook selected (resume)",
            {"runbook_id": runbook_id},
        )

        if runbook_id == "autofix_demo":
            self._run_cycle(
                state, recorder,
                resume_ctx=resume_ctx,
            )
        elif get_runbook(runbook_id) is not None:
            resume_metadata = self._infer_runbook_metadata(trace_id, runbook_id)
            self._run_runbook(
                state, recorder,
                runbook_id=runbook_id,
                metadata=resume_metadata,
                resume_ctx=resume_ctx,
            )
        else:
            state.transition_to(
                RunState.FAILED,
                f"Unknown runbook_id={runbook_id!r} (resume)",
            )
            self.index_store.update_run_state(trace_id, "FAILED")
            recorder.record("error", {
                "error": f"Unknown runbook_id={runbook_id!r} (resume)",
            })

        return RunHandle(
            trace_id=state.trace_id,
            execution_id=state.execution_id,
            created_at=state.created_at,
        )

    # ------------------------------------------------------------------
    # Generic runbook runner (sequential stages)
    # ------------------------------------------------------------------

    def _run_runbook(
        self,
        state: AgentState,
        recorder: TraceRecorder,
        *,
        runbook_id: str,
        metadata: dict[str, Any],
        replay_provider: ReplayProvider | None = None,
        resume_ctx: dict[str, Any] | None = None,
    ) -> None:
        """Execute a registered runbook as a sequence of stage-graphs.

        Each stage goes through COMMIT -> EXECUTE -> CONTROL -> ADAPT.
        If *resume_ctx* is provided, already-completed stages (and
        phases within a stage) are skipped based on the checkpoint
        boundary and gate snapshot.
        """
        try:
            # Strict-fail preflight
            if replay_provider is None and not self._llm_available():
                state.transition_to(RunState.FAILED, "LLM unavailable (strict_fail)")
                self.index_store.update_run_state(state.trace_id, "FAILED")
                recorder.record_decision(
                    "strict_fail: LLM unavailable",
                    {"api_key_env": self.config.llm.api_key_env},
                )
                return

            runbook = get_runbook(runbook_id)
            if runbook is None:
                state.transition_to(RunState.FAILED, f"Runbook not found: {runbook_id!r}")
                self.index_store.update_run_state(state.trace_id, "FAILED")
                recorder.record("error", {"error": f"Runbook not found: {runbook_id!r}"})
                return

            if not resume_ctx:
                state.transition_to(RunState.RUNNING, f"Runbook: {runbook_id} started")
                self.index_store.update_run_state(state.trace_id, "RUNNING")

            objective = state.intention.objective  # type: ignore[union-attr]
            constraints = list(state.intention.constraints)  # type: ignore[union-attr]

            stages = runbook.build_stages(
                objective=objective,
                constraints=constraints,
                metadata=metadata,
                execution_id=state.execution_id,
            )

            # Resume: determine which stage/phase to skip to
            resume_boundary = resume_ctx.get("phase_boundary", "") if resume_ctx else ""
            resume_stage_index = resume_ctx.get("stage_index", 0) if resume_ctx else 0

            for stage_index, stage in enumerate(stages):
                iteration = stage_index + 1  # 1-based for checkpoint compatibility

                # Skip stages completed before the resume checkpoint
                if resume_ctx and stage_index < resume_stage_index:
                    continue

                # Determine which phases to skip within the current stage.
                # Phase order: commit → execute → control.
                # "after_X_iterN" means X is already done — skip it and
                # all preceding phases.  Generalised for N stages.
                skip_commit = False
                skip_execute = False
                skip_control = False
                if resume_ctx and stage_index == resume_stage_index:
                    sfx = f"_iter{iteration}"
                    past_commit = {f"after_execute{sfx}", f"after_control{sfx}"}
                    past_control = {f"after_control{sfx}"}
                    skip_commit = resume_boundary in past_commit
                    skip_execute = resume_boundary in past_commit
                    skip_control = resume_boundary in past_control

                graph = stage.graph
                if stage.dynamic_planner is not None and skip_commit:
                    planned_graph = (resume_ctx or {}).get("planned_graph")
                    if not isinstance(planned_graph, dict):
                        raise ValueError(
                            "Dynamic planner resume requires 'planned_graph' "
                            "in checkpoint resume_context"
                        )
                    graph = ActionGraph.model_validate(planned_graph)
                    graph = validate_planned_graph(
                        graph,
                        spec=stage.dynamic_planner,
                        operator_catalog=self._operator_catalog,
                    )

                # -- stage_start event -----------------------------------------
                recorder.record("stage_start", {
                    "stage_id": stage.stage_id,
                    "stage_index": stage_index,
                    "runbook_id": runbook_id,
                    "plan_id": graph.metadata.plan_id,
                })

                gate = stage.success_gate
                results: dict[str, NodeOutput] = {}
                verdict: DemoCriticVerdict | None = None
                planner_backend_name = ""
                planner_backend_version = ""
                operator_catalog_hash = ""
                planner_trace_payload: dict[str, Any] = {}

                # -- COMMIT ----------------------------------------------------
                if not skip_commit:
                    state.intention.phase = IntentionPhase.COMMIT  # type: ignore[union-attr]
                    if stage.dynamic_planner is not None:
                        planner_outcome = self._plan_dynamic_graph(
                            state=state,
                            recorder=recorder,
                            replay_provider=replay_provider,
                            runbook_id=runbook_id,
                            stage_id=stage.stage_id,
                            metadata=metadata,
                            spec=stage.dynamic_planner,
                        )
                        if isinstance(planner_outcome, PlannerEscalation):
                            self._pause_for_planner_escalation(
                                state=state,
                                recorder=recorder,
                                iteration=iteration,
                                runbook_id=runbook_id,
                                stage_id=stage.stage_id,
                                stage_index=stage_index,
                                escalation=planner_outcome,
                            )
                            return
                        planner_result = planner_outcome
                        graph = planner_result.action_graph
                        planner_backend_name = planner_result.backend_name
                        planner_backend_version = planner_result.backend_version
                        operator_catalog_hash = planner_result.operator_catalog_hash or ""
                        if planner_result.decision_trace is not None:
                            planner_trace_payload = {
                                "subgoals": list(planner_result.decision_trace.subgoals),
                                "selected_methods": list(planner_result.decision_trace.selected_methods),
                                "justification_keys": list(planner_result.decision_trace.justification_keys),
                                "decomposition_steps": list(planner_result.decision_trace.decomposition_steps),
                                "method_expansion_path": list(planner_result.decision_trace.method_expansion_path),
                                "leaf_operators": list(planner_result.decision_trace.leaf_operators),
                                "notes": dict(planner_result.decision_trace.notes),
                            }
                        recorder.record_decision(
                            "Plan created (dynamic)",
                            {
                                "iteration": iteration,
                                "plan_id": graph.metadata.plan_id,
                                "runbook_id": runbook_id,
                                "stage_id": stage.stage_id,
                                "nodes": [n.node_id for n in graph.nodes],
                                "edges": [(e.source, e.target) for e in graph.edges],
                                "planner_backend": planner_backend_name,
                                "planner_backend_version": planner_backend_version,
                                "operator_catalog_hash": operator_catalog_hash,
                                "planner_decision_trace": planner_trace_payload,
                            },
                        )
                    else:
                        recorder.record_decision(
                            f"Plan created ({runbook_id})",
                            {
                                "iteration": iteration,
                                "plan_id": graph.metadata.plan_id,
                                "runbook_id": runbook_id,
                                "stage_id": stage.stage_id,
                                "nodes": [n.node_id for n in graph.nodes],
                                "edges": [(e.source, e.target) for e in graph.edges],
                            },
                        )
                    state.intention.plan_id = graph.metadata.plan_id  # type: ignore[union-attr]
                    commit_extra: dict[str, Any] = {
                        "runbook_id": runbook_id,
                        "stage_id": stage.stage_id,
                        "stage_index": stage_index,
                    }
                    if stage.dynamic_planner is not None:
                        commit_extra["planned_graph"] = graph.model_dump(mode="json")
                        commit_extra["planner_backend"] = planner_backend_name
                        commit_extra["planner_backend_version"] = planner_backend_version
                        commit_extra["operator_catalog_hash"] = operator_catalog_hash
                        commit_extra["planner_decision_trace"] = planner_trace_payload
                    self._write_phase_checkpoint(
                        state, recorder, iteration,
                        f"after_commit_iter{iteration}",
                        extra=commit_extra,
                    )

                # -- EXECUTE ---------------------------------------------------
                if not skip_execute:
                    state.intention.phase = IntentionPhase.EXECUTE  # type: ignore[union-attr]
                    results = self._execute(
                        state, graph, recorder,
                        replay_provider=replay_provider,
                        initial_inputs_override=stage.initial_inputs_override,
                    )
                    # Build gate snapshot for resume
                    gate_snapshot = self._build_gate_snapshot(results, gate)
                    self._write_phase_checkpoint(
                        state, recorder, iteration,
                        f"after_execute_iter{iteration}",
                        extra={
                            "runbook_id": runbook_id,
                            "stage_id": stage.stage_id,
                            "stage_index": stage_index,
                            "gate_snapshot": gate_snapshot,
                        }
                        | (
                            {"planned_graph": graph.model_dump(mode="json")}
                            if stage.dynamic_planner is not None
                            else {}
                        ),
                    )
                else:
                    # Restore gate snapshot from resume context
                    gate_snapshot = (resume_ctx or {}).get("gate_snapshot")

                # -- CONTROL (quality gate) ------------------------------------
                if not skip_control:
                    state.intention.phase = IntentionPhase.CONTROL  # type: ignore[union-attr]

                    if gate.critic_node_id and not skip_execute:
                        verdict = self._extract_verdict(results, gate.critic_node_id)
                    elif gate.critic_node_id and gate_snapshot:
                        # Restore verdict from gate snapshot (resume path)
                        verdict = DemoCriticVerdict(
                            verdict=gate_snapshot.get("critic_verdict", "UNCERTAIN"),
                            confidence=gate_snapshot.get("critic_confidence", 0.0),
                            evidence=gate_snapshot.get("critic_evidence", []),
                            gaps=gate_snapshot.get("critic_gaps", []),
                        )
                    else:
                        verdict = None

                    if verdict is not None:
                        recorder.record("critic_verdict", {
                            "iteration": iteration,
                            "plan_id": graph.metadata.plan_id,
                            "critic_node_id": gate.critic_node_id,
                            "verdict": verdict.verdict,
                            "confidence": verdict.confidence,
                            "evidence": verdict.evidence,
                            "gaps": verdict.gaps,
                        })

                    self._write_phase_checkpoint(
                        state, recorder, iteration,
                        f"after_control_iter{iteration}",
                        extra={
                            "runbook_id": runbook_id,
                            "stage_id": stage.stage_id,
                            "stage_index": stage_index,
                        }
                        | (
                            {"planned_graph": graph.model_dump(mode="json")}
                            if stage.dynamic_planner is not None
                            else {}
                        ),
                    )

                # -- ADAPT (evaluate gate & finish/continue) -------------------
                state.intention.phase = IntentionPhase.ADAPT  # type: ignore[union-attr]
                stage_passed = self._evaluate_gate(
                    results if not skip_execute else {},
                    gate, verdict, gate_snapshot,
                )

                recorder.record("stage_end", {
                    "stage_id": stage.stage_id,
                    "stage_index": stage_index,
                    "runbook_id": runbook_id,
                    "success": stage_passed,
                    "reason": (
                        "quality gate passed"
                        if stage_passed
                        else "quality gate failed"
                    ),
                })

                if not stage_passed:
                    state.transition_to(
                        RunState.FAILED,
                        f"{runbook_id}: quality gate failed at stage {stage.stage_id}",
                    )
                    self.index_store.update_run_state(state.trace_id, "FAILED")
                    state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
                    self._write_phase_checkpoint(
                        state, recorder, iteration, "final",
                        extra={"runbook_id": runbook_id},
                    )
                    if results:
                        self._persist_artifacts(state, results, recorder)
                        self._persist_local_rendered_artifact(
                            state=state,
                            recorder=recorder,
                            results=results,
                            runbook_id=runbook_id,
                        )
                    return

            # All stages passed
            state.transition_to(RunState.COMPLETED, f"{runbook_id}: all stages passed")
            self.index_store.update_run_state(state.trace_id, "COMPLETED")
            state.progress = 1.0
            state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
            self._write_phase_checkpoint(
                state, recorder, len(stages), "final",
                extra={"runbook_id": runbook_id},
            )
            if results:
                self._persist_artifacts(state, results, recorder)
                self._persist_local_rendered_artifact(
                    state=state,
                    recorder=recorder,
                    results=results,
                    runbook_id=runbook_id,
                )

        except Exception as exc:
            logger.exception("Runbook %s failed", runbook_id)
            try:
                state.transition_to(RunState.FAILED, str(exc))
            except ValueError:
                state.run_state = RunState.FAILED
                state.message = str(exc)
            self.index_store.update_run_state(state.trace_id, "FAILED")
            recorder.record("error", {"error": str(exc)})

    # ------------------------------------------------------------------
    # Cognitive Core cycle (fixed 2-iteration demo loop)
    # ------------------------------------------------------------------

    def _run_cycle(
        self,
        state: AgentState,
        recorder: TraceRecorder,
        *,
        replay_provider: ReplayProvider | None = None,
        resume_ctx: dict[str, Any] | None = None,
    ) -> None:
        """Fixed demo loop with phase-boundary checkpoints.

        When *resume_ctx* is provided the cycle skips already-completed
        phases and continues from the checkpoint boundary.  Otherwise
        runs the full iter1 → iter2 sequence.
        """
        # Determine starting point
        skip_to = resume_ctx.get("phase_boundary", "") if resume_ctx else ""

        try:
            # -- Strict-fail preflight: LLM must be available ----------------
            if replay_provider is None and not self._llm_available():
                state.transition_to(RunState.FAILED, "LLM unavailable (strict_fail)")
                self.index_store.update_run_state(state.trace_id, "FAILED")
                recorder.record_decision(
                    "strict_fail: LLM unavailable",
                    {"api_key_env": self.config.llm.api_key_env},
                )
                return

            if not skip_to:
                state.transition_to(RunState.RUNNING, "Planning started")
                self.index_store.update_run_state(state.trace_id, "RUNNING")

            objective = state.intention.objective  # type: ignore[union-attr]
            constraints = list(state.intention.constraints)  # type: ignore[union-attr]

            # Variables that may be filled by iter1 or restored from ctx
            graph1: ActionGraph | None = None
            results1: dict[str, NodeOutput] = {}
            verdict1: DemoCriticVerdict | None = None
            fix_context: dict[str, Any] = {}
            added_constraints: list[str] = []

            # ============================================================
            # ITERATION 1
            # ============================================================

            # -- COMMIT iter1 ------------------------------------------
            if not skip_to or skip_to == "paused":
                state.intention.phase = IntentionPhase.COMMIT  # type: ignore[union-attr]
                graph1 = self._planner.plan_iter1(objective, constraints)
                state.intention.plan_id = graph1.metadata.plan_id  # type: ignore[union-attr]
                recorder.record_decision(
                    "Plan created (iter1)",
                    {
                        "iteration": 1,
                        "plan_id": graph1.metadata.plan_id,
                        "nodes": [n.node_id for n in graph1.nodes],
                        "edges": [(e.source, e.target) for e in graph1.edges],
                    },
                )
                self._write_phase_checkpoint(state, recorder, 1, "after_commit_iter1")

            # -- EXECUTE iter1 -----------------------------------------
            if skip_to not in {
                "after_execute_iter1", "after_control_iter1",
                "after_adapt_iter1",
                "after_commit_iter2", "after_execute_iter2",
                "after_control_iter2",
            }:
                if graph1 is None:
                    graph1 = self._planner.plan_iter1(objective, constraints)
                    state.intention.plan_id = graph1.metadata.plan_id  # type: ignore[union-attr]

                state.intention.phase = IntentionPhase.EXECUTE  # type: ignore[union-attr]
                results1 = self._execute(
                    state, graph1, recorder,
                    replay_provider=replay_provider,
                )
                self._write_phase_checkpoint(state, recorder, 1, "after_execute_iter1")

            # -- CONTROL iter1 -----------------------------------------
            if skip_to not in {
                "after_control_iter1", "after_adapt_iter1",
                "after_commit_iter2", "after_execute_iter2",
                "after_control_iter2",
            }:
                state.intention.phase = IntentionPhase.CONTROL  # type: ignore[union-attr]
                plan_id = state.intention.plan_id or ""  # type: ignore[union-attr]
                verdict1 = self._extract_verdict(results1, "critic")
                recorder.record("critic_verdict", {
                    "iteration": 1,
                    "plan_id": plan_id,
                    "critic_node_id": "critic",
                    "verdict": verdict1.verdict,
                    "confidence": verdict1.confidence,
                    "evidence": verdict1.evidence,
                    "gaps": verdict1.gaps,
                })
                self._write_phase_checkpoint(state, recorder, 1, "after_control_iter1")

                exec_ok_1 = (
                    results1.get("execute") is not None
                    and results1["execute"].status == "COMPLETED"
                )

                if exec_ok_1 and verdict1.verdict == "PASS" and verdict1.evidence:
                    state.intention.phase = IntentionPhase.ADAPT  # type: ignore[union-attr]
                    state.transition_to(RunState.COMPLETED, "Iter1: critic PASS with evidence")
                    self.index_store.update_run_state(state.trace_id, "COMPLETED")
                    state.progress = 1.0
                    state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
                    self._write_phase_checkpoint(state, recorder, 1, "final")
                    self._persist_artifacts(state, results1, recorder)
                    self._persist_local_rendered_artifact(
                        state=state,
                        recorder=recorder,
                        results=results1,
                        runbook_id="autofix_demo",
                    )
                    return

            # -- ADAPT iter1 (replan decision) ----------------------------
            if skip_to not in {
                "after_adapt_iter1",
                "after_commit_iter2", "after_execute_iter2",
                "after_control_iter2",
            }:
                state.intention.phase = IntentionPhase.ADAPT  # type: ignore[union-attr]

                if verdict1 is None:
                    verdict1 = DemoCriticVerdict(
                        verdict="UNCERTAIN",
                        confidence=0.0,
                        evidence=[],
                        gaps=["resume: no verdict1 available"],
                    )

                fix_context = self._build_fix_context(results1, verdict1)
                added_constraints = self._build_added_constraints(results1, verdict1)

                exec_ok_1_adapt = (
                    results1.get("execute") is not None
                    and results1["execute"].status == "COMPLETED"
                )

                recorder.record("replan", {
                    "iteration_from": 1,
                    "iteration_to": 2,
                    "reason": self._replan_reason(exec_ok_1_adapt, verdict1),
                    "added_constraints": added_constraints,
                    "previous_plan_id": state.intention.plan_id or "",  # type: ignore[union-attr]
                    "new_plan_id": "(pending)",
                })
                self._write_phase_checkpoint(
                    state, recorder, 1, "after_adapt_iter1",
                    extra={"fix_context": fix_context,
                           "added_constraints": added_constraints},
                )
            else:
                # Restore fix_context from resume_ctx if skipping
                if resume_ctx:
                    fix_context = resume_ctx.get("fix_context", {})
                    added_constraints = resume_ctx.get("added_constraints", [])

            # ============================================================
            # ITERATION 2 (fix-pipeline)
            # ============================================================

            # -- COMMIT iter2 ------------------------------------------
            if skip_to not in {
                "after_execute_iter2", "after_control_iter2",
            }:
                state.intention.phase = IntentionPhase.COMMIT  # type: ignore[union-attr]
                graph2 = self._planner.plan_iter2_fix(
                    objective,
                    constraints + added_constraints,
                    fix_context=fix_context,
                )
                state.intention.plan_id = graph2.metadata.plan_id  # type: ignore[union-attr]
                recorder.record_decision(
                    "Plan created (iter2 fix-pipeline)",
                    {
                        "iteration": 2,
                        "plan_id": graph2.metadata.plan_id,
                        "nodes": [n.node_id for n in graph2.nodes],
                        "edges": [(e.source, e.target) for e in graph2.edges],
                    },
                )
                self._write_phase_checkpoint(state, recorder, 2, "after_commit_iter2")
            else:
                # Need graph2 for execution
                graph2 = self._planner.plan_iter2_fix(
                    objective,
                    constraints + added_constraints,
                    fix_context=fix_context,
                )
                state.intention.plan_id = graph2.metadata.plan_id  # type: ignore[union-attr]

            # -- EXECUTE iter2 -----------------------------------------
            if skip_to not in {"after_control_iter2"}:
                state.intention.phase = IntentionPhase.EXECUTE  # type: ignore[union-attr]
                results2 = self._execute(
                    state, graph2, recorder,
                    replay_provider=replay_provider,
                    initial_inputs_override=fix_context,
                )
                self._write_phase_checkpoint(state, recorder, 2, "after_execute_iter2")
            else:
                results2 = {}

            # -- CONTROL iter2 -----------------------------------------
            state.intention.phase = IntentionPhase.CONTROL  # type: ignore[union-attr]
            verdict2 = self._extract_verdict(results2, "critic_fix")
            recorder.record("critic_verdict", {
                "iteration": 2,
                "plan_id": graph2.metadata.plan_id,
                "critic_node_id": "critic_fix",
                "verdict": verdict2.verdict,
                "confidence": verdict2.confidence,
                "evidence": verdict2.evidence,
                "gaps": verdict2.gaps,
            })
            self._write_phase_checkpoint(state, recorder, 2, "after_control_iter2")

            exec_ok_2 = (
                results2.get("execute_fix") is not None
                and results2["execute_fix"].status == "COMPLETED"
            )

            state.intention.phase = IntentionPhase.ADAPT  # type: ignore[union-attr]

            if exec_ok_2 and verdict2.verdict == "PASS" and verdict2.evidence:
                state.transition_to(RunState.COMPLETED, "Iter2: critic PASS with evidence")
                self.index_store.update_run_state(state.trace_id, "COMPLETED")
                state.progress = 1.0
            else:
                state.transition_to(
                    RunState.FAILED,
                    "Auto-fix exhausted after 2 iterations",
                )
                self.index_store.update_run_state(state.trace_id, "FAILED")

            state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
            self._write_phase_checkpoint(state, recorder, 2, "final")

            # Persist artifacts from whichever iteration produced results
            if results1:
                self._persist_artifacts(state, results1, recorder)
                self._persist_local_rendered_artifact(
                    state=state,
                    recorder=recorder,
                    results=results1,
                    runbook_id="autofix_demo",
                )
            if results2:
                self._persist_artifacts(state, results2, recorder)
                self._persist_local_rendered_artifact(
                    state=state,
                    recorder=recorder,
                    results=results2,
                    runbook_id="autofix_demo",
                )

        except Exception as exc:
            logger.exception("Orchestrator cycle failed")
            try:
                state.transition_to(RunState.FAILED, str(exc))
            except ValueError:
                state.run_state = RunState.FAILED
                state.message = str(exc)
            self.index_store.update_run_state(state.trace_id, "FAILED")
            recorder.record("error", {"error": str(exc)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_phase_checkpoint(
        self,
        state: AgentState,
        recorder: TraceRecorder,
        iteration: int,
        phase_boundary: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a structured phase-boundary checkpoint event."""
        cp_payload = build_checkpoint_payload(
            state,
            iteration=iteration,
            phase_boundary=phase_boundary,
            extra=extra,
        )
        recorder.record_checkpoint(cp_payload)

    def _get_memory_store(self) -> Any:
        """Lazily create the MemoryStore for the configured backend."""
        if self._memory_store is not None:
            return self._memory_store
        if not getattr(self.config.memory, "enabled", True):
            return None
        backend = getattr(self.config.memory, "graphrag_backend", "sqlite")
        if backend == "sqlite":
            from neuronium_agent.memory.sqlite_memory_store import SqliteMemoryStore
            self._memory_store = SqliteMemoryStore(self.config.storage.sqlite_path)
        elif backend == "postgres":
            from neuronium_agent.memory.postgres_memory_store import PostgresMemoryStore
            dsn = self.config.storage.postgres_dsn or ""
            schema = self.config.storage.postgres_schema
            self._memory_store = PostgresMemoryStore(dsn, schema=schema)
        return self._memory_store

    def _build_tool_runtime(self) -> Any:
        """Build a ToolRuntime with current stores/config."""
        from neuronium_agent.tools.runtime import ToolRuntime
        return ToolRuntime(
            config=self.config,
            index_store=self.index_store,
            blob_store=self.blob_store,
            memory_store=self._get_memory_store(),
        )

    def _llm_available(self) -> bool:
        """Check whether LLM credentials are present (strict_fail preflight)."""
        key = os.environ.get(self.config.llm.api_key_env, "")
        return bool(key)

    def _infer_runbook_id(self, trace_id: str) -> str | None:
        """Best-effort: infer runbook_id from existing trace events."""
        try:
            events = list(self.index_store.get_trace_events(trace_id))
        except Exception:
            return None
        for ev in reversed(events):
            if ev.get("kind") != "decision":
                continue
            payload = ev.get("payload", {})
            if isinstance(payload, dict) and payload.get("runbook_id"):
                return str(payload["runbook_id"])
        return None

    def _infer_runbook_metadata(
        self, trace_id: str, runbook_id: str,
    ) -> dict[str, Any]:
        """Best-effort: reconstruct metadata dict from trace events for replay."""
        metadata: dict[str, Any] = {"runbook_id": runbook_id}
        try:
            events = list(self.index_store.get_trace_events(trace_id))
        except Exception:
            return metadata
        for ev in reversed(events):
            if ev.get("kind") != "decision":
                continue
            payload = ev.get("payload", {})
            if not isinstance(payload, dict):
                continue
            description = str(payload.get("description", ""))
            if description == "control_command: revise":
                control_payload = payload.get("payload", {})
                if isinstance(control_payload, dict):
                    req_aid = str(
                        control_payload.get("clarification_request_artifact_id", "")
                    ).strip()
                    resp_aid = str(
                        control_payload.get("clarification_response_artifact_id", "")
                    ).strip()
                    if req_aid:
                        metadata["clarification_request_artifact_id"] = req_aid
                    if resp_aid:
                        metadata["clarification_response_artifact_id"] = resp_aid

                    answers = control_payload.get("answers", {})
                    if isinstance(answers, dict):
                        if isinstance(answers.get("url"), str) and answers["url"].strip():
                            metadata["url"] = answers["url"].strip()
                            metadata["urls"] = [answers["url"].strip()]
                        raw_urls = answers.get("urls")
                        if isinstance(raw_urls, list):
                            urls = [str(x).strip() for x in raw_urls if str(x).strip()]
                            if urls:
                                metadata["urls"] = urls
                                metadata["url"] = urls[0]
                        raw_doc_paths = answers.get("doc_paths")
                        if isinstance(raw_doc_paths, list):
                            doc_paths = [str(x).strip() for x in raw_doc_paths if str(x).strip()]
                            if doc_paths:
                                metadata["doc_paths"] = doc_paths
                        elif isinstance(raw_doc_paths, str) and raw_doc_paths.strip():
                            parts = [p.strip() for p in raw_doc_paths.split(",") if p.strip()]
                            if parts:
                                metadata["doc_paths"] = parts

                        out_fn = answers.get("output_filename")
                        if isinstance(out_fn, str) and out_fn.strip():
                            metadata["output_filename"] = out_fn.strip()
                        out_text = answers.get("output_text")
                        if isinstance(out_text, str) and out_text.strip():
                            metadata["output_text"] = out_text.strip()
            # Pick up doc_paths for docs_report_v1
            if (
                payload.get("runbook_id") == runbook_id
                and isinstance(payload.get("doc_paths"), list)
            ):
                metadata["doc_paths"] = [str(p) for p in payload["doc_paths"]]
                continue
            if (
                "Plan created" in str(payload.get("description", ""))
                and isinstance(payload.get("doc_paths"), list)
            ):
                metadata["doc_paths"] = [str(p) for p in payload["doc_paths"]]
                continue
        return metadata

    # -- Runbook gate helpers -----------------------------------------------

    @staticmethod
    def _build_gate_snapshot(
        results: dict[str, NodeOutput],
        gate: StageSuccessGate,
    ) -> dict[str, Any]:
        """Build a serialisable gate snapshot for checkpoint resume_context."""
        snapshot: dict[str, Any] = {
            "required_nodes_ok": all(
                results.get(nid) is not None
                and results[nid].status == "COMPLETED"
                for nid in gate.required_completed_nodes
            ),
        }
        if gate.critic_node_id:
            critic_out = results.get(gate.critic_node_id)
            if critic_out and critic_out.status == "COMPLETED":
                raw = critic_out.outputs.get("content", "")
                parsed = critic_out.outputs.get("parsed")
                if isinstance(parsed, dict):
                    snapshot["critic_verdict"] = parsed.get("verdict", "UNCERTAIN")
                    snapshot["critic_confidence"] = parsed.get("confidence", 0.0)
                    snapshot["critic_evidence"] = parsed.get("evidence", [])
                    snapshot["critic_gaps"] = parsed.get("gaps", [])
                else:
                    # Best-effort from raw JSON
                    try:
                        d = json.loads(raw) if isinstance(raw, str) else {}
                    except Exception:
                        d = {}
                    snapshot["critic_verdict"] = d.get("verdict", "UNCERTAIN")
                    snapshot["critic_confidence"] = d.get("confidence", 0.0)
                    snapshot["critic_evidence"] = d.get("evidence", [])
                    snapshot["critic_gaps"] = d.get("gaps", [])
            else:
                snapshot["critic_verdict"] = "UNCERTAIN"
                snapshot["critic_confidence"] = 0.0
                snapshot["critic_evidence"] = []
                snapshot["critic_gaps"] = ["critic node did not complete"]
        return snapshot

    @staticmethod
    def _evaluate_gate(
        results: dict[str, NodeOutput],
        gate: StageSuccessGate,
        verdict: DemoCriticVerdict | None,
        gate_snapshot: dict[str, Any] | None,
    ) -> bool:
        """Evaluate whether a stage's success gate passes.

        Uses live *results* + *verdict* if available, otherwise falls
        back to *gate_snapshot* (resume path).
        """
        # Check required completed nodes
        if results:
            for nid in gate.required_completed_nodes:
                out = results.get(nid)
                if out is None or out.status != "COMPLETED":
                    return False
        elif gate_snapshot is not None:
            if not gate_snapshot.get("required_nodes_ok", False):
                return False
        else:
            # No results and no snapshot — cannot evaluate
            return False

        # Check critic verdict
        if gate.critic_node_id:
            if verdict is not None:
                return verdict.verdict == "PASS" and bool(verdict.evidence)
            if gate_snapshot is not None:
                return (
                    gate_snapshot.get("critic_verdict") == "PASS"
                    and bool(gate_snapshot.get("critic_evidence"))
                )
            return False

        return True

    def _extract_verdict(
        self,
        results: dict[str, NodeOutput],
        critic_node_id: str,
    ) -> DemoCriticVerdict:
        """Parse the critic node output into a ``DemoCriticVerdict``."""
        critic_output = results.get(critic_node_id)
        if critic_output is None or critic_output.status != "COMPLETED":
            error_msg = ""
            if critic_output and critic_output.error:
                error_msg = critic_output.error
            return DemoCriticVerdict(
                verdict="UNCERTAIN",
                confidence=0.0,
                evidence=[],
                gaps=[f"Critic node '{critic_node_id}' failed: {error_msg}"],
            )

        raw = critic_output.outputs.get("content", "")
        # If structured output produced a parsed dict, use that
        parsed = critic_output.outputs.get("parsed")
        if isinstance(parsed, dict):
            try:
                verdict = DemoCriticVerdict(**parsed)
                # Enforce hard rule
                if verdict.verdict == "PASS" and not verdict.evidence:
                    return DemoCriticVerdict(
                        verdict="UNCERTAIN",
                        confidence=verdict.confidence,
                        evidence=[],
                        gaps=["PASS without evidence — downgraded"],
                    )
                return verdict
            except Exception:
                pass
        return parse_critic_verdict(raw)

    def _build_fix_context(
        self,
        results: dict[str, NodeOutput],
        verdict: DemoCriticVerdict,
    ) -> dict[str, Any]:
        """Build initial_inputs for the fix-pipeline from iteration 1 results."""
        ctx: dict[str, Any] = {}

        gen = results.get("generate")
        if gen and gen.status == "COMPLETED":
            ctx["previous_code"] = gen.outputs.get("content", "")

        exe = results.get("execute")
        if exe:
            ctx["previous_exit_code"] = exe.outputs.get("exit_code", -1)
            ctx["previous_stdout"] = exe.outputs.get("stdout", "")
            ctx["previous_stderr"] = exe.outputs.get("stderr", "")

        ctx["previous_verdict"] = verdict.verdict
        ctx["previous_gaps"] = verdict.gaps
        return ctx

    def _build_added_constraints(
        self,
        results: dict[str, NodeOutput],
        verdict: DemoCriticVerdict,
    ) -> list[str]:
        """Derive added constraints from iteration 1 failure."""
        added: list[str] = []
        exe = results.get("execute")
        if exe and exe.status == "FAILED":
            stderr = exe.outputs.get("stderr", "") or exe.error or ""
            if stderr:
                added.append(f"Fix execution error: {stderr[:300]}")
        for gap in verdict.gaps:
            added.append(f"Fix gap: {gap}")
        return added

    @staticmethod
    def _replan_reason(exec_ok: bool, verdict: DemoCriticVerdict) -> str:
        parts: list[str] = []
        if not exec_ok:
            parts.append("exec_failed")
        if verdict.verdict != "PASS":
            parts.append(f"critic_{verdict.verdict.lower()}")
        return ", ".join(parts) or "unknown"

    # ------------------------------------------------------------------
    # Execution (shared between iterations)
    # ------------------------------------------------------------------

    def _plan_dynamic_graph(
        self,
        *,
        state: AgentState,
        recorder: TraceRecorder,
        replay_provider: ReplayProvider | None,
        runbook_id: str,
        stage_id: str,
        metadata: dict[str, Any],
        spec: DynamicPlannerSpec,
    ) -> PlannerOutcome:
        """Run planner backend and return a validated planner result."""
        catalog_hash = self._operator_catalog.catalog_hash()
        if replay_provider is not None:
            replay_hash = replay_provider.latest_operator_catalog_hash()
            if replay_hash and replay_hash != catalog_hash:
                raise ValueError(
                    "Strict replay failed: operator catalog hash mismatch "
                    f"(live={catalog_hash}, replay={replay_hash})"
                )
        request = PlannerRequest(
            objective=state.intention.objective,  # type: ignore[union-attr]
            constraints=list(state.intention.constraints),  # type: ignore[union-attr]
            metadata=metadata,
            runbook_id=runbook_id,
            stage_id=stage_id,
            execution_id=state.execution_id,
            spec=spec,
            operator_catalog_hash=catalog_hash,
            allowed_capabilities={
                "node_types": list(spec.allowed_node_types),
                "tools": list(spec.allowed_tool_names),
            },
        )

        recorder.record_decision(
            "Planner request envelope",
            {
                "runbook_id": runbook_id,
                "stage_id": stage_id,
                "planner_backend": spec.backend_name,
                "planner_backend_version": spec.backend_version,
                "operator_catalog_hash": catalog_hash,
                "allowed_capabilities": request.allowed_capabilities,
            },
        )

        backend = get_planner_backend(spec.backend_name)
        result = backend.plan(
            request=request,
            execute_graph=lambda graph, initial_inputs, suppress_node_events: self._execute(
                state,
                graph,
                recorder,
                replay_provider=replay_provider,
                initial_inputs_override=initial_inputs,
                suppress_node_events=suppress_node_events,
            ),
        )
        if isinstance(result, PlannerEscalation):
            final_escalation = PlannerEscalation(
                reason=result.reason,
                backend_name=result.backend_name,
                backend_version=result.backend_version,
                clarification_request_artifact_id=result.clarification_request_artifact_id,
                missing_fields=list(result.missing_fields),
                evidence_artifact_ids=list(result.evidence_artifact_ids),
                operator_catalog_hash=result.operator_catalog_hash or catalog_hash,
                decision_trace=result.decision_trace,
            )
            recorder.record_decision(
                "Planner escalation envelope",
                {
                    "runbook_id": runbook_id,
                    "stage_id": stage_id,
                    "planner_backend": final_escalation.backend_name,
                    "planner_backend_version": final_escalation.backend_version,
                    "reason": final_escalation.reason,
                    "clarification_request_artifact_id": final_escalation.clarification_request_artifact_id,
                    "missing_fields": final_escalation.missing_fields,
                    "evidence_artifact_ids": final_escalation.evidence_artifact_ids,
                    "operator_catalog_hash": final_escalation.operator_catalog_hash,
                },
            )
            return final_escalation
        validated = validate_planned_graph(
            result.action_graph,
            spec=spec,
            operator_catalog=self._operator_catalog,
        )
        final_result = PlannerResult(
            action_graph=validated,
            backend_name=result.backend_name,
            backend_version=result.backend_version,
            operator_catalog_hash=result.operator_catalog_hash,
            decision_trace=result.decision_trace,
        )
        recorder.record_decision(
            "Planner result envelope",
            {
                "runbook_id": runbook_id,
                "stage_id": stage_id,
                "planner_backend": final_result.backend_name,
                "planner_backend_version": final_result.backend_version,
                "plan_id": validated.metadata.plan_id,
                "operator_catalog_hash": final_result.operator_catalog_hash,
                "decision_trace_notes": (
                    dict(final_result.decision_trace.notes)
                    if final_result.decision_trace is not None
                    else {}
                ),
                "decision_trace_subgoal_count": (
                    len(final_result.decision_trace.subgoals)
                    if final_result.decision_trace is not None
                    else 0
                ),
            },
        )
        return final_result

    def _pause_for_planner_escalation(
        self,
        *,
        state: AgentState,
        recorder: TraceRecorder,
        iteration: int,
        runbook_id: str,
        stage_id: str,
        stage_index: int,
        escalation: PlannerEscalation,
    ) -> None:
        """Handle planner escalation as Commit→Adapt→Escalate suspension."""
        state.intention.phase = IntentionPhase.ADAPT  # type: ignore[union-attr]
        recorder.record_decision(
            "Missing critical parameters",
            {
                "iteration": iteration,
                "runbook_id": runbook_id,
                "stage_id": stage_id,
                "reason": escalation.reason,
                "missing_fields": list(escalation.missing_fields),
                "evidence": [
                    {"artifact_id": aid, "relevance_score": 1.0}
                    for aid in escalation.evidence_artifact_ids
                ],
            },
        )
        recorder.record_decision(
            "Escalation requested",
            {
                "iteration": iteration,
                "runbook_id": runbook_id,
                "stage_id": stage_id,
                "planner_backend": escalation.backend_name,
                "planner_backend_version": escalation.backend_version,
                "clarification_request_artifact_id": escalation.clarification_request_artifact_id,
                "missing_fields": list(escalation.missing_fields),
                "evidence_artifact_ids": list(escalation.evidence_artifact_ids),
            },
        )
        state.transition_to(
            RunState.PAUSED,
            "Clarification required: missing critical parameters",
        )
        self.index_store.update_run_state(state.trace_id, "PAUSED")
        self._write_phase_checkpoint(
            state,
            recorder,
            iteration,
            "paused",
            extra={
                "runbook_id": runbook_id,
                "stage_id": stage_id,
                "stage_index": stage_index,
                "clarification_request_artifact_id": escalation.clarification_request_artifact_id,
                "clarification_missing_fields": list(escalation.missing_fields),
                "clarification_evidence_artifact_ids": list(escalation.evidence_artifact_ids),
                "planner_backend": escalation.backend_name,
                "planner_backend_version": escalation.backend_version,
                "operator_catalog_hash": escalation.operator_catalog_hash or "",
            },
        )

    def _execute(
        self,
        state: AgentState,
        graph: ActionGraph,
        recorder: TraceRecorder,
        *,
        replay_provider: ReplayProvider | None = None,
        initial_inputs_override: dict[str, Any] | None = None,
        suppress_node_events: bool = False,
    ) -> dict[str, NodeOutput]:
        """Build node registry and execute the DAG."""
        registry = self._build_node_registry(graph)

        # Enable recording for replay
        recordings: dict[str, list] = {}
        if replay_provider is None:
            for nid, node in registry.items():
                if hasattr(node, "enable_recording"):
                    recordings[nid] = node.enable_recording()
        else:
            report = replay_provider.inject(registry, strict=True)
            recorder.record_decision("Replay responses injected", report)

        def trace_cb(kind: str, payload: dict[str, Any]) -> None:
            if suppress_node_events and kind in {"node_start", "node_end"}:
                return
            recorder.record(kind, payload)

        executor = DAGExecutor(
            registry,
            max_parallel=self.config.runtime.max_parallel_nodes,
            execution_id=state.execution_id,
            trace_id=state.trace_id,
            random_seed=self.config.determinism.default_random_seed,
            trace_callback=trace_cb,
        )

        objective = state.intention.objective  # type: ignore[union-attr]
        constraints = state.intention.constraints  # type: ignore[union-attr]

        base_inputs: dict[str, Any] = {
            "objective": objective,
            "constraints": constraints,
        }
        if initial_inputs_override:
            base_inputs.update(initial_inputs_override)

        results = executor.execute(graph, initial_inputs=base_inputs)

        # Record replay data as trace events
        for nid, recs in recordings.items():
            if recs:
                recorder.record("replay_data", {
                    "node_id": nid,
                    "recorded_responses": recs,
                })

        return results

    def _build_node_registry(
        self, graph: ActionGraph
    ) -> dict[str, BaseNode]:
        """Instantiate concrete node implementations for each graph node."""
        registry: dict[str, BaseNode] = {}
        for gn in graph.nodes:
            if gn.node_type == "model":
                registry[gn.node_id] = ModelNode(
                    node_id=gn.node_id,
                    parameters=gn.parameters,
                    model=self.config.llm.model,
                    provider=self.config.llm.provider,
                    api_key_env=self.config.llm.api_key_env,
                    base_url=self.config.llm.base_url,
                    structured_output=self.config.llm.structured_output,
                    temperature=self.config.determinism.llm_temperature,
                    timeout=self.config.llm.timeout_seconds,
                    max_retries=self.config.llm.max_retries,
                )
            elif gn.node_type == "code":
                dc = self.config.code_node.docker
                registry[gn.node_id] = CodeNode(
                    node_id=gn.node_id,
                    parameters=gn.parameters,
                    image=dc.image,
                    network_enabled=dc.network_enabled,
                    cpu_limit=dc.cpu_limit,
                    mem_limit=dc.mem_limit,
                    timeout_seconds=dc.timeout_seconds,
                )
            elif gn.node_type == "mcp":
                # v0.2: local transport tools, policy defaults to CWD allowlist.
                roots = []
                if self.config.mcp.enabled:
                    roots = [os.getcwd()]
                registry[gn.node_id] = McpToolNode(
                    node_id=gn.node_id,
                    parameters=gn.parameters,
                    server_name="local",
                    server_url="local://",
                    timeout_seconds=60,
                    policy={
                        "fs_roots_allowlist": roots,
                        "fs_max_read_bytes": 1_000_000,
                        "fs_max_write_bytes": 1_000_000,
                    },
                    tool_runtime=self._build_tool_runtime(),
                )
            elif gn.node_type == "decision":
                from neuronium_agent.nodes.decision_node import DecisionNode

                registry[gn.node_id] = DecisionNode(
                    node_id=gn.node_id, parameters=gn.parameters
                )
            elif gn.node_type == "aggregate":
                from neuronium_agent.nodes.aggregate_node import AggregateNode

                registry[gn.node_id] = AggregateNode(
                    node_id=gn.node_id, parameters=gn.parameters
                )
            else:
                # For other node types, use a pass-through stub
                from neuronium_agent.nodes.aggregate_node import AggregateNode

                registry[gn.node_id] = AggregateNode(
                    node_id=gn.node_id, parameters=gn.parameters
                )
        return registry

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def _persist_control_artifact(
        self,
        *,
        artifact_type: str,
        payload: dict[str, Any],
        produced_by_node_ref: str,
        parent_artifact_ids: list[str],
    ) -> str:
        """Persist control-protocol payload as content-addressed artifact."""
        content = canonical_bytes(payload)
        ctx = {
            "node_ref": produced_by_node_ref,
            "input_artifact_ids": sorted(parent_artifact_ids),
        }
        aid = artifact_id(content, ctx)
        now = datetime.now(timezone.utc).isoformat()
        self.blob_store.put(aid, content, "application/json")
        self.index_store.record_artifact_metadata(
            artifact_id=aid,
            artifact_type=artifact_type,
            created_at=now,
            produced_by_node_ref=produced_by_node_ref,
            inputs_json=canonical_json({"parent_artifact_ids": sorted(parent_artifact_ids)}),
            quality_signals_json="{}",
            blob_key=aid,
            media_type="application/json",
            size_bytes=len(content),
        )
        return aid

    def _persist_artifacts(
        self,
        state: AgentState,
        results: dict[str, NodeOutput],
        recorder: TraceRecorder,
    ) -> None:
        """Persist node outputs as immutable artifacts."""
        for nid, output in results.items():
            if output.status != "COMPLETED":
                continue
            content = canonical_bytes(output.outputs)
            now = datetime.now(timezone.utc)
            ctx = {
                "timestamp": now.isoformat(),
                "node_ref": f"{state.execution_id}:{state.intention.plan_id}/execute/{nid}",  # type: ignore[union-attr]
                "input_artifact_ids": [],
            }
            aid = artifact_id(content, ctx)

            self.blob_store.put(aid, content, "application/json")
            self.index_store.record_artifact_metadata(
                artifact_id=aid,
                artifact_type=f"node_output:{nid}",
                created_at=now.isoformat(),
                produced_by_node_ref=ctx["node_ref"],
                inputs_json="[]",
                quality_signals_json=canonical_json(
                    output.quality_signals.model_dump(mode="json")
                ),
                blob_key=aid,
                media_type="application/json",
                size_bytes=len(content),
            )

    def _persist_local_rendered_artifact(
        self,
        *,
        state: AgentState,
        recorder: TraceRecorder,
        results: dict[str, NodeOutput],
        runbook_id: str,
    ) -> None:
        """Render deterministic artifacts and append local gallery index.

        Produces:
        - debug artifact (all node outputs)
        - user-facing artifact (title + summary + source)
        Index points to user-facing artifact when available.
        """
        objective = state.intention.objective  # type: ignore[union-attr]
        plan_id = state.intention.plan_id or ""  # type: ignore[union-attr]
        debug_rendered = render_run_artifact(
            data_dir=self.config.project.data_dir,
            trace_id=state.trace_id,
            runbook_id=runbook_id,
            objective=objective,
            plan_id=plan_id,
            results=results,
        )
        if debug_rendered is None:
            return

        user_path = render_user_facing_html(
            data_dir=self.config.project.data_dir,
            trace_id=state.trace_id,
            runbook_id=runbook_id,
            objective=objective,
            plan_id=plan_id,
            results=results,
            debug_artifact_path=debug_rendered.path,
        )
        chosen_path = user_path or debug_rendered.path

        # Record a compact user output decision for CLI/demo
        user_summary = extract_user_facing_summary(results)
        recorder.record_decision(
            "User output extracted",
            {
                "runbook_id": runbook_id,
                "trace_id": state.trace_id,
                "title": user_summary.title,
                "summary": user_summary.summary,
                "source_url": user_summary.source_url,
            },
        )

        index = LocalArtifactIndex(self.config.project.data_dir)
        index.append(LocalIndexEntry(
            trace_id=state.trace_id,
            runbook_id=runbook_id,
            objective=objective,
            artifact_path=chosen_path,
            created_at=debug_rendered.created_at,
            plan_id=plan_id,
        ))
        recorder.record_decision(
            "Local rendered artifact saved",
            {
                "runbook_id": runbook_id,
                "trace_id": state.trace_id,
                "artifact_path": chosen_path,
                "artifact_path_debug": debug_rendered.path,
                "artifact_path_user": user_path,
            },
        )
