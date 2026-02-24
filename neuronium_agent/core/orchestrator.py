"""Orchestrator — the Cognitive Core loop (IBS §1.1).

Single execution path: **generic N-stage runbook runner** for any registered
``Runbook`` (including ``autofix_demo``, which is a two-stage runbook).
Each stage goes through COMMIT → EXECUTE → CONTROL → ADAPT with
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
from typing import Any, Callable, Union

from neuronium_agent._canonical import canonical_json, artifact_id, canonical_bytes
from neuronium_agent.artifacts.local_index import LocalArtifactIndex, LocalIndexEntry
from neuronium_agent.artifacts.renderer import render_run_artifact
from neuronium_agent.artifacts.user_renderer import (
    extract_user_facing_summary,
    render_user_facing_html,
)
from neuronium_agent.config import AppConfig
from neuronium_agent.errors import ConfigError
from neuronium_agent.model_catalog_defaults import (
    get_default_catalog,
    resolve_model_for_node,
)
from neuronium_agent.core.state import (
    AgentState,
    Intention,
    IntentionPhase,
    RunState,
)
from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.execution.outcome import ExecutionOutcome
from neuronium_agent.nodes.base import BaseNode, NodeOutput
from neuronium_agent.nodes.code_node import CodeNode
from neuronium_agent.nodes.mcp_node import McpToolNode
from neuronium_agent.nodes.model_node import ModelNode
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode
from neuronium_agent.planning.dynamic_planner import validate_planned_graph
from neuronium_agent.planning.htn import HTNPlanner
from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.planner_backend import get_planner_backend
from neuronium_agent.planning.state_patch import PatchOperation, StatePatchError, apply_patch
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
from neuronium_agent.trace.exporter import TraceExporter
from neuronium_agent.trace.decision_record import (
    DecisionAuthority,
    DecisionRecord,
    DecisionType,
    OptionConsidered,
    OutcomeCorrelation,
    SelectedOption,
)
from neuronium_agent.trace.recorder import TraceRecorder
from neuronium_agent.trace.replay import ReplayProvider
from neuronium_agent.recovery import (
    RecoveryAction,
    compute_rollback_scope,
    decide_recovery,
)
from neuronium_agent.types import ControlCommand, InterruptRequest, RunHandle, RunRequest, RunStatus
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
        self._interrupt_requests: dict[str, InterruptRequest | None] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def start(
        self,
        request: RunRequest,
        *,
        on_handle_ready: Callable[[RunHandle], None] | None = None,
    ) -> RunHandle:
        """Start a new agent run (Commit phase).

        If on_handle_ready is provided (e.g. for interactive CLI), it is called
        with the RunHandle before _run_runbook blocks, so the caller can send
        control commands (pause/stop) during execution.
        """
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
            decision_type=DecisionType.PLANNING,
        )

        runbook_id = (
            (request.metadata or {}).get("runbook_id")  # type: ignore[union-attr]
            or "autofix_demo"
        )
        recorder.record_decision(
            "Runbook selected",
            {"runbook_id": runbook_id},
            record=DecisionRecord(
                decision_type=DecisionType.PLANNING,
                selected_option=SelectedOption(
                    option_id=runbook_id,
                    selection_rationale="Runbook selected",
                    decision_authority=DecisionAuthority.COMPONENT,
                    expected_outcome="Run proceeds with selected runbook",
                ),
            ),
        )

        handle = RunHandle(
            trace_id=trace_id,
            execution_id=execution_id,
            created_at=now,
        )
        if on_handle_ready is not None:
            on_handle_ready(handle)

        # Run synchronously (batch mode)
        if get_runbook(runbook_id) is not None:
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
            decision_type=DecisionType.EXECUTION,
        )

        handle = RunHandle(
            trace_id=trace_id,
            execution_id=execution_id,
            created_at=now,
        )
        # Prefer runbook_id and metadata from original trace's checkpoint
        replay_resume_ctx: dict[str, Any] = {}
        try:
            orig_cp = get_latest_phase_boundary_checkpoint(
                self.index_store, original_trace_id,
            )
            if orig_cp is not None:
                _, replay_resume_ctx = load_state_from_checkpoint(orig_cp)
        except CheckpointError:
            pass
        runbook_id = (
            replay_resume_ctx.get("runbook_id")
            or self._infer_runbook_id(original_trace_id)
            or "autofix_demo"
        )
        recorder.record_decision(
            "Runbook selected (replay)",
            {"runbook_id": runbook_id, "replay_of_trace_id": original_trace_id},
            record=DecisionRecord(
                decision_type=DecisionType.PLANNING,
                selected_option=SelectedOption(
                    option_id=runbook_id or "default",
                    selection_rationale="Runbook selected (replay)",
                    decision_authority=DecisionAuthority.COMPONENT,
                    expected_outcome="Replay proceeds with selected runbook",
                ),
            ),
        )
        if get_runbook(runbook_id) is not None:
            replay_metadata = (
                replay_resume_ctx.get("metadata")
                if isinstance(replay_resume_ctx.get("metadata"), dict)
                else self._infer_runbook_metadata(original_trace_id, runbook_id)
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
    # Declarative meta-control (no user plan DAG execution)
    # ------------------------------------------------------------------

    def apply_control(
        self,
        trace_id: str,
        command: ControlCommand,
    ) -> RunStatus:
        """Apply a control command **declaratively**.

        Performs: state transition → checkpoint → trace decision.
        Does not execute user plan DAGs or trigger stage orchestration.
        Note: revise may perform a bounded internal model conversion step
        (`answer_text -> patch`) when no structured patch/answers are provided.
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
            self._interrupt_requests[trace_id] = InterruptRequest(
                command="pause", mode="graceful"
            )
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
            pl = payload or {}
            stop_mode = pl.get("mode", "graceful")
            export_path = pl.get("export_path")
            if isinstance(export_path, str):
                export_path = export_path.strip() or None
            else:
                export_path = None
            self._interrupt_requests[trace_id] = InterruptRequest(
                command="stop", mode=stop_mode, export_path=export_path
            )
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
            patch_ops = self._normalise_patch_ops(payload.get("patch"))
            if not patch_ops and answers:
                patch_ops = self._legacy_answers_to_patch(answers)
            answer_text = payload.get("answer_text")
            if not isinstance(answer_text, str):
                answer_text = None
            request_artifact_id = str(
                payload.get("clarification_request_artifact_id", "")
            ).strip()
            nl_conversion: dict[str, Any] | None = None
            if not patch_ops and not answers and answer_text:
                patch_ops, nl_conversion = self._convert_nl_answer_to_patch(
                    state=state,
                    recorder=recorder,
                    answer_text=answer_text,
                    clarification_request_artifact_id=request_artifact_id,
                )

            decision_payload = dict(payload)
            decision_payload["patch"] = patch_ops
            if nl_conversion is not None:
                decision_payload["nl_patch_conversion"] = nl_conversion
            if answers or answer_text or patch_ops:
                plan_id = (
                    state.intention.plan_id
                    if state.intention and state.intention.plan_id
                    else "no-plan"
                )
                response_payload = {
                    "request_artifact_id": request_artifact_id,
                    "patch": patch_ops,
                    "answer_text": answer_text,
                }
                if answers:
                    response_payload["legacy_answers"] = answers
                if nl_conversion is not None:
                    response_payload["nl_patch_conversion"] = nl_conversion
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
            record=DecisionRecord(
                decision_type=DecisionType.META_CONTROL,
                selected_option=SelectedOption(
                    option_id=cmd,
                    selection_rationale="control_command: " + cmd,
                    decision_authority=DecisionAuthority.USER,
                    expected_outcome="State and checkpoint updated per command",
                ),
            ),
        )
        iteration = payload.get("iteration", 0)
        extra: dict[str, Any] = {"control_command": cmd, "control_payload": payload}
        # Preserve runbook_id and metadata from latest checkpoint for resume
        try:
            cp = get_latest_phase_boundary_checkpoint(self.index_store, state.trace_id)
            if cp is not None:
                _, resume_ctx = load_state_from_checkpoint(cp)
                if resume_ctx.get("runbook_id"):
                    extra["runbook_id"] = resume_ctx["runbook_id"]
                if isinstance(resume_ctx.get("metadata"), dict):
                    extra["metadata"] = resume_ctx["metadata"]
        except CheckpointError:
            pass
        cp_payload = build_checkpoint_payload(
            state,
            iteration=iteration,
            phase_boundary="paused" if state.run_state == RunState.PAUSED else "after_control_command",
            extra=extra,
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
            decision_type=DecisionType.EXECUTION,
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
            record=DecisionRecord(
                decision_type=DecisionType.PLANNING,
                selected_option=SelectedOption(
                    option_id=runbook_id,
                    selection_rationale="Runbook selected (resume)",
                    decision_authority=DecisionAuthority.COMPONENT,
                    expected_outcome="Resume proceeds with selected runbook",
                ),
            ),
        )

        if get_runbook(runbook_id) is not None:
            resume_metadata = (
                dict(resume_ctx["metadata"])
                if isinstance(resume_ctx.get("metadata"), dict)
                else self._infer_runbook_metadata(trace_id, runbook_id)
            )
            if isinstance(resume_ctx.get("metadata"), dict):
                # Merge inferred (e.g. revise/clarification answers) so post-checkpoint updates are included
                inferred = self._infer_runbook_metadata(trace_id, runbook_id)
                for k, v in inferred.items():
                    if v is not None:
                        resume_metadata[k] = v
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
        self._interrupt_requests.pop(state.trace_id, None)
        try:
            # Strict-fail preflight
            if replay_provider is None and not self._llm_available():
                state.transition_to(RunState.FAILED, "LLM unavailable (strict_fail)")
                self.index_store.update_run_state(state.trace_id, "FAILED")
                recorder.record_decision(
                    "strict_fail: LLM unavailable",
                    {"api_key_env": self.config.llm.api_key_env},
                    decision_type=DecisionType.ESCALATION,
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
            prev_stage_results: dict[str, NodeOutput] = {}
            prev_stage_verdict: DemoCriticVerdict | None = None

            for stage_index, stage in enumerate(stages):
                iteration = stage_index + 1  # 1-based for checkpoint compatibility

                # Skip stages completed before the resume checkpoint
                if resume_ctx and stage_index < resume_stage_index:
                    continue

                # Determine which phases to skip within the current stage.
                # Phase order: commit → execute → control.
                # "after_X_iterN" means X is already done — skip it and
                # all preceding phases.  Generalised for N stages.
                # paused_mid_execute: commit done (use planned_graph), execute continues from exact pause point.
                skip_commit = False
                skip_execute = False
                skip_control = False
                if resume_ctx and stage_index == resume_stage_index:
                    sfx = f"_iter{iteration}"
                    past_commit = {f"after_execute{sfx}", f"after_control{sfx}"}
                    past_control = {f"after_control{sfx}"}
                    if resume_boundary == "paused_mid_execute":
                        skip_commit = True
                        skip_execute = False
                    else:
                        skip_commit = resume_boundary in past_commit
                        skip_execute = resume_boundary in past_commit
                    skip_control = resume_boundary in past_control

                graph = stage.graph
                graph_builder_initial_inputs: dict[str, Any] | None = None
                if stage.graph_builder is not None:
                    if skip_commit:
                        planned_graph = (resume_ctx or {}).get("planned_graph")
                        if not isinstance(planned_graph, dict):
                            raise ValueError(
                                "Resume for graph_builder stage requires "
                                "'planned_graph' in checkpoint resume_context"
                            )
                        graph = ActionGraph.model_validate(planned_graph)
                    else:
                        builder_context: dict[str, Any] = {
                            "objective": objective,
                            "constraints": constraints,
                            "prev_stage_results": prev_stage_results,
                            "prev_stage_verdict": prev_stage_verdict,
                            "execution_id": state.execution_id,
                            "metadata": metadata,
                        }
                        result = stage.graph_builder(builder_context)
                        if isinstance(result, tuple):
                            graph, graph_builder_initial_inputs = result[0], result[1]
                        else:
                            graph = result
                elif stage.dynamic_planner is not None and skip_commit:
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
                stage_retry_count = (
                    int(resume_ctx.get("stage_retry_count", 0))
                    if (resume_ctx and stage_index == resume_stage_index)
                    else 0
                )

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
                            stage_default_model_id=getattr(
                                stage, "default_model_id", None
                            ),
                        )
                        if isinstance(planner_outcome, PlannerEscalation):
                            self._pause_for_planner_escalation(
                                state=state,
                                recorder=recorder,
                                iteration=iteration,
                                runbook_id=runbook_id,
                                metadata=metadata,
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
                            decision_type=DecisionType.PLANNING,
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
                            decision_type=DecisionType.PLANNING,
                        )
                    state.intention.plan_id = graph.metadata.plan_id  # type: ignore[union-attr]
                    commit_extra: dict[str, Any] = {
                        "runbook_id": runbook_id,
                        "metadata": metadata,
                        "stage_id": stage.stage_id,
                        "stage_index": stage_index,
                    }
                    if stage.dynamic_planner is not None or stage.graph_builder is not None:
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

                # -- EXECUTE + CONTROL + ADAPT (with retry-stage loop) ---------
                failure_history_in_stage: list[list[str]] = []
                verdict_fix_attempts = 0
                stage_verdict_fix_override: dict[str, Any] | None = None
                clarification_context_inputs = self._build_clarification_context_inputs(
                    metadata
                )
                while True:
                    if not skip_execute:
                        state.intention.phase = IntentionPhase.EXECUTE  # type: ignore[union-attr]
                        execute_inputs = (stage.initial_inputs_override or {}) | (
                            graph_builder_initial_inputs or {}
                        ) | clarification_context_inputs | (
                            stage_verdict_fix_override or {}
                        )
                        initial_node_results: dict[str, NodeOutput] | None = None
                        if (
                            resume_ctx
                            and stage_index == resume_stage_index
                            and resume_boundary == "paused_mid_execute"
                        ):
                            raw = resume_ctx.get("completed_node_results")
                            if isinstance(raw, dict) and raw:
                                initial_node_results = {
                                    nid: NodeOutput.model_validate(payload)
                                    for nid, payload in raw.items()
                                }
                        results = self._execute(
                            state, graph, recorder,
                            replay_provider=replay_provider,
                            initial_inputs_override=execute_inputs or None,
                            initial_node_results=initial_node_results,
                            stage_default_model_id=getattr(
                                stage, "default_model_id", None
                            ),
                        )
                        stage_verdict_fix_override = None
                        if isinstance(results, ExecutionOutcome) and results.interrupted is not None:
                            outcome = results
                            if outcome.interrupted.command == "pause":
                                gate_snapshot_interrupt = self._build_gate_snapshot(
                                    outcome.results, gate
                                )
                                mid_extra = {
                                    "runbook_id": runbook_id,
                                    "metadata": metadata,
                                    "stage_id": stage.stage_id,
                                    "stage_index": stage_index,
                                    "pending_node_ids": outcome.pending,
                                    "plan_id": graph.metadata.plan_id,
                                    "stage_retry_count": stage_retry_count,
                                    "gate_snapshot": gate_snapshot_interrupt,
                                    "completed_node_results": {
                                        nid: out.model_dump(mode="json")
                                        for nid, out in outcome.results.items()
                                    },
                                    "planned_graph": graph.model_dump(mode="json"),
                                }
                                self._write_phase_checkpoint(
                                    state,
                                    recorder,
                                    iteration,
                                    "paused_mid_execute",
                                    extra=mid_extra,
                                )
                                state.transition_to(
                                    RunState.PAUSED,
                                    "User command: pause (mid-execution)",
                                )
                                self.index_store.update_run_state(
                                    state.trace_id, "PAUSED"
                                )
                                self._record_control_decision(
                                    recorder, state, "pause", {}
                                )
                            else:
                                # stop: graceful = full checkpoint, immediate = minimal
                                if outcome.interrupted.mode == "graceful":
                                    gate_snapshot_interrupt = self._build_gate_snapshot(
                                        outcome.results, gate
                                    )
                                    mid_extra = {
                                        "runbook_id": runbook_id,
                                        "metadata": metadata,
                                        "stage_id": stage.stage_id,
                                        "stage_index": stage_index,
                                        "pending_node_ids": outcome.pending,
                                        "plan_id": graph.metadata.plan_id,
                                        "stage_retry_count": stage_retry_count,
                                        "gate_snapshot": gate_snapshot_interrupt,
                                        "completed_node_results": {
                                            nid: out.model_dump(mode="json")
                                            for nid, out in outcome.results.items()
                                        },
                                        "planned_graph": graph.model_dump(mode="json"),
                                    }
                                    self._write_phase_checkpoint(
                                        state,
                                        recorder,
                                        iteration,
                                        "paused_mid_execute",
                                        extra=mid_extra,
                                    )
                                else:
                                    mid_extra_minimal = {
                                        "runbook_id": runbook_id,
                                        "metadata": metadata,
                                        "stage_id": stage.stage_id,
                                        "stage_index": stage_index,
                                        "plan_id": graph.metadata.plan_id,
                                        "stage_retry_count": stage_retry_count,
                                    }
                                    self._write_phase_checkpoint(
                                        state,
                                        recorder,
                                        iteration,
                                        "cancelled_mid_execute",
                                        extra=mid_extra_minimal,
                                    )
                                state.transition_to(
                                    RunState.CANCELLED,
                                    "User command: stop (mid-execution)",
                                )
                                self.index_store.update_run_state(
                                    state.trace_id, "CANCELLED"
                                )
                                self._record_control_decision(
                                    recorder, state, "stop",
                                    {"mode": outcome.interrupted.mode},
                                )
                                export_path = outcome.interrupted.export_path
                                if export_path:
                                    events = list(
                                        self.index_store.get_trace_events(
                                            state.trace_id
                                        )
                                    )
                                    fmt = "jsonl"
                                    if export_path.endswith(".json"):
                                        fmt = "json"
                                    elif export_path.endswith(".zip"):
                                        fmt = "zip"
                                    try:
                                        TraceExporter().export(
                                            events, export_path, fmt=fmt
                                        )
                                    except Exception as e:  # noqa: BLE001
                                        logger.warning(
                                            "Trace export on stop failed: %s",
                                            e,
                                            exc_info=True,
                                        )
                            self._interrupt_requests.pop(state.trace_id, None)
                            return
                        if isinstance(results, ExecutionOutcome):
                            results = results.results
                        gate_snapshot = self._build_gate_snapshot(results, gate)
                        self._write_phase_checkpoint(
                            state, recorder, iteration,
                            f"after_execute_iter{iteration}",
                            extra={
                                "runbook_id": runbook_id,
                                "metadata": metadata,
                                "stage_id": stage.stage_id,
                                "stage_index": stage_index,
                                "gate_snapshot": gate_snapshot,
                                "stage_retry_count": stage_retry_count,
                            }
                            | (
                                {"planned_graph": graph.model_dump(mode="json")}
                                if stage.dynamic_planner is not None
                                else {}
                            ),
                        )
                    else:
                        gate_snapshot = (resume_ctx or {}).get("gate_snapshot")

                    if not skip_control:
                        state.intention.phase = IntentionPhase.CONTROL  # type: ignore[union-attr]
                        if gate.critic_node_id and not skip_execute:
                            verdict = self._extract_verdict(results, gate.critic_node_id)
                        elif gate.critic_node_id and gate_snapshot:
                            verdict = DemoCriticVerdict(
                                verdict=gate_snapshot.get("critic_verdict", "UNCERTAIN"),
                                confidence=gate_snapshot.get("critic_confidence", 0.0),
                                evidence=gate_snapshot.get("critic_evidence", []),
                                gaps=gate_snapshot.get("critic_gaps", []),
                                suggestions=gate_snapshot.get("critic_suggestions", []),
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
                                "suggestions": verdict.suggestions,
                            })
                        self._write_phase_checkpoint(
                            state, recorder, iteration,
                            f"after_control_iter{iteration}",
                            extra={
                                "runbook_id": runbook_id,
                                "metadata": metadata,
                                "stage_id": stage.stage_id,
                                "stage_index": stage_index,
                                "stage_retry_count": stage_retry_count,
                            }
                            | (
                                {"planned_graph": graph.model_dump(mode="json")}
                                if stage.dynamic_planner is not None
                                else {}
                            ),
                        )

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

                    if stage_passed:
                        prev_stage_results = results
                        prev_stage_verdict = verdict
                        if stage.exit_run_on_success:
                            state.transition_to(
                                RunState.COMPLETED,
                                f"{runbook_id}: stage {stage.stage_id} passed (exit_run_on_success)",
                            )
                            self.index_store.update_run_state(state.trace_id, "COMPLETED")
                            state.progress = 1.0
                            state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
                            self._write_phase_checkpoint(
                                state, recorder, iteration, "final",
                                extra={"runbook_id": runbook_id, "metadata": metadata},
                            )
                            if results:
                                self._finalize_run(
                                    state, recorder, results, runbook_id
                                )
                            return
                        break

                    if stage.proceed_to_next_stage_on_fail:
                        prev_stage_results = results
                        prev_stage_verdict = verdict
                        recorder.record("replan", {
                            "iteration_from": iteration,
                            "iteration_to": iteration + 1,
                            "reason": "proceed_to_next_stage_on_fail",
                            "stage_id": stage.stage_id,
                            "runbook_id": runbook_id,
                        })
                        break

                    # Recovery: collect failed nodes, compute rollback scope, decide action
                    failed_nodes = [
                        (nid, results[nid])
                        for nid in gate.required_completed_nodes
                        if (out := results.get(nid)) and out.status != "COMPLETED"
                    ]
                    critic_failed = bool(
                        gate.critic_node_id
                        and (
                            verdict is None
                            or verdict.verdict != "PASS"
                            or not (verdict.evidence if verdict else False)
                        )
                    )
                    failed_node_ids = {nid for nid, _ in failed_nodes}
                    completed_node_ids = {
                        nid for nid, out in results.items() if out.status == "COMPLETED"
                    }
                    failure_type: str = (
                        "critic_rejection" if critic_failed else "node_execution"
                    )

                    # B2 Part 1: verdict-driven local fix before decide_recovery
                    # Only when we ran execute this iteration (not when resuming from snapshot)
                    has_fix_hints = verdict is not None and (
                        bool(verdict.gaps) or bool(getattr(verdict, "suggestions", []))
                    )
                    if (
                        critic_failed
                        and has_fix_hints
                        and verdict_fix_attempts < self.config.recovery.max_verdict_fix_attempts
                        and not skip_execute
                    ):
                        verdict_fix = {
                            "gaps": verdict.gaps,
                            "suggestions": getattr(verdict, "suggestions", []),
                        }
                        recorder.record("verdict_local_fix_retry", {
                            "stage_id": stage.stage_id,
                            "stage_index": stage_index,
                            "runbook_id": runbook_id,
                            "verdict_fix": verdict_fix,
                            "verdict_fix_attempt": verdict_fix_attempts + 1,
                        })
                        verdict_fix_attempts += 1
                        stage_verdict_fix_override = {"verdict_fix": verdict_fix}
                        skip_execute = False
                        skip_control = False
                        continue

                    rollback_scope = compute_rollback_scope(
                        failure_type,
                        graph,
                        failed_node_ids,
                        critic_failed=critic_failed,
                        completed_node_ids=completed_node_ids,
                        gate_required_node_ids=set(gate.required_completed_nodes),
                    )
                    decision = decide_recovery(
                        failed_nodes,
                        gate_failed=True,
                        stage_retry_count=stage_retry_count,
                        config=self.config,
                        critic_failed=critic_failed,
                        failure_history=failure_history_in_stage,
                        rollback_scope=rollback_scope,
                        has_dynamic_planner=stage.dynamic_planner is not None,
                    )
                    recorder.record_decision(
                        "Recovery decision",
                        {
                            "action": decision.action.value,
                            "reason": decision.reason,
                            "failed_node_ids": [nid for nid, _ in failed_nodes],
                        },
                        record=DecisionRecord(
                            decision_type=DecisionType.ADAPTATION,
                            options_considered=[
                                OptionConsidered(option_id="RETRY_STAGE", description="Retry stage"),
                                OptionConsidered(option_id="REPLAN", description="Replan stage"),
                                OptionConsidered(option_id="ESCALATE", description="Escalate to user"),
                                OptionConsidered(option_id="FAIL", description="Fail run"),
                            ],
                            selected_option=SelectedOption(
                                option_id=decision.action.value,
                                selection_rationale=decision.reason,
                                decision_authority=DecisionAuthority.COMPONENT,
                                expected_outcome="Recovery action applied",
                            ),
                        ),
                    )
                    recovery_payload: dict[str, Any] = {
                        "action": decision.action.value,
                        "reason": decision.reason,
                        "failed_nodes": [nid for nid, _ in failed_nodes],
                        "stage_retry_count": stage_retry_count,
                    }
                    if decision.rollback_scope is not None:
                        rs = decision.rollback_scope
                        recovery_payload["rollback_scope_type"] = rs.scope_type.value
                        recovery_payload["rollback_node_ids"] = list(rs.node_ids)
                        recovery_payload["preservation_node_ids"] = list(
                            rs.preservation_node_ids
                        )
                    recorder.record("recovery_decision", recovery_payload)

                    if decision.action == RecoveryAction.FAIL:
                        state.transition_to(
                            RunState.FAILED,
                            f"{runbook_id}: quality gate failed at stage {stage.stage_id}",
                        )
                        self.index_store.update_run_state(state.trace_id, "FAILED")
                        state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
                        self._write_phase_checkpoint(
                            state, recorder, iteration, "final",
                            extra={"runbook_id": runbook_id, "metadata": metadata},
                        )
                        if results:
                            self._finalize_run(
                                state,
                                recorder,
                                results,
                                runbook_id,
                                rollback_node_ids=(
                                    decision.rollback_scope.node_ids
                                    if decision.rollback_scope else None
                                ),
                            )
                        return

                    if decision.action == RecoveryAction.ESCALATE:
                        state.transition_to(RunState.PAUSED, decision.reason)
                        self.index_store.update_run_state(state.trace_id, "PAUSED")
                        esc_ctx = decision.escalation_context or {}
                        self._write_phase_checkpoint(
                            state, recorder, iteration, "paused",
                            extra={
                                "runbook_id": runbook_id,
                                "metadata": metadata,
                                "stage_id": stage.stage_id,
                                "stage_index": stage_index,
                                "escalation_reason": decision.reason,
                                "failed_node_ids": esc_ctx.get("failed_node_ids", []),
                                "stage_retry_count": stage_retry_count,
                            },
                        )
                        if results:
                            self._finalize_run(
                                state,
                                recorder,
                                results,
                                runbook_id,
                                rollback_node_ids=(
                                    decision.rollback_scope.node_ids
                                    if decision.rollback_scope else None
                                ),
                            )
                        recorder.record_decision(
                            "Escalated (recovery)",
                            {"reason": decision.reason, "escalation_context": esc_ctx},
                            record=DecisionRecord(
                                decision_type=DecisionType.ESCALATION,
                                selected_option=SelectedOption(
                                    option_id="ESCALATE",
                                    selection_rationale=decision.reason,
                                    decision_authority=DecisionAuthority.COMPONENT,
                                    expected_outcome="User resumes or revises",
                                ),
                                outcome_correlation=OutcomeCorrelation(
                                    actual_outcome="escalation",
                                    correlation_timestamp=datetime.now(timezone.utc).isoformat(),
                                    quality_assessment="failure",
                                ),
                            ),
                        )
                        return

                    if decision.action == RecoveryAction.REPLAN:
                        if stage.dynamic_planner is not None:
                            planner_outcome = self._plan_dynamic_graph(
                                state=state,
                                recorder=recorder,
                                replay_provider=replay_provider,
                                runbook_id=runbook_id,
                                stage_id=stage.stage_id,
                                metadata=metadata,
                                spec=stage.dynamic_planner,
                                stage_default_model_id=getattr(
                                    stage, "default_model_id", None
                                ),
                            )
                            if isinstance(planner_outcome, PlannerEscalation):
                                self._pause_for_planner_escalation(
                                    state=state,
                                    recorder=recorder,
                                    iteration=iteration,
                                    runbook_id=runbook_id,
                                    metadata=metadata,
                                    stage_id=stage.stage_id,
                                    stage_index=stage_index,
                                    escalation=planner_outcome,
                                )
                                return
                            graph = planner_outcome.action_graph
                            state.intention.plan_id = graph.metadata.plan_id  # type: ignore[union-attr]
                            results = {}
                            skip_execute = False
                            skip_control = False
                            recorder.record("replan", {
                                "stage_id": stage.stage_id,
                                "reason": decision.reason,
                            })
                            continue
                        # No dynamic planner: treat REPLAN as ESCALATE
                        state.transition_to(RunState.PAUSED, decision.reason)
                        self.index_store.update_run_state(state.trace_id, "PAUSED")
                        esc_ctx = decision.escalation_context or {
                            "failed_node_ids": [nid for nid, _ in failed_nodes],
                            "replan_recommended": True,
                        }
                        self._write_phase_checkpoint(
                            state, recorder, iteration, "paused",
                            extra={
                                "runbook_id": runbook_id,
                                "metadata": metadata,
                                "stage_id": stage.stage_id,
                                "stage_index": stage_index,
                                "escalation_reason": decision.reason,
                                "failed_node_ids": esc_ctx.get("failed_node_ids", []),
                                "stage_retry_count": stage_retry_count,
                            },
                        )
                        if results:
                            self._finalize_run(
                                state,
                                recorder,
                                results,
                                runbook_id,
                                rollback_node_ids=(
                                    decision.rollback_scope.node_ids
                                    if decision.rollback_scope else None
                                ),
                            )
                        recorder.record_decision(
                            "Escalated (recovery, replan recommended)",
                            {"reason": decision.reason, "escalation_context": esc_ctx},
                            record=DecisionRecord(
                                decision_type=DecisionType.ESCALATION,
                                selected_option=SelectedOption(
                                    option_id="ESCALATE",
                                    selection_rationale=decision.reason,
                                    decision_authority=DecisionAuthority.COMPONENT,
                                    expected_outcome="User resumes or replans",
                                ),
                                outcome_correlation=OutcomeCorrelation(
                                    actual_outcome="escalation",
                                    correlation_timestamp=datetime.now(timezone.utc).isoformat(),
                                    quality_assessment="failure",
                                ),
                            ),
                        )
                        return

                    # RETRY_STAGE
                    failure_history_in_stage.append([nid for nid, _ in failed_nodes])
                    stage_retry_count += 1
                    skip_execute = False
                    skip_control = False

            # All stages passed
            state.transition_to(RunState.COMPLETED, f"{runbook_id}: all stages passed")
            self.index_store.update_run_state(state.trace_id, "COMPLETED")
            state.progress = 1.0
            state.intention.phase = IntentionPhase.DONE  # type: ignore[union-attr]
            self._write_phase_checkpoint(
                state, recorder, len(stages), "final",
                extra={"runbook_id": runbook_id, "metadata": metadata},
            )
            if results:
                self._finalize_run(state, recorder, results, runbook_id)

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
        seen_clarification_request = False
        seen_clarification_response = False
        seen_clarification_evidence = False
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
                    if req_aid and not seen_clarification_request:
                        metadata["clarification_request_artifact_id"] = req_aid
                        seen_clarification_request = True
                    if resp_aid and not seen_clarification_response:
                        metadata["clarification_response_artifact_id"] = resp_aid
                        seen_clarification_response = True

                    patch_ops = self._normalise_patch_ops(control_payload.get("patch"))
                    if not patch_ops:
                        answers = control_payload.get("answers", {})
                        if isinstance(answers, dict):
                            patch_ops = self._legacy_answers_to_patch(answers)
                    if patch_ops:
                        try:
                            metadata = apply_patch(metadata, patch_ops)
                        except StatePatchError as exc:
                            logger.warning(
                                "Skipping invalid revise patch during metadata inference: %s",
                                exc,
                            )
            if description == "Escalation requested":
                req_aid = str(payload.get("clarification_request_artifact_id", "")).strip()
                if req_aid and not seen_clarification_request:
                    metadata["clarification_request_artifact_id"] = req_aid
                    seen_clarification_request = True
                evidence_ids = payload.get("evidence_artifact_ids", [])
                if isinstance(evidence_ids, list) and not seen_clarification_evidence:
                    metadata["clarification_evidence_artifact_ids"] = [
                        str(aid) for aid in evidence_ids if str(aid).strip()
                    ]
                    seen_clarification_evidence = True
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

    def _build_clarification_context_inputs(
        self, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        request_artifact_id = str(
            metadata.get("clarification_request_artifact_id", "")
        ).strip()
        response_artifact_id = str(
            metadata.get("clarification_response_artifact_id", "")
        ).strip()
        raw_evidence_ids = metadata.get("clarification_evidence_artifact_ids", [])
        evidence_artifact_ids: list[str] = []
        if isinstance(raw_evidence_ids, list):
            evidence_artifact_ids = [
                str(aid).strip() for aid in raw_evidence_ids if str(aid).strip()
            ]
        if not request_artifact_id and not response_artifact_id and not evidence_artifact_ids:
            return {}

        envelope: dict[str, Any] = {
            "version": "bug4.v1",
            "request_artifact_id": request_artifact_id,
            "response_artifact_id": response_artifact_id,
            "evidence_artifact_ids": evidence_artifact_ids,
            "request": (
                self._load_artifact_json_or_empty(request_artifact_id)
                if request_artifact_id
                else {}
            ),
            "response": (
                self._load_artifact_json_or_empty(response_artifact_id)
                if response_artifact_id
                else {}
            ),
        }
        return {
            "clarification_request_artifact_id": request_artifact_id,
            "clarification_response_artifact_id": response_artifact_id,
            "clarification_evidence_artifact_ids": evidence_artifact_ids,
            "clarification_context_envelope": envelope,
        }

    @staticmethod
    def _normalise_patch_ops(raw_patch: Any) -> list[dict[str, Any]]:
        """Validate and canonicalize patch operations for revise payload."""
        if not isinstance(raw_patch, list):
            return []
        normalized: list[dict[str, Any]] = []
        for raw_op in raw_patch:
            try:
                op = (
                    raw_op
                    if isinstance(raw_op, PatchOperation)
                    else PatchOperation.model_validate(raw_op)
                )
            except Exception:
                continue
            entry: dict[str, Any] = {"op": op.op, "path": op.path}
            if op.op in {"add", "replace"}:
                entry["value"] = op.value
            normalized.append(entry)
        return normalized

    def _convert_nl_answer_to_patch(
        self,
        *,
        state: AgentState,
        recorder: TraceRecorder,
        answer_text: str,
        clarification_request_artifact_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert NL clarification answer into RFC6902 patch operations."""
        context_payload = self._load_artifact_json_or_empty(clarification_request_artifact_id)
        slots = self._extract_clarification_slots(context_payload)
        if not slots:
            return [], {
                "status": "skipped_no_slots",
                "clarification_request_artifact_id": clarification_request_artifact_id,
            }

        prompt = (
            "Преобразуй ответ пользователя в JSON Patch для метаданных запуска.\n"
            "Используй только перечисленные слоты, не добавляй новые пути.\n"
            "Верни строгий JSON по схеме.\n\n"
            f"Answer text: {answer_text}\n"
            f"Clarification slots: {json.dumps(slots, ensure_ascii=False)}\n"
        )
        node_id = "control_nl_to_patch"
        graph = ActionGraph(
            metadata=GraphMetadata(
                plan_id=f"control-nl-patch-{state.execution_id[:12]}",
                description="Convert NL clarification answer to patch",
            ),
            nodes=[
                GraphNode(
                    node_id=node_id,
                    node_type="model",
                    parameters={
                        "system_prompt": (
                            "You map user clarification text to state patch operations. "
                            "If data is ambiguous or insufficient, set needs_clarification=true."
                        ),
                        "json_schema": self._nl_patch_conversion_schema(),
                        "timeout_seconds": 60,
                        "max_retries": 1,
                    },
                )
            ],
            edges=[],
        )
        try:
            raw_outputs = self._execute(
                state,
                graph,
                recorder,
                initial_inputs_override={"prompt": prompt},
            )
            outputs = raw_outputs.results if isinstance(raw_outputs, ExecutionOutcome) else raw_outputs
            parsed = self._parse_json_object_output(outputs.get(node_id))
            if parsed is None:
                return [], {
                    "status": "invalid_json",
                    "clarification_request_artifact_id": clarification_request_artifact_id,
                }

            needs_clarification = bool(parsed.get("needs_clarification", False))
            confidence_raw = parsed.get("confidence", 0.0)
            confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
            rationale = str(parsed.get("rationale", "")).strip()
            if needs_clarification or confidence < self.config.runtime.nl_patch_min_confidence:
                return [], {
                    "status": "needs_clarification",
                    "confidence": confidence,
                    "rationale": rationale,
                    "clarification_request_artifact_id": clarification_request_artifact_id,
                }

            patch_ops = self._normalise_patch_ops(parsed.get("patch"))
            if not patch_ops:
                return [], {
                    "status": "empty_patch",
                    "confidence": confidence,
                    "rationale": rationale,
                    "clarification_request_artifact_id": clarification_request_artifact_id,
                }
            return patch_ops, {
                "status": "ok",
                "confidence": confidence,
                "rationale": rationale,
                "op_count": len(patch_ops),
                "clarification_request_artifact_id": clarification_request_artifact_id,
            }
        except Exception as exc:
            logger.warning("NL->patch conversion failed: %s", exc)
            return [], {
                "status": "error",
                "reason": str(exc),
                "clarification_request_artifact_id": clarification_request_artifact_id,
            }

    def _load_artifact_json_or_empty(self, artifact_id_value: str) -> dict[str, Any]:
        aid = str(artifact_id_value).strip()
        if not aid:
            return {}
        try:
            raw = self.blob_store.get(aid)
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _extract_clarification_slots(cls, clarification_payload: dict[str, Any]) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        questions = clarification_payload.get("questions", [])
        if isinstance(questions, list):
            for item in questions:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", "")).strip()
                if not key:
                    continue
                path = str(item.get("path", "")).strip() or f"/{cls._pointer_escape_token(key)}"
                expected_schema = item.get("expected_schema")
                if not isinstance(expected_schema, dict):
                    expected_schema = cls._legacy_expected_schema(key)
                slots.append(
                    {
                        "key": key,
                        "path": path,
                        "expected_schema": expected_schema,
                        "required": bool(item.get("required", True)),
                    }
                )
        if slots:
            return slots

        missing_fields = clarification_payload.get("missing_fields", [])
        if isinstance(missing_fields, list):
            for item in missing_fields:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("field", "")).strip()
                if not key:
                    continue
                slots.append(
                    {
                        "key": key,
                        "path": f"/{cls._pointer_escape_token(key)}",
                        "expected_schema": cls._legacy_expected_schema(key),
                        "required": bool(item.get("critical", True)),
                    }
                )
        return slots

    @staticmethod
    def _legacy_expected_schema(key: str) -> dict[str, Any]:
        low = key.strip().lower()
        if low.endswith("s") or "paths" in low or "urls" in low:
            return {"type": "array", "items": {"type": "string"}}
        return {"type": "string"}

    @staticmethod
    def _nl_patch_conversion_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["patch", "needs_clarification", "confidence", "rationale"],
            "properties": {
                "patch": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "path"],
                        "properties": {
                            "op": {"type": "string", "enum": ["add", "replace", "remove"]},
                            "path": {"type": "string", "pattern": "^/"},
                            "value": {},
                        },
                    },
                },
                "needs_clarification": {"type": "boolean"},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
            },
        }

    @staticmethod
    def _parse_json_object_output(output: NodeOutput | None) -> dict[str, Any] | None:
        if output is None or output.status != "COMPLETED":
            return None
        parsed = output.outputs.get("parsed")
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                value = json.loads(parsed)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
        content = output.outputs.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def _legacy_answers_to_patch(cls, answers: dict[str, Any]) -> list[dict[str, Any]]:
        """Bridge legacy answers payload to patch operations."""
        patch: list[dict[str, Any]] = []
        for raw_key, raw_value in answers.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key.strip()
            if not key:
                continue
            value = cls._normalize_legacy_answer_value(key, raw_value)
            if value is None:
                continue
            escaped_key = cls._pointer_escape_token(key)
            patch.append({"op": "add", "path": f"/{escaped_key}", "value": value})
            # Keep compatibility for consumers that read both url and urls.
            if key == "url" and isinstance(value, str):
                patch.append({"op": "add", "path": "/urls", "value": [value]})
        return patch

    @staticmethod
    def _normalize_legacy_answer_value(key: str, raw_value: Any) -> Any:
        if key == "doc_paths" and isinstance(raw_value, str):
            parts = [p.strip() for p in raw_value.split(",") if p.strip()]
            return parts or None
        if isinstance(raw_value, str):
            value = raw_value.strip()
            return value or None
        if isinstance(raw_value, list):
            normalized: list[Any] = []
            for item in raw_value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        normalized.append(text)
                    continue
                if item is not None:
                    normalized.append(item)
            return normalized or None
        return raw_value

    @staticmethod
    def _pointer_escape_token(token: str) -> str:
        return token.replace("~", "~0").replace("/", "~1")

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
                    snapshot["critic_suggestions"] = parsed.get("suggestions", [])
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
                    snapshot["critic_suggestions"] = d.get("suggestions", [])
            else:
                snapshot["critic_verdict"] = "UNCERTAIN"
                snapshot["critic_confidence"] = 0.0
                snapshot["critic_evidence"] = []
                snapshot["critic_gaps"] = ["critic node did not complete"]
                snapshot["critic_suggestions"] = []
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
        stage_default_model_id: str | None = None,
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
            decision_type=DecisionType.PLANNING,
        )

        def execute_graph(
            graph: ActionGraph,
            initial_inputs: dict[str, Any],
            suppress_node_events: bool,
        ) -> dict[str, NodeOutput]:
            raw = self._execute(
                state,
                graph,
                recorder,
                replay_provider=replay_provider,
                initial_inputs_override=initial_inputs,
                suppress_node_events=(
                    suppress_node_events and self._trace_event_listener is None
                ),
                stage_default_model_id=stage_default_model_id,
            )
            if isinstance(raw, ExecutionOutcome):
                return raw.results
            return raw

        backend = get_planner_backend(spec.backend_name)
        result = backend.plan(
            request=request,
            execute_graph=execute_graph,
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
                decision_type=DecisionType.ESCALATION,
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
            decision_type=DecisionType.PLANNING,
        )
        return final_result

    def _pause_for_planner_escalation(
        self,
        *,
        state: AgentState,
        recorder: TraceRecorder,
        iteration: int,
        runbook_id: str,
        metadata: dict[str, Any],
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
            decision_type=DecisionType.ESCALATION,
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
            decision_type=DecisionType.ESCALATION,
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
                "metadata": metadata,
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
        initial_node_results: dict[str, NodeOutput] | None = None,
        suppress_node_events: bool = False,
        stage_default_model_id: str | None = None,
    ) -> Union[dict[str, NodeOutput], ExecutionOutcome]:
        """Build node registry and execute the DAG.

        When interrupt_check is used (v1: always), returns ExecutionOutcome;
        otherwise returns dict of node results. Caller must handle both.
        initial_node_results: optional pre-filled results for resume at exact pause point.
        """
        registry = self._build_node_registry(
            graph, stage_default_model_id=stage_default_model_id
        )

        # Enable recording for replay
        recordings: dict[str, list] = {}
        if replay_provider is None:
            for nid, node in registry.items():
                if hasattr(node, "enable_recording"):
                    recordings[nid] = node.enable_recording()
        else:
            report = replay_provider.inject(registry, strict=True)
            recorder.record_decision(
                "Replay responses injected",
                report,
                decision_type=DecisionType.EXECUTION,
            )

        def trace_cb(kind: str, payload: dict[str, Any]) -> None:
            if suppress_node_events and kind in {"node_start", "node_end"}:
                return
            recorder.record(kind, payload)

        def interrupt_check() -> InterruptRequest | None:
            return self._interrupt_requests.get(state.trace_id)

        executor = DAGExecutor(
            registry,
            max_parallel=self.config.runtime.max_parallel_nodes,
            execution_id=state.execution_id,
            trace_id=state.trace_id,
            random_seed=self.config.determinism.default_random_seed,
            trace_callback=trace_cb,
            interrupt_check=interrupt_check,
            max_node_retries=self.config.recovery.max_node_retries,
            retry_backoff_base_seconds=self.config.recovery.retry_backoff_base_seconds,
            retry_count_upgrade_threshold=self.config.recovery.retry_count_upgrade_threshold,
        )

        # B11: determinism audit — record nodes that use seed and that seed is set
        nodes_using_seed = [
            nid
            for nid, node in registry.items()
            if node.get_determinism_contract().uses_seed
        ]
        recorder.record_determinism_audit(
            nodes_using_seed=nodes_using_seed,
            seed_value=self.config.determinism.default_random_seed,
            seed_present=True,
        )

        objective = state.intention.objective  # type: ignore[union-attr]
        raw_constraints = state.intention.constraints  # type: ignore[union-attr]
        # Internal constraints are allowed for planning/control, but should not
        # be shown to LLM prompts by default.
        constraints = [
            c for c in raw_constraints
            if not (isinstance(c, str) and c.startswith("__NEURONIUM_INTERNAL_"))
        ]

        base_inputs: dict[str, Any] = {
            "objective": objective,
            "constraints": constraints,
        }
        if initial_inputs_override:
            base_inputs.update(initial_inputs_override)

        results = executor.execute(
            graph,
            initial_inputs=base_inputs,
            initial_results=initial_node_results,
        )

        # Record replay data as trace events
        for nid, recs in recordings.items():
            if recs:
                recorder.record("replay_data", {
                    "node_id": nid,
                    "recorded_responses": recs,
                })

        return results

    def _build_node_registry(
        self,
        graph: ActionGraph,
        *,
        stage_default_model_id: str | None = None,
    ) -> dict[str, BaseNode]:
        """Instantiate concrete node implementations for each graph node."""
        registry: dict[str, BaseNode] = {}
        for gn in graph.nodes:
            if gn.node_type == "model":
                # B13: resolve model from catalog (or default catalog) with fallback
                effective_catalog = (
                    self.config.model_catalog
                    if self.config.model_catalog is not None
                    else get_default_catalog(self.config.llm)
                )
                model_id = None
                if isinstance(gn.parameters, dict):
                    model_id = gn.parameters.get("model_id")
                model_id = model_id or stage_default_model_id
                resolved = resolve_model_for_node(
                    effective_catalog,
                    self.config.llm,
                    model_id,
                )
                # Allow planner/internal graphs to override timeouts per node
                timeout = self.config.llm.timeout_seconds
                max_retries = self.config.llm.max_retries
                if isinstance(gn.parameters, dict):
                    if gn.parameters.get("timeout_seconds") is not None:
                        try:
                            timeout = int(gn.parameters["timeout_seconds"])
                        except Exception:
                            timeout = self.config.llm.timeout_seconds
                    if gn.parameters.get("max_retries") is not None:
                        try:
                            max_retries = int(gn.parameters["max_retries"])
                        except Exception:
                            max_retries = self.config.llm.max_retries
                registry[gn.node_id] = ModelNode(
                    node_id=gn.node_id,
                    parameters=gn.parameters,
                    model=resolved.model,
                    provider=resolved.provider,
                    api_key_env=resolved.api_key_env,
                    base_url=resolved.base_url,
                    structured_output=self.config.llm.structured_output,
                    temperature=self.config.determinism.llm_temperature,
                    timeout=timeout,
                    max_retries=max_retries,
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
                deterministic = True
                if isinstance(gn.parameters, dict) and gn.parameters.get("deterministic") is False:
                    deterministic = False
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
                    deterministic=deterministic,
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
        # B11: reject declared non-deterministic nodes when strict
        if self.config.determinism.strict:
            allowlist = set(self.config.determinism.mcp_allow_non_deterministic_tool_ids)
            for nid, node in registry.items():
                contract = node.get_determinism_contract()
                if contract.declared_non_deterministic and nid not in allowlist:
                    raise ConfigError(
                        f"Determinism strict: node {nid!r} is declared non-deterministic; "
                        "add to determinism.mcp_allow_non_deterministic_tool_ids or set strict=false"
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

    def _finalize_run(
        self,
        state: AgentState,
        recorder: TraceRecorder,
        results: dict[str, NodeOutput],
        runbook_id: str,
        *,
        rollback_node_ids: set[str] | None = None,
    ) -> None:
        """Single point of run finalization: persist artifacts + render + local index.

        If results is empty, does nothing. Otherwise calls _persist_artifacts
        and _persist_local_rendered_artifact (A3).
        """
        if not results:
            return
        self._persist_artifacts(
            state, results, recorder, rollback_node_ids=rollback_node_ids
        )
        self._persist_local_rendered_artifact(
            state=state,
            recorder=recorder,
            results=results,
            runbook_id=runbook_id,
        )

    def _persist_artifacts(
        self,
        state: AgentState,
        results: dict[str, NodeOutput],
        recorder: TraceRecorder,
        *,
        rollback_node_ids: set[str] | None = None,
    ) -> None:
        """Persist node outputs as immutable artifacts.

        If rollback_node_ids is set, artifacts produced by those nodes are
        marked deprecated for lineage (B1 Part 2 §3.4.1).
        """
        deprecated_aids: list[str] = []
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
            if rollback_node_ids and nid in rollback_node_ids:
                deprecated_aids.append(aid)
        if deprecated_aids:
            self.index_store.mark_artifacts_deprecated(deprecated_aids, "rollback")

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
            decision_type=DecisionType.EXECUTION,
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
            decision_type=DecisionType.EXECUTION,
        )
