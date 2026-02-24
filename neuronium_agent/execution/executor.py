"""Deterministic DAG executor (IBS §5.4).

Executes nodes in topological order.  Independent nodes may run in
parallel (up to ``max_parallel``).  Results are committed in a
deterministic order (tie-breaker: priority, then node_id).
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Union

from neuronium_agent.execution.outcome import ExecutionOutcome
from neuronium_agent.nodes.base import BaseNode, NodeContext, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph
from neuronium_agent.nodes.decision_node import BRANCH_OUTPUT_KEY
from neuronium_agent.recovery.classifier import classify_failure
from neuronium_agent.types import InterruptRequest

logger = logging.getLogger(__name__)

_MAX_RETRY_DELAY_SECONDS = 60.0


class DAGExecutor:
    """Execute an :class:`ActionGraph` using concrete node implementations.

    Parameters
    ----------
    node_registry:
        Mapping of ``node_id`` → :class:`BaseNode` instance.
    max_parallel:
        Maximum concurrent node executions.
    trace_callback:
        Optional callable invoked with (event_kind, payload) for every
        significant execution event.
    interrupt_check:
        Optional callable called after each batch; if it returns an
        :class:`InterruptRequest`, execution stops and an
        :class:`ExecutionOutcome` with partial results and pending is returned.
    """

    def __init__(
        self,
        node_registry: dict[str, BaseNode],
        *,
        max_parallel: int = 4,
        execution_id: str | None = None,
        trace_id: str | None = None,
        random_seed: int = 0,
        trace_callback: Any | None = None,
        interrupt_check: Callable[[], InterruptRequest | None] | None = None,
        max_node_retries: int = 3,
        retry_backoff_base_seconds: float = 1.0,
        retry_count_upgrade_threshold: int = 2,
    ) -> None:
        self._registry = node_registry
        self._max_parallel = max_parallel
        self._execution_id = execution_id or uuid.uuid4().hex
        self._trace_id = trace_id or uuid.uuid4().hex
        self._seed = random_seed
        self._trace_cb = trace_callback
        self._interrupt_check = interrupt_check
        self._max_node_retries = max_node_retries
        self._retry_backoff_base = retry_backoff_base_seconds
        self._retry_count_upgrade_threshold = retry_count_upgrade_threshold

        # Stores outputs keyed by node_id
        self.results: dict[str, NodeOutput] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(
        self,
        graph: ActionGraph,
        *,
        initial_inputs: dict[str, Any] | None = None,
        initial_results: dict[str, NodeOutput] | None = None,
    ) -> Union[dict[str, NodeOutput], ExecutionOutcome]:
        """Run the full DAG and return results or an execution outcome.

        Respects conditional branches (B3): when a decision node completes,
        its output branch value is recorded; nodes in unselected branches
        are never added to ready and are skipped.

        Parameters
        ----------
        initial_results
            Optional pre-filled node results (e.g. from a mid-execution
            checkpoint). When provided, only nodes not in this dict are
            executed; completed and pending are derived from it (resume
            at exact pause point per spec §6.4.2).

        Return type
        ----------
        If ``interrupt_check`` was not passed to the constructor: returns
        ``dict[str, NodeOutput]`` (all node results), same as before.
        If ``interrupt_check`` was passed: returns :class:`ExecutionOutcome`
        with fields ``results`` (completed node outputs), ``pending`` (node
        ids not yet run; empty on normal completion), and ``interrupted``
        (set only when the callback returned an :class:`InterruptRequest`).

        Interrupt contract
        ------------------
        The optional ``interrupt_check`` callable is invoked after each batch.
        If it returns an :class:`InterruptRequest`, the loop exits without
        scheduling further nodes; the current batch is always completed first
        (graceful at batch boundary per spec §9.1.2).
        """
        initial_inputs = initial_inputs or {}
        order = graph.topological_order()
        preds = graph.predecessors()
        nmap = graph.node_map()

        # decision_node_id -> selected branch value (matches ConditionalBranch.branch_label)
        selected_branches: dict[str, str] = {}

        completed: set[str] = set()
        pending: list[str]
        if initial_results:
            self.results = dict(initial_results)
            completed = set(initial_results)
            pending = [nid for nid in order if nid not in initial_results]
            # Restore selected_branches from completed decision nodes so
            # nodes in unselected branches stay pruned during resume.
            for nid, output in initial_results.items():
                gn = nmap.get(nid)
                if gn and gn.node_type == "decision" and output.status == "COMPLETED":
                    branch_key = (gn.parameters or {}).get(
                        "branch_output_key", BRANCH_OUTPUT_KEY
                    )
                    branch_value = output.outputs.get(branch_key)
                    if branch_value is not None:
                        selected_branches[nid] = str(branch_value)
        else:
            pending = list(order)

        while pending:
            # Ready: all predecessors completed and not in an unselected branch
            preds_satisfied = [
                nid
                for nid in pending
                if all(p in completed for p in preds.get(nid, []))
            ]
            ready = [
                nid
                for nid in preds_satisfied
                if not _is_in_unselected_branch(nid, graph, selected_branches)
            ]
            # Skipped: preds satisfied but in unselected branch — remove from pending
            # so the loop terminates (they are never executed, not added to results)
            skipped = [nid for nid in preds_satisfied if nid not in ready]
            for nid in skipped:
                pending.remove(nid)
                completed.add(nid)

            if not ready:
                if pending:
                    raise RuntimeError(
                        "Deadlock: no ready nodes but pending remains"
                    )
                break

            ready.sort(key=lambda nid: (nmap[nid].priority, nid))

            batch = ready[: self._max_parallel]
            batch_results = self._execute_batch(
                batch,
                graph,
                initial_inputs=initial_inputs,
            )

            for nid, output in batch_results:
                self.results[nid] = output
                completed.add(nid)
                pending.remove(nid)
                # Record branch selection for decision nodes (B3)
                gn = nmap.get(nid)
                if gn and gn.node_type == "decision" and output.status == "COMPLETED":
                    branch_key = (gn.parameters or {}).get(
                        "branch_output_key", BRANCH_OUTPUT_KEY
                    )
                    branch_value = output.outputs.get(branch_key)
                    if branch_value is not None:
                        selected_branches[nid] = str(branch_value)
                        self._emit("decision_branch_selected", {
                            "node_id": nid,
                            "branch_value": branch_value,
                            "branch_label": branch_value,
                        })

            if self._interrupt_check is not None:
                request = self._interrupt_check()
                if request is not None:
                    return ExecutionOutcome(
                        results=dict(self.results),
                        pending=list(pending),
                        interrupted=request,
                    )

        if self._interrupt_check is not None:
            return ExecutionOutcome(
                results=dict(self.results),
                pending=[],
                interrupted=None,
            )
        return self.results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_batch(
        self,
        node_ids: list[str],
        graph: ActionGraph,
        *,
        initial_inputs: dict[str, Any],
    ) -> list[tuple[str, NodeOutput]]:
        """Execute a batch of independent nodes, possibly in parallel."""
        if len(node_ids) == 1:
            nid = node_ids[0]
            out = self._execute_single(
                nid,
                graph,
                initial_inputs=initial_inputs,
            )
            return [(nid, out)]

        results: list[tuple[str, NodeOutput]] = []
        with ThreadPoolExecutor(max_workers=self._max_parallel) as pool:
            futures = {
                pool.submit(
                    self._execute_single,
                    nid,
                    graph,
                    initial_inputs=initial_inputs,
                ): nid
                for nid in node_ids
            }
            for fut in as_completed(futures):
                nid = futures[fut]
                results.append((nid, fut.result()))

        # Sort results deterministically before returning
        nmap = graph.node_map()
        results.sort(key=lambda t: (nmap[t[0]].priority, t[0]))
        return results

    def _execute_single(
        self,
        node_id: str,
        graph: ActionGraph,
        *,
        initial_inputs: dict[str, Any],
    ) -> NodeOutput:
        """Execute a single node with retry for TRANSIENT failures, emitting trace events."""
        node_impl = self._registry.get(node_id)
        nmap = graph.node_map()
        graph_node = nmap.get(node_id)

        if node_impl is None:
            logger.error("No implementation for node %s", node_id)
            out = NodeOutput(
                status="FAILED",
                error=f"No implementation registered for node {node_id}",
            )
            fc = classify_failure(
                node_id, "unknown", "FAILED", out.error, 0,
                retry_count_upgrade_threshold=self._retry_count_upgrade_threshold,
            )
            return out.model_copy(update={"failure_class": fc})

        preds = graph.predecessors()
        inputs = self._gather_inputs(node_id, graph, graph_node, initial_inputs)
        node_ref = (
            f"{self._execution_id}:{graph.metadata.plan_id}"
            f"/execute/{node_id}"
        )
        node_type = graph_node.node_type if graph_node else "unknown"

        attempt = 0
        while True:
            ctx = NodeContext(
                execution_id=self._execution_id,
                trace_id=self._trace_id,
                retry_count=attempt,
                random_seed=self._seed,
            )
            node_input = NodeInput(
                inputs=inputs,
                parameters=graph_node.parameters if graph_node else {},
                context=ctx,
            )

            self._emit("node_start", {
                "node_id": node_id,
                "node_ref": node_ref,
                "node_type": node_type,
                "inputs": inputs,
                "parameters": graph_node.parameters if graph_node else {},
            })

            started = time.perf_counter()
            try:
                output = node_impl.execute(node_input)
            except Exception as exc:
                output = NodeOutput(status="FAILED", error=str(exc))

            elapsed_ms = int((time.perf_counter() - started) * 1000)

            if output.status == "COMPLETED":
                self._emit("node_end", {
                    "node_id": node_id,
                    "node_ref": node_ref,
                    "status": output.status,
                    "outputs": output.outputs,
                    "error": output.error,
                    "quality_signals": output.quality_signals.model_dump(mode="json"),
                    "elapsed_ms": elapsed_ms,
                })
                return output

            # Classify failure and attach to output
            fc = classify_failure(
                node_id,
                node_type,
                output.status,
                output.error,
                attempt,
                retry_count_upgrade_threshold=self._retry_count_upgrade_threshold,
            )
            output = output.model_copy(update={"failure_class": fc})

            if fc.kind == "CRITICAL":
                self._emit("node_failure_critical", {
                    "node_id": node_id,
                    "node_ref": node_ref,
                    "failure_class": fc.kind,
                    "message": fc.message,
                })

            self._emit("node_end", {
                "node_id": node_id,
                "node_ref": node_ref,
                "status": output.status,
                "outputs": output.outputs,
                "error": output.error,
                "quality_signals": output.quality_signals.model_dump(mode="json"),
                "elapsed_ms": elapsed_ms,
            })

            if fc.kind == "CRITICAL" or not fc.retryable:
                return output

            if attempt >= self._max_node_retries:
                return output

            delay = min(
                self._retry_backoff_base * (2**attempt),
                _MAX_RETRY_DELAY_SECONDS,
            )
            self._emit("node_retry", {
                "node_id": node_id,
                "node_ref": node_ref,
                "retry_count": attempt + 1,
                "reason": fc.message or "Transient failure",
            })
            time.sleep(delay)
            attempt += 1

    def _gather_inputs(
        self,
        node_id: str,
        graph: ActionGraph,
        graph_node: Any,
        initial_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build inputs dict for a node from predecessors and initial_inputs."""
        preds = graph.predecessors()
        inputs: dict[str, Any] = {}
        for pred_id in preds.get(node_id, []):
            pred_output = self.results.get(pred_id)
            if pred_output:
                inputs.update(pred_output.outputs)
        for k, v in initial_inputs.items():
            inputs.setdefault(k, v)

        if graph_node and graph_node.node_type == "model" and "prompt" not in inputs:
            if graph_node.parameters.get("json_schema"):
                inputs["prompt"] = _build_critic_prompt(
                    inputs, parameters=graph_node.parameters
                )
            elif inputs.get("previous_code"):
                inputs["prompt"] = _build_fix_prompt(inputs)
            else:
                inputs["prompt"] = _build_default_prompt(inputs)

        if graph_node and graph_node.node_type == "code" and "code" not in inputs:
            if "content" in inputs:
                inputs["code"] = inputs["content"]

        return inputs

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._trace_cb:
            self._trace_cb(kind, payload)


# ---------------------------------------------------------------------------
# Module-level helpers (B3 conditional branches)
# ---------------------------------------------------------------------------

def _is_in_unselected_branch(
    node_id: str,
    graph: ActionGraph,
    selected_branches: dict[str, str],
) -> bool:
    """True if node is in a conditional branch that was not selected.

    A node is excluded from ready when it belongs to some ConditionalBranch
    whose decision node has already completed and selected a different
    branch_label.
    """
    for cb in graph.conditional_branches:
        if node_id not in cb.target_node_ids:
            continue
        sel = selected_branches.get(cb.decision_node_id)
        if sel is None:
            continue
        if sel != cb.branch_label:
            return True
    return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_critic_prompt(
    inputs: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Build a structured evaluation prompt for the LLM critic node.

    Aggregates objective, code, and execution results from predecessor
    outputs into a single prompt string.
    """
    parts: list[str] = []

    objective = inputs.get("objective", "")
    if objective:
        parts.append(f"OBJECTIVE:\n{objective}")

    code = inputs.get("content", inputs.get("previous_code", ""))
    if code:
        parts.append(f"CODE:\n{code}")

    exit_code = inputs.get("exit_code", inputs.get("previous_exit_code"))
    if exit_code is not None:
        parts.append(f"EXIT_CODE: {exit_code}")

    stdout = inputs.get("stdout", inputs.get("previous_stdout", ""))
    if stdout:
        parts.append(f"STDOUT:\n{stdout}")

    stderr = inputs.get("stderr", inputs.get("previous_stderr", ""))
    if stderr:
        parts.append(f"STDERR:\n{stderr}")

    constraints = inputs.get("constraints")
    if constraints:
        if isinstance(constraints, list):
            constraints = "\n".join(constraints)
        parts.append(f"CONSTRAINTS:\n{constraints}")

    context_kind = ""
    if parameters and isinstance(parameters.get("context_kind"), str):
        context_kind = str(parameters["context_kind"])
    if context_kind:
        parts.append(f"CONTEXT_KIND: {context_kind}")

    # Include upstream context so critic can be objective-aware for docs/web modes.
    excluded = {
        "objective",
        "constraints",
        "content",
        "previous_code",
        "exit_code",
        "previous_exit_code",
        "stdout",
        "previous_stdout",
        "stderr",
        "previous_stderr",
        "prompt",
        "tool_name",
        "tool_args",
    }
    context_keys = sorted(k for k in inputs if k not in excluded)
    if context_keys:
        parts.append("SOURCE CONTEXT:")
        for key in context_keys:
            if key == "html":
                # Keep critic prompt compact and deterministic.
                continue
            parts.append(f"{key}:\n{_format_context_value(inputs.get(key), max_len=2000)}")

    return "\n\n".join(parts) if parts else "Evaluate the execution result."


def _build_fix_prompt(inputs: dict[str, Any]) -> str:
    """Build a prompt for the fix ModelNode from iteration 1 failure context.

    B2: if ``verdict_fix`` is present, uses its gaps and suggestions in addition
    to (or instead of) ``previous_gaps``.
    """
    parts: list[str] = []

    objective = inputs.get("objective", "")
    if objective:
        parts.append(f"OBJECTIVE:\n{objective}")

    prev_code = inputs.get("previous_code", "")
    if prev_code:
        parts.append(f"PREVIOUS CODE (buggy):\n{prev_code}")

    stderr = inputs.get("previous_stderr", "")
    if stderr:
        parts.append(f"ERROR (stderr):\n{stderr}")

    stdout = inputs.get("previous_stdout", "")
    if stdout:
        parts.append(f"PREVIOUS STDOUT:\n{stdout}")

    exit_code = inputs.get("previous_exit_code")
    if exit_code is not None:
        parts.append(f"EXIT_CODE: {exit_code}")

    verdict_fix = inputs.get("verdict_fix")
    if isinstance(verdict_fix, dict):
        gaps = verdict_fix.get("gaps") or inputs.get("previous_gaps", [])
        suggestions = verdict_fix.get("suggestions") or []
    else:
        gaps = inputs.get("previous_gaps", [])
        suggestions = []

    if gaps:
        parts.append("GAPS IDENTIFIED BY CRITIC:\n" + "\n".join(f"- {g}" for g in gaps))
    if suggestions:
        parts.append("SUGGESTIONS:")
        for s in suggestions:
            if isinstance(s, dict):
                action = s.get("action", "")
                expected = s.get("expected_improvement", s.get("expectedImprovement", ""))
                parts.append(f"- {action}" + (f" (expected: {expected})" if expected else ""))
            else:
                parts.append(f"- {s}")

    parts.append(
        "Fix the code above.  Make the MINIMAL change to resolve the error.  "
        "Output ONLY the corrected Python code."
    )

    return "\n\n".join(parts)


def _build_default_prompt(inputs: dict[str, Any]) -> str:
    """Build a general-purpose prompt from objective + constraints + context.

    Keeps backward compatibility for the simplest case:
    - If only ``objective`` is present, returns it unchanged.
    B2: when ``verdict_fix`` is present (gaps/suggestions from critic), adds
    a PREVIOUS ATTEMPT FEEDBACK section.
    """
    objective = str(inputs.get("objective", "") or "")
    constraints = inputs.get("constraints")

    skip = {"prompt", "objective", "constraints", "verdict_fix"}
    extra_keys = sorted([k for k in inputs.keys() if k not in skip])

    has_constraints = bool(constraints)
    verdict_fix = inputs.get("verdict_fix")
    has_verdict_fix = isinstance(verdict_fix, dict) and verdict_fix
    if objective and not has_constraints and not extra_keys and not has_verdict_fix:
        return objective

    parts: list[str] = []
    if objective:
        parts.append(f"OBJECTIVE:\n{objective}")

    if constraints:
        if isinstance(constraints, list):
            ctext = "\n".join(str(x) for x in constraints)
        else:
            ctext = str(constraints)
        parts.append(f"CONSTRAINTS:\n{ctext}")

    if has_verdict_fix:
        parts.append("PREVIOUS ATTEMPT FEEDBACK (address these to pass the gate):")
        gaps = verdict_fix.get("gaps") or []
        if gaps:
            parts.append("Gaps identified:")
            for g in gaps:
                parts.append(f"- {g}")
        suggestions = verdict_fix.get("suggestions") or []
        if suggestions:
            parts.append("Suggestions:")
            for s in suggestions:
                if isinstance(s, dict):
                    action = s.get("action", "")
                    expected = s.get("expected_improvement", s.get("expectedImprovement", ""))
                    effort = s.get("effort_estimate", s.get("effortEstimate", ""))
                    line = f"- {action}"
                    if expected:
                        line += f" (expected: {expected})"
                    if effort:
                        line += f" [effort: {effort}]"
                    parts.append(line)
                else:
                    parts.append(f"- {s}")

    if extra_keys:
        parts.append("CONTEXT:")
        for k in extra_keys:
            parts.append(f"{k}:\n{_format_context_value(inputs.get(k))}")

    return "\n\n".join(parts).strip()


def _format_context_value(value: Any, *, max_len: int = 4000) -> str:
    """Deterministically format a value for inclusion into prompts."""
    if isinstance(value, str):
        s = value
    else:
        import json

        try:
            s = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s
