"""Deterministic DAG executor (IBS §5.4).

Executes nodes in topological order.  Independent nodes may run in
parallel (up to ``max_parallel``).  Results are committed in a
deterministic order (tie-breaker: priority, then node_id).
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from neuronium_agent.nodes.base import BaseNode, NodeContext, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._registry = node_registry
        self._max_parallel = max_parallel
        self._execution_id = execution_id or uuid.uuid4().hex
        self._trace_id = trace_id or uuid.uuid4().hex
        self._seed = random_seed
        self._trace_cb = trace_callback

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
    ) -> dict[str, NodeOutput]:
        """Run the full DAG and return ``{node_id: NodeOutput}``."""
        initial_inputs = initial_inputs or {}
        order = graph.topological_order()
        adj = graph.adjacency()
        preds = graph.predecessors()
        nmap = graph.node_map()

        # Track completed set for parallel scheduling
        completed: set[str] = set()
        pending = list(order)

        while pending:
            # Find "ready" nodes: all predecessors completed
            ready = [
                nid
                for nid in pending
                if all(p in completed for p in preds.get(nid, []))
            ]
            if not ready:
                raise RuntimeError(
                    "Deadlock: no ready nodes but pending remains"
                )

            # Sort for determinism
            ready.sort(key=lambda nid: (nmap[nid].priority, nid))

            # Execute batch (up to max_parallel)
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
        """Execute a single node, emitting trace events."""
        node_impl = self._registry.get(node_id)
        nmap = graph.node_map()
        graph_node = nmap.get(node_id)

        if node_impl is None:
            logger.error("No implementation for node %s", node_id)
            return NodeOutput(
                status="FAILED",
                error=f"No implementation registered for node {node_id}",
            )

        preds = graph.predecessors()
        # Gather inputs from predecessor outputs
        inputs: dict[str, Any] = {}
        for pred_id in preds.get(node_id, []):
            pred_output = self.results.get(pred_id)
            if pred_output:
                inputs.update(pred_output.outputs)

        # Inject initial inputs for all nodes (as defaults).
        # Upstream outputs always win because we use setdefault().
        for k, v in initial_inputs.items():
            inputs.setdefault(k, v)

        # Inject "prompt" for ModelNode from objective if not present
        if graph_node and graph_node.node_type == "model" and "prompt" not in inputs:
            # Critic nodes have a json_schema parameter — build a structured
            # evaluation prompt from all available inputs.
            if graph_node.parameters.get("json_schema"):
                inputs["prompt"] = _build_critic_prompt(inputs)
            elif inputs.get("previous_code"):
                # Fix node — build prompt with error context
                inputs["prompt"] = _build_fix_prompt(inputs)
            else:
                inputs["prompt"] = _build_default_prompt(inputs)

        # For CodeNode: use "content" from ModelNode output as "code"
        if graph_node and graph_node.node_type == "code" and "code" not in inputs:
            if "content" in inputs:
                inputs["code"] = inputs["content"]

        node_ref = (
            f"{self._execution_id}:{graph.metadata.plan_id}"
            f"/execute/{node_id}"
        )

        ctx = NodeContext(
            execution_id=self._execution_id,
            trace_id=self._trace_id,
            retry_count=0,
            random_seed=self._seed,
        )

        node_input = NodeInput(
            inputs=inputs,
            parameters=graph_node.parameters if graph_node else {},
            context=ctx,
        )

        # Emit node_start
        self._emit("node_start", {
            "node_id": node_id,
            "node_ref": node_ref,
            "node_type": graph_node.node_type if graph_node else "unknown",
            "inputs": inputs,
        })

        now = datetime.now(timezone.utc).isoformat()
        try:
            output = node_impl.execute(node_input)
        except Exception as exc:
            output = NodeOutput(status="FAILED", error=str(exc))

        # Emit node_end
        self._emit("node_end", {
            "node_id": node_id,
            "node_ref": node_ref,
            "status": output.status,
            "outputs": output.outputs,
            "error": output.error,
        })

        return output

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._trace_cb:
            self._trace_cb(kind, payload)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _build_critic_prompt(inputs: dict[str, Any]) -> str:
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

    return "\n\n".join(parts) if parts else "Evaluate the execution result."


def _build_fix_prompt(inputs: dict[str, Any]) -> str:
    """Build a prompt for the fix ModelNode from iteration 1 failure context."""
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

    gaps = inputs.get("previous_gaps", [])
    if gaps:
        parts.append(f"GAPS IDENTIFIED BY CRITIC:\n" + "\n".join(f"- {g}" for g in gaps))

    parts.append(
        "Fix the code above.  Make the MINIMAL change to resolve the error.  "
        "Output ONLY the corrected Python code."
    )

    return "\n\n".join(parts)


def _build_default_prompt(inputs: dict[str, Any]) -> str:
    """Build a general-purpose prompt from objective + constraints + context.

    Keeps backward compatibility for the simplest case:
    - If only ``objective`` is present, returns it unchanged.
    """
    objective = str(inputs.get("objective", "") or "")
    constraints = inputs.get("constraints")

    skip = {"prompt", "objective", "constraints"}
    extra_keys = sorted([k for k in inputs.keys() if k not in skip])

    has_constraints = bool(constraints)
    if objective and not has_constraints and not extra_keys:
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
