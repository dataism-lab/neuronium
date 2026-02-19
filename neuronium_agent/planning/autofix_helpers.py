"""Helpers for autofix-demo runbook: fix context and added constraints from iter1.

Used by AutofixDemoRunbook to build the iter2 graph and initial_inputs.
Internal module; not part of PUBLIC_API_SPEC.
"""

from __future__ import annotations

from typing import Any

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.verification.demo_critic import DemoCriticVerdict


def build_fix_context(
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


def build_added_constraints(
    results: dict[str, NodeOutput],
    verdict: DemoCriticVerdict,
) -> list[str]:
    """Derive added constraints from iteration 1 failure."""
    added: list[str] = []
    exe = results.get("execute")
    if exe and exe.status == "FAILED":
        stderr = exe.outputs.get("stderr", "") or (exe.error or "")
        if stderr:
            added.append(f"Fix execution error: {stderr[:300]}")
    for gap in verdict.gaps:
        added.append(f"Fix gap: {gap}")
    return added
