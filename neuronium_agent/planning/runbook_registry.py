"""Runbook registry — maps runbook_id strings to Runbook instances.

Internal module; not part of PUBLIC_API_SPEC.
"""

from __future__ import annotations

from neuronium_agent.planning.runbook_contract import Runbook

# ---------------------------------------------------------------------------
# Registry (module-level singleton dict)
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Runbook] = {}


def register_runbook(runbook: Runbook) -> None:
    """Register a runbook instance under its ``runbook_id``."""
    _REGISTRY[runbook.runbook_id] = runbook


def get_runbook(runbook_id: str) -> Runbook | None:
    """Return the runbook for *runbook_id*, or ``None`` if not found."""
    _ensure_builtins()
    return _REGISTRY.get(runbook_id)


def list_runbooks() -> list[str]:
    """Return sorted list of registered runbook IDs."""
    _ensure_builtins()
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Lazy registration of built-in runbooks
# ---------------------------------------------------------------------------

_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """Import and register built-in runbooks on first access."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    # Import here to avoid circular imports at module level.
    from neuronium_agent.planning.runbooks import DocsReportV1Runbook  # noqa: F811
    from neuronium_agent.planning.dynamic_planner_demo_runbook import DynamicPlannerDemoV1Runbook  # noqa: F811
    from neuronium_agent.planning.htn_recursive_demo_runbook import HtnRecursiveDemoV0Runbook  # noqa: F811
    from neuronium_agent.planning.super_agent_runbook import SuperAgentV0Runbook  # noqa: F811
    from neuronium_agent.planning.memory_runbook import HybridMemoryReportV1Runbook  # noqa: F811

    register_runbook(DocsReportV1Runbook())
    register_runbook(DynamicPlannerDemoV1Runbook())
    register_runbook(HtnRecursiveDemoV0Runbook())
    register_runbook(SuperAgentV0Runbook())
    register_runbook(HybridMemoryReportV1Runbook())
