"""Backward-compatible shim for shared tool schema registry.

Phase-3 preflight moves ToolSchemaRegistry to a shared module so non-planning
consumers (e.g. orchestrator revise path) can reuse the same contract logic.
"""

from neuronium_agent.schemas.tool_schema_registry import ToolSchemaRegistry
from neuronium_agent.schemas.tool_schema_registry import extract_required_json_pointers
from neuronium_agent.schemas.tool_schema_registry import schema_fragment_at_pointer

__all__ = [
    "ToolSchemaRegistry",
    "extract_required_json_pointers",
    "schema_fragment_at_pointer",
]
