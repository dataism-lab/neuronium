"""Deterministic JSON Schema export from the schema registry.

Produces one ``<name>.schema.json`` file per model in :data:`SCHEMA_REGISTRY`.
Output is canonicalised (sorted keys, compact separators, stable floats)
so that identical code always yields byte-identical files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuronium_agent._canonical import canonical_json
from neuronium_agent.schemas.registry import SCHEMA_REGISTRY


def _canonicalise_schema(raw_schema: dict[str, Any]) -> str:
    """Return a human-readable yet deterministic JSON string.

    Uses ``canonical_json`` for key ordering / float normalisation,
    then re-formats with 2-space indent for readability while keeping
    sorted keys.
    """
    normalised = json.loads(canonical_json(raw_schema))
    return json.dumps(
        normalised,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def export_json_schemas(out_dir: Path) -> list[Path]:
    """Export all registered schemas to *out_dir*.

    Returns the list of written file paths (sorted by name).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name in sorted(SCHEMA_REGISTRY):
        model_cls = SCHEMA_REGISTRY[name]
        raw_schema = model_cls.model_json_schema()
        content = _canonicalise_schema(raw_schema)

        path = out_dir / f"{name}.schema.json"
        path.write_text(content, encoding="utf-8")
        written.append(path)

    return written
