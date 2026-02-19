"""Stage 1 acceptance: reference payloads validate against Pydantic models.

For each ``<Name>.json`` in ``tests/reference_payloads/``, the corresponding
Pydantic model from :data:`SCHEMA_REGISTRY` must successfully parse it via
``model_validate``.

Additionally verifies that schema export is deterministic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from neuronium_agent.schemas.registry import SCHEMA_REGISTRY
from neuronium_agent.schemas.export import export_json_schemas

PAYLOADS_DIR = Path(__file__).parent / "reference_payloads"


def _payload_files() -> list[tuple[str, Path]]:
    """Collect (model_name, path) pairs for parametrised tests."""
    pairs: list[tuple[str, Path]] = []
    for p in sorted(PAYLOADS_DIR.glob("*.json")):
        name = p.stem
        if name in SCHEMA_REGISTRY:
            pairs.append((name, p))
    return pairs


# -- Parametrised: every reference payload validates against its model -------

@pytest.mark.parametrize(
    "model_name,payload_path",
    _payload_files(),
    ids=[n for n, _ in _payload_files()],
)
def test_reference_payload_validates(model_name: str, payload_path: Path) -> None:
    """Reference payload must be accepted by its Pydantic model."""
    model_cls = SCHEMA_REGISTRY[model_name]
    raw = json.loads(payload_path.read_text(encoding="utf-8"))
    instance = model_cls.model_validate(raw)
    # Round-trip: dump → re-validate must also succeed
    dumped = instance.model_dump(mode="json")
    model_cls.model_validate(dumped)


# -- Coverage: every payload file has a matching model -----------------------

def test_all_payload_files_have_matching_models() -> None:
    """No orphan payload files without a corresponding registry entry."""
    orphans: list[str] = []
    for p in PAYLOADS_DIR.glob("*.json"):
        if p.stem not in SCHEMA_REGISTRY:
            orphans.append(p.name)
    assert not orphans, (
        "Payload files without matching SCHEMA_REGISTRY entry: "
        + ", ".join(orphans)
    )


# -- Schema export determinism ----------------------------------------------

def test_schema_export_is_deterministic() -> None:
    """Two sequential exports must produce byte-identical files."""
    dir_a = Path(tempfile.mkdtemp())
    dir_b = Path(tempfile.mkdtemp())

    files_a = export_json_schemas(dir_a)
    files_b = export_json_schemas(dir_b)

    assert len(files_a) == len(files_b)
    for fa, fb in zip(files_a, files_b):
        assert fa.name == fb.name
        content_a = fa.read_text(encoding="utf-8")
        content_b = fb.read_text(encoding="utf-8")
        assert content_a == content_b, (
            "Schema export is not deterministic for " + fa.name
        )


# -- Schema export covers all registry entries ------------------------------

def test_schema_export_covers_registry() -> None:
    """Every model in the registry must produce a schema file."""
    out = Path(tempfile.mkdtemp())
    files = export_json_schemas(out)
    exported_names = {f.stem.replace(".schema", "") for f in files}
    registry_names = set(SCHEMA_REGISTRY.keys())
    assert exported_names == registry_names
