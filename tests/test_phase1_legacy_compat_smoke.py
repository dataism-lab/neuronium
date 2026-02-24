from __future__ import annotations

from neuronium_agent.planning.extraction_contract import extraction_envelope_json_schema


def test_phase1_keeps_legacy_missing_fields_contract_shape() -> None:
    schema = extraction_envelope_json_schema()
    field_schema = (
        schema["properties"]["missing_fields"]["items"]["properties"]["field"]
    )

    assert field_schema["type"] == "string"
    assert "output_text" in field_schema["enum"]
    assert "doc_paths" in field_schema["enum"]
