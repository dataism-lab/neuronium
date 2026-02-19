from __future__ import annotations

from neuronium_agent.tools.local_tools import ToolCall, invoke_local_tool
from neuronium_agent.tools.runtime import ToolRuntime


def test_artifact_put_json_records_blob_metadata_and_lineage(
    app_config,
    blob_store,
    index_store,
) -> None:
    runtime = ToolRuntime(
        config=app_config,
        index_store=index_store,
        blob_store=blob_store,
    )
    parent = invoke_local_tool(
        ToolCall(
            tool_name="artifact.put_json",
            tool_args={
                "artifact_type": "test.parent",
                "json": {"v": 1},
                "produced_by_node_ref": "test:node:parent",
            },
        ),
        policy={},
        runtime=runtime,
    )
    child = invoke_local_tool(
        ToolCall(
            tool_name="artifact.put_json",
            tool_args={
                "artifact_type": "test.child",
                "json": {"v": 2},
                "parent_artifact_ids": [parent["artifact_id"]],
                "produced_by_node_ref": "test:node:child",
            },
        ),
        policy={},
        runtime=runtime,
    )

    child_id = child["artifact_id"]
    assert isinstance(child["canonical_json"], str)
    assert blob_store.exists(child_id)

    meta = index_store.get_artifact(child_id)
    assert meta is not None
    assert meta["artifact_type"] == "test.child"

    rows = index_store._fetchall(  # type: ignore[attr-defined]
        "SELECT parent_artifact_id, child_artifact_id, kind FROM lineage_edges WHERE child_artifact_id=?",
        (child_id,),
    )
    assert rows
    assert rows[0]["parent_artifact_id"] == parent["artifact_id"]
    assert rows[0]["kind"] == "evidence"


def test_text_extract_entities_detects_url_and_path_like_tokens() -> None:
    output = invoke_local_tool(
        ToolCall(
            tool_name="text.extract_entities",
            tool_args={
                "text": "Сделай сводку https://example.com/a и используй ./docs/news.md",
            },
        ),
        policy={},
        runtime=None,
    )
    assert output["urls"] == ["https://example.com/a"]
    assert "./docs/news.md" in output["file_paths"]
    assert output["basenames"] == []


def test_text_extract_entities_treats_protocol_relative_domain_as_url_only() -> None:
    output = invoke_local_tool(
        ToolCall(
            tool_name="text.extract_entities",
            tool_args={"text": "сводка //arxiv.org/html/2511.12869v2"},
        ),
        policy={},
        runtime=None,
    )
    assert output["urls"] == ["https://arxiv.org/html/2511.12869v2"]
    assert output["file_paths"] == []


def test_text_extract_entities_does_not_extract_ambiguous_basenames() -> None:
    output = invoke_local_tool(
        ToolCall(
            tool_name="text.extract_entities",
            tool_args={"text": "обработай data.csv"},
        ),
        policy={},
        runtime=None,
    )
    assert output["urls"] == []
    assert output["file_paths"] == []
    assert output["basenames"] == []


def test_extraction_envelope_json_schema_is_ref_free() -> None:
    from neuronium_agent.planning.extraction_contract import (
        extraction_envelope_json_schema,
    )

    schema = extraction_envelope_json_schema()
    dumped = str(schema)
    assert "$ref" not in dumped
    assert "$defs" not in dumped
    assert "additionalProperties': True" not in dumped
    assert set(schema["properties"]["inputs"]["required"]) == {
        "url",
        "urls",
        "doc_paths",
        "language",
        "output_format",
        "summary_length",
        "output_filename",
        "output_text",
    }
    out_fn_schema = schema["properties"]["inputs"]["properties"]["output_filename"]
    assert out_fn_schema["type"] == ["string", "null"]
    assert "pattern" in out_fn_schema
