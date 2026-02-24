from __future__ import annotations

from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.operator_contracts import OperatorContract
from neuronium_agent.planning.tool_schema_registry import ToolSchemaRegistry
from neuronium_agent.planning.tool_schema_registry import extract_required_json_pointers


def test_tool_schema_registry_get_schema_for_known_tool() -> None:
    registry = ToolSchemaRegistry(operator_catalog=OperatorCatalog.default())

    schema = registry.get_input_schema(tool_name="fs.write_text")
    assert schema["type"] == "object"
    assert sorted(schema["required"]) == ["path", "text"]


def test_extract_required_paths_supports_nested_objects() -> None:
    schema = {
        "type": "object",
        "required": ["inputs"],
        "properties": {
            "inputs": {
                "type": "object",
                "required": ["payload"],
                "properties": {
                    "payload": {
                        "type": "object",
                        "required": ["file_name"],
                        "properties": {
                            "file_name": {"type": "string"},
                        },
                    }
                },
            }
        },
    }

    paths = extract_required_json_pointers(schema)
    assert paths == ["/inputs", "/inputs/payload", "/inputs/payload/file_name"]


def test_tool_schema_registry_merge_input_schemas_union_required() -> None:
    custom_contract = OperatorContract(
        operator_id="mcp.custom.nested",
        node_type="mcp",
        tool_name="custom.nested",
        input_schema={
            "type": "object",
            "required": ["payload"],
            "properties": {
                "payload": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                    },
                }
            },
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
    )

    base = OperatorCatalog.default()
    catalog = OperatorCatalog(
        by_operator_id={**base.by_operator_id, custom_contract.operator_id: custom_contract},
        by_tool_name={**base.by_tool_name, "custom.nested": custom_contract},
        by_node_type=base.by_node_type,
    )
    registry = ToolSchemaRegistry(operator_catalog=catalog)

    merged = registry.merge_input_schemas(
        tool_names=["fs.read_text", "custom.nested"],
    )
    assert merged["required"] == ["path", "payload"]
    assert sorted(merged["properties"]) == ["encoding", "out_key", "path", "payload"]
    assert merged["additionalProperties"] is False
