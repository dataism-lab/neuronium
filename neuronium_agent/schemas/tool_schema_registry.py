"""Shared registry for tool/operator input contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from neuronium_agent.planning.operator_catalog import OperatorCatalog
from neuronium_agent.planning.operator_contracts import OperatorContract


def _escape_json_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _unescape_json_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _pointer_join(base_pointer: str, token: str) -> str:
    escaped = _escape_json_pointer_token(token)
    if not base_pointer:
        return f"/{escaped}"
    return f"{base_pointer}/{escaped}"


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer:
        return []
    if not pointer.startswith("/"):
        msg = f"JSON pointer must start with '/': {pointer!r}"
        raise ValueError(msg)
    if pointer == "/":
        return [""]
    return [_unescape_json_pointer_token(t) for t in pointer[1:].split("/")]


def extract_required_json_pointers(
    input_schema: Mapping[str, Any],
    *,
    base_pointer: str = "",
) -> list[str]:
    """Return deterministic list of required JSON Pointer paths."""

    required_paths: set[str] = set()

    def visit(schema: Mapping[str, Any], pointer: str) -> None:
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return

        required = schema.get("required")
        if not isinstance(required, list):
            return

        for raw_name in required:
            if not isinstance(raw_name, str):
                continue
            path = _pointer_join(pointer, raw_name)
            required_paths.add(path)

            child_schema = properties.get(raw_name)
            if isinstance(child_schema, Mapping):
                visit(child_schema, path)

    visit(input_schema, base_pointer)
    return sorted(required_paths)


def schema_fragment_at_pointer(
    input_schema: Mapping[str, Any],
    pointer: str,
) -> dict[str, Any]:
    """Return schema fragment for a JSON pointer, empty dict if unavailable."""

    tokens = _pointer_tokens(pointer)
    current: Mapping[str, Any] = input_schema
    fragment: Mapping[str, Any] | None = current
    for token in tokens:
        properties = current.get("properties")
        if not isinstance(properties, Mapping):
            fragment = None
            break
        child = properties.get(token)
        if not isinstance(child, Mapping):
            fragment = None
            break
        current = child
        fragment = child

    if not isinstance(fragment, Mapping):
        return {}
    return deepcopy(dict(fragment))


@dataclass(frozen=True)
class ToolSchemaRegistry:
    """Read tool input schemas through operator catalog contracts."""

    operator_catalog: OperatorCatalog

    @classmethod
    def from_default_catalog(cls) -> ToolSchemaRegistry:
        return cls(operator_catalog=OperatorCatalog.default())

    def resolve_contract(
        self,
        *,
        tool_name: str | None = None,
        operator_id: str | None = None,
    ) -> OperatorContract:
        if bool(tool_name) == bool(operator_id):
            msg = "Provide exactly one identifier: tool_name or operator_id"
            raise ValueError(msg)

        if operator_id:
            contract = self.operator_catalog.by_operator_id.get(operator_id)
            if contract is None:
                raise ValueError(f"Unknown operator_id: {operator_id}")
            return contract

        assert tool_name is not None
        contract = self.operator_catalog.by_tool_name.get(tool_name)
        if contract is None:
            raise ValueError(f"Unknown tool_name: {tool_name}")
        return contract

    def get_input_schema(
        self,
        *,
        tool_name: str | None = None,
        operator_id: str | None = None,
    ) -> dict[str, Any]:
        contract = self.resolve_contract(tool_name=tool_name, operator_id=operator_id)
        return deepcopy(contract.input_schema)

    def get_required_paths(
        self,
        *,
        tool_name: str | None = None,
        operator_id: str | None = None,
        base_pointer: str = "",
    ) -> list[str]:
        schema = self.get_input_schema(tool_name=tool_name, operator_id=operator_id)
        return extract_required_json_pointers(schema, base_pointer=base_pointer)

    def merge_input_schemas(
        self,
        *,
        tool_names: Sequence[str] | None = None,
        operator_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Merge schemas for multiple candidate operators/tools.

        Phase-1 strategy: union of required and object properties.
        """
        names = list(tool_names or [])
        ids = list(operator_ids or [])
        if not names and not ids:
            raise ValueError("Provide at least one tool_name or operator_id")

        schemas: list[dict[str, Any]] = []
        for tool_name in names:
            schemas.append(self.get_input_schema(tool_name=tool_name))
        for operator_id in ids:
            schemas.append(self.get_input_schema(operator_id=operator_id))

        merged_required: set[str] = set()
        merged_properties: dict[str, Any] = {}
        additional_properties = True

        for schema in schemas:
            required = schema.get("required", [])
            if isinstance(required, list):
                merged_required.update(str(x) for x in required if isinstance(x, str))

            properties = schema.get("properties", {})
            if isinstance(properties, Mapping):
                for key in sorted(properties):
                    value = properties[key]
                    if (
                        key in merged_properties
                        and isinstance(merged_properties[key], dict)
                        and isinstance(value, Mapping)
                    ):
                        nested = self._merge_property_fragments(
                            merged_properties[key],
                            dict(value),
                        )
                        merged_properties[key] = nested
                    else:
                        merged_properties[key] = deepcopy(value)

            if schema.get("additionalProperties") is False:
                additional_properties = False

        return {
            "type": "object",
            "required": sorted(merged_required),
            "properties": dict(sorted(merged_properties.items())),
            "additionalProperties": additional_properties,
        }

    @staticmethod
    def _merge_property_fragments(
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> dict[str, Any]:
        merged = deepcopy(left)

        right_required = right.get("required", [])
        if isinstance(right_required, list):
            left_required = merged.get("required", [])
            if not isinstance(left_required, list):
                left_required = []
            merged["required"] = sorted(
                set(str(x) for x in left_required if isinstance(x, str))
                | set(str(x) for x in right_required if isinstance(x, str))
            )

        left_props = merged.get("properties", {})
        right_props = right.get("properties", {})
        if isinstance(left_props, Mapping) and isinstance(right_props, Mapping):
            combined: dict[str, Any] = dict(left_props)
            for key in sorted(right_props):
                if key not in combined:
                    combined[key] = deepcopy(right_props[key])
            merged["properties"] = combined

        for k, v in right.items():
            if k in {"required", "properties"}:
                continue
            if k not in merged:
                merged[k] = deepcopy(v)

        return merged
