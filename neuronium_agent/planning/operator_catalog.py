"""Operator catalog for planner-safe DAG validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from neuronium_agent.planning.dag import GraphNode
from neuronium_agent.planning.operator_contracts import OperatorContract


@dataclass(frozen=True)
class OperatorCatalog:
    """Catalog of allowed operators with contracts and policies."""

    by_operator_id: dict[str, OperatorContract]
    by_tool_name: dict[str, OperatorContract]
    by_node_type: dict[str, OperatorContract]

    @classmethod
    def default(cls) -> OperatorCatalog:
        contracts = [
            OperatorContract(
                operator_id="model.default",
                node_type="model",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                deterministic=False,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="code.default",
                node_type="code",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="decision.default",
                node_type="decision",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=False,
            ),
            OperatorContract(
                operator_id="aggregate.default",
                node_type="aggregate",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=False,
            ),
            OperatorContract(
                operator_id="mcp.fs.read_text",
                node_type="mcp",
                tool_name="fs.read_text",
                input_schema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": {"type": "string"},
                        "out_key": {"type": "string"},
                        "encoding": {"type": "string"},
                    },
                },
                output_schema={"type": "object"},
                policy={"fs_roots_allowlist": "required"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.fs.write_text",
                node_type="mcp",
                tool_name="fs.write_text",
                input_schema={
                    "type": "object",
                    "required": ["path", "text"],
                    "properties": {
                        "path": {"type": "string"},
                        "text": {"type": "string"},
                        "encoding": {"type": "string"},
                        "overwrite": {"type": "boolean"},
                    },
                },
                output_schema={"type": "object"},
                policy={"fs_roots_allowlist": "required"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.fs.list_dir",
                node_type="mcp",
                tool_name="fs.list_dir",
                input_schema={
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
                output_schema={"type": "object"},
                policy={"fs_roots_allowlist": "required"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.fs.glob",
                node_type="mcp",
                tool_name="fs.glob",
                input_schema={
                    "type": "object",
                    "required": ["root", "pattern"],
                    "properties": {
                        "root": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                },
                output_schema={"type": "object"},
                policy={"fs_roots_allowlist": "required"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.web.fetch_html",
                node_type="mcp",
                tool_name="web.fetch_html",
                input_schema={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string"},
                        "timeout_seconds": {"type": "integer"},
                        "max_bytes": {"type": "integer"},
                    },
                },
                output_schema={"type": "object"},
                deterministic=False,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.web.extract_article",
                node_type="mcp",
                tool_name="web.extract_article",
                input_schema={
                    "type": "object",
                    "required": ["url", "html"],
                    "properties": {
                        "url": {"type": "string"},
                        "html": {"type": "string"},
                    },
                },
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.text.extract_entities",
                node_type="mcp",
                tool_name="text.extract_entities",
                input_schema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                    },
                },
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.artifact.put_json",
                node_type="mcp",
                tool_name="artifact.put_json",
                input_schema={
                    "type": "object",
                    "required": ["artifact_type", "json"],
                    "properties": {
                        "artifact_type": {"type": "string"},
                        "json": {"type": "object"},
                        "parent_artifact_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "produced_by_node_ref": {"type": "string"},
                        "media_type": {"type": "string"},
                    },
                },
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.export.write_text",
                node_type="mcp",
                tool_name="export.write_text",
                input_schema={
                    "type": "object",
                    "required": ["run_id"],
                    "properties": {
                        "run_id": {"type": "string"},
                        "kind": {"type": "string"},
                        "filename": {"type": "string"},
                        "stem": {"type": "string"},
                        "name": {"type": "string"},
                        "ext": {"type": "string"},
                        "format": {"type": "string"},
                        "text": {"type": "string"},
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "title": {"type": "string"},
                        "title_guess": {"type": "string"},
                        "source_url": {"type": "string"},
                        "final_url": {"type": "string"},
                        "url": {"type": "string"},
                        "encoding": {"type": "string"},
                        "overwrite": {"type": "boolean"},
                    },
                },
                output_schema={"type": "object"},
                deterministic=True,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.memory.ingest_files",
                node_type="mcp",
                tool_name="memory.ingest_files",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                deterministic=False,
                replay_required=True,
            ),
            OperatorContract(
                operator_id="mcp.memory.query",
                node_type="mcp",
                tool_name="memory.query",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                deterministic=False,
                replay_required=True,
            ),
        ]
        by_operator_id = {c.operator_id: c for c in contracts}
        by_tool_name = {c.tool_name: c for c in contracts if c.tool_name}
        by_node_type = {
            c.node_type: c for c in contracts if c.node_type != "mcp"
        }
        return cls(
            by_operator_id=by_operator_id,
            by_tool_name=by_tool_name,
            by_node_type=by_node_type,
        )

    def catalog_hash(self) -> str:
        payload = {
            k: self.by_operator_id[k].to_dict()
            for k in sorted(self.by_operator_id)
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def assert_node_allowed(self, node: GraphNode) -> None:
        if node.node_type == "mcp":
            tool_name = str(node.parameters.get("tool_name", "")).strip()
            if not tool_name:
                raise ValueError(
                    f"Dynamic plan MCP node '{node.node_id}' is missing tool_name"
                )
            if tool_name not in self.by_tool_name:
                raise ValueError(
                    f"Dynamic plan MCP tool '{tool_name}' has no operator contract"
                )
            return

        if node.node_type not in self.by_node_type:
            raise ValueError(
                f"Dynamic plan node_type '{node.node_type}' has no operator contract"
            )
