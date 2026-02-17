from __future__ import annotations

from neuronium_agent.core.state import RunState
from neuronium_agent.nodes.base import NodeContext, NodeInput
from neuronium_agent.nodes.mcp_node import McpToolNode


def test_mcp_local_fs_read_text_allowed(tmp_path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")

    node = McpToolNode(
        node_id="t1",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": [str(tmp_path)]},
    )

    out = node.execute(
        NodeInput(
            inputs={},
            parameters={
                "tool_name": "fs.read_text",
                "tool_args": {"path": str(p), "out_key": "doc_000"},
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "COMPLETED"
    assert out.outputs["doc_000"] == "hello"
    assert str(p) in out.outputs["doc_000__path"]


def test_mcp_local_fs_read_text_denied_by_policy(tmp_path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("hello", encoding="utf-8")

    node = McpToolNode(
        node_id="t1",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": []},
    )

    out = node.execute(
        NodeInput(
            inputs={},
            parameters={
                "tool_name": "fs.read_text",
                "tool_args": {"path": str(p), "out_key": "doc_000"},
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "FAILED"
    assert out.error and "Policy denied" in out.error

