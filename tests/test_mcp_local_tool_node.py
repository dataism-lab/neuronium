from __future__ import annotations

from pathlib import Path

from neuronium_agent.config import AppConfig, ProjectConfig
from neuronium_agent.nodes.base import NodeContext, NodeInput
from neuronium_agent.nodes.mcp_node import McpToolNode
from neuronium_agent.tools.runtime import ToolRuntime


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


def test_mcp_local_fs_write_text_allowed_with_auto_mkdir(tmp_path) -> None:
    target = tmp_path / "nested" / "review.html"
    content = "<h1>hello</h1>"

    node = McpToolNode(
        node_id="t2",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": [str(tmp_path)]},
    )

    out = node.execute(
        NodeInput(
            inputs={},
            parameters={
                "tool_name": "fs.write_text",
                "tool_args": {
                    "path": str(target),
                    "text": content,
                    "encoding": "utf-8",
                    "overwrite": True,
                },
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "COMPLETED"
    assert out.outputs["path"] == str(target.resolve())
    assert out.outputs["bytes_written"] == len(content.encode("utf-8"))
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content


def test_mcp_local_fs_write_text_denied_by_policy(tmp_path) -> None:
    target = tmp_path / "review.html"

    node = McpToolNode(
        node_id="t3",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": []},
    )

    out = node.execute(
        NodeInput(
            inputs={},
            parameters={
                "tool_name": "fs.write_text",
                "tool_args": {"path": str(target), "text": "abc"},
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "FAILED"
    assert out.error and "Policy denied" in out.error


def test_mcp_local_fs_write_text_overwrite_false_fails_if_exists(tmp_path) -> None:
    target = tmp_path / "review.html"
    target.write_text("old", encoding="utf-8")

    node = McpToolNode(
        node_id="t4",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": [str(tmp_path)]},
    )

    out = node.execute(
        NodeInput(
            inputs={},
            parameters={
                "tool_name": "fs.write_text",
                "tool_args": {
                    "path": str(target),
                    "text": "new",
                    "overwrite": False,
                },
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "FAILED"
    assert out.error and "overwrite is false" in out.error
    assert target.read_text(encoding="utf-8") == "old"


def test_mcp_local_fs_write_text_size_limit(tmp_path) -> None:
    target = tmp_path / "big.txt"

    node = McpToolNode(
        node_id="t5",
        server_name="local",
        server_url="local://",
        policy={
            "fs_roots_allowlist": [str(tmp_path)],
            "fs_max_write_bytes": 5,
        },
    )

    out = node.execute(
        NodeInput(
            inputs={},
            parameters={
                "tool_name": "fs.write_text",
                "tool_args": {"path": str(target), "text": "123456"},
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "FAILED"
    assert out.error and "payload too large" in out.error
    assert not target.exists()


def test_mcp_local_export_write_text_writes_under_project_data_dir(tmp_path) -> None:
    # Arrange a runtime with project.data_dir under tmp_path
    data_dir = tmp_path / ".neuronium"
    cfg = AppConfig(project=ProjectConfig(name="test", data_dir=str(data_dir)))

    node = McpToolNode(
        node_id="t6",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": [str(tmp_path)]},
        tool_runtime=ToolRuntime(config=cfg),
    )

    # Provide upstream-like inputs that auto-wire into tool_args (content/title/url).
    out = node.execute(
        NodeInput(
            inputs={
                "content": "Hello summary",
                "title_guess": "Title",
                "final_url": "https://example.com/x",
            },
            parameters={
                "tool_name": "export.write_text",
                "tool_args": {
                    "run_id": "run123",
                    "kind": "news_summary",
                    "filename": "summary.md",
                    "overwrite": True,
                },
            },
            context=NodeContext(execution_id="e1", trace_id="t1"),
        )
    )

    assert out.status == "COMPLETED"
    path = out.outputs["path"]
    assert str(data_dir.resolve()) in path
    assert Path(path).parent.name == "run123"
    assert Path(path).name == "summary.md"

