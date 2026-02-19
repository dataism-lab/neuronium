from __future__ import annotations

import os
from typing import Any

import pytest

from neuronium_agent.nodes.base import NodeContext, NodeInput
from neuronium_agent.nodes.mcp_node import McpToolNode
from neuronium_agent.tools.local_tools import ToolCall, invoke_local_tool


class _FakeHeaders:
    def __init__(self, content_type: str) -> None:
        self._content_type = content_type

    def get(self, key: str, default: str = "") -> str:
        if key.lower() == "content-type":
            return self._content_type
        return default

    def get_content_charset(self) -> str:
        if "charset=" in self._content_type.lower():
            return self._content_type.split("charset=", 1)[1].strip()
        return "utf-8"


class _FakeResponse:
    def __init__(
        self,
        *,
        body: bytes,
        final_url: str = "https://example.com/article",
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self._body = body
        self._index = 0
        self._final_url = final_url
        self.status = status
        self.headers = _FakeHeaders(content_type)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            return self._body
        data = self._body[self._index:self._index + amount]
        self._index += amount
        return data

    def geturl(self) -> str:
        return self._final_url


def _contains_sensitive_key(data: Any) -> bool:
    if isinstance(data, dict):
        for k, v in data.items():
            key = str(k).lower()
            if any(token in key for token in ("token", "secret", "api_key", "password", "key")):
                return True
            if _contains_sensitive_key(v):
                return True
    elif isinstance(data, list):
        return any(_contains_sensitive_key(x) for x in data)
    return False


def test_web_fetch_html_via_local_tool_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuronium_agent.tools import web_tools

    payload = b"<html><head><title>Demo</title></head><body>ok</body></html>"

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse(body=payload, final_url="https://example.com/final")

    monkeypatch.setattr(web_tools, "urlopen", fake_urlopen)

    out = invoke_local_tool(ToolCall(
        tool_name="web.fetch_html",
        tool_args={"url": "https://example.com/start", "timeout_seconds": 5, "max_bytes": 1024},
    ))
    assert out["final_url"] == "https://example.com/final"
    assert out["status_code"] == 200
    assert "<title>Demo</title>" in out["html"]
    assert out["warnings"] == []


def test_web_fetch_html_truncates_when_max_bytes_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuronium_agent.tools import web_tools

    payload = b"x" * 300

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse(body=payload)

    monkeypatch.setattr(web_tools, "urlopen", fake_urlopen)
    out = invoke_local_tool(ToolCall(
        tool_name="web.fetch_html",
        tool_args={"url": "https://example.com", "max_bytes": 128},
    ))
    assert len(out["html"]) == 128
    assert out["warnings"] == ["response_truncated_to_128_bytes"]


def test_web_extract_article_returns_text_and_images() -> None:
    html = """
    <html>
      <head>
        <title>Fallback title</title>
        <meta property="og:title" content="OpenGraph title"/>
      </head>
      <body>
        <main>
          <h1>Article heading</h1>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
          <img src="/img/cover.jpg" alt="Cover image"/>
          <img src="/img/cover.jpg" alt="Cover image duplicate"/>
        </main>
      </body>
    </html>
    """
    out = invoke_local_tool(ToolCall(
        tool_name="web.extract_article",
        tool_args={"url": "https://example.com/post", "html": html},
    ))
    assert out["title_guess"] == "OpenGraph title"
    assert "First paragraph." in out["text"]
    assert "Second paragraph." in out["text"]
    assert out["images"] == [{"src": "https://example.com/img/cover.jpg", "alt": "Cover image"}]


def test_web_extract_article_via_mcp_tool_node() -> None:
    node = McpToolNode(
        node_id="t-web",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": []},
    )
    html = "<html><body><article><p>Hello world</p><img src='a.png' alt='A'/></article></body></html>"
    out = node.execute(NodeInput(
        inputs={},
        parameters={"tool_name": "web.extract_article", "tool_args": {"url": "https://x.test/base", "html": html}},
        context=NodeContext(execution_id="e1", trace_id="t1"),
    ))
    assert out.status == "COMPLETED"
    assert "Hello world" in out.outputs["text"]
    # Relative a.png resolved against document URL (with path /base) → /base/a.png
    assert out.outputs["images"][0]["src"] == "https://x.test/base/a.png"


def test_web_extract_article_via_mcp_tool_node_accepts_html_from_node_inputs() -> None:
    node = McpToolNode(
        node_id="t-web-flow",
        server_name="local",
        server_url="local://",
        policy={"fs_roots_allowlist": []},
    )
    html = "<html><body><article><p>Edge propagated text</p></article></body></html>"
    out = node.execute(NodeInput(
        inputs={"html": html},
        parameters={"tool_name": "web.extract_article", "tool_args": {"url": "https://x.test/base"}},
        context=NodeContext(execution_id="e1", trace_id="t1"),
    ))
    assert out.status == "COMPLETED"
    assert "Edge propagated text" in out.outputs["text"]


def test_web_extract_article_normalizes_arxiv_like_base_for_relative_images() -> None:
    html = "<html><body><main><img src='x1.png' alt='Refer to caption'/></main></body></html>"
    out = invoke_local_tool(ToolCall(
        tool_name="web.extract_article",
        tool_args={"url": "https://arxiv.org/html/2511.12869v2", "html": html},
    ))
    assert out["images"] == [
        {
            "src": "https://arxiv.org/html/2511.12869v2/x1.png",
            "alt": "Refer to caption",
        }
    ]


def test_web_tools_outputs_do_not_include_sensitive_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from neuronium_agent.tools import web_tools

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse(body=b"<html><body>safe</body></html>")

    monkeypatch.setattr(web_tools, "urlopen", fake_urlopen)
    fetched = invoke_local_tool(ToolCall(
        tool_name="web.fetch_html",
        tool_args={"url": "https://example.com"},
    ))
    extracted = invoke_local_tool(ToolCall(
        tool_name="web.extract_article",
        tool_args={"url": "https://example.com", "html": fetched["html"]},
    ))
    assert not _contains_sensitive_key(fetched)
    assert not _contains_sensitive_key(extracted)


@pytest.mark.skipif(
    os.getenv("NEURONIUM_RUN_LIVE_WEB_TESTS") != "1",
    reason="Set NEURONIUM_RUN_LIVE_WEB_TESTS=1 to enable live web smoke test.",
)
def test_web_fetch_html_live_smoke() -> None:
    out = invoke_local_tool(ToolCall(
        tool_name="web.fetch_html",
        tool_args={"url": "https://example.com", "timeout_seconds": 10, "max_bytes": 200_000},
    ))
    assert out["status_code"] == 200
    assert "<html" in out["html"].lower()
