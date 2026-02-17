from __future__ import annotations

from pathlib import Path

from neuronium_agent.config import AppConfig, ProjectConfig
from neuronium_agent.tools.export_tools import invoke_export_write_text
from neuronium_agent.tools.runtime import ToolRuntime


def _runtime_with_data_dir(tmp_path: Path) -> ToolRuntime:
    data_dir = tmp_path / ".neuronium"
    cfg = AppConfig(project=ProjectConfig(name="test", data_dir=str(data_dir)))
    return ToolRuntime(config=cfg)


def test_export_news_summary_html_prefers_model_content_over_raw_text(tmp_path: Path) -> None:
    runtime = _runtime_with_data_dir(tmp_path)

    out = invoke_export_write_text(
        {
            "run_id": "run123",
            "kind": "news_summary",
            "filename": "summary.html",
            "text": "RAW SOURCE ARTICLE TEXT",
            "content": "<!DOCTYPE html><html><body><h1>Ready summary</h1></body></html>",
        },
        policy={},
        runtime=runtime,
    )

    payload = Path(out["path"]).read_text(encoding="utf-8")
    assert Path(out["path"]).parent.name == "run123"
    assert Path(out["path"]).name == "summary.html"
    assert "<!DOCTYPE html>" in payload
    assert "Ready summary" in payload
    assert "RAW SOURCE ARTICLE TEXT" not in payload


def test_export_news_summary_html_unwraps_fenced_html_block(tmp_path: Path) -> None:
    runtime = _runtime_with_data_dir(tmp_path)

    out = invoke_export_write_text(
        {
            "run_id": "run456",
            "kind": "news_summary",
            "filename": "summary.html",
            "content": (
                "```html\n"
                "<!DOCTYPE html>\n"
                "<html><body><p>Short summary</p></body></html>\n"
                "```"
            ),
        },
        policy={},
        runtime=runtime,
    )

    payload = Path(out["path"]).read_text(encoding="utf-8")
    assert Path(out["path"]).parent.name == "run456"
    assert Path(out["path"]).name == "summary.html"
    assert payload.lstrip().startswith("<!DOCTYPE html>")
    assert "```" not in payload
    assert "<p>Short summary</p>" in payload
