"""Deterministic rendering of final run artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from neuronium_agent.nodes.base import NodeOutput


@dataclass(frozen=True)
class RenderedArtifact:
    """Materialized final artifact for a run."""

    path: str
    title: str
    created_at: str


def render_run_artifact(
    *,
    data_dir: str,
    trace_id: str,
    runbook_id: str,
    objective: str,
    plan_id: str,
    results: dict[str, NodeOutput],
) -> RenderedArtifact | None:
    """Render a deterministic HTML summary artifact from node outputs."""
    completed = {
        nid: out for nid, out in sorted(results.items())
        if out.status == "COMPLETED"
    }
    if not completed:
        return None

    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    ts_prefix = now.strftime("%Y%m%dT%H%M%SZ")
    title = f"{runbook_id} :: {trace_id[:12]}"
    out_dir = Path(data_dir) / "rendered"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_plan = plan_id.replace("/", "_").replace(":", "_") if plan_id else "no-plan"
    file_name = f"{ts_prefix}_{trace_id[:12]}_{safe_plan[:24]}.html"
    out_path = out_dir / file_name

    rows: list[str] = []
    for nid, out in completed.items():
        payload = json.dumps(out.outputs, ensure_ascii=False, sort_keys=True, indent=2)
        rows.append(
            "<section>"
            f"<h3>{escape(nid)}</h3>"
            f"<pre>{escape(payload)}</pre>"
            "</section>"
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:980px;margin:24px auto;line-height:1.5}"
        "pre{background:#f5f5f5;padding:12px;overflow:auto}section{margin:24px 0}</style>"
        "</head><body>"
        f"<h1>{escape(title)}</h1>"
        f"<p><strong>trace_id:</strong> {escape(trace_id)}</p>"
        f"<p><strong>runbook:</strong> {escape(runbook_id)}</p>"
        f"<p><strong>plan_id:</strong> {escape(plan_id)}</p>"
        f"<p><strong>objective:</strong> {escape(objective)}</p>"
        f"<p><strong>created_at:</strong> {escape(created_at)}</p>"
        + "".join(rows) +
        "</body></html>"
    )
    out_path.write_text(html, encoding="utf-8")
    return RenderedArtifact(path=str(out_path), title=title, created_at=created_at)
