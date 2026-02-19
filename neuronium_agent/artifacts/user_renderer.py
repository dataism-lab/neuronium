"""User-facing deterministic renderers for runs.

Keeps debug-dump renderer intact, but provides a minimal artifact that
looks like what an end-user expects: title + summary + source link.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from neuronium_agent.nodes.base import NodeOutput


@dataclass(frozen=True)
class UserFacingSummary:
    title: str | None
    summary: str | None
    source_url: str | None


def extract_user_facing_summary(results: dict[str, NodeOutput]) -> UserFacingSummary:
    """Best-effort extract title/summary/url from known node outputs."""
    title: str | None = None
    summary: str | None = None
    source_url: str | None = None

    extract = results.get("extract_article")
    if extract and extract.status == "COMPLETED":
        raw_title = extract.outputs.get("title_guess")
        if isinstance(raw_title, str) and raw_title.strip():
            title = raw_title.strip()
        raw_url = extract.outputs.get("final_url")
        if isinstance(raw_url, str) and raw_url.strip():
            source_url = raw_url.strip()

    draft = results.get("draft_report")
    if draft and draft.status == "COMPLETED":
        raw = draft.outputs.get("content")
        if isinstance(raw, str) and raw.strip():
            summary = raw.strip()

    return UserFacingSummary(title=title, summary=summary, source_url=source_url)


def render_user_facing_html(
    *,
    data_dir: str,
    trace_id: str,
    runbook_id: str,
    objective: str,
    plan_id: str,
    results: dict[str, NodeOutput],
    debug_artifact_path: str | None = None,
) -> str | None:
    """Render a small user-facing HTML artifact.

    Returns written path or None if required fields are missing.
    """
    extracted = extract_user_facing_summary(results)
    if not extracted.summary:
        return None

    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    ts_prefix = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(data_dir) / "rendered"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_plan = plan_id.replace("/", "_").replace(":", "_") if plan_id else "no-plan"
    file_name = f"{ts_prefix}_{trace_id[:12]}_{safe_plan[:24]}_user.html"
    out_path = out_dir / file_name

    title = extracted.title or f"{runbook_id} :: {trace_id[:12]}"
    summary_html = "<pre>" + escape(extracted.summary) + "</pre>"
    source_html = ""
    if extracted.source_url:
        url = escape(extracted.source_url)
        source_html = f"<p><strong>source:</strong> <a href='{url}'>{url}</a></p>"

    debug_link_html = ""
    if debug_artifact_path:
        debug_name = Path(debug_artifact_path).name
        debug_link_html = (
            "<p><strong>debug:</strong> "
            f"<a href='{escape(debug_name)}'>{escape(debug_name)}</a></p>"
        )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:980px;margin:24px auto;line-height:1.5}"
        "pre{background:#f5f5f5;padding:12px;overflow:auto;white-space:pre-wrap}"
        "</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f"<p><strong>trace_id:</strong> {escape(trace_id)}</p>"
        f"<p><strong>runbook:</strong> {escape(runbook_id)}</p>"
        f"<p><strong>plan_id:</strong> {escape(plan_id)}</p>"
        f"<p><strong>objective:</strong> {escape(objective)}</p>"
        f"<p><strong>created_at:</strong> {escape(created_at)}</p>"
        + source_html
        + "<h2>Summary</h2>"
        + summary_html
        + debug_link_html
        + "</body></html>"
    )

    out_path.write_text(html, encoding="utf-8")
    return str(out_path)

