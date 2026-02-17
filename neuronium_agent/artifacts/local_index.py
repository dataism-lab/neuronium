"""Local gallery index for rendered artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalIndexEntry:
    trace_id: str
    runbook_id: str
    objective: str
    artifact_path: str
    created_at: str
    plan_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "runbook_id": self.runbook_id,
            "objective": self.objective,
            "artifact_path": self.artifact_path,
            "created_at": self.created_at,
            "plan_id": self.plan_id,
        }


class LocalArtifactIndex:
    """Deterministic local index writer (JSONL + HTML)."""

    def __init__(self, data_dir: str) -> None:
        root = Path(data_dir) / "rendered"
        root.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = root / "index.jsonl"
        self._html_path = root / "index.html"

    def append(self, entry: LocalIndexEntry) -> None:
        existing = self._load_all()
        existing.append(entry)
        # Deterministic ordering by timestamp then trace_id.
        sorted_entries = sorted(
            existing,
            key=lambda x: (x.created_at, x.trace_id),
        )
        self._jsonl_path.write_text(
            "\n".join(json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) for e in sorted_entries) + "\n",
            encoding="utf-8",
        )
        self._render_html(sorted_entries)

    def _load_all(self) -> list[LocalIndexEntry]:
        if not self._jsonl_path.exists():
            return []
        lines = self._jsonl_path.read_text(encoding="utf-8").splitlines()
        entries: list[LocalIndexEntry] = []
        for line in lines:
            if not line.strip():
                continue
            d = json.loads(line)
            entries.append(LocalIndexEntry(
                trace_id=str(d.get("trace_id", "")),
                runbook_id=str(d.get("runbook_id", "")),
                objective=str(d.get("objective", "")),
                artifact_path=str(d.get("artifact_path", "")),
                created_at=str(d.get("created_at", "")),
                plan_id=str(d.get("plan_id", "")),
            ))
        return entries

    def _render_html(self, entries: list[LocalIndexEntry]) -> None:
        items: list[str] = []
        for e in entries:
            artifact_name = Path(e.artifact_path).name
            items.append(
                "<tr>"
                f"<td>{escape(e.created_at)}</td>"
                f"<td>{escape(e.runbook_id)}</td>"
                f"<td>{escape(e.trace_id)}</td>"
                f"<td>{escape(e.plan_id)}</td>"
                f"<td>{escape(e.objective)}</td>"
                f"<td><a href='{escape(artifact_name)}'>{escape(artifact_name)}</a></td>"
                "</tr>"
            )
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Neuronium Local Index</title>"
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:24px auto}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}"
            "th{background:#f2f2f2;text-align:left}</style>"
            "</head><body><h1>Neuronium Local Index</h1>"
            "<table><thead><tr>"
            "<th>created_at</th><th>runbook</th><th>trace_id</th><th>plan_id</th><th>objective</th><th>artifact</th>"
            "</tr></thead><tbody>"
            + "".join(items) +
            "</tbody></table></body></html>"
        )
        self._html_path.write_text(html, encoding="utf-8")
