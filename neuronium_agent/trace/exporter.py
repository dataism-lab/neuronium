"""Trace exporter — export events to JSONL / JSON / ZIP (PUBLIC_API_SPEC §2.2)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Literal

from neuronium_agent._canonical import canonical_json


TraceExportFormat = Literal["jsonl", "json", "zip"]


class TraceExporter:
    """Export trace events to various formats."""

    def export(
        self,
        events: Iterable[dict[str, Any]],
        path: str | Path,
        fmt: TraceExportFormat = "jsonl",
    ) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        event_list = list(events)

        if fmt == "jsonl":
            self._export_jsonl(event_list, p)
        elif fmt == "json":
            self._export_json(event_list, p)
        elif fmt == "zip":
            self._export_zip(event_list, p)
        else:
            raise ValueError(f"Unknown export format: {fmt}")

    # ------------------------------------------------------------------

    @staticmethod
    def _export_jsonl(events: list[dict], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(canonical_json(ev) + "\n")

    @staticmethod
    def _export_json(events: list[dict], path: Path) -> None:
        path.write_text(
            json.dumps(events, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _export_zip(events: list[dict], path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            content = "\n".join(canonical_json(ev) for ev in events)
            zf.writestr("trace.jsonl", content)
        path.write_bytes(buf.getvalue())
