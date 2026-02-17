"""Export tools: write user-facing outputs to deterministic locations.

Rationale:
- End users expect a small "final" file (md/html/txt/...) in a stable place.
- The debug rendered artifact remains available separately.
- The export itself should be part of the DAG (Audit-by-Construction).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from neuronium_agent.tools.local_tools import ToolExecutionError


_SAFE_EXT_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_FENCED_BLOCK_RE = re.compile(
    r"^\s*```(?:[A-Za-z0-9_-]+)?\s*\n(?P<body>[\s\S]*?)\n```\s*$",
    flags=re.IGNORECASE,
)


def _safe_ext(value: str) -> str:
    ext = value.strip().lstrip(".")
    if not ext:
        return "md"
    if not _SAFE_EXT_RE.match(ext):
        raise ToolExecutionError(f"export.write_text: invalid ext={value!r}")
    return ext.lower()


def _safe_stem(value: str) -> str:
    stem = value.strip()
    if not stem:
        return "output"
    stem = _SAFE_STEM_RE.sub("_", stem).strip("._-")
    return stem or "output"


def _safe_filename(value: str) -> str:
    """Validate a filename (no directories), allowing a single extension."""
    name = value.strip()
    if not name:
        raise ToolExecutionError("export.write_text: filename is empty")
    # No path separators or traversal.
    if any(sep in name for sep in ("/", "\\", ":", "..")):
        raise ToolExecutionError(f"export.write_text: filename must not contain paths: {value!r}")
    if not _SAFE_FILENAME_RE.match(name):
        raise ToolExecutionError(f"export.write_text: invalid filename={value!r}")
    return name


def _pick_text_payload(args: dict[str, Any], *, kind: str) -> str:
    """Pick best text candidate from merged tool args.

    For summary-like exports, prefer model output (`summary`/`content`)
    over raw extractor text (`text`) to avoid exporting source dumps.
    """
    summary_kinds = {"news_summary", "web_summary", "summary"}
    if kind in summary_kinds:
        candidate_keys = ("summary", "content", "text")
    else:
        candidate_keys = ("text", "content", "summary")

    for key in candidate_keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ToolExecutionError("export.write_text: missing text (expected 'text'/'content'/'summary')")


def _normalize_html_candidate(text: str) -> str:
    """Unwrap fenced HTML blocks and keep deterministic formatting."""
    s = text.strip()
    match = _FENCED_BLOCK_RE.match(s)
    if match:
        s = match.group("body").strip()
    return s


def invoke_export_write_text(
    args: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Write a text file into `project.data_dir/exports` and return its path.

    Supports any text extension, e.g. md/html/txt/py.
    """
    _policy = policy or {}
    if runtime is None or getattr(runtime, "config", None) is None:
        raise ToolExecutionError("export.write_text: ToolRuntime.config is required")
    config = runtime.config

    data_dir = getattr(getattr(config, "project", None), "data_dir", None)
    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ToolExecutionError("export.write_text: config.project.data_dir is missing")

    filename = args.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        filename = None
    if filename is not None:
        # filename includes extension, e.g. "summary.md"
        safe_name = _safe_filename(filename)
        stem = safe_name.rsplit(".", 1)[0] if "." in safe_name else safe_name
        ext = safe_name.rsplit(".", 1)[1] if "." in safe_name else "md"
        ext = _safe_ext(ext)
    else:
        ext = _safe_ext(str(args.get("ext") or args.get("format") or "md"))
        stem = _safe_stem(str(args.get("stem") or args.get("name") or "summary"))
    run_id = str(args.get("run_id") or "").strip()
    if not run_id:
        raise ToolExecutionError("export.write_text: missing 'run_id'")
    run_id = _safe_stem(run_id)[:32]

    kind = str(args.get("kind") or "text").strip()

    # Inputs can arrive via DAG auto-wiring (predecessor outputs).
    text = _pick_text_payload(args, kind=kind)

    title = args.get("title")
    if not isinstance(title, str) or not title.strip():
        title = args.get("title_guess")
    title = title.strip() if isinstance(title, str) and title.strip() else None

    source_url = args.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        source_url = args.get("final_url")
    if not isinstance(source_url, str) or not source_url.strip():
        source_url = args.get("url")
    source_url = source_url.strip() if isinstance(source_url, str) and source_url.strip() else None

    # Build file body deterministically.
    body: str
    if kind in {"news_summary", "web_summary", "summary"} and ext in {"md", "markdown"}:
        header = f"# {title}\n\n" if title else ""
        src = f"Источник: {source_url}\n\n" if source_url else ""
        body = header + src + text.strip() + "\n"
    elif kind in {"news_summary", "web_summary", "summary"} and ext in {"html", "htm"}:
        html_candidate = _normalize_html_candidate(text)
        low = html_candidate.lower().lstrip()
        if low.startswith("<!doctype html") or low.startswith("<html"):
            body = html_candidate + ("" if html_candidate.endswith("\n") else "\n")
        else:
            h = f"<h1>{title}</h1>\n" if title else ""
            src = (
                f"<p><strong>Источник:</strong> <a href='{source_url}'>{source_url}</a></p>\n"
                if source_url
                else ""
            )
            body = (
                "<!doctype html><meta charset='utf-8'>\n"
                + h
                + src
                + "<pre>\n"
                + html_candidate
                + "\n</pre>\n"
            )
    else:
        body = text.strip() + "\n"

    encoding = str(args.get("encoding", "utf-8"))
    overwrite = bool(args.get("overwrite", True))

    root = Path(data_dir) / "exports" / run_id
    root.mkdir(parents=True, exist_ok=True)
    out_path = (root / f"{stem}.{ext}").resolve()

    # Size guard consistent with fs.write_text
    data = body.encode(encoding, errors="replace")
    max_write_bytes = int(_policy.get("fs_max_write_bytes", 1_000_000))
    if len(data) > max_write_bytes:
        raise ToolExecutionError(
            f"export.write_text: payload too large ({len(data)} bytes > {max_write_bytes})"
        )

    if out_path.exists() and not overwrite:
        raise ToolExecutionError(
            f"export.write_text: target exists and overwrite is false: {str(out_path)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return {
        "path": str(out_path),
        "bytes_written": len(data),
        "ext": ext,
        "kind": kind,
        "title": title,
        "source_url": source_url,
    }

