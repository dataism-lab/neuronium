"""Deterministic text parsing tools for planning-time extraction."""

from __future__ import annotations

import re
from typing import Any

from neuronium_agent.tools.local_tools import ToolExecutionError

_URL_RE = re.compile(r"(?P<url>https?://[^\s<>\"]+)", flags=re.IGNORECASE)
_PROTO_REL_URL_RE = re.compile(r"(?P<url>//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s<>\"]*)")
_UNIX_PATH_RE = re.compile(r"(?P<path>(?:\./|\.\./|/(?!/))[^\s\"']+)")
_WIN_PATH_RE = re.compile(r"(?P<path>[A-Za-z]:\\[^\s\"']+)")


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = v.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def invoke_text_extract_entities(
    args: dict[str, Any],
    *,
    policy: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """Extract candidate entities from free-form text deterministically."""
    _ = policy, runtime
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolExecutionError("text.extract_entities: missing 'text'")

    url_matches = list(_URL_RE.finditer(text))
    proto_matches = list(_PROTO_REL_URL_RE.finditer(text))
    urls = _stable_unique([m.group("url") for m in url_matches])
    proto_rel_urls = [m.group("url") for m in proto_matches]
    # Normalize protocol-relative URLs for downstream web tools.
    urls = _stable_unique(urls + [f"https:{u}" for u in proto_rel_urls])

    protected_spans = [m.span() for m in url_matches] + [m.span() for m in proto_matches]

    def _in_protected_span(start: int, end: int) -> bool:
        for p_start, p_end in protected_spans:
            if start >= p_start and end <= p_end:
                return True
        return False

    unix_paths = [
        m.group("path")
        for m in _UNIX_PATH_RE.finditer(text)
        if not _in_protected_span(*m.span())
    ]
    win_paths = [
        m.group("path")
        for m in _WIN_PATH_RE.finditer(text)
        if not _in_protected_span(*m.span())
    ]
    file_paths = _stable_unique(unix_paths + win_paths)

    return {
        "urls": urls,
        "file_paths": file_paths,
        # Basenames are intentionally not extracted deterministically.
        "basenames": [],
    }
