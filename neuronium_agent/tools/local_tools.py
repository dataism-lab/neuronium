"""Local tool registry used by McpToolNode (v0.2).

This provides an in-process "local transport" that mimics MCP tool calls
without implementing the MCP protocol yet. It is intentionally small and
deterministic to support strict replay and easy codegen.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolPolicyError(RuntimeError):
    pass


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    tool_args: dict[str, Any]


def _normalize_path(p: str) -> Path:
    # Resolve without requiring the path to exist.
    return Path(p).expanduser().resolve()


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _ensure_allowed_path(path: Path, *, roots_allowlist: list[str]) -> None:
    if not roots_allowlist:
        raise ToolPolicyError("fs_roots_allowlist is empty (deny by default)")
    roots = [_normalize_path(r) for r in roots_allowlist]
    if not any(_is_under_root(path, rt) for rt in roots):
        raise ToolPolicyError(
            f"Path not allowed by fs_roots_allowlist: {str(path)}"
        )


def invoke_local_tool(
    call: ToolCall,
    *,
    policy: dict[str, Any] | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute a local tool call and return JSON-serializable outputs.

    Parameters
    ----------
    runtime:
        Optional :class:`~neuronium_agent.tools.runtime.ToolRuntime` for
        tools that need access to stores/config (e.g. ``memory.*``).
    """
    policy = policy or {}
    tool_name = call.tool_name
    args = call.tool_args or {}

    # -- Memory tools (require runtime) ------------------------------------
    if tool_name.startswith("memory."):
        from neuronium_agent.tools.memory_tools import (
            invoke_memory_ingest_files,
            invoke_memory_query,
        )

        if tool_name == "memory.ingest_files":
            return invoke_memory_ingest_files(
                args, policy=policy, runtime=runtime,
            )
        if tool_name == "memory.query":
            return invoke_memory_query(
                args, policy=policy, runtime=runtime,
            )
        raise ToolExecutionError(f"Unknown memory tool: {tool_name}")

    # -- Web tools -----------------------------------------------------------
    if tool_name.startswith("web."):
        from neuronium_agent.tools.web_tools import (
            invoke_web_extract_article,
            invoke_web_fetch_html,
        )

        if tool_name == "web.fetch_html":
            return invoke_web_fetch_html(args, policy=policy, runtime=runtime)
        if tool_name == "web.extract_article":
            return invoke_web_extract_article(args, policy=policy, runtime=runtime)
        raise ToolExecutionError(f"Unknown web tool: {tool_name}")

    # -- Text tools ----------------------------------------------------------
    if tool_name.startswith("text."):
        from neuronium_agent.tools.text_tools import invoke_text_extract_entities

        if tool_name == "text.extract_entities":
            return invoke_text_extract_entities(args, policy=policy, runtime=runtime)
        raise ToolExecutionError(f"Unknown text tool: {tool_name}")

    # -- Artifact tools ------------------------------------------------------
    if tool_name.startswith("artifact."):
        from neuronium_agent.tools.artifact_tools import invoke_artifact_put_json

        if tool_name == "artifact.put_json":
            return invoke_artifact_put_json(args, policy=policy, runtime=runtime)
        raise ToolExecutionError(f"Unknown artifact tool: {tool_name}")

    # -- Export tools --------------------------------------------------------
    if tool_name.startswith("export."):
        from neuronium_agent.tools.export_tools import invoke_export_write_text

        if tool_name == "export.write_text":
            return invoke_export_write_text(args, policy=policy, runtime=runtime)
        raise ToolExecutionError(f"Unknown export tool: {tool_name}")

    if tool_name == "fs.read_text":
        raw_path = str(args.get("path", ""))
        if not raw_path:
            raise ToolExecutionError("fs.read_text: missing 'path'")
        encoding = str(args.get("encoding", "utf-8"))
        out_key = args.get("out_key")

        p = _normalize_path(raw_path)
        roots_allowlist = list(policy.get("fs_roots_allowlist", []))
        _ensure_allowed_path(p, roots_allowlist=roots_allowlist)

        # Size guard: keep prompts bounded and avoid huge reads.
        max_bytes = int(policy.get("fs_max_read_bytes", 1_000_000))
        data = p.read_bytes()
        if len(data) > max_bytes:
            raise ToolExecutionError(
                f"fs.read_text: file too large ({len(data)} bytes > {max_bytes})"
            )
        text = data.decode(encoding, errors="replace")

        if isinstance(out_key, str) and out_key:
            return {
                out_key: text,
                f"{out_key}__path": str(p),
            }
        return {"text": text, "path": str(p)}

    if tool_name == "fs.write_text":
        raw_path = str(args.get("path", ""))
        if not raw_path:
            raise ToolExecutionError("fs.write_text: missing 'path'")
        text = args.get("text")
        if not isinstance(text, str):
            raise ToolExecutionError("fs.write_text: missing or invalid 'text'")
        encoding = str(args.get("encoding", "utf-8"))
        overwrite = bool(args.get("overwrite", True))

        p = _normalize_path(raw_path)
        roots_allowlist = list(policy.get("fs_roots_allowlist", []))
        _ensure_allowed_path(p, roots_allowlist=roots_allowlist)

        data = text.encode(encoding, errors="replace")
        max_write_bytes = int(policy.get("fs_max_write_bytes", 1_000_000))
        if len(data) > max_write_bytes:
            raise ToolExecutionError(
                f"fs.write_text: payload too large ({len(data)} bytes > {max_write_bytes})"
            )

        if p.exists() and not overwrite:
            raise ToolExecutionError(
                f"fs.write_text: target exists and overwrite is false: {str(p)}"
            )

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return {"path": str(p), "bytes_written": len(data)}

    if tool_name == "fs.list_dir":
        raw_path = str(args.get("path", ""))
        if not raw_path:
            raise ToolExecutionError("fs.list_dir: missing 'path'")
        p = _normalize_path(raw_path)
        roots_allowlist = list(policy.get("fs_roots_allowlist", []))
        _ensure_allowed_path(p, roots_allowlist=roots_allowlist)
        if not p.exists():
            raise ToolExecutionError("fs.list_dir: path does not exist")
        if not p.is_dir():
            raise ToolExecutionError("fs.list_dir: path is not a directory")
        entries = sorted([e.name for e in p.iterdir()])
        return {"path": str(p), "entries": entries}

    if tool_name == "fs.glob":
        raw_root = str(args.get("root", ""))
        pattern = str(args.get("pattern", ""))
        if not raw_root or not pattern:
            raise ToolExecutionError("fs.glob: requires 'root' and 'pattern'")
        root = _normalize_path(raw_root)
        roots_allowlist = list(policy.get("fs_roots_allowlist", []))
        _ensure_allowed_path(root, roots_allowlist=roots_allowlist)
        # Deterministic output ordering.
        paths = sorted(glob.glob(str(root / pattern), recursive=True))
        # Only return paths under root.
        safe: list[str] = []
        for s in paths:
            p = _normalize_path(s)
            if _is_under_root(p, root):
                safe.append(str(p))
        return {"root": str(root), "pattern": pattern, "paths": safe}

    raise ToolExecutionError(f"Unknown local tool: {tool_name}")

