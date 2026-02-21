#!/usr/bin/env python3
"""
Diagnostic script for inspecting a specific Neuronium run from the SQLite DB.

Usage:
    # List recent runs (default 10)
    python scripts/debug_run.py --list
    python scripts/debug_run.py --list --limit 20

    # Full diagnosis for a run (prefix match on trace_id)
    python scripts/debug_run.py 1e40d4f2

    # Only specific sections
    python scripts/debug_run.py 1e40d4f2 --section events
    python scripts/debug_run.py 1e40d4f2 --section llm
    python scripts/debug_run.py 1e40d4f2 --section clarification
    python scripts/debug_run.py 1e40d4f2 --section metadata

    # Custom DB path (default: .neuronium/index.sqlite3)
    python scripts/debug_run.py 1e40d4f2 --db path/to/index.sqlite3
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DEFAULT_DB = ".neuronium/index.sqlite3"
SEPARATOR = "=" * 80


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        print(f"ERROR: DB not found at {path.resolve()}", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(str(path))


def _json_pretty(data: object, max_chars: int = 0) -> str:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n  ... (truncated)"
    return text


def _resolve_trace_id(conn: sqlite3.Connection, prefix: str) -> str:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT trace_id FROM runs WHERE trace_id LIKE ? ORDER BY created_at DESC LIMIT 2",
        (prefix + "%",),
    )
    rows = cursor.fetchall()
    if not rows:
        print(f"ERROR: no run found matching prefix '{prefix}'", file=sys.stderr)
        sys.exit(1)
    if len(rows) > 1:
        print(f"WARNING: multiple runs match '{prefix}', using most recent", file=sys.stderr)
    return rows[0][0]


# ── List runs ────────────────────────────────────────────────────────────────

def cmd_list(conn: sqlite3.Connection, limit: int) -> None:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT trace_id, created_at, state, objective FROM runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    if not rows:
        print("No runs found.")
        return
    print(f"{'trace_id':<36s}  {'state':<10s}  {'created_at':<28s}  objective")
    print("-" * 120)
    for trace_id, created_at, state, objective in rows:
        obj_short = (objective or "")[:60]
        print(f"{trace_id:<36s}  {state:<10s}  {created_at:<28s}  {obj_short}")


# ── Trace events summary ────────────────────────────────────────────────────

def section_events(conn: sqlite3.Connection, trace_id: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"TRACE EVENTS for {trace_id}")
    print(SEPARATOR)

    cursor = conn.cursor()
    cursor.execute(
        "SELECT event_id, ts, kind, payload_json FROM trace_events WHERE trace_id = ? ORDER BY event_id",
        (trace_id,),
    )
    for event_id, ts, kind, payload_json in cursor.fetchall():
        payload = json.loads(payload_json) if payload_json else {}

        parts: list[str] = []
        desc = payload.get("description", "")
        if desc:
            parts.append(desc[:100])
        phase = payload.get("phase", "")
        if phase:
            parts.append(f"phase={phase}")
        ctrl = payload.get("control_command") or payload.get("command", "")
        if ctrl:
            parts.append(f"control={ctrl}")
        if "missing_fields" in payload:
            fields = [mf.get("field", "?") for mf in payload["missing_fields"] if isinstance(mf, dict)]
            parts.append(f"missing=[{', '.join(fields)}]")
        if "answers" in payload and isinstance(payload["answers"], dict):
            keys = list(payload["answers"].keys())
            parts.append(f"answer_keys={keys}")
        if "metadata" in payload and isinstance(payload["metadata"], dict) and payload["metadata"]:
            parts.append(f"meta_keys={list(payload['metadata'].keys())}")

        node_id = payload.get("node_id", "")
        if node_id and kind in ("node_start", "node_end"):
            elapsed = payload.get("elapsed_ms", "")
            status = payload.get("status", "")
            extra = f" [{status}]" if status else ""
            extra += f" {elapsed}ms" if elapsed != "" else ""
            parts.insert(0, f"{node_id}{extra}")

        summary = " | ".join(parts) if parts else json.dumps(payload, ensure_ascii=False)[:150]
        print(f"  [{event_id:5d}] {kind:<25s} {summary}")


# ── LLM calls detail ────────────────────────────────────────────────────────

def section_llm(conn: sqlite3.Connection, trace_id: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"LLM CALLS for {trace_id}")
    print(SEPARATOR)

    cursor = conn.cursor()
    cursor.execute(
        """SELECT event_id, kind, payload_json FROM trace_events
           WHERE trace_id = ? AND kind IN ('node_start', 'node_end')
           ORDER BY event_id""",
        (trace_id,),
    )

    # Pair start/end events sequentially: each node_start for a model node
    # is matched with the next node_end sharing the same node_id.
    pending_starts: dict[str, list[tuple[int, dict]]] = {}
    calls: list[tuple[int, dict, int, dict]] = []  # (start_eid, start_p, end_eid, end_p)

    for event_id, kind, pj in cursor.fetchall():
        p = json.loads(pj) if pj else {}
        nid = p.get("node_id", "")
        if kind == "node_start" and p.get("node_type") == "model":
            pending_starts.setdefault(nid, []).append((event_id, p))
        elif kind == "node_end" and nid in pending_starts and pending_starts[nid]:
            start_eid, start_p = pending_starts[nid].pop(0)
            calls.append((start_eid, start_p, event_id, p))

    if not calls:
        print("  (no LLM calls found)")
        return

    for call_num, (start_eid, start_p, end_eid, end_p) in enumerate(calls, 1):
        node_id = start_p.get("node_id", "")
        prompt = start_p.get("inputs", {}).get("prompt", "")
        sys_prompt = start_p.get("parameters", {}).get("system_prompt", "")
        schema = start_p.get("parameters", {}).get("json_schema")

        print(f"\n  -- LLM Call #{call_num}: {node_id} (events {start_eid}->{end_eid}) --")
        if sys_prompt:
            print(f"  SYSTEM: {sys_prompt}")
        if prompt:
            print(f"  PROMPT:\n{textwrap.indent(prompt, '    ')}")
        if schema:
            required = schema.get("required", [])
            props = list(schema.get("properties", {}).keys())
            print(f"  SCHEMA: required={required}, props={props}")

        elapsed = end_p.get("elapsed_ms", "?")
        tokens = end_p.get("quality_signals", {}).get("tokens_used", "?")
        status = end_p.get("status", "?")
        raw_content = end_p.get("outputs", {}).get("content", "")
        parsed = end_p.get("outputs", {}).get("parsed")

        print(f"  RESPONSE: status={status}, {elapsed}ms, {tokens} tokens")
        if parsed:
            print(f"  PARSED:\n{textwrap.indent(_json_pretty(parsed, 2000), '    ')}")
        elif raw_content:
            print(f"  RAW:\n{textwrap.indent(raw_content[:2000], '    ')}")


# ── Clarification flow ──────────────────────────────────────────────────────

def section_clarification(conn: sqlite3.Connection, trace_id: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"CLARIFICATION FLOW for {trace_id}")
    print(SEPARATOR)

    cursor = conn.cursor()
    cursor.execute(
        """SELECT event_id, kind, payload_json FROM trace_events
           WHERE trace_id = ?
           ORDER BY event_id""",
        (trace_id,),
    )

    escalations = []
    revises = []
    checkpoints_meta = []

    for event_id, kind, pj in cursor.fetchall():
        p = json.loads(pj) if pj else {}

        if kind == "decision" and "missing_fields" in p:
            escalations.append((event_id, p))
        if kind == "decision" and p.get("command") == "revise":
            revises.append((event_id, p))
        if kind == "checkpoint":
            meta = p.get("metadata")
            if isinstance(meta, dict):
                checkpoints_meta.append((event_id, meta))

    if not escalations and not revises:
        print("  (no clarification events found)")
        return

    for eid, p in escalations:
        mfs = p.get("missing_fields", [])
        clar_aid = p.get("clarification_request_artifact_id", "")
        desc = p.get("description", "")
        print(f"\n  -- Escalation (event {eid}): {desc} --")
        print(f"  clarification_request_artifact_id: {clar_aid}")
        print("  Missing fields:")
        for mf in mfs:
            if isinstance(mf, dict):
                print(f"    - field={mf.get('field')!r}, critical={mf.get('critical')}, reason={mf.get('reason')!r}")

    for eid, p in revises:
        payload = p.get("payload", {})
        answers = payload.get("answers", {})
        clar_req = payload.get("clarification_request_artifact_id", "")
        clar_resp = payload.get("clarification_response_artifact_id", "")
        print(f"\n  -- Revise (event {eid}) --")
        print(f"  clarification_request_artifact_id:  {clar_req}")
        print(f"  clarification_response_artifact_id: {clar_resp}")
        print("  Answers:")
        for k, v in answers.items():
            print(f"    {k!r} -> {v!r}")

    if checkpoints_meta:
        print("\n  -- Checkpoint metadata snapshots --")
        for eid, meta in checkpoints_meta:
            if meta:
                print(f"  event {eid}: {_json_pretty(meta, 500)}")


# ── Metadata at resume ──────────────────────────────────────────────────────

def section_metadata(conn: sqlite3.Connection, trace_id: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"PLANNER REQUEST METADATA for {trace_id}")
    print(SEPARATOR)

    cursor = conn.cursor()
    cursor.execute(
        """SELECT event_id, payload_json FROM trace_events
           WHERE trace_id = ? AND kind = 'decision'
           ORDER BY event_id""",
        (trace_id,),
    )
    call_num = 0
    for event_id, pj in cursor.fetchall():
        p = json.loads(pj) if pj else {}
        if p.get("description") != "Planner request envelope":
            continue
        call_num += 1
        print(f"\n  -- Planner request #{call_num} (event {event_id}) --")
        for key in ("runbook_id", "stage_id", "planner_backend"):
            if key in p:
                print(f"  {key}: {p[key]}")

    # Show what the extraction prompt contained (Raw metadata) for each LLM extraction call
    cursor.execute(
        """SELECT event_id, payload_json FROM trace_events
           WHERE trace_id = ? AND kind = 'node_start'
           ORDER BY event_id""",
        (trace_id,),
    )
    ext_num = 0
    for event_id, pj in cursor.fetchall():
        p = json.loads(pj) if pj else {}
        node_id = p.get("node_id", "")
        if "extract_envelope" not in node_id:
            continue
        ext_num += 1
        prompt = p.get("inputs", {}).get("prompt", "")
        meta_line = ""
        for line in prompt.split("\n"):
            if line.startswith("Raw metadata:"):
                meta_line = line
                break
        print(f"\n  -- Extraction #{ext_num} prompt metadata (event {event_id}, {node_id}) --")
        if meta_line:
            raw_meta_str = meta_line.replace("Raw metadata: ", "", 1)
            try:
                raw_meta = json.loads(raw_meta_str)
                print(f"  Raw metadata:\n{textwrap.indent(_json_pretty(raw_meta, 1000), '    ')}")
            except json.JSONDecodeError:
                print(f"  Raw metadata (unparsed): {meta_line[:500]}")
        else:
            print("  (Raw metadata line not found in prompt)")

    # Show extraction outputs (what LLM returned as inputs)
    cursor.execute(
        """SELECT event_id, payload_json FROM trace_events
           WHERE trace_id = ? AND kind = 'node_end'
           ORDER BY event_id""",
        (trace_id,),
    )
    ext_num = 0
    for event_id, pj in cursor.fetchall():
        p = json.loads(pj) if pj else {}
        node_id = p.get("node_id", "")
        if "extract_envelope" not in node_id:
            continue
        ext_num += 1
        parsed = p.get("outputs", {}).get("parsed", {})
        inputs = parsed.get("inputs", {}) if isinstance(parsed, dict) else {}
        missing = parsed.get("missing_fields", []) if isinstance(parsed, dict) else []
        print(f"\n  -- Extraction #{ext_num} LLM output (event {event_id}) --")
        print(f"  inputs: {_json_pretty(inputs, 500)}")
        if missing:
            print(f"  missing_fields: {_json_pretty(missing, 500)}")


# ── Main ─────────────────────────────────────────────────────────────────────

ALL_SECTIONS = ("events", "llm", "clarification", "metadata")

SECTION_FNS = {
    "events": section_events,
    "llm": section_llm,
    "clarification": section_clarification,
    "metadata": section_metadata,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Neuronium run from the SQLite trace DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/debug_run.py --list
              python scripts/debug_run.py 1e40d4f2
              python scripts/debug_run.py 1e40d4f2 --section llm
              python scripts/debug_run.py 1e40d4f2 --section clarification --section metadata
        """),
    )
    parser.add_argument("trace_prefix", nargs="?", help="Prefix of trace_id to inspect")
    parser.add_argument("--list", action="store_true", help="List recent runs")
    parser.add_argument("--limit", type=int, default=10, help="Number of runs to list (default: 10)")
    parser.add_argument(
        "--section", action="append", choices=ALL_SECTIONS,
        help="Show only specific section(s). Can be repeated. Default: all.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Path to SQLite DB (default: {DEFAULT_DB})")

    args = parser.parse_args()

    conn = _connect(args.db)

    if args.list:
        cmd_list(conn, args.limit)
        return

    if not args.trace_prefix:
        parser.print_help()
        sys.exit(1)

    trace_id = _resolve_trace_id(conn, args.trace_prefix)
    print(f"Run: {trace_id}")

    cursor = conn.cursor()
    cursor.execute(
        "SELECT created_at, state, objective FROM runs WHERE trace_id = ?",
        (trace_id,),
    )
    row = cursor.fetchone()
    if row:
        print(f"Created: {row[0]}")
        print(f"State:   {row[1]}")
        print(f"Objective: {row[2]}")

    sections = args.section or list(ALL_SECTIONS)
    for s in sections:
        SECTION_FNS[s](conn, trace_id)

    conn.close()


if __name__ == "__main__":
    main()
