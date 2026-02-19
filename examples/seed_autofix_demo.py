#!/usr/bin/env python3
"""Seed a SQLite index with deterministic replay data for the autofix demo.

Usage::

    python examples/seed_autofix_demo.py [--data-dir .neuronium]
    neuronium-agent replay --trace-id trace-seeded-autofix

The seeded trace encodes:

- Iteration 1: ``generate`` produces buggy code (``print(x)``),
  ``execute`` fails with NameError, ``critic`` returns FAIL.
- Iteration 2: ``fix`` produces corrected code (``x = 42\\nprint(x)``),
  ``execute_fix`` succeeds, ``critic_fix`` returns PASS with evidence.

All timestamps and IDs are fixed constants — no randomness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronium_agent.storage.sqlite_store import SqliteIndexStore


# -- Deterministic constants -------------------------------------------------

FIXED_TS = "2000-01-01T00:00:00+00:00"
TRACE_ID = "trace-seeded-autofix"
EXEC_ID = "exec-seeded"
OBJECTIVE = "Print the value of x"

BUGGY_CODE = "print(x)"
FIXED_CODE = "x = 42\nprint(x)"

REPLAY_DATA: dict[str, list[dict]] = {
    # -- Iteration 1 --
    "generate": [{
        "outputs": {"content": BUGGY_CODE},
        "quality_signals": {},
    }],
    "execute": [{
        "outputs": {
            "stdout": "",
            "stderr": (
                "Traceback (most recent call last):\n"
                '  File "<string>", line 1, in <module>\n'
                "NameError: name 'x' is not defined"
            ),
            "exit_code": 1,
        },
        "quality_signals": {},
        "status": "FAILED",
    }],
    "critic": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "FAIL",
                "confidence": 0.95,
                "evidence": ["exit_code=1", "NameError in stderr"],
                "gaps": ["Variable 'x' is not defined before use"],
            }, sort_keys=True),
        },
        "quality_signals": {},
    }],
    # -- Iteration 2 (fix-pipeline) --
    "fix": [{
        "outputs": {"content": FIXED_CODE},
        "quality_signals": {},
    }],
    "execute_fix": [{
        "outputs": {"stdout": "42\n", "exit_code": 0},
        "quality_signals": {},
        "status": "COMPLETED",
    }],
    "critic_fix": [{
        "outputs": {
            "content": json.dumps({
                "verdict": "PASS",
                "confidence": 0.99,
                "evidence": ["exit_code=0", "stdout contains '42'"],
                "gaps": [],
            }, sort_keys=True),
        },
        "quality_signals": {},
    }],
}


def seed(data_dir: str) -> None:
    """Write the seeded run + replay_data events into SQLite."""
    db_path = Path(data_dir) / "index.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteIndexStore(str(db_path))

    store.upsert_run(
        trace_id=TRACE_ID,
        execution_id=EXEC_ID,
        state="COMPLETED",
        objective=OBJECTIVE,
        config_snapshot_json="{}",
        created_at=FIXED_TS,
    )

    for node_id, responses in REPLAY_DATA.items():
        store.append_trace_event(
            TRACE_ID,
            {
                "ts": FIXED_TS,
                "span_id": f"seed-{node_id}",
                "kind": "replay_data",
                "payload": {
                    "node_id": node_id,
                    "recorded_responses": responses,
                },
            },
        )

    store.close()
    print(f"Seeded trace '{TRACE_ID}' in {db_path}")
    print(f"Run: neuronium-agent replay --trace-id {TRACE_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed autofix demo trace")
    parser.add_argument(
        "--data-dir",
        default=".neuronium",
        help="Data directory (default: .neuronium)",
    )
    args = parser.parse_args()
    seed(args.data_dir)


if __name__ == "__main__":
    main()
