"""Lightweight built-in schema migrator (STORAGE_SCHEMA §4).

Reads numbered SQL files from ``migrations/<dialect>/NNNN_*.sql`` and
applies them in order, tracking ``schema_version``.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _migration_dir(dialect: str) -> Path:
    """Return the absolute path to the migration SQL directory."""
    pkg = resources.files("neuronium_agent.storage.migrations").joinpath(dialect)
    return Path(str(pkg))


def _ensure_version_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version  INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT MAX(version) AS v FROM schema_version"
    ).fetchone()
    return row[0] if row and row[0] is not None else 0


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def apply_migrations(
    conn: sqlite3.Connection,
    dialect: str = "sqlite",
) -> int:
    """Apply pending migrations and return the new schema version.

    Parameters
    ----------
    conn:
        Open SQLite or Postgres-via-psycopg connection.
    dialect:
        ``"sqlite"`` or ``"postgres"`` — determines which SQL directory
        to read from.

    Returns
    -------
    int
        Schema version after applying all pending migrations.
    """
    _ensure_version_table(conn)
    current = _current_version(conn)

    mig_dir = _migration_dir(dialect)
    if not mig_dir.exists():
        return current

    sql_files = sorted(mig_dir.glob("*.sql"))
    for sql_file in sql_files:
        # Extract version number from filename, e.g. ``0001_init.sql → 1``
        version = int(sql_file.stem.split("_", maxsplit=1)[0])
        if version <= current:
            continue

        sql = sql_file.read_text(encoding="utf-8")
        conn.executescript(sql)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (version, now),
        )
        conn.commit()
        current = version

    return current
