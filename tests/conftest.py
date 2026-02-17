"""Shared fixtures for neuronium_agent tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from neuronium_agent.config import AppConfig, StorageConfig, ProjectConfig
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture()
def blob_store(tmp_dir: Path) -> FsCasStore:
    """Create a temporary FS CAS blob store."""
    return FsCasStore(tmp_dir / "blobs")


@pytest.fixture()
def index_store(tmp_dir: Path) -> SqliteIndexStore:
    """Create a temporary SQLite index store."""
    return SqliteIndexStore(tmp_dir / "index.sqlite3")


@pytest.fixture()
def app_config(tmp_dir: Path) -> AppConfig:
    """Create a test AppConfig pointing to temp storage."""
    return AppConfig(
        project=ProjectConfig(
            name="test",
            data_dir=str(tmp_dir / ".neuronium"),
        ),
        storage=StorageConfig(
            blob_backend="fs_cas",
            fs_cas_root=str(tmp_dir / "blobs"),
            index_backend="sqlite",
            sqlite_path=str(tmp_dir / "index.sqlite3"),
        ),
    )
