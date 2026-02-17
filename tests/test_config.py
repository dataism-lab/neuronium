"""Config loading tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from neuronium_agent.config import AppConfig, load_config
from neuronium_agent.errors import ConfigError


class TestAppConfigDefaults:
    """Default config values match CONFIG_SPEC."""

    def test_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.project.name == "neuronium"
        assert cfg.project.data_dir == ".neuronium"
        assert cfg.storage.blob_backend == "fs_cas"
        assert cfg.storage.index_backend == "sqlite"
        assert cfg.queue.enabled is False
        assert cfg.llm.provider == "openai"
        assert cfg.llm.model == "gpt-4.1-mini"
        assert cfg.determinism.llm_temperature == 0.0
        assert cfg.determinism.default_random_seed == 0
        assert cfg.code_node.docker.network_enabled is False
        assert cfg.memory.semantic_search.enabled is False


class TestConfigLoading:
    """Config loading with TOML file + env overrides."""

    def test_load_from_toml(self, tmp_path: Path) -> None:
        toml_content = """
[project]
name = "my-project"
data_dir = "/tmp/test"

[storage]
index_backend = "sqlite"
sqlite_path = "/tmp/test/index.db"
"""
        toml_file = tmp_path / "neuronium.toml"
        toml_file.write_text(toml_content, encoding="utf-8")

        cfg = load_config(config_path=str(toml_file))
        assert cfg.project.name == "my-project"
        assert cfg.storage.sqlite_path == "/tmp/test/index.db"

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEURONIUM_STORAGE_INDEX_BACKEND", "sqlite")
        monkeypatch.setenv("NEURONIUM_QUEUE_ENABLED", "true")
        monkeypatch.setenv("NEURONIUM_RUNTIME_MODE", "supervised")

        # No config file
        cfg = load_config(config_path="nonexistent.toml")
        assert cfg.queue.enabled is True
        assert cfg.runtime.mode == "supervised"

    def test_cli_overrides_win(self) -> None:
        cfg = load_config(
            cli_overrides={"runtime": {"mode": "supervised"}}
        )
        assert cfg.runtime.mode == "supervised"

    def test_priority_cli_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEURONIUM_RUNTIME_MODE", "supervised")
        cfg = load_config(cli_overrides={"runtime": {"mode": "batch"}})
        assert cfg.runtime.mode == "batch"
