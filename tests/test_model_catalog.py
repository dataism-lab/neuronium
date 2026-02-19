"""B13 Model Catalog: default catalog, resolve, and registry integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from neuronium_agent.config import (
    AppConfig,
    LLMConfig,
    ModelCatalogConfig,
    ModelCatalogEntry,
    ProjectConfig,
    StorageConfig,
)
from neuronium_agent.model_catalog_defaults import (
    get_default_catalog,
    resolve_model_for_node,
)
from neuronium_agent.planning.dag import ActionGraph, GraphEdge, GraphMetadata, GraphNode
from neuronium_agent.storage.fs_cas import FsCasStore
from neuronium_agent.storage.sqlite_store import SqliteIndexStore


class TestGetDefaultCatalog:
    def test_returns_single_default_entry_from_llm(self) -> None:
        llm = LLMConfig(provider="openai", model="gpt-4.1-mini", api_key_env="NEURONIUM_OPENAI_API_KEY")
        catalog = get_default_catalog(llm)
        assert catalog.default_model_id == "default"
        assert len(catalog.models) == 1
        entry = catalog.models[0]
        assert entry.id == "default"
        assert entry.provider == "openai"
        assert entry.model == "gpt-4.1-mini"
        assert entry.api_key_env is None


class TestResolveModelForNode:
    def test_uses_entry_when_available(self) -> None:
        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[
                ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini"),
                ModelCatalogEntry(id="gpt4", provider="openai", model="gpt-4.1"),
            ],
        )
        llm = LLMConfig()
        env = {"NEURONIUM_OPENAI_API_KEY": "sk-test"}

        def env_get(key: str, default: str = "") -> str | None:
            return env.get(key, default)

        r = resolve_model_for_node(catalog, llm, "gpt4", env_get=env_get)
        assert r.model == "gpt-4.1"
        assert r.provider == "openai"

    def test_fallback_to_default_when_model_id_unknown(self) -> None:
        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[
                ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini"),
            ],
        )
        llm = LLMConfig()
        env = {"NEURONIUM_OPENAI_API_KEY": "sk-test"}

        r = resolve_model_for_node(
            catalog, llm, "nonexistent", env_get=lambda k, d="": env.get(k, d)
        )
        assert r.model == "gpt-4.1-mini"

    def test_fallback_to_llm_when_api_key_missing(self) -> None:
        catalog = ModelCatalogConfig(
            default_model_id="other",
            models=[
                ModelCatalogEntry(id="other", provider="openai", model="gpt-4.1", api_key_env="OTHER_KEY"),
            ],
        )
        llm = LLMConfig(model="gpt-4.1-mini", api_key_env="NEURONIUM_OPENAI_API_KEY")
        env = {"NEURONIUM_OPENAI_API_KEY": "sk-llm"}  # other key not set

        r = resolve_model_for_node(catalog, llm, "other", env_get=lambda k, d="": env.get(k, d))
        assert r.model == "gpt-4.1-mini"
        assert r.api_key_env == "NEURONIUM_OPENAI_API_KEY"

    def test_none_model_id_uses_default_model_id(self) -> None:
        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini")],
        )
        llm = LLMConfig()
        env = {"NEURONIUM_OPENAI_API_KEY": "sk-x"}

        r = resolve_model_for_node(catalog, llm, None, env_get=lambda k, d="": env.get(k, d))
        assert r.model == "gpt-4.1-mini"

    def test_empty_string_model_id_treated_as_default(self) -> None:
        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini")],
        )
        llm = LLMConfig()
        env = {"NEURONIUM_OPENAI_API_KEY": "sk-x"}

        r = resolve_model_for_node(catalog, llm, "  ", env_get=lambda k, d="": env.get(k, d))
        assert r.model == "gpt-4.1-mini"


class TestOrchestratorRegistryModelCatalog:
    """Integration: orchestrator _build_node_registry uses catalog and model_id from graph."""

    def test_registry_uses_resolved_model_when_model_id_in_graph(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-key")
        from neuronium_agent.core.orchestrator import Orchestrator

        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[
                ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini"),
                ModelCatalogEntry(id="gpt4", provider="openai", model="gpt-4.1"),
            ],
        )
        config = AppConfig(
            model_catalog=catalog,
            project=ProjectConfig(name="t", data_dir=str(tmp_path / ".n")),
            storage=StorageConfig(
                fs_cas_root=str(tmp_path / "blobs"),
                sqlite_path=str(tmp_path / "idx.sqlite3"),
            ),
        )
        blob = FsCasStore(config.storage.fs_cas_root)
        idx = SqliteIndexStore(config.storage.sqlite_path)
        orch = Orchestrator(config, blob, idx)
        graph = ActionGraph(
            metadata=GraphMetadata(plan_id="p1", description=""),
            nodes=[
                GraphNode(
                    node_id="m1",
                    node_type="model",
                    label="Test model",
                    parameters={"model_id": "gpt4", "system_prompt": "You are helpful."},
                ),
            ],
            edges=[],
        )
        registry = orch._build_node_registry(graph)
        assert "m1" in registry
        node = registry["m1"]
        assert getattr(node, "model", None) == "gpt-4.1"

    def test_registry_uses_default_when_no_model_id_in_graph(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-key")
        from neuronium_agent.core.orchestrator import Orchestrator

        config = AppConfig(  # model_catalog=None -> get_default_catalog used
            project=ProjectConfig(name="t", data_dir=str(tmp_path / ".n")),
            storage=StorageConfig(
                fs_cas_root=str(tmp_path / "blobs"),
                sqlite_path=str(tmp_path / "idx.sqlite3"),
            ),
        )
        blob = FsCasStore(config.storage.fs_cas_root)
        idx = SqliteIndexStore(config.storage.sqlite_path)
        orch = Orchestrator(config, blob, idx)
        graph = ActionGraph(
            metadata=GraphMetadata(plan_id="p1", description=""),
            nodes=[
                GraphNode(
                    node_id="m1",
                    node_type="model",
                    label="Test model",
                    parameters={"system_prompt": "You are helpful."},
                ),
            ],
            edges=[],
        )
        registry = orch._build_node_registry(graph)
        assert "m1" in registry
        node = registry["m1"]
        assert getattr(node, "model", None) == config.llm.model

    def test_registry_uses_stage_default_model_id_when_no_model_id_in_graph(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """B13 Part 2: stage_default_model_id is used for model nodes without parameters.model_id."""
        monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-key")
        from neuronium_agent.core.orchestrator import Orchestrator

        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[
                ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini"),
                ModelCatalogEntry(id="gpt4", provider="openai", model="gpt-4.1"),
            ],
        )
        config = AppConfig(
            model_catalog=catalog,
            project=ProjectConfig(name="t", data_dir=str(tmp_path / ".n")),
            storage=StorageConfig(
                fs_cas_root=str(tmp_path / "blobs"),
                sqlite_path=str(tmp_path / "idx.sqlite3"),
            ),
        )
        blob = FsCasStore(config.storage.fs_cas_root)
        idx = SqliteIndexStore(config.storage.sqlite_path)
        orch = Orchestrator(config, blob, idx)
        graph = ActionGraph(
            metadata=GraphMetadata(plan_id="p1", description=""),
            nodes=[
                GraphNode(
                    node_id="m1",
                    node_type="model",
                    label="Test model",
                    parameters={"system_prompt": "You are helpful."},
                ),
            ],
            edges=[],
        )
        registry = orch._build_node_registry(
            graph, stage_default_model_id="gpt4"
        )
        assert "m1" in registry
        node = registry["m1"]
        assert getattr(node, "model", None) == "gpt-4.1"

    def test_registry_stage_default_model_id_none_uses_catalog_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """B13 Part 2: stage_default_model_id=None preserves previous behavior (catalog default)."""
        monkeypatch.setenv("NEURONIUM_OPENAI_API_KEY", "test-key")
        from neuronium_agent.core.orchestrator import Orchestrator

        catalog = ModelCatalogConfig(
            default_model_id="default",
            models=[
                ModelCatalogEntry(id="default", provider="openai", model="gpt-4.1-mini"),
            ],
        )
        config = AppConfig(
            model_catalog=catalog,
            project=ProjectConfig(name="t", data_dir=str(tmp_path / ".n")),
            storage=StorageConfig(
                fs_cas_root=str(tmp_path / "blobs"),
                sqlite_path=str(tmp_path / "idx.sqlite3"),
            ),
        )
        blob = FsCasStore(config.storage.fs_cas_root)
        idx = SqliteIndexStore(config.storage.sqlite_path)
        orch = Orchestrator(config, blob, idx)
        graph = ActionGraph(
            metadata=GraphMetadata(plan_id="p1", description=""),
            nodes=[
                GraphNode(
                    node_id="m1",
                    node_type="model",
                    label="Test model",
                    parameters={},
                ),
            ],
            edges=[],
        )
        registry = orch._build_node_registry(graph, stage_default_model_id=None)
        assert "m1" in registry
        node = registry["m1"]
        assert getattr(node, "model", None) == "gpt-4.1-mini"
