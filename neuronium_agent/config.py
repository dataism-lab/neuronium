"""Configuration model and loader (CONFIG_SPEC §1-4).

Priority (highest → lowest): CLI flags → env vars → TOML file → defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from neuronium_agent.errors import ConfigError

try:
    # Optional at runtime, but included in core deps for OSS DX.
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    name: str = "neuronium"
    data_dir: str = ".neuronium"


class DeterminismConfig(BaseModel):
    canonical_json: str = "neuronium-v1"
    default_random_seed: int = 0
    llm_temperature: float = 0.0
    strict: bool = False
    """When True, reject nodes with declared_non_deterministic at registry build (Spec §1.2.1)."""
    mcp_allow_non_deterministic_tool_ids: list[str] = Field(default_factory=list)
    """Tool/node IDs allowed to be non-deterministic when strict is True (allowlist)."""


class RuntimeConfig(BaseModel):
    mode: Literal["batch", "supervised", "interactive"] = "batch"
    max_parallel_nodes: int = 4
    checkpoint_policy: Literal[
        "on_transition", "periodic", "node_boundary"
    ] = "on_transition"
    checkpoint_interval_seconds: int | None = None
    pause_grace_period_seconds: int = 30
    """Grace period for pause: allow active nodes to reach checkpoint (Spec §9.1.2)."""
    stop_grace_period_seconds: int = 5
    """Grace period for cooperative stop before forced termination (Spec §6.2.4)."""
    nl_patch_min_confidence: float = 0.5
    """Minimum confidence for applying NL->patch conversion in revise flow."""


class StorageConfig(BaseModel):
    blob_backend: Literal["fs_cas", "s3"] = "fs_cas"
    fs_cas_root: str = ".neuronium/blobs"
    index_backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = ".neuronium/index.sqlite3"
    postgres_dsn: str | None = None
    postgres_schema: str = "neuronium_agent"
    migrations_auto_apply: bool = True


class QueueConfig(BaseModel):
    enabled: bool = False
    backend: str = "rq"
    redis_url: str | None = None
    queue_name: str = "neuronium"
    job_timeout_seconds: int = 900
    result_ttl_seconds: int = 86400


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    base_url: str | None = None
    api_key_env: str = "NEURONIUM_OPENAI_API_KEY"
    structured_output: bool = True
    timeout_seconds: int = 60
    max_retries: int = 2


class ModelCatalogEntry(BaseModel):
    """Single model entry in the catalog (B13, Spec §5.2.1)."""

    id: str
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key_env: str | None = None
    """If None, fall back to llm.api_key_env."""
    base_url: str | None = None
    description: str | None = None


class ModelCatalogConfig(BaseModel):
    """Model catalog: default model id + list of entries (B13)."""

    default_model_id: str = "default"
    models: list[ModelCatalogEntry] = Field(default_factory=list)


class McpServerPolicy(BaseModel):
    fs_roots_allowlist: list[str] = Field(default_factory=list)
    network_allowlist: list[str] = Field(default_factory=list)
    require_approval_for: list[str] = Field(
        default_factory=lambda: [
            "destructive",
            "exfiltration_risk",
            "high_cost",
        ]
    )


class McpServerConfig(BaseModel):
    name: str
    url: str
    timeout_seconds: int = 60
    rate_limit_rps: float | None = None
    policy: McpServerPolicy = Field(default_factory=McpServerPolicy)


class McpConfig(BaseModel):
    enabled: bool = True
    servers: list[McpServerConfig] = Field(default_factory=list)


class DockerConfig(BaseModel):
    enabled: bool = True
    image: str = "python:3.11-slim"
    network_enabled: bool = False
    cpu_limit: str | None = None
    mem_limit: str | None = None
    timeout_seconds: int = 120
    fs_roots_allowlist: list[str] = Field(default_factory=list)


class CodeNodeConfig(BaseModel):
    enabled: bool = True
    runtime: str = "python"
    docker: DockerConfig = Field(default_factory=DockerConfig)


class PgVectorConfig(BaseModel):
    enabled: bool = False
    vector_dim: int = 1536


class LocalEmbeddingsConfig(BaseModel):
    enabled: bool = False
    embedding_provider: Literal["openai", "sentence_transformers"] = (
        "sentence_transformers"
    )
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dim: int | None = None
    index: Literal["bruteforce", "hnsw"] = "bruteforce"
    store_in_sqlite: bool = True


class SemanticSearchConfig(BaseModel):
    enabled: bool = False
    backend: Literal["pgvector", "local"] = "local"
    pgvector: PgVectorConfig = Field(default_factory=PgVectorConfig)
    local: LocalEmbeddingsConfig = Field(default_factory=LocalEmbeddingsConfig)


class MemoryConfig(BaseModel):
    enabled: bool = True
    graphrag_backend: Literal["sqlite", "postgres"] = "sqlite"
    semantic_search: SemanticSearchConfig = Field(
        default_factory=SemanticSearchConfig
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # NOTE: keep TOML/env compatibility with key name "json",
    # but avoid shadowing BaseModel.json() which triggers a warning.
    model_config = ConfigDict(populate_by_name=True)
    json_logs: bool = Field(default=True, alias="json")
    path: str = ".neuronium/logs/neuronium.jsonl"


class RecoveryConfig(BaseModel):
    """Recovery and retry policy (B1 Part 1 + B1 Part 2, B2 verdict local fix)."""

    max_node_retries: int = 3
    max_stage_retries: int = 2
    retry_backoff_base_seconds: float = 1.0
    retry_count_upgrade_threshold: int = 2  # After this many node retries → PERSISTENT
    # B1 Part 2: configurable escalation (§3.4.2)
    repeated_rollback_threshold: int = 3  # Same node fails this many times → escalate
    allow_auto_replan: bool = False  # When True, may return REPLAN instead of ESCALATE
    # B2 Part 1: max retries of stage with verdict fix context (gaps/suggestions) before normal recovery
    max_verdict_fix_attempts: int = 1


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Root application configuration (public)."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    determinism: DeterminismConfig = Field(default_factory=DeterminismConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    model_catalog: ModelCatalogConfig | None = None
    """If None, model nodes use llm.* only; no catalog resolution."""
    mcp: McpConfig = Field(default_factory=McpConfig)
    code_node: CodeNodeConfig = Field(default_factory=CodeNodeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_PREFIX = "NEURONIUM_"

_FLAT_ENV_MAP: dict[str, tuple[str, ...]] = {
    "NEURONIUM_PROJECT_NAME": ("project", "name"),
    "NEURONIUM_PROJECT_DATA_DIR": ("project", "data_dir"),
    "NEURONIUM_STORAGE_BLOB_BACKEND": ("storage", "blob_backend"),
    "NEURONIUM_STORAGE_FS_CAS_ROOT": ("storage", "fs_cas_root"),
    "NEURONIUM_STORAGE_INDEX_BACKEND": ("storage", "index_backend"),
    "NEURONIUM_STORAGE_SQLITE_PATH": ("storage", "sqlite_path"),
    "NEURONIUM_STORAGE_POSTGRES_DSN": ("storage", "postgres_dsn"),
    "NEURONIUM_STORAGE_POSTGRES_SCHEMA": ("storage", "postgres_schema"),
    "NEURONIUM_QUEUE_ENABLED": ("queue", "enabled"),
    "NEURONIUM_QUEUE_REDIS_URL": ("queue", "redis_url"),
    "NEURONIUM_QUEUE_QUEUE_NAME": ("queue", "queue_name"),
    "NEURONIUM_RUNTIME_MODE": ("runtime", "mode"),
    "NEURONIUM_RUNTIME_PAUSE_GRACE_PERIOD_SECONDS": ("runtime", "pause_grace_period_seconds"),
    "NEURONIUM_RUNTIME_STOP_GRACE_PERIOD_SECONDS": ("runtime", "stop_grace_period_seconds"),
    "NEURONIUM_RUNTIME_NL_PATCH_MIN_CONFIDENCE": ("runtime", "nl_patch_min_confidence"),
    "NEURONIUM_LLM_PROVIDER": ("llm", "provider"),
    "NEURONIUM_LLM_MODEL": ("llm", "model"),
    "NEURONIUM_LLM_BASE_URL": ("llm", "base_url"),
    "NEURONIUM_LOGGING_LEVEL": ("logging", "level"),
    "NEURONIUM_RECOVERY_MAX_NODE_RETRIES": ("recovery", "max_node_retries"),
    "NEURONIUM_RECOVERY_MAX_STAGE_RETRIES": ("recovery", "max_stage_retries"),
    "NEURONIUM_RECOVERY_MAX_VERDICT_FIX_ATTEMPTS": ("recovery", "max_verdict_fix_attempts"),
    "NEURONIUM_MODEL_CATALOG_DEFAULT_MODEL_ID": ("model_catalog", "default_model_id"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive copy)."""
    result = dict(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _set_nested(data: dict, path: tuple[str, ...], value: str) -> None:
    """Set a nested dict value by key path, coercing booleans."""
    d = data
    for part in path[:-1]:
        d = d.setdefault(part, {})
    raw: str | bool = value
    if value.lower() in ("true", "1", "yes"):
        raw = True
    elif value.lower() in ("false", "0", "no"):
        raw = False
    d[path[-1]] = raw


def _apply_env_overrides(data: dict) -> dict:
    """Apply NEURONIUM_* env vars over *data*."""
    result = dict(data)
    for env_key, path in _FLAT_ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            _set_nested(result, path, val)
    return result


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

_SEARCH_PATHS = ("neuronium.toml", "config/neuronium.toml")


def _maybe_load_dotenv(config_path: str | None) -> None:
    """Load `.env` for local development / CLI usage.

    We intentionally do not override already-defined environment variables.

    Search order:
    - next to an explicit `--config` TOML file (if provided)
    - current working directory
    """
    if load_dotenv is None:
        return

    candidates: list[Path] = []
    if config_path:
        try:
            candidates.append(Path(config_path).resolve().parent / ".env")
        except Exception:
            pass
    candidates.append(Path.cwd() / ".env")

    seen: set[str] = set()
    for p in candidates:
        ps = str(p)
        if ps in seen:
            continue
        seen.add(ps)
        if p.is_file():
            load_dotenv(dotenv_path=p, override=False)


def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load and validate configuration.

    Priority: CLI overrides > env vars > TOML file > built-in defaults.
    """
    _maybe_load_dotenv(config_path)
    toml_data: dict[str, Any] = {}

    # 1. TOML file -----------------------------------------------------------
    candidates = [config_path] if config_path else list(_SEARCH_PATHS)
    for candidate in candidates:
        if candidate is None:
            continue
        p = Path(candidate)
        if p.is_file():
            try:
                with p.open("rb") as fh:
                    toml_data = tomllib.load(fh)
            except Exception as exc:
                raise ConfigError(f"Failed to parse {p}: {exc}") from exc
            break

    # 2. Env overrides -------------------------------------------------------
    toml_data = _apply_env_overrides(toml_data)

    # 3. CLI overrides -------------------------------------------------------
    if cli_overrides:
        toml_data = _deep_merge(toml_data, cli_overrides)

    # 4. Build & validate ----------------------------------------------------
    try:
        return AppConfig(**toml_data)
    except Exception as exc:
        raise ConfigError(f"Config validation failed: {exc}") from exc
