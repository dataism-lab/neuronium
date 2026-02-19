"""Default model catalog and model resolution (B13).

When no [model_catalog] is set, get_default_catalog(llm) provides a single
entry so that model_id resolution and fallback behave consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from neuronium_agent.config import (
    LLMConfig,
    ModelCatalogConfig,
    ModelCatalogEntry,
)


def get_default_catalog(llm_config: LLMConfig) -> ModelCatalogConfig:
    """Build a catalog with a single 'default' entry from llm config."""
    return ModelCatalogConfig(
        default_model_id="default",
        models=[
            ModelCatalogEntry(
                id="default",
                provider=llm_config.provider,
                model=llm_config.model,
                api_key_env=None,
                base_url=llm_config.base_url,
                description="Default LLM from llm config",
            ),
        ],
    )


@dataclass(frozen=True)
class ResolvedModel:
    """Resolved model binding for a ModelNode (B13, Spec §5.2.1)."""

    model: str
    provider: str
    api_key_env: str
    base_url: str | None


def _is_available(entry: ModelCatalogEntry, llm_config: LLMConfig, env_get: Callable[[str, str], str | None]) -> bool:
    key_env = entry.api_key_env or llm_config.api_key_env
    val = env_get(key_env, "") or ""
    return len(val.strip()) > 0


def resolve_model_for_node(
    catalog: ModelCatalogConfig,
    llm_config: LLMConfig,
    model_id: str | None,
    *,
    env_get: Callable[[str, str], str | None] | None = None,
) -> ResolvedModel:
    """Resolve model_id to a concrete model binding with availability fallback.

    - If model_id is None or not in catalog, use catalog.default_model_id.
    - If the chosen entry is unavailable (no API key in env), try default_model_id.
    - If still unavailable, fall back to llm_config model/provider/api_key_env/base_url.
    """
    import os
    get_env = env_get if env_get is not None else os.environ.get
    by_id = {e.id: e for e in catalog.models}

    def entry_for(id_key: str) -> ModelCatalogEntry | None:
        return by_id.get(id_key)

    def to_resolved(entry: ModelCatalogEntry) -> ResolvedModel:
        return ResolvedModel(
            model=entry.model,
            provider=entry.provider,
            api_key_env=entry.api_key_env or llm_config.api_key_env,
            base_url=entry.base_url if entry.base_url is not None else llm_config.base_url,
        )

    # Prefer graph model_id, then default
    candidate_id = model_id if model_id and model_id.strip() else catalog.default_model_id
    entry = entry_for(candidate_id)
    if entry and _is_available(entry, llm_config, get_env):
        return to_resolved(entry)
    # Try default from catalog
    default_entry = entry_for(catalog.default_model_id)
    if default_entry and _is_available(default_entry, llm_config, get_env):
        return to_resolved(default_entry)
    # Fallback to llm config
    return ResolvedModel(
        model=llm_config.model,
        provider=llm_config.provider,
        api_key_env=llm_config.api_key_env,
        base_url=llm_config.base_url,
    )
