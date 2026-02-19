"""ToolRuntime — dependency-injection context for local tools (Stage 5).

Holds references to system-level services that memory tools need
(config, stores) without creating global singletons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuronium_agent.config import AppConfig
    from neuronium_agent.memory.store import MemoryStore
    from neuronium_agent.storage.blob_store import BlobStore
    from neuronium_agent.storage.index_store import IndexStore


@dataclass
class ToolRuntime:
    """Lightweight context passed through McpToolNode → local tools."""

    config: AppConfig | None = None
    index_store: IndexStore | None = None
    blob_store: BlobStore | None = None
    memory_store: MemoryStore | None = None
    extras: dict[str, Any] = field(default_factory=dict)
