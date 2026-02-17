"""Artifact tools for content-addressed evidence persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from neuronium_agent._canonical import artifact_id, canonical_bytes, canonical_json
from neuronium_agent.tools.local_tools import ToolExecutionError


def invoke_artifact_put_json(
    args: dict[str, Any],
    *,
    policy: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """Persist JSON payload as a content-addressed artifact."""
    _ = policy
    blob_store = getattr(runtime, "blob_store", None)
    index_store = getattr(runtime, "index_store", None)
    if blob_store is None or index_store is None:
        raise ToolExecutionError(
            "artifact.put_json: ToolRuntime must provide blob_store and index_store"
        )

    payload = args.get("json")
    if payload is None:
        raise ToolExecutionError("artifact.put_json: missing 'json'")

    artifact_type = str(args.get("artifact_type", "evidence.json")).strip() or "evidence.json"
    produced_by_node_ref = str(args.get("produced_by_node_ref", "artifact.put_json"))
    media_type = str(args.get("media_type", "application/json")).strip() or "application/json"
    parent_ids_raw = args.get("parent_artifact_ids", [])
    if parent_ids_raw is None:
        parent_ids_raw = []
    if not isinstance(parent_ids_raw, list):
        raise ToolExecutionError("artifact.put_json: 'parent_artifact_ids' must be a list")
    parent_ids = sorted(str(x) for x in parent_ids_raw if str(x).strip())

    content = canonical_bytes(payload)
    creation_context = {
        "produced_by_node_ref": produced_by_node_ref,
        "parent_artifact_ids": parent_ids,
    }
    aid = artifact_id(content, creation_context)
    now = datetime.now(timezone.utc).isoformat()

    blob_store.put(aid, content, media_type)
    index_store.record_artifact_metadata(
        artifact_id=aid,
        artifact_type=artifact_type,
        created_at=now,
        produced_by_node_ref=produced_by_node_ref,
        inputs_json=canonical_json({"parent_artifact_ids": parent_ids}),
        quality_signals_json="{}",
        blob_key=aid,
        media_type=media_type,
        size_bytes=len(content),
    )
    for parent_id in parent_ids:
        index_store.record_lineage_edge(parent_id, aid, "evidence")

    return {
        "artifact_id": aid,
        "artifact_type": artifact_type,
        "size_bytes": len(content),
        "media_type": media_type,
        "parent_artifact_ids": parent_ids,
        "canonical_json": canonical_json(payload),
    }
