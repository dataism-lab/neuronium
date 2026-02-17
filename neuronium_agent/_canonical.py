"""Canonical JSON serialization and content-addressed artifact IDs.

Rules (IBS §3.1):
- sorted keys
- no NaN / Infinity
- stable number representation
- UTF-8 encoding
- compact separators (no extra whitespace)
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, date
from typing import Any


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Canonical JSON forbids NaN/Infinity, got {obj!r}"
            )
        # normalise -0.0 → 0.0
        if obj == 0.0:
            return 0.0
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump(mode="json")
    raise TypeError(f"Cannot canonically serialize {type(obj).__name__}")


def _check_special_floats(obj: Any) -> None:
    """Recursively check for NaN/Infinity in *obj* before json.dumps.

    json.dumps handles float natively and never calls the ``default``
    callback for them, so we must validate up-front.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ValueError("Canonical JSON forbids NaN/Infinity, got NaN")
        if math.isinf(obj):
            raise ValueError(
                "Canonical JSON forbids NaN/Infinity, got Infinity"
            )
    elif isinstance(obj, dict):
        for v in obj.values():
            _check_special_floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check_special_floats(v)


def canonical_json(obj: Any) -> str:
    """Return a canonical JSON string (deterministic, sorted keys, compact)."""
    _check_special_floats(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_default_serializer,
        allow_nan=False,
    )


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON as UTF-8 bytes."""
    return canonical_json(obj).encode("utf-8")


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def artifact_id(content_bytes: bytes, creation_context: dict[str, Any]) -> str:
    """Compute a content-addressed Artifact ID.

    Format: ``sha256:<hex>``

    Hash input = canonical(content_bytes) + canonical(creation_context).
    """
    ctx_bytes = canonical_bytes(creation_context)
    h = hashlib.sha256()
    h.update(content_bytes)
    h.update(ctx_bytes)
    return f"sha256:{h.hexdigest()}"
