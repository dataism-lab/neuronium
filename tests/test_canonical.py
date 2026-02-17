"""Tests for canonical JSON and artifact ID computation."""

from __future__ import annotations

import math

import pytest

from neuronium_agent._canonical import (
    artifact_id,
    canonical_bytes,
    canonical_json,
    content_hash,
)


class TestCanonicalJson:
    """Canonical JSON must produce deterministic, sorted, compact output."""

    def test_sorted_keys(self) -> None:
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        assert result == '{"a":2,"m":3,"z":1}'

    def test_compact_separators(self) -> None:
        obj = {"key": "value"}
        assert " " not in canonical_json(obj)

    def test_nested_sorting(self) -> None:
        obj = {"b": {"z": 1, "a": 2}, "a": 1}
        result = canonical_json(obj)
        assert result == '{"a":1,"b":{"a":2,"z":1}}'

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_json({"x": float("nan")})

    def test_infinity_raises(self) -> None:
        with pytest.raises(ValueError, match="Infinity"):
            canonical_json({"x": float("inf")})

    def test_deterministic(self) -> None:
        """Same input → same output, always."""
        obj = {"list": [3, 1, 2], "nested": {"b": True, "a": False}}
        r1 = canonical_json(obj)
        r2 = canonical_json(obj)
        assert r1 == r2

    def test_bytes_are_base64(self) -> None:
        obj = {"data": b"hello"}
        result = canonical_json(obj)
        assert "aGVsbG8=" in result  # base64 of "hello"

    def test_utf8_encoding(self) -> None:
        obj = {"text": "Привет мир"}
        b = canonical_bytes(obj)
        assert isinstance(b, bytes)
        assert "Привет мир" in b.decode("utf-8")


class TestArtifactId:
    """Artifact ID is a content-addressed hash."""

    def test_deterministic(self) -> None:
        content = b'{"key":"value"}'
        ctx = {"timestamp": "2026-01-01T00:00:00Z", "node_ref": "test/node"}
        id1 = artifact_id(content, ctx)
        id2 = artifact_id(content, ctx)
        assert id1 == id2

    def test_format(self) -> None:
        content = b"test"
        ctx = {"timestamp": "t"}
        aid = artifact_id(content, ctx)
        assert aid.startswith("sha256:")
        assert len(aid) == len("sha256:") + 64  # 64 hex chars

    def test_different_content_different_id(self) -> None:
        ctx = {"timestamp": "t"}
        id1 = artifact_id(b"aaa", ctx)
        id2 = artifact_id(b"bbb", ctx)
        assert id1 != id2

    def test_different_context_different_id(self) -> None:
        content = b"same"
        id1 = artifact_id(content, {"ts": "1"})
        id2 = artifact_id(content, {"ts": "2"})
        assert id1 != id2


class TestContentHash:
    def test_sha256(self) -> None:
        h = content_hash(b"hello")
        assert len(h) == 64
        # Known SHA-256 of "hello"
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
