# tests/test_jsonutil_canonical_hash.py
from __future__ import annotations

import pytest

from veriscope.core.jsonutil import canonical_bytes, canonical_dumps, canonical_json_sha256

pytestmark = pytest.mark.unit


def test_canonical_dumps_is_order_invariant_and_hash_matches() -> None:
    a = {"b": 2, "a": 1, "nested": {"z": 9, "y": 8}}
    b = {"nested": {"y": 8, "z": 9}, "a": 1, "b": 2}

    sa = canonical_dumps(a)
    sb = canonical_dumps(b)

    assert sa == sb
    assert canonical_json_sha256(a) == canonical_json_sha256(b)


def test_canonical_dumps_preserves_unicode_and_uses_compact_separators() -> None:
    obj = {"msg": "café ☃", "k": 1}
    s = canonical_dumps(obj)

    # ensure_ascii=False: unicode stays unescaped
    assert "café" in s
    assert "☃" in s

    # separators=(",", ":") -> no spaces after colon/comma
    assert ": " not in s
    assert ", " not in s


def test_canonical_bytes_is_utf8() -> None:
    obj = {"msg": "café"}
    b = canonical_bytes(obj)

    assert isinstance(b, bytes)
    assert b.decode("utf-8") == canonical_dumps(obj)


def test_canonical_dumps_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_dumps({"x": float("-inf")})
