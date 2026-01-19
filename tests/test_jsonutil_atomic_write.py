# tests/test_jsonutil_atomic_write.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from veriscope.core.jsonutil import atomic_write_json, atomic_write_text

pytestmark = pytest.mark.unit


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_atomic_write_json_creates_file_and_no_tmp_left(tmp_path: Path) -> None:
    target = tmp_path / "results.json"
    payload = {"b": 2, "a": 1}

    atomic_write_json(target, payload)

    assert target.exists()
    s = _read_text(target)

    # module promises newline termination for JSON writes
    assert s.endswith("\n")

    loaded = json.loads(s)
    assert loaded == payload

    leftovers = list(tmp_path.glob(f".{target.name}.vs-tmp.*.tmp"))
    assert leftovers == []


def test_atomic_write_json_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "results.json"

    atomic_write_json(target, {"x": 1})
    first = json.loads(_read_text(target))

    atomic_write_json(target, {"x": 2, "y": True})
    second = json.loads(_read_text(target))

    assert first == {"x": 1}
    assert second == {"x": 2, "y": True}


def test_atomic_write_text_writes_exact_text_and_no_tmp_left(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    text = "hello\nworld\n"

    atomic_write_text(target, text)

    assert target.exists()
    assert _read_text(target) == text

    leftovers = list(tmp_path.glob(f".{target.name}.vs-tmp.*.tmp"))
    assert leftovers == []


def test_atomic_write_text_fsync_path_smoke(tmp_path: Path) -> None:
    target = tmp_path / "fsync.txt"
    atomic_write_text(target, "x\n", fsync=True)

    assert target.exists()
    assert _read_text(target) == "x\n"
