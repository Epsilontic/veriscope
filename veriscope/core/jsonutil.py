# veriscope/core/jsonutil.py
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union

try:
    # Pydantic v2 (core dep in your repo now); kept optional to avoid hard import coupling.
    from pydantic import BaseModel  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore


def canonical_dumps(obj: Any) -> str:
    """
    Canonical JSON string per productization spec:

        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    Notes:
    - This intentionally does NOT pretty-print.
    - This does not attempt to normalize non-JSON-serializable objects; callers must.
    - Float rendering is CPython-stable in practice but not a formal cross-runtime consensus guarantee.
    - NaN/Infinity are forbidden.
    """
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except TypeError as e:
        # Non-serializable object (e.g., Path, datetime, numpy scalar) -> TypeError.
        raise TypeError(
            "canonical_dumps() requires a JSON-serializable object (normalize Paths/datetimes/numpy scalars, etc.)"
        ) from e
    except ValueError:
        # With allow_nan=False, NaN/Infinity raise ValueError; preserve that signal.
        raise


def canonical_bytes(obj: Any) -> bytes:
    """UTF-8 bytes of canonical JSON."""
    return canonical_dumps(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """SHA256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    """SHA256 hex digest of canonical JSON bytes (canonical_dumps -> utf-8 -> sha256)."""
    return sha256_hex(canonical_dumps(obj).encode("utf-8"))


WINDOW_SIGNATURE_VOLATILE_KEYS = frozenset({"created_ts_utc"})


def _is_iso8601_utc_z(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value.strip()
    if not raw.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except Exception:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def canonicalize_window_signature(window_signature: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a canonicalized window signature for hashing/comparability.

    Volatile metadata (e.g., created_ts_utc) is excluded so identical evidence spaces remain comparable.
    """
    canonical: dict[str, Any] = {}
    for key, value in window_signature.items():
        if key in WINDOW_SIGNATURE_VOLATILE_KEYS:
            # created_ts_utc is intentionally excluded from comparability hashing, but only
            # when it is a valid UTC timestamp string (prevents semantic smuggling via objects).
            if key == "created_ts_utc" and not _is_iso8601_utc_z(value):
                raise ValueError("window_signature.created_ts_utc must be ISO8601 UTC with trailing 'Z'")
            continue
        canonical[key] = value
    return canonical


def window_signature_sha256(window_signature: Mapping[str, Any]) -> str:
    """SHA256 hash of the canonicalized window signature."""
    return canonical_json_sha256(canonicalize_window_signature(window_signature))


def sanitize_json_value(obj: Any) -> Any:
    """Convert non-finite floats to None and recursively sanitize JSON containers."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_value(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _unique_tmp_path_for(path: Path) -> Path:
    """
    Choose a unique temp path in the same directory to preserve atomicity of os.replace.

    Concurrency-safe: multiple writers targeting the same final path will never clobber each other's temp file.
    """
    token = uuid.uuid4().hex[:12]
    tmp_name = f".{path.name}.vs-tmp.{token}.tmp"
    # Defensive: avoid pathological name-length issues.
    if len(tmp_name) > 240:
        tmp_name = f".vs-tmp.{token}.tmp"
    return path.with_name(tmp_name)


def _fsync_dir_best_effort(dir_path: Path) -> None:
    """Best-effort fsync of a directory to improve rename durability on POSIX."""
    try:
        if os.name != "posix":
            return
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # Best-effort only.
        return


def atomic_write_text(path: Path, text: str, *, fsync: bool = False) -> None:
    """
    Atomic text write using a unique temp file:

      1) write to <path>.<uuid>.tmp in the same directory
      2) os.replace(tmp, path) (atomic on POSIX; overwrites existing)
      3) best-effort cleanup of temp file on failure

    Parameters
    ----------
    fsync:
        If True, flush and fsync the file before replace. This reduces the chance of zero-byte files
        after abrupt power loss, at a performance cost. Default False for Phase 1.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = _unique_tmp_path_for(path)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())

        os.replace(tmp, path)
        if fsync:
            _fsync_dir_best_effort(path.parent)

    except Exception:
        # Best-effort cleanup. If the process is SIGKILLed, cleanup is impossible; leftovers are acceptable.
        try:
            if tmp.exists():
                os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any, *, fsync: bool = False) -> None:
    """
    Atomic write of canonical JSON.

    Writes canonical JSON (not pretty JSON) to ensure deterministic hashing and diffs.
    Appends a trailing newline for POSIX-friendly text conventions.
    """
    atomic_write_text(path, canonical_dumps(obj) + "\n", fsync=fsync)


def atomic_append_jsonl(path: Path, obj: Any, *, fsync: bool = True) -> None:
    """
    Append a JSON object as a single JSONL line.

    Writes use O_APPEND and loop until all bytes are written to reduce partial-line interleaves on POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_json_value(obj)
    line = canonical_dumps(payload) + "\n"
    data = line.encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY)
    try:
        offset = 0
        data_len = len(data)
        while offset < data_len:
            written = os.write(fd, data[offset:])
            if written == 0:
                raise OSError("atomic_append_jsonl() short write (0 bytes)")
            offset += written
        if fsync:
            os.fsync(fd)
            _fsync_dir_best_effort(path.parent)
    finally:
        os.close(fd)


def atomic_write_pydantic_json(
    path: Path, model: "BaseModel", *, by_alias: bool = True, exclude_none: bool = False, fsync: bool = False
) -> None:
    """
    Atomic write for Pydantic models using stable JSON-compatible dumping.

    Intended for Pydantic v2:
      - uses model_dump(mode="json") for JSON-compatible primitives

    Note: by_alias=True is the default to preserve contract keys.

    If a non-v2 model is passed, raises TypeError (explicit failure is preferable to silently changing semantics).
    """
    model_dump = getattr(model, "model_dump", None)
    if model_dump is None or not callable(model_dump):
        raise TypeError("atomic_write_pydantic_json() requires a Pydantic v2 BaseModel instance (with model_dump).")

    payload: Union[dict[str, Any], list[Any]] = model_dump(
        mode="json",
        by_alias=by_alias,
        exclude_none=exclude_none,
    )
    atomic_write_json(path, payload, fsync=fsync)


def read_json_obj(path: Path) -> dict[str, Any]:
    """
    Read a JSON file and require a top-level object.

    Raises TypeError if the JSON is not an object.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{Path(path).name} must be a JSON object")
    return obj


def cleanup_tmp_siblings(path: Path, *, include_legacy: bool = False) -> int:
    """
    Optional utility: remove stale sibling temp files created by this module.

    Deletes files matching: ".<name>.vs-tmp.*.tmp" in the same directory by default.
    Optionally deletes legacy files matching "<name>.*.tmp".

    Returns count of removed files.

    This is NOT used automatically, but can be called by maintenance tooling or validation routines.
    """
    path = Path(path)
    parent = path.parent
    stem_new = f".{path.name}.vs-tmp."
    stem_legacy = path.name + "."
    removed = 0

    if not parent.exists():
        return 0

    for p in parent.iterdir():
        name = p.name
        match_new = name.startswith(stem_new) and name.endswith(".tmp")
        match_legacy = include_legacy and name.startswith(stem_legacy) and name.endswith(".tmp")
        if p.is_file() and (match_new or match_legacy):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass

    return removed
