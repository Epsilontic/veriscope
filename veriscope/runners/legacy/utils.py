# veriscope/runners/legacy/utils.py
"""
Centralized utility functions for IO, numeric conversions, and quantiles.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.floating]


# --- IO helpers ---


def save_json(obj: Any, path: Path) -> None:
    """Atomic JSON write with indentation."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Directory creation failure will surface on write.
        pass

    def _jsonable(o: Any) -> Any:
        try:
            json.dumps(o)
            return o
        except Exception:
            return str(o)

    if isinstance(obj, dict):
        obj = {k: _jsonable(v) for k, v in obj.items()}

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def update_json(path: Path, patch: Dict[str, Any]) -> None:
    """Read-modify-write JSON dict at path with a shallow update."""
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cur, dict):
            cur = {}
    except Exception:
        cur = {}
    cur.update(patch or {})
    save_json(cur, path)


def file_md5(path: Path) -> str:
    """Return hex MD5 checksum for the given file path, or '' if missing."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Numeric helpers ---


def as_int(x: Any, default: int = 0) -> int:
    """Best-effort int conversion (handles NumPy scalars and strings)."""
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return int(x)
        if isinstance(x, (str, bytes, bytearray)):
            s = str(x).strip()
            if not s:
                return default
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(s, 10)
    except Exception:
        pass
    return default


def as_float(x: Any, default: float = 0.0) -> float:
    """Best-effort float conversion (handles NumPy scalars and strings)."""
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, (str, bytes, bytearray)):
            s = str(x).strip()
            if not s:
                return default
            return float(s)
    except Exception:
        pass
    return default


def to_numeric_series(
    s: Optional[pd.Series],
    *,
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
) -> pd.Series:
    """
    Optional Series -> numeric Series of float dtype.

    errors="ignore" is treated like "coerce" but leaves dtype float.
    """
    if s is None:
        return pd.Series(dtype=float)
    eno: Literal["raise", "coerce"] = "coerce" if errors == "ignore" else errors
    return pd.to_numeric(s, errors=eno).astype(float, copy=False)


# --- Quantile helpers ---


def _as_float_array(x: Iterable[float] | npt.ArrayLike) -> FloatArray:
    """Coerce to a 1-D float array; scalars become length-1 arrays."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def qlin(a: Iterable[float] | npt.ArrayLike, q: float) -> float:
    """
    Deterministic linear quantile on floats.

    Uses NumPy's modern API (method="linear") with a fallback to
    interpolation="linear" for older NumPy versions.
    """
    arr = _as_float_array(a)
    qf = float(q)
    if not (0.0 <= qf <= 1.0):
        raise ValueError(f"qlin: q={qf} outside [0,1]")
    try:
        v = np.quantile(arr, qf, method="linear")
    except TypeError:  # NumPy < 1.22
        v = np.quantile(arr, qf, interpolation="linear")  # type: ignore[arg-type]
    return float(v)


def quantile2(
    a: Iterable[float] | npt.ArrayLike,
    qlo: float,
    qhi: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return (lo, hi) quantiles as floats.

    If qhi is None, returns (qlo, 1 - qlo) quantiles.
    If qhi is provided, returns (qlo, qhi) quantiles.
    """
    arr = _as_float_array(a)
    if qhi is None:
        lo_q = float(qlo)
        hi_q = 1.0 - lo_q
        qs = np.asarray([lo_q, hi_q], dtype=float)
    else:
        qs = np.asarray([float(qlo), float(qhi)], dtype=float)

    try:
        res = np.quantile(arr, qs, method="linear")
    except TypeError:
        res = np.quantile(arr, qs, interpolation="linear")  # type: ignore[arg-type]

    r = np.asarray(res, dtype=float).reshape(-1)
    lo = float(r[0]) if r.size > 0 else float("nan")
    hi = float(r[1]) if r.size > 1 else lo
    return lo, hi


def series_or_empty(x: Optional[pd.Series]) -> pd.Series:
    """Return the Series if given, else an empty float Series."""
    return x if isinstance(x, pd.Series) else pd.Series(dtype=float)
