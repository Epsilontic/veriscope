from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def iso_z(dt: Optional[datetime] = None) -> str:
    resolved = dt or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    else:
        resolved = resolved.astimezone(timezone.utc)
    return resolved.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return out


def coerce_nonneg_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer (got bool)")
    out = int(value)
    if out < 0:
        raise ValueError(f"{field} must be >= 0")
    return out


def coerce_optional_nonneg_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return coerce_nonneg_int(value, field="value")
    except Exception:
        return None
