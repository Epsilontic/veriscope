from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def metric_snapshot(
    metric_history: list[dict[str, Any]],
    gate_window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (past, recent) slices once a full 2*window history exists."""
    if int(gate_window) <= 0:
        return [], []
    if len(metric_history) < (2 * int(gate_window)):
        return [], list(metric_history)
    gw = int(gate_window)
    past = metric_history[-2 * gw : -gw]
    recent = metric_history[-gw:]
    return past, recent


def extract_metric_array(
    slice_data: Sequence[Mapping[str, Any]],
    key: str,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Extract finite float values for one metric from a history slice.

    When strict=True, non-coercible present values raise ValueError instead of
    being silently dropped.
    """
    vals: list[float] = []
    for row in slice_data:
        raw = row.get(key, np.nan)
        try:
            v = float(raw)
        except Exception:
            if strict and key in row:
                iter_value = row.get("iter")
                raise ValueError(
                    f"Non-coercible metric value for key={key!r} at iter={iter_value!r}: {raw!r}"
                ) from None
            v = float("nan")
        vals.append(v)
    arr = np.asarray(vals, dtype=float)
    return arr[np.isfinite(arr)]


def compute_evidence_counts(
    past_by_metric: Mapping[str, np.ndarray],
    recent_by_metric: Mapping[str, np.ndarray],
    metrics: Iterable[str],
) -> tuple[Dict[str, int], int]:
    """Count comparable finite evidence per metric as min(len(past), len(recent))."""
    counts: Dict[str, int] = {}
    total = 0
    for metric in metrics:
        past_arr = np.asarray(past_by_metric.get(metric, np.array([], dtype=float)), dtype=float)
        recent_arr = np.asarray(recent_by_metric.get(metric, np.array([], dtype=float)), dtype=float)
        n = int(min(len(past_arr), len(recent_arr)))
        counts[str(metric)] = n
        total += n
    return counts, int(total)


def compute_loss_delta_z(
    loss: float,
    prior_losses: Sequence[float],
    gate_window: int,
) -> float:
    """HF-runner-compatible loss z-score against recent finite prior losses."""
    try:
        loss_f = float(loss)
    except Exception:
        return float("nan")
    if not math.isfinite(loss_f):
        return float("nan")

    gw = int(gate_window)
    if gw <= 0 or not prior_losses:
        return float("nan")

    ref_window = prior_losses[-min(len(prior_losses), gw) :]
    ref_vals: list[float] = []
    for value in ref_window:
        try:
            vf = float(value)
        except Exception:
            continue
        if math.isfinite(vf):
            ref_vals.append(vf)
    ref_arr = np.asarray(ref_vals, dtype=float)
    if ref_arr.size < 2:
        return float("nan")

    ref_mean = float(ref_arr.mean())
    ref_std = float(ref_arr.std(ddof=1))
    loss_delta = loss_f - ref_mean
    loss_delta_z = loss_delta / max(ref_std, 1e-4)
    return float(np.clip(loss_delta_z, -6.0, 6.0))
