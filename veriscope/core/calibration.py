# veriscope/core/calibration.py — FR window calibration & ε-statistics
#
# Title: Controls-to-epsilon and ε_stat aggregation for fixed-partition TV
# Summary: Resolve epsilon from control segments (quantile) and aggregate
#          finite-sample ε_stat across Φ_W using declared weights and bins.
# Invariants: window-relative; weights defensively normalized; ε_stat ∈ [0,1];
#             NumPy-only (no torch) to keep core light and reusable.
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .window import WindowDecl

__all__ = ["epsilon_statistic_bhc", "aggregate_epsilon_stat"]


def resolve_epsilon_from_controls(vals: Iterable[Any], q: float, fallback: float) -> Tuple[float, int]:
    a = np.asarray([float(x) for x in vals if isinstance(x, (int, float, np.floating))], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float(fallback), 0
    try:
        eps = float(np.quantile(a, q, method="linear"))
    except TypeError:
        eps = float(np.quantile(a, q))  # numpy<1.22 fallback
    return eps, int(a.size)


# --- ε-statistics for histogram TV ---
def epsilon_statistic_bhc(n: int, k: int, alpha: float = 0.05) -> float:
    # No samples => maximum uncertainty.
    # Downstream code should clamp this to (max_frac * epsilon) anyway.
    if n <= 0:
        return 1.0
    t = max(0.0, math.log(max(2.0, float(k))) + math.log(1.0 / max(alpha, 1e-12)))
    eps = math.sqrt(max(0.0, t / max(2.0 * float(n), 1e-12)))
    return float(min(max(eps, 0.0), 1.0))


def aggregate_epsilon_stat(window: "WindowDecl", counts_by_metric: Dict[str, int], alpha: float = 0.05) -> float:
    """Weighted aggregation across Φ_W using declared weights and bins."""
    weights = dict(getattr(window, "weights", {}) or {})
    s = sum(abs(v) for v in weights.values()) or 1.0
    bins = int(getattr(window, "bins", 1))
    total = 0.0
    for m, w in weights.items():
        n_m = int(counts_by_metric.get(m, 0))
        eps_m = epsilon_statistic_bhc(n=n_m, k=bins, alpha=alpha)
        total += (float(w) / s) * eps_m
    return float(total)
