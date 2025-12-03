# veriscope/core/ipm.py
from __future__ import annotations

from typing import Any, Dict, Callable

import numpy as np

from .window import FRWindow, WindowDecl

__all__ = ["tv_hist_fixed", "d_hyb", "dPi_product_tv", "d_Pi", "D_W"]


def tv_hist_fixed(z0: Any, z1: Any, bins: int) -> float:
    """Total variation distance between fixed-range [0,1] histograms built from z0,z1.
    Uses counts (density=False) then normalizes to probability mass; robust to empties."""
    a = np.asarray(z0, float)
    b = np.asarray(z1, float)
    a = np.clip(a[np.isfinite(a)], 0.0, 1.0)
    b = np.clip(b[np.isfinite(b)], 0.0, 1.0)
    ha, _ = np.histogram(a, bins=bins, range=(0.0, 1.0), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=False)
    # If both sides are empty after filtering, treat them as identical (distance 0).
    if ha.sum() == 0 and hb.sum() == 0:
        return 0.0

    # If either side has no mass, TV is undefined for this window.
    # Return NaN so callers can treat it as "not evaluated" rather than fabricating a uniform baseline.
    if ha.sum() == 0 or hb.sum() == 0:
        return float("nan")
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    return 0.5 * float(np.abs(ha - hb).sum())


def d_hyb(obs: Dict[str, np.ndarray], pred: Dict[str, np.ndarray], weights: Dict[str, float], bins: int) -> float:
    mx, acc = 0.0, 0.0
    for m, w in weights.items():
        if (m in obs) and (m in pred):
            tv = tv_hist_fixed(obs[m], pred[m], bins)
            mx = max(mx, tv)
            acc += float(w) * tv
    return mx + acc


def d_Pi(
    decl: WindowDecl,
    P: Dict[str, np.ndarray],
    Q: Dict[str, np.ndarray],
    apply: Callable[[str, np.ndarray], np.ndarray],
) -> float:
    # product-TV under fixed partition; caller passes apply = frwin.transport.apply
    s = 0.0
    for m, w in decl.weights.items():
        if (m in P) and (m in Q):
            p = apply(m, P[m])
            q = apply(m, Q[m])
            s += float(w) * tv_hist_fixed(p, q, decl.bins)
    return s


def dPi_product_tv(
    decl: WindowDecl,
    P: Dict[str, np.ndarray],
    Q: Dict[str, np.ndarray],
    apply: Callable[[str, np.ndarray], np.ndarray],
) -> float:
    """Canonical name for the fixed-partition product-TV distance.

    This is an alias of `d_Pi` to keep a stable import surface across the repo.
    """
    return d_Pi(decl, P, Q, apply)


def D_W(frwin: FRWindow, P: Dict[str, np.ndarray], Q: Dict[str, np.ndarray]) -> float:
    # Restricted operational distinguishability on Φ_W.
    # In the current fixed-partition regime, define D_W as max TV across declared metrics.
    mx = 0.0
    for m in frwin.decl.metrics:
        if (m in P) and (m in Q):
            p = frwin.transport.apply(m, P[m])
            q = frwin.transport.apply(m, Q[m])
            tv = tv_hist_fixed(p, q, frwin.decl.bins)
            mx = max(mx, tv)
    return mx
