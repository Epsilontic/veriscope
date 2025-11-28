# veriscope/runners/legacy/detectors/baselines.py
from __future__ import annotations

"""
Baseline detector implementations for legacy runners.

Phase-1 boundary:
- This module owns the implementations formerly in legacy_cli_refactor.py.
- Nothing here imports from legacy_cli_refactor.py.
- Public facades in veriscope/detectors/* re-export from this module.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from veriscope.runners.legacy import runtime as _rt
from veriscope.runners.legacy.utils import as_int, as_float

# Runtime config (safe if runtime not installed yet).
# IMPORTANT: keep a shared reference so later CFG mutations are visible here.
try:
    CFG: Dict[str, Any] = getattr(_rt, "CFG", {}) or {}
except Exception:
    CFG = {}

# Scheduled metrics (cadenced/missing by design) — never fed to the learner.
# Moved from legacy_cli_refactor.py
SCHEDULED_METRICS: List[str] = ["sw2", "pers_H0", "mon_entropy", "avg_max_prob"]

# TTL for scheduled metrics propagation to avoid stale ffill artifacts
CFG.setdefault("scheduled_ttl", 2 * CFG.get("heavy_every", 6))


# ---------------------------
# PH series preprocessing helper (centralized)
# ---------------------------
def _prep_series_for_ph(g: pd.DataFrame, metric: str) -> List[float]:
    """Prepare a metric series for PH detection: ffill scheduled metrics, apply validity masks,
    and gate pers_H0 by min repeats and time budget. Returns a Python list with NaNs for invalid epochs.
    """
    s = g[metric].copy()
    if metric in SCHEDULED_METRICS:
        s = s.ffill()
        # apply TTL to scheduled metrics to avoid stale carry-over
        arr = s.to_numpy(dtype=float)
        age = np.full_like(arr, np.inf, dtype=float)
        last = -1
        for i, v in enumerate(arr):
            if np.isfinite(v):
                last = i
            age[i] = (i - last) if last >= 0 else np.inf
        ttl = float(CFG.get("scheduled_ttl", 2 * CFG.get("heavy_every", 6)))
        arr = np.where(age <= ttl, arr, np.nan)
    else:
        arr = s.to_numpy(dtype=float)

    vcol = f"{metric}_valid"
    if vcol in g.columns:
        mask = g[vcol].astype(bool).to_numpy()
        arr = np.where(mask, arr, np.nan)

    if metric == "pers_H0":
        if "topo_done" in g.columns:
            min_rep = int(math.ceil(CFG.get("rp_repeats", 8) / 2))
            td = g["topo_done"].astype(float).to_numpy()
            arr = np.where((td >= min_rep) & np.isfinite(arr), arr, np.nan)

        if "topo_ms" in g.columns:
            ms = g["topo_ms"].astype(float).to_numpy()
            budget = float(CFG.get("ripser_budget_ms", 250))
            arr = np.where(np.isfinite(arr) & np.isfinite(ms) & (ms <= 0.9 * budget), arr, np.nan)

        # Require enough samples for the topology estimate:
        # prefer the minimum used across repeats if available (topo_n_used_min),
        # falling back to legacy columns.
        nused = None
        if "topo_n_used_min" in g.columns:
            nused = g["topo_n_used_min"].astype(float).to_numpy()
        elif "n_topo_sampled" in g.columns:
            nused = g["n_topo_sampled"].astype(float).to_numpy()
        elif "topo_n_used" in g.columns:
            nused = g["topo_n_used"].astype(float).to_numpy()

        if nused is not None:
            min_n_required = int(CFG.get("topo_min_n", 64))
            arr = np.where(np.isfinite(arr) & (nused >= min_n_required), arr, np.nan)

    return [float(x) if np.isfinite(x) else float("nan") for x in arr]


# ---------------------------
# PH & sequential helpers
# ---------------------------
def _ph_on_z(zs: List[float], lam: float, direction: str) -> Tuple[Optional[int], List[float]]:
    s = 0.0
    track: List[float] = []
    for t, z in enumerate(zs):
        if direction == "up":
            s = max(0.0, s + z)
            track.append(s)
            if s > lam:
                return t, track
        else:
            s = min(0.0, s + z)
            track.append(s)
            if s < -lam:
                return t, track
    return None, track


def robust_z_series(xs: List[float], win: int, burn_in: int) -> List[float]:
    thr = max(burn_in, win, 2)
    zs: List[float] = []
    for t, x in enumerate(xs):
        if t < thr:
            zs.append(0.0)
            continue
        a = max(0, t - win)
        b = t
        w = [v for v in xs[a:b] if np.isfinite(v)]
        if len(w) < 4:
            zs.append(0.0)
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(np.array(w) - med))) + 1e-8
        z = (x - med) / (1.4826 * mad)
        zs.append(float(z))
    return zs


def ph_window_sparse(
    xs: List[float],
    win: int,
    lam: float,
    direction: str,
    burn_in: int,
    min_points: int,
    two_sided: bool,
) -> Tuple[Optional[int], List[float], List[float]]:
    """
    Sparse CUSUM-on-robust-z over a series that may contain NaNs.
    (Verbatim from legacy_cli_refactor.py.)
    """
    thr = max(burn_in, win, 2)

    # indices of finite observations in original time space
    idxs = [i for i, x in enumerate(xs) if np.isfinite(x)]
    if not idxs:
        return None, [0.0] * len(xs), [0.0] * len(xs)

    # compacted finite-valued series
    comp = [xs[i] for i in idxs]

    # enforce availability after burn-in in comp-space
    comp_after_thr = [v for v in comp[thr:] if np.isfinite(v)]
    if len(comp_after_thr) < int(min_points):
        zs = robust_z_series(comp, win, burn_in)
        return None, zs, [0.0] * len(xs)

    # compute robust z in comp-space
    zs = robust_z_series(comp, win, burn_in)

    def _detect(zs_list, dir_label):
        t_comp, tr_comp = _ph_on_z(zs_list, lam, dir_label)
        if t_comp is None:
            return None, tr_comp
        if t_comp < thr:
            return None, tr_comp
        return t_comp, tr_comp

    if two_sided:
        t_up, tr_up = _detect(zs, "up")
        t_dn, tr_dn = _detect(zs, "down")
        if t_up is None and t_dn is None:
            t_comp = None
            tr_use = [0.0] * len(comp)
        else:
            if t_up is None:
                t_comp, tr_use = t_dn, tr_dn
            elif t_dn is None:
                t_comp, tr_use = t_up, tr_up
            else:
                t_comp = t_up if t_up <= t_dn else t_dn
                tr_use = tr_up if t_up <= t_dn else tr_dn
    else:
        t_comp, tr_use = _detect(zs, direction)

    track_full = [0.0] * len(xs)
    if t_comp is not None:
        # map the comp-space track starting at comp index thr back to time indices
        start = thr
        end = min(len(tr_use), len(comp))
        for k in range(start, end):
            ti = idxs[k]
            if 0 <= ti < len(track_full):
                track_full[ti] = tr_use[k]
        t_time = idxs[t_comp]
        return t_time, zs, track_full
    else:
        return None, zs, track_full


def _delta(xs: List[float]) -> List[float]:
    out = [np.nan] * len(xs)
    for t in range(1, len(xs)):
        a, b = xs[t - 1], xs[t]
        out[t] = (b - a) if (np.isfinite(a) and np.isfinite(b)) else np.nan
    return out


def cusum_one_sided(zs: List[float], lam: float, direction: str = "down") -> Tuple[Optional[int], List[float]]:
    s = 0.0
    track = []
    for t, z in enumerate(zs):
        if not np.isfinite(z):
            track.append(s)
            continue
        if direction == "down":
            s = min(0.0, s + z)
            track.append(s)
            if s < -lam:
                return t, track
        else:
            s = max(0.0, s + z)
            track.append(s)
            if s > lam:
                return t, track
    return None, track


def newma_warn_epoch(xs: List[float], fast: float, slow: float, lam: float, burn_in: int) -> Optional[int]:
    mu_f = 0.0
    mu_s = 0.0
    for t, x in enumerate(xs):
        if not np.isfinite(x):
            continue
        a = float(x)
        mu_f = (1 - fast) * mu_f + fast * a
        mu_s = (1 - slow) * mu_s + slow * a
        if t >= burn_in:
            s = mu_f - mu_s
            if abs(s) > lam:
                return t
    return None


def calibrate_ph_directions(df_cal: pd.DataFrame, metrics: List[str]) -> Dict[str, str]:
    dir_map: Dict[str, str] = {}
    warm = as_int(CFG.get("warmup"), default=0) + as_int(CFG.get("ph_burn"), default=0)
    win_default = as_int(CFG.get("ph_win"), default=0)
    win_short_default = as_int(CFG.get("ph_win_short"), default=win_default)

    from typing import cast, Any

    for m in metrics:
        # Make sure win_m is always an int for mypy-clean comparisons
        win_m = win_short_default if (m in SCHEDULED_METRICS) else win_default
        best: Optional[float] = None
        best_dir = "up"
        for d in ["up", "down"]:
            leads: List[int] = []
            for key, g in df_cal.groupby(["seed", "factor"]):
                seed, factor = cast(tuple[Any, Any], key)
                g = g.sort_values("epoch")
                t_c_raw = g["t_collapse_gt"].iloc[0] if "t_collapse_gt" in g.columns else np.nan
                ctag = g["collapse_tag_gt"].iloc[0] if "collapse_tag_gt" in g.columns else "none"
                t_c_i = as_int(t_c_raw, default=-1)
                if (t_c_i < 0) or ctag != "soft":
                    continue
                xs_all = _prep_series_for_ph(g, m)
                pre = g["epoch"].to_numpy(dtype=int) < t_c_i
                xs = np.where(pre, np.array(xs_all, dtype=float), np.nan).tolist()
                t, _, _ = ph_window_sparse(
                    xs,
                    win=int(win_m),
                    lam=as_float(CFG.get("ph_lambda"), default=0.0),
                    direction=d,
                    burn_in=int(warm),
                    min_points=as_int(CFG.get("ph_min_points"), default=0),
                    two_sided=bool(CFG.get("ph_two_sided")),
                )
                if t is not None:
                    leads.append(t_c_i - int(t))
            if leads:
                avg = float(np.mean(leads))
                if (best is None) or (avg > best):
                    best = avg
                    best_dir = d
        dir_map[m] = best_dir
    return dir_map


__all__ = [
    "SCHEDULED_METRICS",
    "_prep_series_for_ph",
    "_ph_on_z",
    "_delta",
    "robust_z_series",
    "ph_window_sparse",
    "cusum_one_sided",
    "newma_warn_epoch",
    "calibrate_ph_directions",
]
