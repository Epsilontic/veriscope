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
from veriscope.runners.legacy.utils import as_float, as_int

# Runtime config (safe if runtime not installed yet).
# IMPORTANT: keep a shared reference so later CFG mutations are visible here.
try:
    CFG: Dict[str, Any] = getattr(_rt, "CFG", {}) or {}
except Exception:
    CFG = {}


def _scheduled_ttl_default() -> float:
    heavy_every_i = as_int(CFG.get("heavy_every"), default=6)
    return float(2 * heavy_every_i)


# Scheduled metrics (cadenced/missing by design) — never fed to the learner.
# Moved from legacy_cli_refactor.py
SCHEDULED_METRICS: List[str] = ["sw2", "pers_H0", "mon_entropy", "avg_max_prob"]


# ---------------------------
# PH series preprocessing helper (centralized)
# ---------------------------
def _prep_series_for_ph(g: pd.DataFrame, metric: str, *, ffill_scheduled: bool = False) -> List[float]:
    """Prepare a metric series for PH detection.

    CRITICAL DETECTOR CONTRACT:
      - By default we DO NOT forward-fill scheduled metrics.
        Missing epochs must remain NaN so detectors cannot use stale values as evidence.
      - Optional ffill_scheduled=True exists only for plotting/debug convenience and applies a TTL.
    """
    # Guard: return all-NaN if metric column is missing
    if metric not in g.columns:
        return [float("nan")] * len(g)

    s = g[metric].copy()
    if metric in SCHEDULED_METRICS:
        raw_arr = s.to_numpy(dtype=float)
        if ffill_scheduled:
            # Forward-fill is allowed ONLY when explicitly requested (plots/debug).
            # Still apply TTL to prevent unbounded staleness.
            s_ff = s.ffill()
            arr = s_ff.to_numpy(dtype=float)
            age = np.full_like(arr, np.inf, dtype=float)
            last = -1
            for i, v in enumerate(raw_arr):
                if np.isfinite(v):
                    last = i
                age[i] = (i - last) if last >= 0 else np.inf
            # scheduled_ttl may be missing or non-numeric; fall back safely.
            ttl = as_float(CFG.get("scheduled_ttl"), default=_scheduled_ttl_default())
            arr = np.where(age <= ttl, arr, np.nan)
        else:
            # Detector-safe: keep the series sparse; DO NOT manufacture observations.
            arr = raw_arr
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
    """Sparse CUSUM-on-robust-z over a series that may contain NaNs.
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


def newma_warn_epoch(
    xs: List[float],
    fast: float,
    slow: float,
    lam: float,
    burn_in: int,
    min_points: int = 0,
) -> Optional[int]:
    """NEWMA-style fast/slow EWMA crossover detector.

    Smoke-critical fix: initialize both EWMAs to the first finite observation
    (prevents deterministic cold-start divergence when slow EWMA begins at 0).

    min_points: require at least this many finite observations after burn_in
    before allowing a trigger (0 disables the guard).
    """
    mu_f: Optional[float] = None
    mu_s: Optional[float] = None
    n_finite_after_burn = 0

    for t, x in enumerate(xs):
        if not np.isfinite(x):
            continue
        a = float(x)

        # Cold-start: initialize to the first observed value; do not evaluate on init epoch.
        if mu_f is None:
            mu_f = a
            mu_s = a
            continue

        mu_f = (1.0 - float(fast)) * mu_f + float(fast) * a
        mu_s = (1.0 - float(slow)) * mu_s + float(slow) * a

        if t >= int(burn_in):
            n_finite_after_burn += 1
            if int(min_points) > 0 and n_finite_after_burn < int(min_points):
                continue
            s = mu_f - mu_s
            if abs(s) > float(lam):
                return t

    return None


def calibrate_ph_directions(df_cal: pd.DataFrame, metrics: List[str]) -> Dict[str, str]:
    dir_map: Dict[str, str] = {}
    warm = as_int(CFG.get("warmup"), default=0) + as_int(CFG.get("ph_burn"), default=0)
    win_default = as_int(CFG.get("ph_win"), default=0)
    win_short_default = as_int(CFG.get("ph_win_short"), default=win_default)

    from typing import Any, cast

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

                if "collapse_tag_gt" in g.columns:
                    use = g
                    if "epoch" in g.columns:
                        ep = pd.to_numeric(g["epoch"], errors="coerce")
                        post = g[ep >= int(warm)]
                        if not post.empty:
                            use = post
                    tags = {str(x) for x in use["collapse_tag_gt"].dropna().astype(str).unique().tolist()}
                    if "hard" in tags:
                        ctag = "hard"
                    elif "soft" in tags:
                        ctag = "soft"
                    else:
                        ctag = "none"
                else:
                    ctag = "none"

                t_c_i = as_int(t_c_raw, default=-1)
                if (t_c_i < 0) or ctag != "soft":
                    continue
                xs_all = _prep_series_for_ph(g, m, ffill_scheduled=False)
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
