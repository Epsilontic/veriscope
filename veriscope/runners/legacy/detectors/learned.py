# veriscope/runners/legacy/detectors/learned.py
"""
Learned detector training and inference for legacy runners.

Phase-1 boundary:
- This module owns the implementations formerly in legacy_cli_refactor.py.
- Nothing here imports from legacy_cli_refactor.py.
- Public facades in veriscope/detectors/* re-export from this module.

Notes:
- We do NOT bind CFG at import time; use _get_cfg() if/when needed.
- Public surface is intentionally small: only map_threshold_to_gated_fp is stable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get_cfg() -> Dict[str, Any]:
    """Fetch runtime CFG lazily (avoids stale import-time snapshots)."""
    try:
        from veriscope.runners.legacy import runtime as _rt  # local import on purpose

        cfg = getattr(_rt, "CFG", None)
        return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


# Utilities (fallbacks keep runner resilient)
try:
    from veriscope.runners.legacy.utils import as_int, as_float
except Exception:  # pragma: no cover

    def as_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def as_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default


def _sigmoid_stable(z: np.ndarray, cap: float = 60.0) -> np.ndarray:
    """Numerically stable sigmoid with clipping."""
    zc = np.clip(z, -float(cap), float(cap))
    return 1.0 / (1.0 + np.exp(-zc))


def _metrics_matrix_with_missing(g: pd.DataFrame, metric_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and missingness indicator from DataFrame."""
    X = g[metric_cols].to_numpy(dtype=np.float32)
    M = np.isnan(X).astype(np.float32)
    return X, M


def _fit_global_robust_norm_precollapse(df_cal: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """Robust med/MAD estimated only on pre-collapse epochs (or all epochs for none-runs)."""
    stats: Dict[str, Tuple[float, float]] = {}
    if df_cal.empty:
        return {m: (0.0, 1.0) for m in cols}

    for m in cols:
        sub = df_cal.copy()
        if {"collapse_tag_gt", "t_collapse_gt"}.issubset(sub.columns):
            cfg = _get_cfg()
            try:
                warm_idx = as_int(cfg.get("warmup"), default=0) + as_int(cfg.get("ph_burn"), default=0)
            except Exception:
                warm_idx = 0

            pre_parts: List[pd.DataFrame] = []
            for _, g in sub.groupby(["seed", "factor"]):
                tag = _runlevel_tag_from_epochwise(g, warm_idx)
                tc = g["t_collapse_gt"].iloc[0]
                if (tag == "soft") and pd.notna(tc):
                    pre_parts.append(g[g["epoch"] < int(tc)])
                elif tag == "hard":
                    # exclude hard-collapse runs from normalization
                    continue
                else:
                    pre_parts.append(g)
            sub = pd.concat(pre_parts, ignore_index=True) if pre_parts else sub.iloc[0:0]

        arr = sub[m].to_numpy(dtype=np.float32) if m in sub.columns else np.array([], dtype=np.float32)
        fin = np.isfinite(arr)
        if fin.sum() >= 16:
            med = float(np.median(arr[fin]))
            mad = float(np.median(np.abs(arr[fin] - med))) + 1e-8
        else:
            med, mad = 0.0, 1.0
        stats[m] = (med, 1.4826 * mad)
    return stats


def _fit_global_robust_norm(df_cal: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    """Robust med/MAD over all data (no pre-collapse filtering)."""
    stats: Dict[str, Tuple[float, float]] = {}
    X = df_cal[cols].to_numpy(dtype=np.float32)
    for j, m in enumerate(cols):
        col = X[:, j]
        finite = np.isfinite(col)
        if finite.sum() >= 16:
            med = float(np.median(col[finite]))
            mad = float(np.median(np.abs(col[finite] - med))) + 1e-8
        else:
            med, mad = 0.0, 1.0
        stats[m] = (med, 1.4826 * mad)
    return stats


def _apply_global_norm_impute(
    X: np.ndarray,
    stats: Dict[str, Tuple[float, float]],
    cols: List[str],
) -> np.ndarray:
    """Apply robust normalization and impute missing values with median."""
    Xn = X.copy()
    for j, m in enumerate(cols):
        med, scale = stats.get(m, (0.0, 1.0))
        if scale <= 0:
            scale = 1.0
        col = Xn[:, j]
        mask = ~np.isfinite(col)
        col[mask] = med
        Xn[:, j] = (col - med) / scale
    Xn[np.isinf(Xn)] = 0.0
    return Xn


# Helper: reduce epochwise collapse_tag_gt to a run-level tag (hard > soft > none)
def _runlevel_tag_from_epochwise(df: pd.DataFrame, warm_idx: int) -> str:
    """Reduce epochwise collapse_tag_gt to a run-level tag (hard > soft > none).

    Preference is to consider post-warm epochs only when available.
    """
    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty:
        return "none"
    if "collapse_tag_gt" not in df.columns:
        return "none"

    use = df
    if "epoch" in df.columns:
        ep = pd.to_numeric(df["epoch"], errors="coerce")
        post = df[ep >= int(warm_idx)]
        if not post.empty:
            use = post

    tags = {str(x) for x in use["collapse_tag_gt"].dropna().astype(str).unique().tolist()}
    if "hard" in tags:
        return "hard"
    if "soft" in tags:
        return "soft"
    return "none"


def _train_logistic_ridge_balanced(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, float]:
    """Train balanced logistic regression with L2 regularization via SGD."""
    _ = groups  # groups is currently unused; kept for signature parity

    # Guard: degenerate labels -> deterministic "disabled" classifier
    if (y.sum() == 0) or (y.sum() == len(y)):
        w = np.zeros(X.shape[1], dtype=np.float32)
        b = -10.0 if y.sum() == 0 else 10.0
        return w, float(b)

    pos = max(1e-6, float(y.mean()))
    w_pos = 0.5 / pos
    w_neg = 0.5 / (1 - pos)

    w = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0

    for _ in range(int(steps)):
        z = np.clip(X @ w + b, -20.0, 20.0)
        p = _sigmoid_stable(z)
        w_i = np.where(y > 0.5, w_pos, w_neg)
        grad_w = (X.T @ ((p - y) * w_i)) / len(y) + float(l2) * w
        grad_b = float(np.mean((p - y) * w_i))
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b

    return w, float(b)


def _partition_groups_with_positives(
    uniq_groups: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int = 5,
) -> List[np.ndarray]:
    """
    Deterministic stratified partition of group ids into folds.

    Strategy: put all positive groups into k buckets round-robin, then
    fill remaining buckets with negatives round-robin. Falls back to
    even split if there are no positive groups.
    """
    uniq = np.array(sorted(uniq_groups.astype(np.int64)))
    k = int(max(2, min(len(uniq), as_int(folds, default=2))))

    pos_groups: List[int] = []
    neg_groups: List[int] = []
    for g_id in uniq:
        mask = groups.astype(np.int64) == g_id
        if mask.any() and (y[mask].sum() > 0):
            pos_groups.append(int(g_id))
        else:
            neg_groups.append(int(g_id))

    buckets: List[List[int]] = [[] for _ in range(k)]
    for i, g_id in enumerate(pos_groups):
        buckets[i % k].append(g_id)
    for i, g_id in enumerate(neg_groups):
        buckets[i % k].append(g_id)

    return [np.array(sorted(b), dtype=np.int64) for b in buckets]


def _first_run_end(hit_idx: np.ndarray, L: int) -> int:
    """Return index into hit_idx for the end of the first run of length >= L, or -1."""
    L = int(L)
    if len(hit_idx) < L:
        return -1
    r = 1
    for j in range(1, len(hit_idx)):
        if hit_idx[j] == hit_idx[j - 1] + 1:
            r += 1
            if r >= L:
                return j
        else:
            r = 1
    return -1


def _oof_probs_for_params(
    Xn: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    steps: int,
    l2: float,
    lr: float,
    folds: int,
) -> np.ndarray:
    """
    Deterministic grouped OOF probabilities for a fixed (steps, l2) setting.

    Groups are seeds; we partition them deterministically so every train split
    contains positives (when available).

    Returns a float32 vector of OOF probabilities aligned with `y`.
    """
    uniq = np.unique(groups.astype(np.int64))
    k = int(max(2, min(len(uniq), as_int(folds, default=2))))
    folds_idx = _partition_groups_with_positives(uniq.copy(), y, groups, folds=k)

    p = np.full(y.shape, np.nan, dtype=np.float32)

    for va_seeds in folds_idx:
        va_mask = np.isin(groups, va_seeds)
        tr_mask = ~va_mask
        if tr_mask.sum() == 0 or va_mask.sum() == 0:
            continue
        w, b = _train_logistic_ridge_balanced(Xn[tr_mask], y[tr_mask], groups[tr_mask], steps=steps, lr=lr, l2=l2)
        z = Xn[va_mask] @ w + b
        p[va_mask] = _sigmoid_stable(z)

    return p


def _cv_grouped_fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fp_mask: np.ndarray,
    steps_grid: List[int],
    l2_grid: List[float],
    lr: float,
    folds: int,
    fp_cap: float,
    epochs: np.ndarray,
    tc_soft: np.ndarray,
    warm_idx: int,
) -> Optional[Tuple[int, float, float]]:
    """
    Pick (steps, l2, threshold) by maximizing expected OOF lead time under an FP cap.
    Tie-break: sensitivity.
    """
    if y.sum() == 0:
        return None

    uniq = np.unique(groups)
    if len(uniq) < max(2, folds):
        folds = max(2, min(len(uniq), folds))
    folds_idx = _partition_groups_with_positives(uniq.copy(), y, groups, folds)

    best_lead = -np.inf
    best_sens = -np.inf
    best_info: Optional[Tuple[int, float, float]] = None

    for l2 in l2_grid:
        for steps in steps_grid:
            oof_p = np.full_like(y, np.nan, dtype=np.float32)
            for va_seeds in folds_idx:
                tr = ~np.isin(groups, va_seeds)
                va = ~tr
                if va.sum() == 0 or tr.sum() == 0:
                    continue
                w, b = _train_logistic_ridge_balanced(X[tr], y[tr], groups[tr], steps=steps, lr=lr, l2=l2)
                z_va = X[va] @ w + b
                oof_p[va] = _sigmoid_stable(z_va)

            mask = np.isfinite(oof_p)
            if mask.sum() == 0:
                continue

            elig = (fp_mask == 1) & (y == 0) & mask

            rel = mask & (elig | (y == 1))
            scores = oof_p[rel]
            if scores.size > 0:
                try:
                    ts = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 61)))
                except Exception:
                    ts = np.linspace(0.05, 0.95, 91)
            else:
                ts = np.linspace(0.05, 0.95, 91)

            for t in ts:
                pred = oof_p >= t

                fp_rate = pred[elig].mean() if elig.any() else 0.0
                if not np.isfinite(fp_rate) or fp_rate > fp_cap:
                    continue

                leads: List[float] = []
                hits = 0
                total_pos = 0

                for seed in uniq.astype(np.int64):
                    gmask = (groups == seed) & mask
                    if not gmask.any():
                        continue
                    tc = np.nanmean(tc_soft[groups == seed])
                    if np.isfinite(tc):
                        total_pos += 1
                        idx = np.where(gmask & (epochs >= warm_idx) & pred)[0]
                        if idx.size > 0:
                            t_warn = int(epochs[idx.min()])
                            if t_warn < tc:
                                leads.append(float(tc - t_warn))
                                hits += 1

                sens = (hits / max(1, total_pos)) if total_pos > 0 else 0.0
                mean_lead = float(np.mean(leads)) if len(leads) > 0 else -np.inf

                if (mean_lead > best_lead + 1e-9) or (abs(mean_lead - best_lead) <= 1e-9 and sens > best_sens):
                    best_lead, best_sens, best_info = mean_lead, sens, (int(steps), float(l2), float(t))

    return best_info


def _fam_alarm_at(
    i: int,
    Z: np.ndarray,
    det_features: List[str],
    dir_map: Dict[str, str],
    cols: List[str],
    z_thr: float,
    K: int,
) -> bool:
    """Check if family alarm is triggered at index i."""
    if not cols:
        return False
    lo, hi = max(0, i - int(K)), min(Z.shape[0] - 1, i + int(K))
    for m in cols:
        if m not in det_features:
            continue
        j = det_features.index(m)
        zwin = Z[lo : hi + 1, j]
        d = dir_map.get(m)
        if d is None:
            continue
        if d == "up" and np.nanmax(zwin) >= float(z_thr):
            return True
        if d == "down" and np.nanmin(zwin) <= -float(z_thr):
            return True
    return False


def _gated_runlevel_fp_for_threshold(
    meta_rows: pd.DataFrame,
    X_raw: np.ndarray,
    p: np.ndarray,
    det_features: List[str],
    stats: Dict[str, Tuple[float, float]],
    dir_map: Dict[str, str],
    rp_flags: Dict[Tuple[int, str], int],
    warm: int,
    z_thr: float,
    K: int,
    warn_consec: int,
) -> float:
    """
    Run-level FP on factor=='none' after warm, under deployed gate at fixed τ.
    Threshold is applied by masking p to -inf *before* this is called.
    """
    Xz = _apply_global_norm_impute(X_raw, stats, det_features)
    geom_cols = [c for c in ("cos_disp", "var_out_k") if c in det_features]
    dyn_cols = [c for c in ("ftle", "ftle_lowent") if c in det_features]

    run_warn: Dict[Tuple[int, str], Optional[int]] = {}

    for (sd, fc), indices in meta_rows.groupby(["seed", "factor"]).groups.items():
        idx = np.asarray(sorted(indices), dtype=np.int64)
        sub = meta_rows.iloc[idx].sort_values("epoch")
        order = np.argsort(sub["epoch"].to_numpy())
        idx = idx[order]
        epochs = meta_rows.iloc[idx]["epoch"].to_numpy().astype(np.int64)
        Zg = Xz[idx]
        pg = p[idx]

        elig = (epochs >= int(warm)) & np.isfinite(pg) & (pg > -np.inf)
        hit_idx = np.where(elig)[0]
        j_end = _first_run_end(hit_idx, int(warn_consec))

        t_warn: Optional[int] = None
        if j_end >= 0:
            i = int(hit_idx[j_end])
            geom_ok = _fam_alarm_at(i, Zg, det_features, dir_map, geom_cols, z_thr, K)
            dyn_ok = _fam_alarm_at(i, Zg, det_features, dir_map, dyn_cols, z_thr, K)
            rp_under = bool(rp_flags.get((int(sd), str(fc)), 0))
            gate_ok = dyn_ok if rp_under else (geom_ok or dyn_ok)
            if gate_ok:
                t_warn = int(epochs[i])

        run_warn[(int(sd), str(fc))] = t_warn

    none_runs = [(int(sd), str(fc)) for (sd, fc), _ in meta_rows.groupby(["seed", "factor"]) if str(fc) == "none"]

    flags: List[bool] = []
    for k in none_runs:
        tw = run_warn.get(k)
        flags.append(bool(tw is not None and tw >= int(warm)))

    return float(np.mean(flags)) if flags else float("nan")


def map_threshold_to_gated_fp(
    meta_rows: pd.DataFrame,
    X_raw: np.ndarray,
    p: np.ndarray,
    det_features: List[str],
    stats: Dict[str, Tuple[float, float]],
    dir_map: Dict[str, str],
    rp_flags: Dict[Tuple[int, str], int],
    warm: int,
    z_thr: float,
    K: int,
    warn_consec: int,
    fp_cap: float,
) -> Tuple[float, float]:
    """
    Deterministically map τ→τ′ so gated run-level FP on factor=='none' ≤ fp_cap.
    Returns (tau_prime, measured_fp).
    """
    ep = meta_rows["epoch"].to_numpy().astype(np.int64)
    fac = meta_rows["factor"].astype(str).to_numpy()
    elig = (ep >= int(warm)) & np.isfinite(p) & (fac == "none")
    scores = np.asarray(p[elig], dtype=np.float32)

    if scores.size == 0:
        print("[WARN] τ′ calibration skipped: no factor=='none' rows after warm; failing closed with τ′=1.0")
        return 1.0, float("nan")

    qs = np.linspace(0.05, 0.995, 191, dtype=np.float64)
    try:
        ts = list(map(float, np.unique(np.quantile(scores, qs)).tolist()))
    except Exception:
        ts = list(np.linspace(0.05, 0.95, 91))

    best_tau, best_fp = float(ts[-1]), float("inf")

    for t in ts:
        p_masked = np.where(p >= float(t), p, -np.inf).astype(np.float32)
        fp = _gated_runlevel_fp_for_threshold(
            meta_rows,
            X_raw,
            p_masked,
            det_features,
            stats,
            dir_map,
            rp_flags,
            warm,
            z_thr,
            K,
            warn_consec,
        )
        if np.isnan(fp):
            continue
        if fp <= float(fp_cap):
            best_tau, best_fp = float(t), float(fp)
            break
        best_fp = float(fp)

    return best_tau, best_fp


# Stable public surface for phase-1: keep it minimal.
__all__ = ["map_threshold_to_gated_fp"]
