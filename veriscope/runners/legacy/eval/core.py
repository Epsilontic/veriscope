# veriscope/runners/legacy/eval/core.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

# ---------------------------
# Imports from refactored legacy modules (no CLI imports)
# ---------------------------

# ---------------------------
# Lightweight defaults; runtime fetched lazily to avoid import-time stalls
# ---------------------------
CFG: Mapping[str, Any] = {}
OUTDIR: Path = Path(".")
SUCCESS_TARGET: Mapping[str, Any] = {"min_lead": 2}


def _get_runtime():
    try:
        from veriscope.runners.legacy import runtime as r  # local import

        return r
    except Exception:
        return None


def _get_cfg() -> Mapping[str, Any]:
    r = _get_runtime()
    return getattr(r, "CFG", CFG) if r is not None else CFG


def _get_outdir() -> Path:
    r = _get_runtime()
    return Path(getattr(r, "OUTDIR", OUTDIR)) if r is not None else Path(OUTDIR)


def _get_success_target() -> Mapping[str, Any]:
    r = _get_runtime()
    return getattr(r, "SUCCESS_TARGET", SUCCESS_TARGET) if r is not None else SUCCESS_TARGET

# Numeric / config / IO shims.
# Prefer the shared helpers in legacy.utils; fall back to local definitions if missing.
try:
    from veriscope.runners.legacy.utils import (
        as_int,
        as_float,
        save_json,
        to_numeric_series,
        to_numeric_opt,
        qlin,
        quantile2,
    )
except Exception:  # pragma: no cover

    def as_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def as_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def save_json(obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    def to_numeric_series(s: Any, errors: str = "coerce") -> pd.Series:
        return pd.to_numeric(s, errors=errors)

    def to_numeric_opt(s: Any) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def qlin(arr: np.ndarray, q: float) -> float:
        return float(np.quantile(arr, q))

    def quantile2(arr: np.ndarray, qlo: float, qhi: float) -> Tuple[float, float]:
        return (float(np.quantile(arr, qlo)), float(np.quantile(arr, qhi)))


if TYPE_CHECKING:
    from veriscope.core.window import WindowDecl  # pragma: no cover
else:
    WindowDecl = Any  # runtime type


# Local tiny numeric helper
def _as_float_array(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        arr = np.asarray([], dtype=float)
    return arr


# Optional global DeclTransport instance (may be installed by runner)
_DECL_TRANSPORT: Any = None


def install_decl_transport(transport: Any) -> None:
    """Allow runner to install a DeclTransport-like adapter for offline eval."""
    global _DECL_TRANSPORT
    _DECL_TRANSPORT = transport


# ---------------------------
# Events & evaluation (UNIFIED GT)
# ---------------------------
def compute_events(
    df: pd.DataFrame,
    metrics_for_ph: Sequence[str],
    dir_map: Mapping[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    dbg: List[Dict[str, Any]] = []

    cfg = _get_cfg()
    burn = as_int(cfg.get("warmup"), default=0) + as_int(cfg.get("ph_burn"), default=0)
    win_default = as_int(cfg.get("ph_win"), default=0)
    lam = as_float(cfg.get("ph_lambda"), default=0.0)
    min_points = as_int(cfg.get("ph_min_points"), default=0)

    if df is None or df.empty:
        return pd.DataFrame([]), pd.DataFrame([])

    try:
        from veriscope.runners.legacy.detectors.baselines import (
            _prep_series_for_ph,
            ph_window_sparse,
            SCHEDULED_METRICS,
        )
    except Exception as e:
        raise RuntimeError("PH helpers are unavailable; check baselines module imports.") from e

    from typing import cast

    for key, g in df.groupby(["seed", "factor"]):
        seed, factor = cast(Tuple[Any, Any], key)
        g0 = g[g.epoch >= 0].sort_values("epoch").copy()

        # Coerce t_collapse safely
        t_raw = g0["t_collapse_gt"].iloc[0] if "t_collapse_gt" in g0.columns else np.nan
        t_collapse_i = as_int(t_raw, default=-1) if pd.notna(t_raw) else -1
        t_collapse = t_collapse_i if t_collapse_i >= 0 else None

        ctag = str(g0["collapse_tag_gt"].iloc[0]) if "collapse_tag_gt" in g0.columns else "none"
        t_map: Dict[str, Optional[int]] = {}

        for m in metrics_for_ph:
            xs = _prep_series_for_ph(g0, m)
            win_m = (
                as_int(cfg.get("ph_win_short"), default=win_default)
                if (m in SCHEDULED_METRICS)
                else int(win_default)
            )
            d = dir_map.get(m, "up")
            t, zs, cs = ph_window_sparse(
                xs,
                win=int(win_m),
                lam=float(lam),
                direction=d,
                burn_in=int(burn),
                min_points=int(min_points),
                two_sided=bool(cfg.get("ph_two_sided")),
            )
            thr = int(max(int(burn), int(win_m)))
            t_i = as_int(t, default=-1) if t is not None else -1
            violation = 1 if (t_i >= 0 and t_i < thr) else 0
            t_out: Optional[int] = None if violation == 1 else (t_i if t_i >= 0 else None)
            zsa = np.asarray(zs if zs is not None else [], dtype=float)
            csa = np.asarray(cs if cs is not None else [], dtype=float)
            t_map[m] = t_out
            dbg.append(
                dict(
                    seed=int(seed),
                    factor=str(factor),
                    metric=m,
                    z_scores=json.dumps([float(z) for z in zsa]),
                    cusum=json.dumps([float(s) for s in csa]),
                    retro_violation=int(violation),
                )
            )

        rows.append(
            dict(
                run_id=f"s{int(seed)}-{str(factor)}",
                seed=int(seed),
                factor=str(factor),
                collapse_tag=ctag,
                t_collapse=t_collapse,
                ph_win=win_default,
                ph_lambda=lam,
                ph_two_sided=int(bool(cfg.get("ph_two_sided"))),
                heavy_every=cfg.get("heavy_every"),
                metric_batches=cfg.get("metric_batches"),
                var_k_energy=cfg.get("var_k_energy"),
                var_k_max=cfg.get("var_k_max"),
                **{f"t_{k}": v for k, v in t_map.items()},
            )
        )

    return pd.DataFrame(rows), pd.DataFrame(dbg)


def _first_t_column(tr_df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in tr_df.columns if c.startswith("t_") and c != "t_collapse"]
    if not cols:
        return None
    if len(cols) == 1:
        return cols[0]
    counts = {c: int(tr_df[c].notna().sum()) for c in cols}
    return max(counts, key=lambda c: counts[c])


def mark_events_epochwise(
    df_runs: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Add boolean overlays to df_runs for warn/collapse based on `events`:
    - is_warn_epoch_<prefix>: True on the K-length window ending at t_warn (inclusive)
    - is_collapse_epoch_<prefix>: True exactly at t_collapse

    Auto-picks warn source:
      * uses 't_warn' if present; otherwise falls back to the most-populated 't_<metric>' column
        (via _first_t_column), ignoring 't_collapse'.

    Robust if multiple rows exist per (seed, factor): the first row is used.
    No-ops cleanly on empty frames.
    """
    df = df_runs.copy()
    warn_col = f"is_warn_epoch_{prefix}"
    col_col = f"is_collapse_epoch_{prefix}"
    df[warn_col] = False
    df[col_col] = False

    if events is None or events.empty or df.empty:
        return df

    warn_key = "t_warn" if ("t_warn" in events.columns) else _first_t_column(events)
    ev = events.set_index(["seed", "factor"])

    cfg = _get_cfg()
    K = int(cfg.get("warn_consec", 3))
    from typing import cast

    for key, g in df.groupby(["seed", "factor"], sort=False):
        seed, factor = cast(Tuple[Any, Any], key)
        sf_key = (int(seed), str(factor))
        try:
            row_df = ev.loc[[sf_key]]  # always DataFrame
        except KeyError:
            continue
        row = row_df.iloc[0]

        # warn window
        if isinstance(warn_key, str) and (warn_key in row.index):
            tw = row[warn_key]
            twi = as_int(tw, default=-1) if pd.notna(tw) else -1
            if (twi >= 0) and K > 0:
                e = g["epoch"].to_numpy(dtype=int)
                win = (e >= twi - (K - 1)) & (e <= twi)
                if win.any():
                    df.loc[g.index[win], warn_col] = True

        # collapse point
        tc_raw = row["t_collapse"] if "t_collapse" in row.index else row.get("t_collapse_gt", np.nan)
        tci = as_int(tc_raw, default=-1) if pd.notna(tc_raw) else -1
        if tci >= 0:
            df.loc[g.index, col_col] = g["epoch"].to_numpy(dtype=int) == tci

    return df


def assert_overlay_consistency(
    df_epoch: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
) -> None:
    warn_col = f"is_warn_epoch_{prefix}"
    col_col = f"is_collapse_epoch_{prefix}"
    if df_epoch is None or events is None or df_epoch.empty or events.empty:
        return

    ev = events.set_index(["seed", "factor"])
    mismatches: List[Dict[str, Any]] = []

    from typing import cast

    for key, g in df_epoch.groupby(["seed", "factor"]):
        seed, factor = cast(Tuple[Any, Any], key)
        sf_key = (int(seed), str(factor))
        try:
            row_df = ev.loc[[sf_key]]
        except KeyError:
            continue

        tw_flags = g[g[warn_col]].sort_values("epoch")["epoch"].tolist()
        tc_flags = g[g[col_col]].sort_values("epoch")["epoch"].tolist()

        tw = int(tw_flags[-1]) if tw_flags else None
        tc = int(tc_flags[0]) if tc_flags else None

        row_ev = row_df.iloc[0]
        e_tw_raw = row_ev["t_warn"] if "t_warn" in row_ev.index else np.nan
        e_tc_raw = row_ev["t_collapse"] if "t_collapse" in row_ev.index else np.nan

        e_tw_i = as_int(e_tw_raw, default=-1) if pd.notna(e_tw_raw) else -1
        e_tc_i = as_int(e_tc_raw, default=-1) if pd.notna(e_tc_raw) else -1
        tw_i = tw if tw is not None else -1
        tc_i = tc if tc is not None else -1

        bad_warn = (
            ((e_tw_i >= 0) and (tw_i < 0))
            or ((e_tw_i < 0) and (tw_i >= 0))
            or ((e_tw_i >= 0) and (tw_i != e_tw_i))
        )
        bad_col = (
            ((e_tc_i >= 0) and (tc_i < 0))
            or ((e_tc_i < 0) and (tc_i >= 0))
            or ((e_tc_i >= 0) and (tc_i != e_tc_i))
        )

        if bad_warn or bad_col:
            mismatches.append(
                dict(
                    seed=int(seed),
                    factor=str(factor),
                    warn_overlay=tw,
                    warn_events=(int(e_tw_i) if e_tw_i >= 0 else None),
                    collapse_overlay=tc,
                    collapse_events=(int(e_tc_i) if e_tc_i >= 0 else None),
                )
            )

    if mismatches:
        cap = 50
        path = _get_outdir() / f"overlay_mismatches_{prefix}.json"
        save_json({"count": len(mismatches), "items": mismatches[:cap]}, path)
        print(f"[WARN] overlay mismatches: {len(mismatches)} (showing first {cap}; details in {path})")


def bootstrap_stratified(
    rows: pd.DataFrame,
    B: int = 200,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(123456)
    factors = sorted(rows["factor"].unique().tolist())
    if not factors:
        return {}

    vals_detect: List[float] = []
    vals_fp: List[float] = []
    vals_med: List[float] = []

    cfg = _get_cfg()
    warm = as_int(cfg.get("warmup"), default=0) + as_int(cfg.get("ph_burn"), default=0)

    for _ in range(B):
        boot_parts = []
        for f in factors:
            seeds_f = sorted(rows[rows["factor"] == f]["seed"].unique().tolist())
            if not seeds_f:
                continue
            draw = rng.choice(seeds_f, size=len(seeds_f), replace=True)
            for s in draw:
                boot_parts.append(rows[(rows["factor"] == f) & (rows["seed"] == s)])
        if not boot_parts:
            continue
        boot = pd.concat(boot_parts, ignore_index=True)

        trig = boot[boot["collapse_tag"] == "soft"]
        ncol = len(trig)
        succ = int(
            (
                (trig["t_warn"].notna())
                & ((trig["t_collapse"] - trig["t_warn"]) >= _get_success_target().get("min_lead", 2))
            ).sum()
        )
        vals_detect.append(float(succ / max(1, ncol)))

        non_trig = boot[boot["collapse_tag"] == "none"]
        denom = max(1, len(non_trig))
        fp = (
            float(np.mean((non_trig["t_warn"].notna()) & (non_trig["t_warn"] >= warm)))
            if denom > 0
            else 1.0
        )
        vals_fp.append(float(fp))

        leads = (trig["t_collapse"] - trig["t_warn"]).dropna().to_numpy()
        vals_med.append(float(np.median(leads)) if leads.size > 0 else float("nan"))

    def ci(v: List[float]) -> Tuple[float, float]:
        arr = np.array(v, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return (np.nan, np.nan)
        lo, hi = quantile2(arr, 0.025, 0.975)
        return (float(lo), float(hi))

    return dict(
        detect_rate_ci=ci(vals_detect),
        fp_rate_ci=ci(vals_fp),
        lead_median_ci=ci(vals_med),
    )


def summarize_detection(rows: pd.DataFrame, warm_idx: int) -> pd.DataFrame:
    if rows is None or len(rows) == 0:
        return pd.DataFrame(columns=["kind", "n", "successes", "value", "lo", "hi"])

    if "collapse_tag" not in rows.columns:
        if "collapse_tag_gt" in rows.columns:
            rows = rows.rename(columns={"collapse_tag_gt": "collapse_tag"})
        else:
            rows = rows.copy()
            rows["collapse_tag"] = "none"

    out: List[Dict[str, Any]] = []

    trig = rows[rows["collapse_tag"] == "soft"].copy()
    n_collapse = len(trig)

    if int(n_collapse) <= 0:
        non_trig = rows[rows["collapse_tag"] == "none"]
        _ntw = to_numeric_series(non_trig["t_warn"], errors="coerce")
        mask_nt = (_ntw.notna()) & (_ntw >= int(warm_idx))
        fp_nontrig = float(np.mean(mask_nt.to_numpy(dtype=bool))) if len(non_trig) > 0 else 1.0

        out.append(dict(kind="detect_rate", n=0, successes=0, value=0.0, lo=np.nan, hi=np.nan))
        out.append(dict(kind="fp_nontriggered_after_warm", n=int(len(non_trig)), value=fp_nontrig))
        out.append(dict(kind="lead_time", n=0, med=np.nan, q1=np.nan, q3=np.nan))

        boot = bootstrap_stratified(rows)
        out.append(dict(kind="detect_rate_ci_boot", lo=boot.get("detect_rate_ci", (np.nan, np.nan))[0],
                        hi=boot.get("detect_rate_ci", (np.nan, np.nan))[1]))
        out.append(dict(kind="fp_rate_ci_boot", lo=boot.get("fp_rate_ci", (np.nan, np.nan))[0],
                        hi=boot.get("fp_rate_ci", (np.nan, np.nan))[1]))
        out.append(dict(kind="lead_median_ci_boot", lo=boot.get("lead_median_ci", (np.nan, np.nan))[0],
                        hi=boot.get("lead_median_ci", (np.nan, np.nan))[1]))
        return pd.DataFrame(out)

    _tw = to_numeric_series(trig["t_warn"], errors="coerce")
    _tc = to_numeric_series(trig["t_collapse"], errors="coerce")
    lead_min = as_int(_get_success_target().get("min_lead", 2), default=2)
    mask = (_tw.notna()) & ((_tc - _tw) >= int(lead_min))

    successes = int(np.count_nonzero(mask.to_numpy(dtype=bool)))
    detect_rate = successes / max(1, n_collapse)

    if n_collapse > 0:
        z = 1.96
        phat = detect_rate
        denom = 1 + z**2 / n_collapse
        center = (phat + z * z / (2 * n_collapse)) / denom
        half = (
            z
            * math.sqrt((phat * (1 - phat) / n_collapse) + z * z / (4 * n_collapse * n_collapse))
            / denom
        )
        d_lo, d_hi = center - half, center + half
    else:
        d_lo = d_hi = np.nan

    non_trig = rows[rows["collapse_tag"] == "none"]
    _ntw = to_numeric_series(non_trig["t_warn"], errors="coerce")
    mask_nt = (_ntw.notna()) & (_ntw >= int(warm_idx))
    fp_nontrig = float(np.mean(mask_nt.to_numpy(dtype=bool))) if len(non_trig) > 0 else 1.0

    leads = (trig["t_collapse"] - trig["t_warn"]).dropna().to_numpy(dtype=float)
    med = q1 = q3 = np.nan
    if leads.size > 0:
        med = float(np.median(leads))
        q1 = float(qlin(leads, 0.25))
        q3 = float(qlin(leads, 0.75))

    out.append(dict(kind="detect_rate", n=n_collapse, successes=successes, value=detect_rate, lo=d_lo, hi=d_hi))
    out.append(dict(kind="fp_nontriggered_after_warm", n=int(len(non_trig)), value=fp_nontrig))
    out.append(dict(kind="lead_time", n=int(leads.size), med=med, q1=q1, q3=q3))

    boot = bootstrap_stratified(rows)
    out.append(dict(kind="detect_rate_ci_boot", lo=boot.get("detect_rate_ci", (np.nan, np.nan))[0],
                    hi=boot.get("detect_rate_ci", (np.nan, np.nan))[1]))
    out.append(dict(kind="fp_rate_ci_boot", lo=boot.get("fp_rate_ci", (np.nan, np.nan))[0],
                    hi=boot.get("fp_rate_ci", (np.nan, np.nan))[1]))
    out.append(dict(kind="lead_median_ci_boot", lo=boot.get("lead_median_ci", (np.nan, np.nan))[0],
                    hi=boot.get("lead_median_ci", (np.nan, np.nan))[1]))

    return pd.DataFrame(out)


def summarize_runlevel_fp(events: pd.DataFrame, warm_idx: int) -> float:
    """
    Run-level FP = fraction of 'none' runs that have any t_warn at/after warm_idx.
    Robust to empty frames / missing columns.
    """
    if events is None or events.empty:
        return float("nan")

    req = {"seed", "factor", "collapse_tag", "t_warn"}
    if not req.issubset(set(events.columns)):
        return float("nan")

    none_runs = events[events["collapse_tag"] == "none"].copy()
    if none_runs.empty:
        return float("nan")

    flags: List[bool] = []
    from typing import cast

    for key, g in none_runs.groupby(["seed", "factor"]):
        sd, fc = cast(Tuple[Any, Any], key)
        tw = g["t_warn"].dropna()
        hit = (len(tw) > 0) and (
            as_int(tw.iloc[0], default=-1) >= as_int(warm_idx, default=-1)
        )
        flags.append(bool(hit))

    return float(np.mean(flags)) if flags else float("nan")


# ---------------------------
# RP adequacy: JL vs native agreement pre-warm
# ---------------------------
def rp_adequacy_flags(
    df: pd.DataFrame,
    warm: int,
    corr_min: float = 0.9,
    min_pts: int = 8,
) -> Dict[Tuple[int, str], int]:
    """Return {(seed,factor): 1/0} flag where geometry appears under-resolved in JL space."""
    flags: Dict[Tuple[int, str], int] = {}
    from typing import cast

    for key, g in df.groupby(["seed", "factor"]):
        seed, factor = cast(Tuple[Any, Any], key)
        gg = g.sort_values("epoch")
        pre = gg[gg["epoch"] < warm]

        # eff_dim vs eff_dim_gt
        x1 = pre["eff_dim"].to_numpy(dtype=float) if "eff_dim" in pre.columns else np.array([], dtype=float)
        if "eff_dim_gt" in pre.columns:
            y1 = pre["eff_dim_gt"].to_numpy(dtype=float)
        else:
            y1 = pre["eff_dim"].to_numpy(dtype=float) if "eff_dim" in pre.columns else np.array([], dtype=float)

        m1 = np.isfinite(x1) & np.isfinite(y1)
        corr1 = np.nan
        if int(m1.sum()) >= int(min_pts):
            corr1 = float(np.corrcoef(x1[m1], y1[m1])[0, 1])

        # var_out_k vs native
        x2 = pre["var_out_k"].to_numpy(dtype=float) if "var_out_k" in pre.columns else np.array([], dtype=float)
        y2 = (
            pre["var_out_k_native"].to_numpy(dtype=float)
            if "var_out_k_native" in pre.columns
            else np.full_like(x2, np.nan)
        )

        m2 = np.isfinite(x2) & np.isfinite(y2)
        corr2 = np.nan
        if int(m2.sum()) >= int(min_pts):
            corr2 = float(np.corrcoef(x2[m2], y2[m2])[0, 1])

        flag = int(
            (np.isfinite(corr1) and corr1 < corr_min)
            or (np.isfinite(corr2) and corr2 < corr_min)
        )
        flags[(as_int(seed, default=0), str(factor))] = flag

    return flags


# ---------------------------
# Helper: series_or_empty
# ---------------------------
def series_or_empty(obj: Any) -> pd.Series:
    """Return a pandas Series for downstream numeric helpers."""
    if isinstance(obj, pd.Series):
        return obj
    if obj is None:
        return pd.Series([], dtype=float)
    try:
        return pd.Series(obj, dtype=float)
    except Exception:
        return pd.Series([], dtype=float)


def _count_finite_pairs(
    window: WindowDecl,
    past: np.ndarray,
    recent: np.ndarray,
    name: str,
) -> int:
    """Finite-pair count under DeclTransport if installed."""
    try:
        adapter = getattr(window, "_DECL_TRANSPORT", None)
        if adapter is not None:
            tp = adapter.apply(name, past)   # type: ignore[attr-defined]
            tr = adapter.apply(name, recent) # type: ignore[attr-defined]
        elif _DECL_TRANSPORT is not None:
            tp = _DECL_TRANSPORT.apply(name, past)   # type: ignore[attr-defined]
            tr = _DECL_TRANSPORT.apply(name, recent) # type: ignore[attr-defined]
        else:
            tp = np.asarray(past, float)
            tr = np.asarray(recent, float)

        ta = np.asarray(tp, dtype=float)
        ra = np.asarray(tr, dtype=float)
        return int(min(int(np.isfinite(ta).sum()), int(np.isfinite(ra).sum())))
    except Exception:
        return 0


def recompute_gate_series_under_decl(
    df_eval: pd.DataFrame,
    window_decl: WindowDecl,
    W: int,
) -> pd.DataFrame:
    """
    Offline recompute of gate diagnostics under a fixed WindowDecl.

    Returns a COPY of df_eval with:
      gate_worst_tv_calib, gate_eps_stat_calib, gate_gain_calib, gate_warn_calib
    """
    try:
        from veriscope.core.ipm import dPi_product_tv  # product-TV under Φ_W
    except Exception as e:
        raise RuntimeError("dPi_product_tv unavailable; core.ipm import failed.") from e

    try:
        from veriscope.core.calibration import aggregate_epsilon_stat as AGGREGATE_EPSILON
    except Exception:
        AGGREGATE_EPSILON = None  # type: ignore

    df = df_eval.copy()
    cfg = _get_cfg()

    # Normalize key columns to numeric for sorting / slicing
    for c in ("epoch", "seed"):
        if c in df.columns:
            df[c] = to_numeric_opt(df.get(c))

    # Initialize calibrated columns
    df["gate_worst_tv_calib"] = np.nan
    df["gate_eps_stat_calib"] = np.nan
    df["gate_gain_calib"] = np.nan
    df["gate_warn_calib"] = 0

    max_frac = as_float(cfg.get("gate_eps_stat_max_frac", 0.25), default=0.25)
    thr_gain = as_float(cfg.get("gate_gain_thresh", 0.1), default=0.1)
    eps_sens = as_float(cfg.get("gate_epsilon_sens", 0.04), default=0.04)
    alpha = as_float(cfg.get("gate_eps_stat_alpha", 0.05), default=0.05)
    ln2 = math.log(2.0)

    mets = [m for m in getattr(window_decl, "metrics", []) if m in df.columns]
    agg_fn = AGGREGATE_EPSILON

    for _, g in df.groupby(["seed", "factor"], sort=False):
        g = g.sort_values("epoch").copy()
        if len(g) < 2 * W:
            continue

        idxs = list(g.index)

        for pos, idx in enumerate(idxs):
            if pos < 2 * W - 1:
                continue

            ps = slice(pos - 2 * W + 1, pos - W + 1)
            rs = slice(pos - W + 1, pos + 1)

            past = {
                m: _as_float_array(to_numeric_opt(g.get(m)).iloc[ps].to_numpy())
                for m in mets
            }
            recent = {
                m: _as_float_array(to_numeric_opt(g.get(m)).iloc[rs].to_numpy())
                for m in mets
            }

            try:
                tv = dPi_product_tv(window_decl, past, recent)
            except Exception:
                tv = float("nan")

            counts = {
                m: _count_finite_pairs(window_decl, past[m], recent[m], m)
                for m in mets
            }

            try:
                if agg_fn is not None:
                    eps_stat = float(agg_fn(window_decl, counts_by_metric=counts, alpha=alpha))
                else:
                    eps_stat = 0.0
            except Exception:
                eps_stat = 0.0

            if not np.isfinite(eps_stat):
                eps_stat = 0.0
            eps_stat = float(
                min(
                    max(0.0, eps_stat),
                    max_frac * float(getattr(window_decl, "epsilon", 0.0)),
                )
            )

            eps_eff = max(
                0.0,
                float(getattr(window_decl, "epsilon", 0.0)) - eps_stat,
            )

            try:
                s_train = series_or_empty(g.get("train_loss"))
                s_ewma = series_or_empty(g.get("ewma_loss"))
                model_losses = to_numeric_opt(s_train).iloc[rs].to_numpy(dtype=np.float64)
                base_losses = to_numeric_opt(s_ewma).iloc[rs].to_numpy(dtype=np.float64)
                msk = np.isfinite(model_losses) & np.isfinite(base_losses)
                gain_bits = float(((base_losses[msk] - model_losses[msk]).mean()) / ln2) if msk.any() else float("nan")
            except Exception:
                gain_bits = float("nan")

            if "gate_kappa" in g.columns:
                s_kappa = series_or_empty(g.get("gate_kappa"))
                try:
                    kappa = as_float(s_kappa.iloc[pos], default=float("nan"))
                except Exception:
                    kappa = float("nan")
            else:
                kappa = float("nan")

            ok_gain = np.isfinite(gain_bits) and (gain_bits >= thr_gain)
            ok_tv = np.isfinite(tv) and (tv <= eps_eff)
            ok_kappa = (not np.isfinite(kappa)) or (kappa <= eps_sens)
            flag = int(bool(ok_gain and ok_tv and ok_kappa))

            df.loc[idx, "gate_worst_tv_calib"] = float(tv) if np.isfinite(tv) else np.nan
            df.loc[idx, "gate_eps_stat_calib"] = float(eps_stat)
            df.loc[idx, "gate_gain_calib"] = float(gain_bits) if np.isfinite(gain_bits) else np.nan
            df.loc[idx, "gate_warn_calib"] = int(flag)

    return df


__all__ = [
    "compute_events",
    "mark_events_epochwise",
    "assert_overlay_consistency",
    "bootstrap_stratified",
    "summarize_detection",
    "summarize_runlevel_fp",
    "rp_adequacy_flags",
    "series_or_empty",
    "recompute_gate_series_under_decl",
    "install_decl_transport",
]
