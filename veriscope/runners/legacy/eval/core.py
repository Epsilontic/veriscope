# veriscope/runners/legacy/eval/core.py
from __future__ import annotations

import os
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


def _effective_min_lead() -> int:
    """
    Effective minimum required lead time for counting a detection as a success.

    Policy:
      - Base from SUCCESS_TARGET (possibly runtime-overridden)
      - In smoke mode: default relax to 0 (warn at/before collapse counts as success)
      - Optional explicit override: SCAR_MIN_LEAD (non-negative int)
    """
    import os

    base = _get_success_target()
    lead = as_int(base.get("min_lead", 2), default=2)

    # Optional explicit override wins (useful for quick experiments)
    v = os.environ.get("SCAR_MIN_LEAD")
    if v is not None:
        try:
            return max(0, int(str(v).strip()))
        except Exception:
            return max(0, int(lead))

    # Smoke default: relax to 0 unless explicitly overridden above
    env_val = os.environ.get("SCAR_SMOKE", "0").strip().lower()
    if env_val in ("1", "true", "yes", "on"):
        return 0

    return max(0, int(lead))


# Small truthy-env helper.
def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


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


def _get_apply_for_decl(window_decl: WindowDecl):
    """Return an apply(ctx, x) callable for offline eval under a WindowDecl.

    Priority:
      1) window_decl._DECL_TRANSPORT (if present)
      2) module-level _DECL_TRANSPORT (installed via install_decl_transport)
      3) identity
    """
    adapter = getattr(window_decl, "_DECL_TRANSPORT", None)
    if adapter is None:
        adapter = _DECL_TRANSPORT

    if adapter is not None and hasattr(adapter, "apply"):
        # Normalize x to float array before applying
        return lambda ctx, x: adapter.apply(ctx, np.asarray(x, float))  # type: ignore[attr-defined]

    return lambda ctx, x: np.asarray(x, float)


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

        if "collapse_tag_gt" in g0.columns:
            tags = {str(x) for x in g0["collapse_tag_gt"].dropna().unique().tolist()}
            if "hard" in tags:
                ctag = "hard"
            elif "soft" in tags:
                ctag = "soft"
            else:
                ctag = "none"
        else:
            ctag = "none"
        t_map: Dict[str, Optional[int]] = {}

        for m in metrics_for_ph:
            xs = _prep_series_for_ph(g0, m)
            win_m = (
                as_int(cfg.get("ph_win_short"), default=win_default) if (m in SCHEDULED_METRICS) else int(win_default)
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
            ((e_tw_i >= 0) and (tw_i < 0)) or ((e_tw_i < 0) and (tw_i >= 0)) or ((e_tw_i >= 0) and (tw_i != e_tw_i))
        )
        bad_col = (
            ((e_tc_i >= 0) and (tc_i < 0)) or ((e_tc_i < 0) and (tc_i >= 0)) or ((e_tc_i >= 0) and (tc_i != e_tc_i))
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
    warm_idx: int,
    B: int = 200,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(123456)
    factors = sorted(rows["factor"].unique().tolist())
    if not factors:
        return {}

    vals_detect: List[float] = []
    vals_fp: List[float] = []
    vals_med: List[float] = []

    warm = int(warm_idx)

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

        trig0 = boot[boot["collapse_tag"].isin(["soft", "hard"])]
        # Admissibility guard: do not score collapses that occur before warm (gate not yet evaluable).
        if (trig0 is not None) and (len(trig0) > 0) and ("t_collapse" in trig0.columns):
            _tc0 = to_numeric_series(trig0["t_collapse"], errors="coerce")
            trig = trig0[_tc0.notna() & (_tc0 >= float(warm))].copy()
        else:
            trig = trig0

        ncol = len(trig)
        lead_min = int(_effective_min_lead())

        # Optional diagnostic policy: count late warnings as detections.
        # When enabled, any finite t_warn on a triggered run counts as success.
        allow_late = _env_truthy("SCAR_ALLOW_LATE_WARN", default="0")
        if allow_late:
            succ = int(trig["t_warn"].notna().sum())
        else:
            succ = int(((trig["t_warn"].notna()) & ((trig["t_collapse"] - trig["t_warn"]) >= lead_min)).sum())

        vals_detect.append(float(succ / max(1, ncol)))

        non_trig = boot[boot["collapse_tag"] == "none"]
        if len(non_trig) > 0:
            fp = float(np.mean((non_trig["t_warn"].notna()) & (non_trig["t_warn"] >= warm)))
        else:
            fp = float("nan")
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
        rows = rows.copy()
        if "collapse_tag_gt" in rows.columns:
            tags = {str(x) for x in rows["collapse_tag_gt"].dropna().unique().tolist()}
            if "hard" in tags:
                rows["collapse_tag"] = "hard"
            elif "soft" in tags:
                rows["collapse_tag"] = "soft"
            else:
                rows["collapse_tag"] = "none"
        else:
            rows["collapse_tag"] = "none"

    # === Scoring filter (single source of truth) ===
    # Keep all non-triggered runs for FP denominators, and only score collapses that occur
    # at/after warm_idx (gate not yet evaluable pre-warm).
    rows_scored = rows
    try:
        _tag = rows_scored["collapse_tag"].astype(str)
    except Exception:
        _tag = pd.Series(["none"] * len(rows_scored), index=rows_scored.index)

    if "t_collapse" in rows_scored.columns:
        _tc = to_numeric_series(rows_scored["t_collapse"], errors="coerce")
    else:
        _tc = pd.Series([np.nan] * len(rows_scored), index=rows_scored.index)

    keep = (_tag == "none") | (_tc.notna() & (_tc >= float(warm_idx)))
    rows_scored = rows_scored[keep].copy()

    out: List[Dict[str, Any]] = []

    # Include both soft and hard collapses in detection scoring
    trig = rows_scored[rows_scored["collapse_tag"].isin(["soft", "hard"])].copy()

    n_collapse = len(trig)

    if int(n_collapse) <= 0:
        non_trig = rows_scored[rows_scored["collapse_tag"] == "none"]
        _ntw = to_numeric_series(non_trig["t_warn"], errors="coerce")
        mask_nt = (_ntw.notna()) & (_ntw >= int(warm_idx))
        fp_nontrig = float(np.mean(mask_nt.to_numpy(dtype=bool))) if len(non_trig) > 0 else 1.0

        out.append(dict(kind="detect_rate", n=0, successes=0, value=0.0, lo=np.nan, hi=np.nan))
        out.append(dict(kind="fp_nontriggered_after_warm", n=int(len(non_trig)), value=fp_nontrig))
        out.append(dict(kind="lead_time", n=0, med=np.nan, q1=np.nan, q3=np.nan))

        boot = bootstrap_stratified(rows_scored, warm_idx=warm_idx)
        out.append(
            dict(
                kind="detect_rate_ci_boot",
                lo=boot.get("detect_rate_ci", (np.nan, np.nan))[0],
                hi=boot.get("detect_rate_ci", (np.nan, np.nan))[1],
            )
        )
        out.append(
            dict(
                kind="fp_rate_ci_boot",
                lo=boot.get("fp_rate_ci", (np.nan, np.nan))[0],
                hi=boot.get("fp_rate_ci", (np.nan, np.nan))[1],
            )
        )
        out.append(
            dict(
                kind="lead_median_ci_boot",
                lo=boot.get("lead_median_ci", (np.nan, np.nan))[0],
                hi=boot.get("lead_median_ci", (np.nan, np.nan))[1],
            )
        )
        return pd.DataFrame(out)

    _tw = to_numeric_series(trig["t_warn"], errors="coerce")
    _tc = to_numeric_series(trig["t_collapse"], errors="coerce")
    lead_min = int(_effective_min_lead())

    # Optional diagnostic policy: count late warnings as detections.
    # When enabled, any finite t_warn on a triggered run counts as success.
    allow_late = _env_truthy("SCAR_ALLOW_LATE_WARN", default="0")
    if allow_late:
        mask = _tw.notna()
    else:
        mask = (_tw.notna()) & ((_tc - _tw) >= lead_min)

    successes = int(np.count_nonzero(mask.to_numpy(dtype=bool)))
    detect_rate = successes / max(1, n_collapse)

    if n_collapse > 0:
        z = 1.96
        phat = detect_rate
        denom = 1 + z**2 / n_collapse
        center = (phat + z * z / (2 * n_collapse)) / denom
        half = z * math.sqrt((phat * (1 - phat) / n_collapse) + z * z / (4 * n_collapse * n_collapse)) / denom
        d_lo, d_hi = center - half, center + half
    else:
        d_lo = d_hi = np.nan

    non_trig = rows_scored[rows_scored["collapse_tag"] == "none"]
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

    boot = bootstrap_stratified(rows_scored, warm_idx=warm_idx)
    out.append(
        dict(
            kind="detect_rate_ci_boot",
            lo=boot.get("detect_rate_ci", (np.nan, np.nan))[0],
            hi=boot.get("detect_rate_ci", (np.nan, np.nan))[1],
        )
    )
    out.append(
        dict(
            kind="fp_rate_ci_boot",
            lo=boot.get("fp_rate_ci", (np.nan, np.nan))[0],
            hi=boot.get("fp_rate_ci", (np.nan, np.nan))[1],
        )
    )
    out.append(
        dict(
            kind="lead_median_ci_boot",
            lo=boot.get("lead_median_ci", (np.nan, np.nan))[0],
            hi=boot.get("lead_median_ci", (np.nan, np.nan))[1],
        )
    )

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
        hit = (len(tw) > 0) and (as_int(tw.iloc[0], default=-1) >= as_int(warm_idx, default=-1))
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

        flag = int((np.isfinite(corr1) and corr1 < corr_min) or (np.isfinite(corr2) and corr2 < corr_min))
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
        apply = _get_apply_for_decl(window)
        tp = apply(name, past)
        tr = apply(name, recent)
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

    Returns a COPY of df_eval with calibrated gate diagnostics:
      - Core scalars: gate_worst_tv_calib, gate_eps_stat_calib, gate_gain_calib,
        gate_eps_eff_calib, gate_eps_scaled_calib, gate_total_evidence_calib
      - Pass/fail channels: gate_ok_calib, gate_fail_calib, gate_evaluated_calib, gate_reason_calib
      - Warning channels: gate_warn_calib (warning-trigger bit), gate_warn_pending_calib,
        gate_dw_exceeds_threshold_calib, gate_consecutive_exceedances_calib

    Semantics / provenance:
      - If the ONLINE run used FR (GateEngine via SCAR_FR=1), the deployed gate
        does NOT apply legacy evidence-scaled epsilon inflation.
        In that case, the main *_calib columns mirror FR:
          eps_scaled := eps_base
          eps_stat cap := max_frac * eps_base
      - We also compute a clearly labeled legacy-inflated counterfactual into
        *_legacy_calib columns for comparison (do not silently mix semantics).
      - FR/legacy detection is derived from training-time provenance columns
        (gate_lane / scar_fr / gate_min_evidence_used) and is robust to early blank rows.
    """
    try:
        # GateEngine-consistent product-TV (ignores NaN per-metric TVs; abs-weight normalization)
        from veriscope.core.ipm import dPi_product_tv_robust
    except Exception as e:
        raise RuntimeError("dPi_product_tv_robust unavailable; core.ipm import failed.") from e

    try:
        from veriscope.core.calibration import aggregate_epsilon_stat as AGGREGATE_EPSILON
    except Exception:
        AGGREGATE_EPSILON = None  # type: ignore

    df = df_eval.copy()
    cfg = _get_cfg()

    apply = _get_apply_for_decl(window_decl)

    # Normalize key columns to numeric for sorting / slicing
    for c in ("epoch", "seed"):
        if c in df.columns:
            df[c] = to_numeric_opt(df.get(c))

    # Initialize calibrated columns
    df["gate_worst_tv_calib"] = np.nan
    df["gate_eps_stat_calib"] = np.nan
    df["gate_gain_calib"] = np.nan

    # Debug/parity fields
    df["gate_eps_eff_calib"] = np.nan
    df["gate_eps_scaled_calib"] = np.nan

    # Legacy-inflated (counterfactual under FR unless explicitly enabled)
    df["gate_eps_scaled_legacy_calib"] = np.nan
    df["gate_eps_eff_legacy_calib"] = np.nan
    df["gate_eps_stat_legacy_calib"] = np.nan
    # Counterfactual parameter capsule (auditable)
    df["gate_min_evidence_used_legacy_calib"] = np.nan
    df["gate_min_evidence_full_eps_legacy_calib"] = np.nan
    df["gate_eps_inflation_max_legacy_calib"] = np.nan

    # Per-row mirror bit (1 if row is treated as FR mirror)
    df["gate_fr_mirror_calib"] = 0

    df["gate_total_evidence_calib"] = 0
    df["gate_kappa_checked_calib"] = 0
    df["gate_kappa_sens_calib"] = np.nan

    # Calibrated gate channels (neutral defaults for non-evaluated epochs)
    # Semantics:
    #   gate_ok_calib=1: pass OR not evaluated (neutral)
    #   gate_fail_calib=1: evaluated AND failed
    #   gate_warn_calib=1: evaluated AND warning condition triggered (pending or exceeded threshold)
    df["gate_ok_calib"] = 1
    df["gate_fail_calib"] = 0
    df["gate_evaluated_calib"] = 0
    df["gate_reason_calib"] = "insufficient_history"

    # Warning channels (1 means warning triggered)
    df["gate_warn_calib"] = 0
    df["gate_warn_pending_calib"] = 0
    df["gate_dw_exceeds_threshold_calib"] = 0
    df["gate_consecutive_exceedances_calib"] = 0

    max_frac = as_float(cfg.get("gate_eps_stat_max_frac", 0.25), default=0.25)
    thr_gain = as_float(cfg.get("gate_gain_thresh", 0.05), default=0.05)
    eps_sens = as_float(cfg.get("gate_epsilon_sens", 0.04), default=0.04)
    alpha = as_float(cfg.get("gate_eps_stat_alpha", 0.05), default=0.05)
    ln2 = math.log(2.0)

    num_classes = as_int(cfg.get("num_classes", 10), default=10)
    # Precedence: env > cfg (so exports affect offline recompute too)
    gain_baseline = (
        str(os.environ.get("SCAR_GATE_GAIN_BASELINE", cfg.get("gate_gain_baseline", "uniform"))).strip().lower()
    )
    policy_requested = str(os.environ.get("SCAR_GATE_POLICY", cfg.get("gate_policy", "either"))).strip().lower()
    try:
        persistence_k = int(os.environ.get("SCAR_GATE_PERSISTENCE_K", cfg.get("gate_persistence_k", 2)))
    except Exception:
        persistence_k = 2
    persistence_k = max(1, int(persistence_k))

    # Map legacy-only policy to GateEngine-equivalent semantics
    if policy_requested == "stability_only":
        policy = "persistence_stability"
        persistence_k = 1
    else:
        policy = policy_requested

    df["gate_policy_requested_calib"] = policy_requested
    df["gate_policy_effective_calib"] = policy
    df["gate_gain_baseline_calib"] = gain_baseline
    df["gate_persistence_k_calib"] = int(persistence_k)

    # FIX #1: Evidence-based epsilon inflation for offline recompute.
    # Must match online _evidence_scaled_epsilon byte-for-byte, including:
    #   - gate_eps_inflation_max default = 4.0 (not 2.0)
    #   - min_full >= min_evidence + 1 guard
    #   - clamp max_infl >= 1.0
    #   - identical interpolation form
    gate_eps_inflation_max = as_float(cfg.get("gate_eps_inflation_max", 4.0), default=4.0)
    gate_eps_inflation_max = float(max(1.0, gate_eps_inflation_max))  # clamp >= 1
    # NOTE: min_evidence_full_eps becomes per-run/per-row (because min_evidence can be per-run provenance).
    min_evidence_full_eps_cfg = cfg.get("gate_min_evidence_full_eps", None)

    def _evidence_scaled_epsilon_offline(
        eps_base: float,
        total_evidence: int,
        min_evidence_full: int,
        max_inflation: float,
    ) -> float:
        """
        Offline mirror of legacy_cli._evidence_scaled_epsilon (byte-for-byte logic).
        Returns inflated epsilon when evidence is below min_evidence_full.
        """
        eps_base = float(max(0.0, eps_base))
        total_evidence = int(max(0, total_evidence))
        min_evidence_full = int(max(1, min_evidence_full))
        max_inflation = float(max(1.0, max_inflation))
        if total_evidence >= min_evidence_full:
            return eps_base
        if total_evidence <= 0:
            return eps_base * max_inflation
        # Linear interpolation: inflation decreases as evidence increases
        frac = float(total_evidence) / float(min_evidence_full)
        infl = 1.0 + (max_inflation - 1.0) * (1.0 - frac)
        return eps_base * float(infl)

    def _parse_evidence_from_row(row_idx: int, g: pd.DataFrame, mets: List[str]) -> int:
        """
        Extract total evidence robustly.

        Precedence:
          1) gate_total_evidence / total_evidence scalar column (if present)
          2) gate_counts / counts_by_metric JSON dict columns (sum only known metric keys when possible)
          3) fallback: sum integer-like non-negative metric columns
        """
        # (1) Prefer explicit evidence column if present (strictly safest)
        for col in ("gate_total_evidence", "total_evidence"):
            if col in g.columns:
                try:
                    v = g[col].iloc[row_idx]
                    if pd.notna(v):
                        return int(max(0, int(v)))
                except Exception:
                    pass

        total = 0
        allowed = set(map(str, mets)) if mets else None

        # (2) Prefer gate_counts JSON (stringified dict with integer counts)
        for col_name in ("gate_counts", "counts_by_metric"):
            if col_name in g.columns:
                try:
                    raw = g[col_name].iloc[row_idx]
                    if pd.notna(raw) and raw:
                        parsed = json.loads(str(raw)) if isinstance(raw, str) else raw
                        if isinstance(parsed, dict):
                            for k, v in parsed.items():
                                # Sum only intended metric keys when we know them.
                                if allowed is not None and str(k) not in allowed:
                                    continue
                                try:
                                    total += int(v)
                                except (TypeError, ValueError):
                                    pass
                            if total > 0:
                                return total
                except Exception:
                    pass

        # (3) Fallback: sum per-metric columns only if they look like integer counts
        for met in mets:
            if met in g.columns:
                try:
                    val = g[met].iloc[row_idx]
                    if pd.notna(val):
                        # Only treat as count if it's a non-negative integer-like value
                        fval = float(val)
                        if fval >= 0 and fval == int(fval):
                            total += int(fval)
                except Exception:
                    pass
        return total

    def _gain_flags_like_core(gain_bits: float, thr: float) -> Tuple[bool, bool, bool]:
        """Return (gain_checked, ok_gain, gain_below) matching core semantics: non-finite => not checked => OK."""
        gain_checked = bool(np.isfinite(gain_bits))
        ok_gain = True if (not gain_checked) else bool(float(gain_bits) >= float(thr))
        gain_below = False if (not gain_checked) else bool(float(gain_bits) < float(thr))
        return gain_checked, ok_gain, gain_below

    mets = [m for m in getattr(window_decl, "metrics", []) if m in df.columns]
    agg_fn = AGGREGATE_EPSILON

    # Optional explicit override: allow evidence-scaled epsilon under FR too.
    # If False, FR rows mirror GateEngine (no inflation).
    use_evidence_scaled_eps = bool(cfg.get("gate_use_evidence_scaled_epsilon", False))

    for _, g in df.groupby(["seed", "factor"], sort=False):
        g = g.sort_values("epoch").copy()
        if len(g) < 2 * W:
            continue

        idxs = list(g.index)
        consec = 0

        # Determine per-run lane provenance (stable across epochs), robust to early blank rows.
        # Prefer: mode over non-empty gate_lane values; else max(scar_fr).
        run_is_fr = False
        try:
            if "gate_lane" in g.columns:
                s = g["gate_lane"].fillna("").astype(str).str.strip().str.lower()
                s = s[s != ""]
                if len(s) > 0:
                    # mode() is robust to a few stray values; take the first mode.
                    run_is_fr = str(s.mode().iloc[0]) == "fr"
                else:
                    run_is_fr = False
            elif "scar_fr" in g.columns:
                v = pd.to_numeric(g["scar_fr"], errors="coerce").fillna(0.0)
                run_is_fr = bool(float(v.max()) >= 0.5)
        except Exception:
            run_is_fr = False

        # Set gate_fr_mirror_calib for ALL rows in this run for clarity (even pre-2W).
        mirror_fr_run = bool(run_is_fr and (not use_evidence_scaled_eps))
        try:
            df.loc[idxs, "gate_fr_mirror_calib"] = int(mirror_fr_run)
        except Exception:
            pass

        # min_evidence used online (per run): prefer logged value if present, else cfg
        min_evidence_run = as_int(cfg.get("gate_min_evidence", 0), default=0)
        try:
            if "gate_min_evidence_used" in g.columns:
                vv = pd.to_numeric(g["gate_min_evidence_used"], errors="coerce").to_numpy(dtype=float)
                vv = vv[np.isfinite(vv)]
                if vv.size > 0:
                    min_evidence_run = int(np.max(vv))
        except Exception:
            pass
        # Record for audit clarity: this is the evidence floor used to decide evaluability.
        try:
            df.loc[idxs, "gate_min_evidence_used_legacy_calib"] = int(min_evidence_run)
        except Exception:
            pass

        for pos, idx in enumerate(idxs):
            # Align with ONLINE semantics:
            # At epoch E (row position pos), compare only prior epochs [0..E-1].
            # Requires at least 2W prior points -> pos >= 2W.
            if pos < (2 * W):
                continue

            # Exclude current epoch from both windows:
            # past   = epochs [pos-2W, pos-W)
            # recent = epochs [pos-W,  pos)
            ps = slice(pos - (2 * W), pos - W)
            rs = slice(pos - W, pos)

            past = {m: _as_float_array(to_numeric_opt(g.get(m)).iloc[ps].to_numpy()) for m in mets}
            recent = {m: _as_float_array(to_numeric_opt(g.get(m)).iloc[rs].to_numpy()) for m in mets}

            # Evidence accounting (matches online): count finite pairs under DeclTransport.
            counts = {m: _count_finite_pairs(window_decl, past[m], recent[m], m) for m in mets}
            evidence_n = int(sum(int(v) for v in (counts or {}).values()))

            # Prefer explicit per-row evidence if present; otherwise fall back to recomputed evidence.
            total_evidence = _parse_evidence_from_row(pos, g, mets)
            if int(total_evidence) <= 0:
                total_evidence = int(evidence_n)
            df.loc[idx, "gate_total_evidence_calib"] = int(total_evidence)

            # Per-row mirror flag: FR rows mirror GateEngine unless explicitly overridden to use inflation.
            mirror_fr_row = bool(mirror_fr_run)

            # Legacy counterfactual parameters should be auditable per-row.
            eps_base = float(getattr(window_decl, "epsilon", 0.0))
            min_full_row = min_evidence_run
            try:
                if min_evidence_full_eps_cfg is None:
                    min_full_row = int(min_evidence_run)
                else:
                    min_full_row = as_int(min_evidence_full_eps_cfg, default=int(min_evidence_run))
            except Exception:
                min_full_row = int(min_evidence_run)
            # Guard: inflation should still be active at the first evaluated evidence
            min_full_row = max(int(min_full_row), int(min_evidence_run) + 1)
            df.loc[idx, "gate_min_evidence_full_eps_legacy_calib"] = int(min_full_row)
            df.loc[idx, "gate_eps_inflation_max_legacy_calib"] = float(gate_eps_inflation_max)

            eps_scaled_legacy = _evidence_scaled_epsilon_offline(
                eps_base=eps_base,
                total_evidence=int(total_evidence),
                min_evidence_full=int(min_full_row),
                max_inflation=float(gate_eps_inflation_max),
            )
            df.loc[idx, "gate_eps_scaled_legacy_calib"] = float(eps_scaled_legacy)

            # Main eps_scaled used for *_calib columns:
            #   - mirror FR: eps_scaled == eps_base
            #   - legacy:   eps_scaled == eps_scaled_legacy
            eps_scaled = float(eps_base) if mirror_fr_row else float(eps_scaled_legacy)
            df.loc[idx, "gate_eps_scaled_calib"] = float(eps_scaled)

            # Evidence gate: mirror online semantics using per-run min_evidence_run
            if int(min_evidence_run) > 0 and int(total_evidence) < int(min_evidence_run):
                df.loc[idx, "gate_ok_calib"] = 1
                df.loc[idx, "gate_fail_calib"] = 0
                df.loc[idx, "gate_warn_calib"] = 0
                df.loc[idx, "gate_evaluated_calib"] = 0
                df.loc[idx, "gate_reason_calib"] = "not_evaluated_insufficient_evidence"
                df.loc[idx, "gate_warn_pending_calib"] = 0
                df.loc[idx, "gate_dw_exceeds_threshold_calib"] = 0
                df.loc[idx, "gate_consecutive_exceedances_calib"] = int(consec)
                continue

            # GateEngine-consistent: worst over interventions, robust product-TV semantics
            tv = float("nan")
            try:
                intervs = getattr(window_decl, "interventions", None) or (lambda x: x,)
                tvs: List[float] = []
                for T in intervs:

                    def _apply_T(ctx: str, arr: np.ndarray, _T=T) -> np.ndarray:
                        return np.asarray(_T(apply(ctx, arr)), dtype=float)

                    tvi = dPi_product_tv_robust(window_decl, past, recent, apply=_apply_T)
                    if np.isfinite(tvi):
                        tvs.append(float(tvi))
                if tvs:
                    tv = float(max(tvs))
            except Exception:
                tv = float("nan")

            # If TV is not finite, treat as not evaluated (prevents NaN-propagation into events).
            if not np.isfinite(tv):
                df.loc[idx, "gate_ok_calib"] = 1
                df.loc[idx, "gate_fail_calib"] = 0
                df.loc[idx, "gate_warn_calib"] = 0
                df.loc[idx, "gate_evaluated_calib"] = 0
                df.loc[idx, "gate_reason_calib"] = "not_evaluated_nan_tv"
                df.loc[idx, "gate_warn_pending_calib"] = 0
                df.loc[idx, "gate_dw_exceeds_threshold_calib"] = 0
                df.loc[idx, "gate_consecutive_exceedances_calib"] = int(consec)
                continue

            # ε_stat: compute raw once, then dual-cap for FR vs legacy from the same eps_stat_raw.
            try:
                if agg_fn is not None:
                    eps_stat_raw = float(agg_fn(window_decl, counts_by_metric=counts, alpha=alpha))
                else:
                    eps_stat_raw = 0.0
            except Exception:
                eps_stat_raw = 0.0
            if not np.isfinite(eps_stat_raw):
                eps_stat_raw = 0.0

            eps_stat_fr = float(min(max(0.0, eps_stat_raw), float(max_frac) * float(eps_base)))
            eps_stat_legacy = float(min(max(0.0, eps_stat_raw), float(max_frac) * float(eps_scaled_legacy)))
            df.loc[idx, "gate_eps_stat_legacy_calib"] = float(eps_stat_legacy)

            # Main eps_stat mirrors online lane:
            eps_stat = float(eps_stat_fr) if mirror_fr_row else float(eps_stat_legacy)
            df.loc[idx, "gate_eps_stat_calib"] = float(eps_stat)

            # Canonical GateEngine stability slack:
            # eps_eff = eps_scaled + eps_sens - eps_stat_capped
            eps_eff = max(
                0.0,
                float(eps_scaled) + float(eps_sens) - float(eps_stat),
            )
            df.loc[idx, "gate_eps_eff_calib"] = float(eps_eff)

            # Legacy eps_eff for explicit comparison
            try:
                eps_eff_legacy = max(
                    0.0,
                    float(eps_scaled_legacy) + float(eps_sens) - float(eps_stat_legacy),
                )
                df.loc[idx, "gate_eps_eff_legacy_calib"] = float(eps_eff_legacy)
            except Exception:
                pass

            try:
                s_train = series_or_empty(g.get("train_loss"))
                s_ewma = series_or_empty(g.get("ewma_loss"))
                model_losses = to_numeric_opt(s_train).iloc[rs].to_numpy(dtype=np.float64)
                base_losses = to_numeric_opt(s_ewma).iloc[rs].to_numpy(dtype=np.float64)
                msk = np.isfinite(model_losses) & np.isfinite(base_losses)

                if gain_baseline == "uniform":
                    mm = np.isfinite(model_losses)
                    if mm.any():
                        base_nats = float(math.log(max(2, int(num_classes))))
                        gain_bits = float((base_nats - float(np.mean(model_losses[mm]))) / ln2)
                    else:
                        gain_bits = float("nan")
                else:
                    gain_bits = (
                        float(((base_losses[msk] - model_losses[msk]).mean()) / ln2) if msk.any() else float("nan")
                    )
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

            gain_checked, ok_gain, gain_below = _gain_flags_like_core(gain_bits, thr_gain)
            ok_tv = bool(np.isfinite(tv) and (tv <= eps_eff))
            dw_exceeds = bool(np.isfinite(tv) and (tv > eps_eff))

            # Canonical GateEngine κ semantics: non-finite => not checked => OK
            kappa_checked = bool(np.isfinite(kappa))
            ok_kappa = bool((not kappa_checked) or (float(kappa) <= float(eps_sens)))
            df.loc[idx, "gate_kappa_checked_calib"] = int(kappa_checked)
            df.loc[idx, "gate_kappa_sens_calib"] = float(kappa) if kappa_checked else np.nan

            warn_pending = False
            persistence_fail = False

            if policy in ("persistence", "persistence_stability"):
                # Update streak ONLY on evaluated checks (tv finite, evidence OK)
                if dw_exceeds:
                    consec += 1
                else:
                    consec = 0

                persistence_fail = bool(consec >= int(persistence_k))
                warn_pending = bool(dw_exceeds and (not persistence_fail))

                if policy == "persistence":
                    # Gain is an immediate veto in GateEngine.PERSISTENCE.
                    if gain_checked and gain_below:
                        consec = 0
                        persistence_fail = False
                        warn_pending = False

                    ok = bool(ok_kappa and ok_gain and (not persistence_fail))
                    if not ok_kappa:
                        reason = "evaluated_fail_kappa"
                    elif gain_checked and gain_below:
                        reason = "evaluated_fail_gain"
                    elif persistence_fail:
                        reason = "evaluated_fail_stability"
                    else:
                        reason = "evaluated_ok"
                else:
                    # persistence_stability: gain is audited but does not veto ok/warn/fail
                    ok = bool(ok_kappa and (not persistence_fail))
                    if not ok_kappa:
                        reason = "evaluated_fail_kappa"
                    elif persistence_fail:
                        reason = "evaluated_fail_stability"
                    else:
                        reason = "evaluated_ok"

                flag = int(ok)

            elif policy == "either":
                # Match GateEngine: "either" is strict (fail if gain OR stability fails)
                ok = bool(ok_kappa and ok_gain and ok_tv)
                flag = int(ok)
                if flag == 1:
                    reason = "evaluated_ok"
                elif not ok_kappa:
                    reason = "evaluated_fail_kappa"
                elif not ok_gain:
                    reason = "evaluated_fail_gain"
                else:
                    reason = "evaluated_fail_stability"

            elif policy == "conjunction":
                # GateEngine.CONJUNCTION: fail only if BOTH gain AND stability fail (kappa veto stays)
                ok = bool(ok_kappa and (ok_gain or ok_tv))
                flag = int(ok)

                # GateEngine warns when stability exceeds but gain is OK and overall ok holds.
                warn_pending = bool(dw_exceeds and ok_gain and ok)

                if flag == 1:
                    reason = "evaluated_ok"
                elif not ok_kappa:
                    reason = "evaluated_fail_kappa"
                else:
                    # Here ok_kappa==True and (ok_gain or ok_tv)==False => BOTH failed.
                    # Prefer stability label for drift-only analysis lanes.
                    reason = "evaluated_fail_stability"

            df.loc[idx, "gate_worst_tv_calib"] = float(tv)
            df.loc[idx, "gate_gain_calib"] = float(gain_bits) if np.isfinite(gain_bits) else np.nan

            # PASS/FAIL
            df.loc[idx, "gate_ok_calib"] = int(flag)
            df.loc[idx, "gate_fail_calib"] = int(1 - int(flag))

            # Warning channels
            df.loc[idx, "gate_dw_exceeds_threshold_calib"] = int(bool(dw_exceeds))
            df.loc[idx, "gate_warn_pending_calib"] = int(bool(warn_pending))
            df.loc[idx, "gate_consecutive_exceedances_calib"] = int(consec)

            # Warning bit used by event extraction
            df.loc[idx, "gate_warn_calib"] = int(bool(dw_exceeds) or bool(warn_pending))

            df.loc[idx, "gate_evaluated_calib"] = 1
            df.loc[idx, "gate_reason_calib"] = str(reason)

    # Optional explicit alias to reduce semantic confusion downstream.
    # gate_pass_calib is identical to gate_ok_calib.
    try:
        df["gate_pass_calib"] = df["gate_ok_calib"]
    except Exception:
        pass
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
