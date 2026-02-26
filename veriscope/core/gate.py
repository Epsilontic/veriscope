# veriscope/core/gate.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import warnings
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .ipm import tv_hist_fixed

if TYPE_CHECKING:
    from .window import FRWindow


_POLICY_PARSE_WARNED: bool = False
_REGIME_STATE_PARSE_WARNED: bool = False
_REGIME_STATE_ALLOWED: frozenset[str] = frozenset({"disabled", "building", "established", "active"})
_REGIME_STATE_ALLOWED_SORTED: tuple[str, ...] = tuple(sorted(_REGIME_STATE_ALLOWED))


def _warn_policy_parse_once(msg: str) -> None:
    global _POLICY_PARSE_WARNED
    if _POLICY_PARSE_WARNED:
        return
    _POLICY_PARSE_WARNED = True
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _warn_regime_state_parse_once(msg: str) -> None:
    global _REGIME_STATE_PARSE_WARNED
    if _REGIME_STATE_PARSE_WARNED:
        return
    _REGIME_STATE_PARSE_WARNED = True
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _trend_slope_pairs(pairs: Iterable[Tuple[int, float]]) -> float:
    """Slope using true indices (avoids time compression when margins intermittently NaN)."""
    xs: list[float] = []
    ys: list[float] = []
    for t, v in pairs:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            xs.append(float(t))
            ys.append(float(fv))
    if len(xs) < 2:
        return float("nan")
    dt = xs[-1] - xs[0]
    if dt <= 0:
        return float("nan")
    return float((ys[-1] - ys[0]) / dt)


def _normalize_regime_state(rs_in: Any) -> str:
    """Normalize regime_state to canonical string; canonical enum lives elsewhere."""
    if rs_in is None:
        return "disabled"
    # Handle enum instances directly (avoids Enum.__str__ differences across versions)
    if isinstance(rs_in, Enum):
        rs_in = rs_in.value
    s = str(rs_in).strip()
    if "." in s:
        s = s.split(".")[-1]
    s = s.lower()
    if s in _REGIME_STATE_ALLOWED:
        return s
    _warn_regime_state_parse_once(
        f"Unknown regime_state input={rs_in!r}. Treating as 'disabled'. "
        f"Valid values={list(_REGIME_STATE_ALLOWED_SORTED)}."
    )
    return "disabled"


def _gain_flags(gain_bits: Any, gain_thr: float) -> tuple[bool, bool, bool, float]:
    """Return (gain_is_finite, ok_gain, gain_below, gain_bits_float).

    Convention: non-numeric or non-finite gain => not evaluated => ok_gain True, gain_below False.
    """
    try:
        gb = float(gain_bits)
    except Exception:
        return (False, True, False, float("nan"))
    if not np.isfinite(gb):
        return (False, True, False, float("nan"))
    thr = float(gain_thr)
    return (True, gb >= thr, gb < thr, float(gb))


def _array_equal_nan_safe(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN-safe array equality helper with backward-compatible NumPy fallback."""
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        if a.shape != b.shape:
            return False
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def _rescue_zero_tv_from_bin_collapse(
    tv: float,
    a: np.ndarray,
    b: np.ndarray,
    bins: int,
    *,
    cal_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, Dict[str, bool]]:
    """
    Preserve fixed-bin TV as primary, but rescue all-zero collapse on nonidentical windows.

    This is only active when fixed-bin TV returns exactly 0 while finite transformed arrays are
    nonempty and not elementwise identical.
    """
    rescue_meta: Dict[str, bool] = {}
    try:
        tv0 = float(tv)
    except Exception:
        return float("nan"), rescue_meta
    if not (np.isfinite(tv0) and np.isclose(tv0, 0.0, atol=1e-12)):
        return tv0, rescue_meta

    a_f = np.asarray(a, float)
    b_f = np.asarray(b, float)
    a_f = a_f[np.isfinite(a_f)]
    b_f = b_f[np.isfinite(b_f)]

    if a_f.size == 0 and b_f.size == 0:
        return 0.0, rescue_meta
    if a_f.size == 0 or b_f.size == 0:
        return float("nan"), rescue_meta
    if a_f.size == b_f.size and _array_equal_nan_safe(a_f, b_f):
        return 0.0, rescue_meta

    lo = float(min(np.min(a_f), np.min(b_f)))
    hi = float(max(np.max(a_f), np.max(b_f)))
    if cal_range is not None:
        try:
            cal_lo = float(cal_range[0])
            cal_hi = float(cal_range[1])
        except Exception:
            cal_lo = float("nan")
            cal_hi = float("nan")
        if np.isfinite(cal_lo) and np.isfinite(cal_hi) and (cal_hi > cal_lo):
            # Clamp to intersection of observed span and cal range so we don't widen
            # bins beyond where data lives.
            obs_lo, obs_hi = lo, hi
            lo = max(obs_lo, cal_lo)
            hi = min(obs_hi, cal_hi)
            if not (np.isfinite(lo) and np.isfinite(hi) and (hi > lo)):
                lo, hi = obs_lo, obs_hi
                # Audit marker: cal-range intersection was empty, so rescue used observed span.
                rescue_meta["tv_rescue_used_observed_range"] = True
    if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
        fine_bins = int(max(256, 8 * max(1, int(bins))))
        fine_bins = min(fine_bins, 4096)
        try:
            ha, _ = np.histogram(a_f, bins=fine_bins, range=(lo, hi), density=False)
            hb, _ = np.histogram(b_f, bins=fine_bins, range=(lo, hi), density=False)
        except ValueError:
            # Extremely tiny nonzero spans can be smaller than float bin resolution,
            # causing NumPy to reject a large uniform bin count. This path is a best-effort
            # rescue for tv==0 only; if rescue cannot be computed safely, preserve tv==0.
            rescue_meta["tv_rescue_tiny_range"] = True
            return 0.0, rescue_meta
        sa = int(np.sum(ha))
        sb = int(np.sum(hb))
        if sa > 0 and sb > 0:
            pa = ha / float(sa)
            pb = hb / float(sb)
            tv_fine = 0.5 * float(np.abs(pa - pb).sum())
            if np.isfinite(tv_fine) and tv_fine > 0.0:
                return float(min(max(tv_fine, 0.0), 1.0)), rescue_meta

    # KS distance is a different metric than TV; mixing them into a TV-calibrated
    # threshold is unsound.
    return 0.0, rescue_meta


def _sha16_of_values(arr: np.ndarray) -> str:
    """sha256 over repr(value) joined by newlines; return 16-char prefix."""
    vals = np.asarray(arr, float).ravel()
    payload = "\n".join(repr(float(v)) for v in vals.tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _head_tail(arr: np.ndarray, n: int = 5) -> tuple[list[float], list[float]]:
    vals = np.asarray(arr, float).ravel()
    head = [float(v) for v in vals[:n].tolist()]
    tail = [float(v) for v in vals[-n:].tolist()]
    return head, tail


def _finite_min_max(arr: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    vals = np.asarray(arr, float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    return float(np.min(vals)), float(np.max(vals))


def _tv_input_fingerprint_payload(past_arr: np.ndarray, recent_arr: np.ndarray) -> Dict[str, Any]:
    p = np.asarray(past_arr, float)
    q = np.asarray(recent_arr, float)
    p_head, p_tail = _head_tail(p, n=5)
    q_head, q_tail = _head_tail(q, n=5)
    p_min, p_max = _finite_min_max(p)
    q_min, q_max = _finite_min_max(q)
    return {
        "tv_input_dtype": {"past": str(p.dtype), "recent": str(q.dtype)},
        "tv_input_shape": {"past": list(p.shape), "recent": list(q.shape)},
        "tv_input_head": {"past": p_head, "recent": q_head},
        "tv_input_tail": {"past": p_tail, "recent": q_tail},
        "tv_input_sha16": {"past": _sha16_of_values(p), "recent": _sha16_of_values(q)},
        "tv_input_min": {"past": p_min, "recent": q_min},
        "tv_input_max": {"past": p_max, "recent": q_max},
    }


def _derive_window_ranges_from_counts(
    per_metric_n: Dict[str, Tuple[int, int]],
    t_idx: int,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not isinstance(per_metric_n, dict) or not per_metric_n:
        return None, None
    recent_n = max((int(v[1]) for v in per_metric_n.values()), default=0)
    past_n = max((int(v[0]) for v in per_metric_n.values()), default=0)
    if recent_n <= 0:
        return None, None
    cur_end = int(t_idx)
    cur_start = int(cur_end - recent_n + 1)
    cur = {
        "start_idx": cur_start,
        "end_idx": cur_end,
        "n_points": int(recent_n),
        "source": "derived_from_iter_num_and_counts",
    }
    if past_n <= 0:
        return None, cur
    ref_end = int(cur_start - 1)
    ref_start = int(ref_end - past_n + 1)
    ref = {
        "start_idx": ref_start,
        "end_idx": ref_end,
        "n_points": int(past_n),
        "source": "derived_from_iter_num_and_counts",
    }
    return ref, cur


class GatePolicy(Enum):
    """Gate failure policy."""

    EITHER = "either"  # Fail if gain OR stability fails (original behavior)
    CONJUNCTION = "conjunction"  # Fail if gain AND stability both fail

    # Legacy semantics: persistence on stability, BUT gain/kappa still veto immediately.
    PERSISTENCE = "persistence"  # Fail if stability fails for K consecutive evaluated checks

    # NEW opt-in: stability-only persistence (kappa veto), gain audited only.
    PERSISTENCE_STABILITY = "persistence_stability"


@dataclass
class GateResult:
    ok: bool
    audit: Dict[str, Any]
    warn: bool = False  # Non-fatal warning (e.g., persistence warn-pending or gain below threshold)


class GateEngine:
    def __init__(
        self,
        frwin: "FRWindow",
        gain_thresh: float,
        eps_stat_alpha: float,
        eps_stat_max_frac: float,
        eps_sens: float,
        min_evidence: int = 0,
        policy: str = "either",
        persistence_k: int = 2,
        min_metrics_exceeding: int = 1,
        trend_n: int = 8,
    ):
        self.win = frwin
        self.gain_thr = float(gain_thresh)
        self.alpha = float(eps_stat_alpha)
        self.cap_frac = float(eps_stat_max_frac)
        self.eps_sens = float(eps_sens)
        self.min_evidence = int(max(0, min_evidence))

        # Multi-metric consensus: require >=K metrics with per-metric TV > eps_eff
        # before treating stability exceedance as "real". Applied BEFORE persistence bookkeeping.
        try:
            self.min_metrics_exceeding = max(1, int(min_metrics_exceeding))
        except Exception:
            self.min_metrics_exceeding = 1

        # Parse policy (string -> enum, normalize aliases; conservative default on invalid)
        policy_in = policy  # preserve original for audit/debug
        policy_str_raw = str(policy_in).lower().strip()

        aliases = {
            "persist": "persistence",
            "persistence_stab": "persistence_stability",
            "persistence_only": "persistence_stability",
            "persistence-stability": "persistence_stability",
        }
        policy_str_norm = aliases.get(policy_str_raw, policy_str_raw)

        try:
            self.policy = GatePolicy(policy_str_norm)
        except ValueError:
            _warn_policy_parse_once(
                f"Unknown gate policy input={policy_in!r} normalized={policy_str_norm!r}. "
                f"Defaulting to 'persistence' (conservative). Valid={[p.value for p in GatePolicy]}."
            )
            self.policy = GatePolicy.PERSISTENCE
        self.persistence_k = max(1, int(persistence_k))

        # State for persistence policy (only updated on evaluated checks)
        self._consecutive_exceedances: int = 0

        # Trend tracking with proper indices
        self._trend_n = max(2, int(trend_n))
        self._margins_change: Deque[Tuple[int, float]] = deque(maxlen=self._trend_n)
        # _trend_idx always increments on check(); used as fallback when iter_num not provided
        self._trend_idx: int = 0

    def check(
        self,
        past: Dict[str, np.ndarray],
        recent: Dict[str, np.ndarray],
        counts_by_metric: Dict[str, int],
        gain_bits: float,
        kappa_sens: float,
        eps_stat_value: float,
        *,
        regime_state: Any = None,
        iter_num: Optional[int] = None,
        window_ref_range: Optional[Dict[str, Any]] = None,
        window_cur_range: Optional[Dict[str, Any]] = None,
        ref_window_id: Optional[str] = None,
        cur_window_id: Optional[str] = None,
    ) -> GateResult:
        wd = self.win.decl

        # Regime state normalization (string only; canonical enum lives elsewhere)
        rs = _normalize_regime_state(regime_state)

        # Monotone trend index tracking
        self._trend_idx += 1  # check count (always increments)
        check_idx = self._trend_idx
        if iter_num is not None:
            t_idx = int(iter_num)  # use provided iteration number for trend slope
            t_idx_source = "iter_num"
        else:
            t_idx = check_idx  # fallback to check count
            t_idx_source = "check_idx"

        gain_is_finite, ok_gain, gain_below, gain_bits_f = _gain_flags(gain_bits, self.gain_thr)
        persistence_fail_flag = False  # always emitted in audit

        # Check evidence first (early return if insufficient)
        # NOTE: This changes behavior from original (which set ok_evidence=False).
        # New semantic: insufficient evidence → can't evaluate → ok=True, evaluated=False.
        # Persistence counter is NOT touched on unevaluated checks.
        evidence_n = int(sum(int(v) for v in (counts_by_metric or {}).values()))
        if self.min_evidence > 0 and evidence_n < self.min_evidence:
            return GateResult(
                ok=True,
                warn=False,
                audit=dict(
                    evaluated=False,
                    # Legacy-compatible reason string
                    reason="not_evaluated_insufficient_evidence",
                    # Keep existing key, but also provide legacy alias
                    evidence_total=int(evidence_n),
                    total_evidence=int(evidence_n),
                    min_evidence=int(self.min_evidence),
                    policy=self.policy.value,
                    consecutive_exceedances=self._consecutive_exceedances,
                    consecutive_exceedances_before=self._consecutive_exceedances,
                    consecutive_exceedances_after=self._consecutive_exceedances,
                    persistence_k=self.persistence_k,
                    # Include what we can compute
                    gain_bits=float(gain_bits_f),
                    gain_thr=float(self.gain_thr),
                    gain_evaluated=bool(gain_is_finite),
                    ok_gain=bool(ok_gain),
                    eps=float(wd.epsilon),
                    eps_sens=float(self.eps_sens),
                    kappa_sens=float(kappa_sens) if np.isfinite(kappa_sens) else float("nan"),
                    kappa_checked=bool(np.isfinite(kappa_sens)),
                    persistence_fail=bool(persistence_fail_flag),
                    counts_by_metric={k: int(v) for k, v in (counts_by_metric or {}).items()},
                    # Schema consistency (unevaluated → placeholders)
                    worst_DW=float("nan"),
                    eps_stat=float("nan"),
                    eps_eff=float("nan"),
                    per_metric_tv={},
                    per_metric_n={},
                    drifts={},
                    dw_exceeds_threshold=False,
                    ok_stab=True,
                    # Multi-metric consensus fields (schema-stable)
                    min_metrics_exceeding=int(getattr(self, "min_metrics_exceeding", 1) or 1),
                    min_metrics_exceeding_effective=int(getattr(self, "min_metrics_exceeding", 1) or 1),
                    n_metrics_exceeding=0,
                    metrics_exceeding=[],
                    multi_metric_filtered=False,
                    dw_exceeds_threshold_raw=False,
                    ok_stab_raw=True,
                    gain_below_threshold=bool(gain_below),
                    # Trend/margin schema (placeholders for unevaluated check)
                    trend_x=int(t_idx),
                    trend_x_source=t_idx_source,
                    trend_n=int(self._trend_n),
                    check_idx=int(check_idx),
                    margin_change_raw=float("nan"),
                    margin_change_eff=float("nan"),
                    margin_change_slope_eff=float("nan"),
                    margin_change=float("nan"),  # legacy alias
                    margin_change_slope=float("nan"),  # legacy alias
                    margin_change_rel_raw=float("nan"),
                    margin_change_rel_eff=float("nan"),
                    # Regime state (string); margin_regime* injected by regime engine
                    regime_state=rs,
                ),
            )

        # Interventions (default to identity)
        intervs: Sequence = tuple(getattr(wd, "interventions", None) or (lambda x: x,))

        # Safe transport apply hook: prefer self.win.transport.apply, fall back to identity
        _transport = getattr(self.win, "transport", None)
        _apply = getattr(_transport, "apply", None)
        if _apply is None:
            _apply = lambda name, arr: np.asarray(arr, float)

        # Weight sum for normalization (keeps aggregated TV in [0,1] for nonnegative weights)
        # Use abs() defensively so an accidental negative weight cannot invert the distance.
        w_sum = float(sum(abs(float(v)) for v in wd.weights.values())) or 1.0

        # Track per-metric TV and n for the worst intervention
        worst = 0.0
        worst_any = False
        worst_per_metric_tv: Dict[str, float] = {}
        worst_per_metric_n: Dict[str, Tuple[int, int]] = {}
        worst_tv_inputs: Dict[str, Dict[str, np.ndarray]] = {}
        raw_windows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        raw_window_debug: Dict[str, Dict[str, Any]] = {}

        # Raw window diagnostics are interval-invariant (pre-transform), so compute once per metric.
        for m in wd.weights.keys():
            past_obj = past.get(m, np.array([], float))
            recent_obj = recent.get(m, np.array([], float))
            past_raw = np.asarray(past_obj, float)
            recent_raw = np.asarray(recent_obj, float)
            raw_windows[m] = (past_raw, recent_raw)
            past_f = past_raw[np.isfinite(past_raw)]
            recent_f = recent_raw[np.isfinite(recent_raw)]
            raw_n_points = {"past": int(past_raw.size), "recent": int(recent_raw.size)}
            finite_n_points = {"past": int(past_f.size), "recent": int(recent_f.size)}
            raw_window_debug[m] = {
                "raw_empty_strict": raw_n_points["past"] == 0 and raw_n_points["recent"] == 0,
                "raw_exact_match": past_f.size == recent_f.size and _array_equal_nan_safe(past_f, recent_f),
                "raw_n_points": raw_n_points,
                "finite_n_points": finite_n_points,
                "raw_ranges": {
                    "past": [float(np.min(past_f)), float(np.max(past_f))] if past_f.size > 0 else None,
                    "recent": [float(np.min(recent_f)), float(np.max(recent_f))] if recent_f.size > 0 else None,
                },
                # Debug-only alias hint (identity only; do NOT treat as hard failure).
                "raw_identity_alias": past_obj is recent_obj,
            }

        for T in intervs:
            tv_sum = 0.0
            tv_finite = 0
            this_per_metric_tv: Dict[str, float] = {}
            this_per_metric_n: Dict[str, Tuple[int, int]] = {}
            this_tv_inputs: Dict[str, Dict[str, np.ndarray]] = {}

            for m, w in wd.weights.items():
                past_raw, recent_raw = raw_windows[m]

                # Ensure arrays after transport + intervention
                a = np.asarray(T(_apply(m, past_raw)), float)
                b = np.asarray(T(_apply(m, recent_raw)), float)

                tv = tv_hist_fixed(a, b, wd.bins)

                # Per-metric diagnostics (TUPLE FORMAT PRESERVED)
                n_past = int(np.sum(np.isfinite(a)))
                n_recent = int(np.sum(np.isfinite(b)))
                this_per_metric_n[m] = (n_past, n_recent)
                this_per_metric_tv[m] = float(tv) if np.isfinite(tv) else float("nan")
                if (
                    np.isfinite(tv)
                    and float(tv) == 0.0
                    and n_past > 0
                    and n_recent > 0
                    and (not _array_equal_nan_safe(a, b))
                ):
                    this_tv_inputs[m] = {"past": np.array(a, copy=True), "recent": np.array(b, copy=True)}

                if np.isfinite(tv):
                    tv_sum += (abs(float(w)) / w_sum) * float(tv)
                    tv_finite += 1

            if tv_finite > 0 and np.isfinite(tv_sum):
                if (not worst_any) or (float(tv_sum) > float(worst)):
                    worst = float(tv_sum)
                    worst_per_metric_tv = dict(this_per_metric_tv)
                    worst_per_metric_n = dict(this_per_metric_n)
                    worst_tv_inputs = dict(this_tv_inputs)
                worst_any = True

        if not worst_any:
            # Not evaluated: no finite metric TVs were produced. Do not touch persistence.
            return GateResult(
                ok=True,
                warn=False,
                audit=dict(
                    evaluated=False,
                    reason="not_evaluated_no_finite_metrics",
                    policy=self.policy.value,
                    consecutive_exceedances=self._consecutive_exceedances,
                    consecutive_exceedances_before=self._consecutive_exceedances,
                    consecutive_exceedances_after=self._consecutive_exceedances,
                    persistence_k=self.persistence_k,
                    # Provide expected numeric fields as NaN where appropriate
                    gain_bits=float(gain_bits_f),
                    gain_thr=float(self.gain_thr),
                    gain_evaluated=bool(gain_is_finite),
                    ok_gain=bool(ok_gain),
                    worst_DW=float("nan"),
                    eps=float(wd.epsilon),
                    eps_sens=float(self.eps_sens),
                    persistence_fail=bool(persistence_fail_flag),
                    eps_stat=float("nan"),
                    eps_eff=float("nan"),
                    kappa_sens=float(kappa_sens) if np.isfinite(kappa_sens) else float("nan"),
                    kappa_checked=bool(np.isfinite(kappa_sens)),
                    counts_by_metric={k: int(v) for k, v in (counts_by_metric or {}).items()},
                    evidence_total=int(evidence_n),
                    total_evidence=int(evidence_n),
                    min_evidence=int(self.min_evidence),
                    eps_aggregation="cap_to_frac",
                    per_metric_tv={},
                    per_metric_n={},
                    drifts={},
                    # Schema-stable threshold fields
                    dw_exceeds_threshold=False,
                    dw_exceeds_threshold_raw=False,
                    ok_stab=True,
                    ok_stab_raw=True,
                    gain_below_threshold=bool(gain_below),
                    # Multi-metric consensus fields (schema-stable)
                    min_metrics_exceeding=int(getattr(self, "min_metrics_exceeding", 1) or 1),
                    min_metrics_exceeding_effective=int(getattr(self, "min_metrics_exceeding", 1) or 1),
                    n_metrics_exceeding=0,
                    metrics_exceeding=[],
                    multi_metric_filtered=False,
                    # Trend/margin schema (placeholders for unevaluated check)
                    trend_x=int(t_idx),
                    trend_x_source=t_idx_source,
                    trend_n=int(self._trend_n),
                    check_idx=int(check_idx),
                    margin_change_raw=float("nan"),
                    margin_change_eff=float("nan"),
                    margin_change_slope_eff=float("nan"),
                    margin_change=float("nan"),  # legacy alias
                    margin_change_slope=float("nan"),  # legacy alias
                    margin_change_rel_raw=float("nan"),
                    margin_change_rel_eff=float("nan"),
                    # Regime state (string); margin_regime* injected by regime engine
                    regime_state=rs,
                ),
            )

        empty_metrics = sorted(str(m) for m, dbg in raw_window_debug.items() if dbg.get("raw_empty_strict", False))
        nonidentical_metrics = sorted(
            str(m) for m, dbg in raw_window_debug.items() if not dbg.get("raw_exact_match", True)
        )
        nonidentical_with_points = sorted(
            str(m)
            for m in nonidentical_metrics
            if (
                int((raw_window_debug.get(m, {}).get("finite_n_points", {}) or {}).get("past", 0)) > 0
                and int((raw_window_debug.get(m, {}).get("finite_n_points", {}) or {}).get("recent", 0)) > 0
            )
        )

        # Gate rescue so the normal path remains unchanged and cheap.
        per_metric_vals_pre = list(worst_per_metric_tv.values())
        per_metric_tv_rescue: Dict[str, Dict[str, bool]] = {}
        finite_tv_vals_pre: list[float] = []
        for v in per_metric_vals_pre:
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                finite_tv_vals_pre.append(fv)
        all_finite_zero_tv = bool(finite_tv_vals_pre) and all(
            np.isclose(v, 0.0, atol=1e-12) for v in finite_tv_vals_pre
        )
        impossible_zero_tv_pre = bool(all_finite_zero_tv and nonidentical_with_points)
        if impossible_zero_tv_pre and worst_tv_inputs:
            cal_ranges = getattr(wd, "cal_ranges", {}) or {}
            for m in wd.weights:
                if m not in worst_tv_inputs:
                    continue
                payload = worst_tv_inputs[m]
                prior_tv = worst_per_metric_tv.get(m, float("nan"))
                past_arr = np.asarray(payload.get("past", np.array([], float)), float)
                recent_arr = np.asarray(payload.get("recent", np.array([], float)), float)
                cal_range = cal_ranges.get(m) if isinstance(cal_ranges, dict) else None
                rescued_out = _rescue_zero_tv_from_bin_collapse(
                    prior_tv,
                    past_arr,
                    recent_arr,
                    wd.bins,
                    cal_range=cal_range,
                )
                rescued, rescue_meta = rescued_out
                worst_per_metric_tv[m] = float(rescued) if np.isfinite(rescued) else float("nan")
                if isinstance(rescue_meta, dict) and rescue_meta:
                    per_metric_tv_rescue[str(m)] = dict(rescue_meta)

            rescued_sum = 0.0
            rescued_finite = 0
            for m, w in wd.weights.items():
                try:
                    tv_m = float(worst_per_metric_tv.get(m, float("nan")))
                except Exception:
                    continue
                if np.isfinite(tv_m):
                    rescued_sum += (abs(float(w)) / w_sum) * tv_m
                    rescued_finite += 1
            if rescued_finite > 0 and np.isfinite(rescued_sum):
                worst = float(rescued_sum)

        # Guardrail intent: if all per-metric TVs collapse to ~0, fail loudly unless raw windows are truly identical.
        per_metric_vals = list(worst_per_metric_tv.values())
        all_zero_tv = bool(per_metric_vals) and all(
            np.isfinite(float(v)) and np.isclose(float(v), 0.0, atol=1e-12) for v in per_metric_vals
        )
        degenerate_reason: Optional[str] = None
        if all_zero_tv:
            if raw_window_debug and (len(empty_metrics) == len(raw_window_debug)):
                degenerate_reason = "empty_window"
            elif nonidentical_with_points:
                degenerate_reason = "zero_tv_nonidentical_windows"

        eps_cap = float(wd.epsilon) * float(min(max(self.cap_frac, 0.0), 1.0))
        eps_stat = float(min(max(0.0, eps_stat_value), eps_cap))
        # Legacy semantics: eps_eff = epsilon + eps_sens - eps_stat_capped
        eps_eff = max(0.0, float(wd.epsilon) + float(self.eps_sens) - eps_stat)

        # --- margin (CHANGE, raw pre-consensus) ---
        margin_change_raw = float(worst - eps_eff) if (np.isfinite(worst) and np.isfinite(eps_eff)) else float("nan")
        if np.isfinite(worst) and np.isfinite(eps_eff) and float(eps_eff) > 0.0:
            margin_change_rel_raw = float(worst / eps_eff - 1.0)
        else:
            margin_change_rel_raw = float("nan")

        # Threshold checks
        ok_stab = np.isfinite(worst) and (worst <= eps_eff)
        dw_exceeds = np.isfinite(worst) and (worst > eps_eff)
        kappa_is_finite = np.isfinite(kappa_sens)
        ok_k = (not kappa_is_finite) or (kappa_sens <= self.eps_sens)

        # "Evaluated" means we computed at least one finite metric
        evaluated = worst_any

        # ============================================================
        # Multi-metric consensus filter (MUST run before persistence)
        # ============================================================
        min_m_req = int(getattr(self, "min_metrics_exceeding", 1) or 1)
        min_m_req = max(1, min_m_req)

        # Clamp-to-total-metrics: prefer the metrics we actually computed TVs for.
        # Priority: wd.weights (computed loop) → wd.metrics (declared) → worst_per_metric_tv (fallback).
        weights_dict = getattr(wd, "weights", None) or {}
        if isinstance(weights_dict, dict) and len(weights_dict) > 0:
            n_metrics_total = len(weights_dict)
        else:
            metrics_list = getattr(wd, "metrics", None) or []
            n_metrics_total = len(metrics_list) if isinstance(metrics_list, (list, tuple)) else 0
            if n_metrics_total <= 0 and isinstance(worst_per_metric_tv, dict):
                n_metrics_total = len(worst_per_metric_tv)
        n_metrics_total = max(1, int(n_metrics_total))

        min_m_eff = min(min_m_req, n_metrics_total)

        exceeding: list[str] = []
        worst_metric: Optional[str] = None
        worst_metric_tv: Optional[float] = None
        per_metric_tv = worst_per_metric_tv if isinstance(worst_per_metric_tv, dict) else None
        if per_metric_tv is not None and np.isfinite(float(eps_eff)):
            for m, v in per_metric_tv.items():
                try:
                    # tolerate structured payloads like {"tv": ...}
                    if isinstance(v, dict) and "tv" in v:
                        tv = float(v["tv"])
                    else:
                        tv = float(v)
                except Exception:
                    continue
                if not np.isfinite(tv):
                    continue
                if (worst_metric_tv is None) or (tv > worst_metric_tv):
                    worst_metric_tv = float(tv)
                    worst_metric = str(m)
                if tv > float(eps_eff):
                    exceeding.append(str(m))

        # Preserve raw exceedance for debugging/auditing
        dw_exceeds_raw = bool(dw_exceeds)
        ok_stab_raw = bool(ok_stab)

        multi_metric_filtered = False
        if dw_exceeds_raw and (min_m_eff > 1) and (len(exceeding) < min_m_eff):
            # Crucial: override BEFORE persistence updates / policy logic
            dw_exceeds = False
            ok_stab = True
            multi_metric_filtered = True

        # --- margin (CHANGE, effective post-consensus) ---
        # Semantics:
        #   - When exceeding (dw_exceeds=True): positive margin shows exceedance magnitude
        #   - When filtered (worst > eps_eff but dw_exceeds=False): margin collapses to 0
        #   - When below threshold (worst < eps_eff): negative margin preserves safety headroom
        if np.isfinite(worst) and np.isfinite(eps_eff):
            if bool(dw_exceeds):
                worst_eff = float(worst)  # preserve positive exceedance magnitude
            else:
                # Filtered exceedances collapse to 0; otherwise preserves negative headroom
                worst_eff = float(min(float(worst), float(eps_eff)))
            margin_change_eff = float(worst_eff - eps_eff)
        else:
            worst_eff = float("nan")
            margin_change_eff = float("nan")

        if np.isfinite(worst_eff) and np.isfinite(eps_eff) and float(eps_eff) > 0.0:
            margin_change_rel_eff = float(worst_eff / eps_eff - 1.0)
        else:
            margin_change_rel_eff = float("nan")

        self._margins_change.append((t_idx, margin_change_eff))
        margin_change_slope_eff = _trend_slope_pairs(self._margins_change)

        # Policy-based decision
        warn = False
        warn_stab = False
        persistence_fail = False
        persistence_fail_flag = False
        gain_warn_raw = bool(gain_is_finite and gain_below)
        gain_warn = bool(evaluated and gain_warn_raw)
        consecutive_before = self._consecutive_exceedances

        if self.policy == GatePolicy.EITHER:
            # Original behavior: fail if gain OR stability fails (plus kappa)
            ok = bool(ok_gain and ok_stab and ok_k)

        elif self.policy == GatePolicy.CONJUNCTION:
            # Fail only if BOTH gain AND stability fail
            # (kappa still independently checked)
            ok = bool((ok_gain or ok_stab) and ok_k)
            # Warn if stability fails but gain is OK
            warn = dw_exceeds and ok_gain and ok

        elif self.policy == GatePolicy.PERSISTENCE:
            # Only update persistence counter if we actually computed something
            if evaluated:
                if dw_exceeds:
                    self._consecutive_exceedances += 1
                else:
                    # Gain-only warnings do not reset the persistence counter.
                    if not gain_warn_raw:
                        self._consecutive_exceedances = 0
            # else: don't touch counter (no finite metrics to evaluate)

            persistence_fail = self._consecutive_exceedances >= self.persistence_k
            persistence_fail_flag = bool(persistence_fail)

            # Stability check uses persistence; kappa veto remains immediate; gain is WARN-only.
            ok_stab_persist = not persistence_fail
            ok = bool(ok_stab_persist and ok_k)

            # Warn only when evaluated=True.
            warn_stab = bool(dw_exceeds and (not persistence_fail))
            warn = bool(ok and evaluated and (warn_stab or gain_warn))

        elif self.policy == GatePolicy.PERSISTENCE_STABILITY:
            # Only update persistence counter on evaluated checks
            if evaluated:
                if dw_exceeds:
                    self._consecutive_exceedances += 1
                else:
                    # Gain-only warnings do not reset the persistence counter.
                    if not gain_warn_raw:
                        self._consecutive_exceedances = 0

            persistence_fail = self._consecutive_exceedances >= self.persistence_k
            persistence_fail_flag = bool(persistence_fail)

            # Kappa veto remains immediate. Gain is WARN-only (never FAIL-capable).
            ok = bool((not persistence_fail) and ok_k)

            # Warn = current exceedance before sustained fail (consistent with persistence)
            warn_stab = bool(dw_exceeds and (not persistence_fail))
            warn = bool(ok and evaluated and (warn_stab or gain_warn))

        else:
            # Fallback to EITHER
            ok = bool(ok_gain and ok_stab and ok_k)

        consecutive_after = self._consecutive_exceedances
        reason_gain_only = bool(warn and gain_warn and (not warn_stab))
        if degenerate_reason is not None:
            # Degenerate all-zero TV checks are hard failures.
            # Keep persistence behavior monotone by counting this as an exceedance-like failure.
            if self.policy in (GatePolicy.PERSISTENCE, GatePolicy.PERSISTENCE_STABILITY):
                self._consecutive_exceedances = int(consecutive_before) + 1
                consecutive_after = self._consecutive_exceedances
                persistence_fail_flag = bool(consecutive_after >= self.persistence_k)
            ok = False
            warn = False
            ok_stab = False
            ok_stab_raw = False

        audit = dict(
            # Existing fields (UNCHANGED)
            gain_bits=float(gain_bits_f),
            gain_thr=float(self.gain_thr),
            gain_evaluated=bool(gain_is_finite),
            ok_gain=bool(ok_gain),
            # Use the same weighted aggregate that drives dw_exceeds_threshold / decision logic.
            worst_DW=(float(worst) if np.isfinite(worst) else float("nan")),
            eps=float(wd.epsilon),
            eps_sens=float(self.eps_sens),
            eps_stat=float(eps_stat),
            eps_eff=float(eps_eff),
            kappa_sens=(float(kappa_sens) if bool(kappa_is_finite) else float("nan")),
            kappa_checked=bool(kappa_is_finite),
            counts_by_metric={k: int(v) for k, v in (counts_by_metric or {}).items()},
            evidence_total=int(evidence_n),
            total_evidence=int(evidence_n),
            min_evidence=int(self.min_evidence),
            eps_aggregation="cap_to_frac",
            # Existing per-metric diagnostics (TUPLE FORMAT PRESERVED)
            per_metric_tv=worst_per_metric_tv,
            per_metric_n=worst_per_metric_n,
            drifts=worst_per_metric_tv,  # backward-compat alias for regime.py
            # NEW fields (additive only)
            evaluated=evaluated,
            policy=self.policy.value,
            dw_exceeds_threshold=bool(dw_exceeds),
            dw_exceeds_threshold_raw=bool(dw_exceeds_raw),
            gain_below_threshold=bool(gain_below),
            ok_stab=bool(ok_stab),
            ok_stab_raw=bool(ok_stab_raw),
            # Multi-metric consensus audit
            min_metrics_exceeding=int(min_m_req),
            min_metrics_exceeding_effective=int(min_m_eff),
            n_metrics_exceeding=int(len(exceeding)),
            metrics_exceeding=sorted(exceeding),
            multi_metric_filtered=bool(multi_metric_filtered),
            worst_metric=worst_metric,
            worst_metric_tv=worst_metric_tv,
            persistence_fail=bool(persistence_fail_flag),
            consecutive_exceedances=consecutive_after,
            consecutive_exceedances_before=consecutive_before,
            consecutive_exceedances_after=consecutive_after,
            persistence_k=self.persistence_k,
            # Margin/trend (explicit naming + legacy aliases)
            trend_x=int(t_idx),
            trend_x_source=t_idx_source,
            trend_n=int(self._trend_n),
            check_idx=int(check_idx),
            margin_change_raw=float(margin_change_raw),
            margin_change_eff=float(margin_change_eff),
            margin_change_slope_eff=float(margin_change_slope_eff),
            margin_change=float(margin_change_eff),  # legacy alias
            margin_change_slope=float(margin_change_slope_eff),  # legacy alias
            margin_change_rel_raw=float(margin_change_rel_raw),
            margin_change_rel_eff=float(margin_change_rel_eff),
            # Regime state (string); margin_regime* injected by regime engine
            regime_state=rs,
            **({"reason": "gain_below_threshold"} if reason_gain_only else {}),
        )

        if per_metric_tv_rescue:
            # Audit key: per-metric rescue diagnostics (e.g., observed-range fallback marker).
            audit["per_metric_tv_rescue"] = per_metric_tv_rescue

        if degenerate_reason is not None:
            # `evaluated` is already correct from the normal audit path; evidence-insufficient
            # cases return before reaching this point.
            audit["reason"] = str(degenerate_reason)
            audit["degenerate_window"] = True
            # Audit key: disambiguates diagnostic hard-failure from true threshold divergence.
            audit["failure_mode"] = "diagnostic"
            audit["dw_exceeds_threshold"] = False
            audit["ok_stab"] = False
            audit["ok_stab_raw"] = False
            # Minimal debug payload for proving raw-window degeneracy.
            audit["window_debug_n_points"] = {
                str(m): dict(dbg.get("finite_n_points", {"past": 0, "recent": 0}))
                for m, dbg in raw_window_debug.items()
            }
            audit["window_debug_raw_n_points"] = {
                str(m): dict(dbg.get("raw_n_points", {"past": 0, "recent": 0})) for m, dbg in raw_window_debug.items()
            }
            audit["window_debug_raw_ranges"] = {
                str(m): dict(dbg.get("raw_ranges", {"past": None, "recent": None}))
                for m, dbg in raw_window_debug.items()
            }
            audit["window_debug_nonidentical_metrics"] = list(nonidentical_metrics)
            audit["window_debug_nonidentical_with_points_metrics"] = list(nonidentical_with_points)
            audit["window_debug_empty_metrics"] = list(empty_metrics)
            identity_alias_metrics = sorted(
                str(m) for m, dbg in raw_window_debug.items() if dbg.get("raw_identity_alias", False)
            )
            if identity_alias_metrics:
                audit["window_debug_identity_alias_metrics"] = identity_alias_metrics

            derived_ref_range, derived_cur_range = _derive_window_ranges_from_counts(worst_per_metric_n, int(t_idx))
            ref_range_payload: Any = window_ref_range if window_ref_range is not None else derived_ref_range
            cur_range_payload: Any = window_cur_range if window_cur_range is not None else derived_cur_range

            if "window_ref_range" not in audit and ref_range_payload is not None:
                audit["window_ref_range"] = ref_range_payload
            if "window_cur_range" not in audit and cur_range_payload is not None:
                audit["window_cur_range"] = cur_range_payload

            if "ref_window_id" not in audit:
                if ref_window_id is not None:
                    audit["ref_window_id"] = str(ref_window_id)
                elif isinstance(ref_range_payload, dict):
                    rs = ref_range_payload.get("start_iter", ref_range_payload.get("start_idx"))
                    r_end = ref_range_payload.get("end_iter", ref_range_payload.get("end_idx"))
                    if rs is not None and r_end is not None:
                        audit["ref_window_id"] = f"window:{rs}:{r_end}"
                    else:
                        audit["ref_window_id"] = f"window_ref_check:{check_idx}"
                else:
                    audit["ref_window_id"] = f"window_ref_check:{check_idx}"

            if "cur_window_id" not in audit:
                if cur_window_id is not None:
                    audit["cur_window_id"] = str(cur_window_id)
                elif isinstance(cur_range_payload, dict):
                    cs = cur_range_payload.get("start_iter", cur_range_payload.get("start_idx"))
                    c_end = cur_range_payload.get("end_iter", cur_range_payload.get("end_idx"))
                    if cs is not None and c_end is not None:
                        audit["cur_window_id"] = f"window:{cs}:{c_end}"
                    else:
                        audit["cur_window_id"] = f"window_cur_check:{check_idx}"
                else:
                    audit["cur_window_id"] = f"window_cur_check:{check_idx}"

            impossible_zero_tv = (
                str(degenerate_reason) == "zero_tv_nonidentical_windows"
                and bool(nonidentical_with_points)
                and bool(all_zero_tv)
            )
            if impossible_zero_tv:
                tv_fp: Dict[str, Dict[str, Any]] = {}
                for m in sorted(worst_tv_inputs.keys()):
                    payload = worst_tv_inputs.get(m, {})
                    past_arr = np.asarray(payload.get("past", np.array([], float)), float)
                    recent_arr = np.asarray(payload.get("recent", np.array([], float)), float)
                    tv_fp[str(m)] = _tv_input_fingerprint_payload(past_arr, recent_arr)
                if tv_fp:
                    audit["window_debug_tv_input_fingerprint"] = tv_fp

        return GateResult(ok=ok, warn=warn, audit=audit)

    def save_persistence_state(self) -> Dict[str, int]:
        """Snapshot all persistence counters for temporary override/rollback flows."""
        return {
            "consecutive_exceedances": int(self._consecutive_exceedances),
        }

    def restore_persistence_state(self, state: Any) -> None:
        """Restore persistence counters from a previous save_persistence_state() snapshot."""
        if isinstance(state, dict):
            val = state.get("consecutive_exceedances", self._consecutive_exceedances)
        else:
            val = state
        try:
            self._consecutive_exceedances = max(0, int(val))
        except Exception:
            # Ignore malformed snapshots to keep gating robust under mixed callers.
            return

    def reset_persistence(self) -> int:
        """Reset persistence counter. Returns previous count.
        Call after intentional intervention or regime change."""
        prev = self._consecutive_exceedances
        self._consecutive_exceedances = 0
        return prev

    @property
    def consecutive_exceedances(self) -> int:
        """Current persistence counter value."""
        return self._consecutive_exceedances
