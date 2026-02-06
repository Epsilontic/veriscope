# veriscope/core/gate.py
from __future__ import annotations

from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING
import warnings

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

        for T in intervs:
            tv_sum = 0.0
            tv_finite = 0
            this_per_metric_tv: Dict[str, float] = {}
            this_per_metric_n: Dict[str, Tuple[int, int]] = {}

            for m, w in wd.weights.items():
                past_arr = past.get(m, np.array([], float))
                recent_arr = recent.get(m, np.array([], float))

                # Ensure arrays after transport + intervention
                a = np.asarray(T(_apply(m, past_arr)), float)
                b = np.asarray(T(_apply(m, recent_arr)), float)

                tv = tv_hist_fixed(a, b, wd.bins)

                # Per-metric diagnostics (TUPLE FORMAT PRESERVED)
                n_past = int(np.sum(np.isfinite(a)))
                n_recent = int(np.sum(np.isfinite(b)))
                this_per_metric_n[m] = (n_past, n_recent)
                this_per_metric_tv[m] = float(tv) if np.isfinite(tv) else float("nan")

                if np.isfinite(tv):
                    tv_sum += (abs(float(w)) / w_sum) * float(tv)
                    tv_finite += 1

            if tv_finite > 0 and np.isfinite(tv_sum):
                if (not worst_any) or (float(tv_sum) > float(worst)):
                    worst = float(tv_sum)
                    worst_per_metric_tv = dict(this_per_metric_tv)
                    worst_per_metric_n = dict(this_per_metric_n)
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

        return GateResult(
            ok=ok,
            warn=warn,
            audit=dict(
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
            ),
        )

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
