# veriscope/core/gate.py
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .ipm import tv_hist_fixed

if TYPE_CHECKING:
    from .window import FRWindow


class GatePolicy(Enum):
    """Gate failure policy."""

    EITHER = "either"  # Fail if gain OR stability fails (original behavior)
    CONJUNCTION = "conjunction"  # Fail if gain AND stability both fail
    PERSISTENCE = "persistence"  # Fail if stability fails for K consecutive evaluated checks


@dataclass
class GateResult:
    ok: bool
    audit: Dict[str, Any]
    warn: bool = False  # Threshold exceeded but not yet FAIL (persistence mode)


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
    ):
        self.win = frwin
        self.gain_thr = float(gain_thresh)
        self.alpha = float(eps_stat_alpha)
        self.cap_frac = float(eps_stat_max_frac)
        self.eps_sens = float(eps_sens)
        self.min_evidence = int(max(0, min_evidence))

        # Parse policy (string -> enum, default to EITHER on invalid)
        policy_str = str(policy).lower().strip()
        try:
            self.policy = GatePolicy(policy_str)
        except ValueError:
            self.policy = GatePolicy.EITHER
        self.persistence_k = max(1, int(persistence_k))

        # State for persistence policy (only updated on evaluated checks)
        self._consecutive_exceedances: int = 0

    def check(
        self,
        past: Dict[str, np.ndarray],
        recent: Dict[str, np.ndarray],
        counts_by_metric: Dict[str, int],
        gain_bits: float,
        kappa_sens: float,
        eps_stat_value: float,
    ) -> GateResult:
        wd = self.win.decl

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
                    gain_bits=float(gain_bits) if np.isfinite(gain_bits) else float("nan"),
                    eps=float(wd.epsilon),
                    eps_sens=float(self.eps_sens),
                    kappa_sens=float(kappa_sens) if np.isfinite(kappa_sens) else float("nan"),
                    kappa_checked=bool(np.isfinite(kappa_sens)),
                    counts_by_metric={k: int(v) for k, v in (counts_by_metric or {}).items()},
                    # Schema consistency (unevaluated → placeholders)
                    worst_DW=float("nan"),
                    eps_stat=float("nan"),
                    eps_eff=float("nan"),
                    per_metric_tv={},
                    per_metric_n={},
                    drifts={},
                    dw_exceeds_threshold=False,
                    gain_below_threshold=bool(np.isfinite(gain_bits) and (gain_bits < self.gain_thr)),
                ),
            )

        # Interventions (default to identity)
        intervs: Sequence = tuple(getattr(wd, "interventions", ()) or (lambda x: x,))

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
                    gain_bits=float(gain_bits) if np.isfinite(gain_bits) else float("nan"),
                    worst_DW=float("nan"),
                    eps=float(wd.epsilon),
                    eps_sens=float(self.eps_sens),
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
                ),
            )

        eps_cap = float(wd.epsilon) * float(min(max(self.cap_frac, 0.0), 1.0))
        eps_stat = float(min(max(0.0, eps_stat_value), eps_cap))
        # Legacy semantics: eps_eff = epsilon + eps_sens - eps_stat_capped
        eps_eff = max(0.0, float(wd.epsilon) + float(self.eps_sens) - eps_stat)

        # Threshold checks
        ok_gain = np.isfinite(gain_bits) and (gain_bits >= self.gain_thr)
        ok_stab = np.isfinite(worst) and (worst <= eps_eff)
        dw_exceeds = np.isfinite(worst) and (worst > eps_eff)
        gain_below = np.isfinite(gain_bits) and (gain_bits < self.gain_thr)

        kappa_is_finite = np.isfinite(kappa_sens)
        ok_k = (not kappa_is_finite) or (kappa_sens <= self.eps_sens)

        # "Evaluated" means we computed at least one finite metric
        evaluated = worst_any

        # Policy-based decision
        warn = False
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
                    self._consecutive_exceedances = 0
            # else: don't touch counter (no finite metrics to evaluate)

            # Fail only after K consecutive exceedances
            persistence_fail = self._consecutive_exceedances >= self.persistence_k

            # Stability check uses persistence; gain and kappa still immediate
            ok_stab_persist = not persistence_fail
            ok = bool(ok_gain and ok_stab_persist and ok_k)

            # Warn on exceedance that hasn't yet triggered fail
            warn = dw_exceeds and not persistence_fail

            # If gain fails immediately, reset persistence counter (clean semantics)
            if gain_below:
                self._consecutive_exceedances = 0

        else:
            # Fallback to EITHER
            ok = bool(ok_gain and ok_stab and ok_k)

        consecutive_after = self._consecutive_exceedances

        return GateResult(
            ok=ok,
            warn=warn,
            audit=dict(
                # Existing fields (UNCHANGED)
                gain_bits=float(gain_bits) if np.isfinite(gain_bits) else float("nan"),
                worst_DW=float(worst) if np.isfinite(worst) else float("nan"),
                eps=float(wd.epsilon),
                eps_sens=float(self.eps_sens),
                eps_stat=float(eps_stat),
                eps_eff=float(eps_eff),
                kappa_sens=float(kappa_sens if kappa_is_finite else 0.0),
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
                dw_exceeds_threshold=dw_exceeds,
                gain_below_threshold=gain_below,
                consecutive_exceedances=consecutive_after,
                consecutive_exceedances_before=consecutive_before,
                consecutive_exceedances_after=consecutive_after,
                persistence_k=self.persistence_k,
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
