# veriscope/core/gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .ipm import tv_hist_fixed

if TYPE_CHECKING:
    from .window import FRWindow


@dataclass
class GateResult:
    ok: bool
    audit: Dict[str, Any]


class GateEngine:
    def __init__(
        self,
        frwin: "FRWindow",
        gain_thresh: float,
        eps_stat_alpha: float,
        eps_stat_max_frac: float,
        eps_sens: float,
        min_evidence: int = 0,
    ):
        self.win = frwin
        self.gain_thr = float(gain_thresh)
        self.alpha = float(eps_stat_alpha)
        self.cap_frac = float(eps_stat_max_frac)
        self.eps_sens = float(eps_sens)
        self.min_evidence = int(max(0, min_evidence))

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

                # Per-metric diagnostics
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
            worst = float("nan")

        eps_cap = float(wd.epsilon) * float(min(max(self.cap_frac, 0.0), 1.0))
        eps_stat = float(min(max(0.0, eps_stat_value), eps_cap))
        eps_eff = max(0.0, float(wd.epsilon) - eps_stat)

        ok_gain = np.isfinite(gain_bits) and (gain_bits >= self.gain_thr)
        ok_stab = np.isfinite(worst) and (worst <= eps_eff)
        kappa_is_finite = np.isfinite(kappa_sens)
        ok_k = (not kappa_is_finite) or (kappa_sens <= self.eps_sens)
        evidence_n = int(sum(int(v) for v in (counts_by_metric or {}).values()))
        ok_evidence = (evidence_n >= self.min_evidence) if self.min_evidence > 0 else True

        return GateResult(
            ok=bool(ok_gain and ok_stab and ok_k and ok_evidence),
            audit=dict(
                gain_bits=float(gain_bits) if np.isfinite(gain_bits) else float("nan"),
                worst_DW=float(worst) if np.isfinite(worst) else float("nan"),
                eps=float(wd.epsilon),
                eps_stat=float(eps_stat),
                eps_eff=float(eps_eff),
                kappa_sens=float(kappa_sens if kappa_is_finite else 0.0),
                kappa_checked=bool(kappa_is_finite),
                counts_by_metric={k: int(v) for k, v in (counts_by_metric or {}).items()},
                evidence_total=int(evidence_n),
                min_evidence=int(self.min_evidence),
                eps_aggregation="cap_to_frac",
                # NEW: per-metric diagnostics
                per_metric_tv=worst_per_metric_tv,
                per_metric_n=worst_per_metric_n,
                drifts=worst_per_metric_tv,  # backward-compat alias for regime.py
            ),
        )
