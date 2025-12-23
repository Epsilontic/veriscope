# veriscope/core/regime.py
"""Reference-anchored regime detection extension for FR gating.

Reuses existing GateEngine machinery to ensure consistent divergence
calculations and eps_stat semantics. No reimplementation of D_W.

Key design principles:
1. Reuses identical GateEngine.check() for divergence - no gauge slippage
2. Uses aggregate_epsilon_stat() with correct per-comparison counts
3. Per-metric evidence requirements for reference establishment
4. Bounded memory for reference and accumulator storage
5. Reference established ONLY during explicit build phase (fixes "bad but stationary" bug)
6. Reproducible seeding across processes (no Python hash())
7. Canonical metric set from weights.keys() to match GateEngine
8. Auto-disable if no valid metrics after cal_ranges validation

Final revision addressing:
- RegimeReference dataclass included
- Auto-disable when no metrics survive cal_ranges validation
- eps_sens fallback tracked and warned
- "not_evaluated" added to skip_reasons
- Missing/non-finite worst_DW fails quality gate (conservative)
- Missing/non-finite gain_bits fails quality gate (conservative)
- min_windows_for_reference only counts complete windows
- cal_ranges validated for value correctness, not just key presence
"""

from __future__ import annotations

import copy
import hashlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import multiprocessing as _mp

from veriscope.core.window import WindowDecl, FRWindow
from veriscope.core.transport import DeclTransport
from veriscope.core.gate import GateEngine, GateResult  # GatePolicy not needed here
from veriscope.core.calibration import aggregate_epsilon_stat


# --- Transport saturation diagnostic (rate-limited) ---
_SATURATION_WARN_COUNT: Dict[str, int] = {}
_SATURATION_WARN_LIMIT = 5

# --- Regime sanity diagnostics (rate-limited) ---
_SANITY_WARN_COUNT: Dict[str, int] = {}
_SANITY_WARN_LIMIT = 5


def _is_main_process() -> bool:
    """Check if we're in the main process (stdlib, no torch dependency)."""
    try:
        return _mp.current_process().name == "MainProcess"
    except Exception:
        return True  # fail open


def _validate_cal_range(entry) -> Tuple[bool, float, float]:
    """Validate a cal_ranges entry. Returns (valid, lo, hi).

    A cal_range is valid if:
    - It's a list/tuple of exactly 2 elements
    - Both elements are finite floats
    - lo < hi (strict inequality)
    """
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        return False, 0.0, 1.0
    lo, hi = entry
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return False, 0.0, 1.0
    if not (np.isfinite(lo_f) and np.isfinite(hi_f)):
        return False, 0.0, 1.0
    if lo_f >= hi_f:
        return False, 0.0, 1.0
    return True, lo_f, hi_f


def diagnose_transport_saturation(
    arr: np.ndarray,
    lo: float,
    hi: float,
    name: str = "metric",
) -> Dict[str, float]:
    """Compute fraction of samples clipped by transport.

    Uses inclusive bounds with small epsilon for boundary cases.

    NOTE: This diagnostic assumes `arr` contains RAW metric values (not yet
    transported/binned). In the GPT runner, `recent[m]` comes from
    metric_history which stores raw scalars, so this is correct. If reusing
    this engine with already-transported values, the diagnostic is meaningless.
    """
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"name": name, "n": 0, "clip_lo_frac": np.nan, "clip_hi_frac": np.nan}
    eps = 1e-9
    n_below = float((x <= lo + eps).sum())
    n_above = float((x >= hi - eps).sum())
    n = int(len(x))
    n_f = float(n)
    return {
        "name": name,
        "n": n,
        "lo": lo,
        "hi": hi,
        "actual_min": float(np.min(x)),
        "actual_max": float(np.max(x)),
        "actual_q01": float(np.quantile(x, 0.01)) if len(x) >= 10 else float(np.min(x)),
        "actual_q99": float(np.quantile(x, 0.99)) if len(x) >= 10 else float(np.max(x)),
        "clip_lo_frac": n_below / n_f,
        "clip_hi_frac": n_above / n_f,
        "clip_total_frac": (n_below + n_above) / n_f,
    }


def _maybe_warn_saturation(
    metric_name: str,
    diag: Dict[str, float],
    threshold: float = 0.10,
) -> None:
    """Rate-limited warning for transport saturation. Main process only."""
    # Main process check FIRST (before accounting)
    if not _is_main_process():
        return

    clip_frac = diag.get("clip_total_frac", 0.0)
    if not np.isfinite(clip_frac) or clip_frac < threshold:
        return

    # Rate limit accounting
    global _SATURATION_WARN_COUNT
    count = _SATURATION_WARN_COUNT.get(metric_name, 0)
    if count >= _SATURATION_WARN_LIMIT:
        return
    _SATURATION_WARN_COUNT[metric_name] = count + 1

    print(
        f"[REGIME DIAG] {metric_name}: {clip_frac * 100:.1f}% clipped, "
        f"actual_range=[{diag['actual_min']:.4f}, {diag['actual_max']:.4f}], "
        f"cal_range=[{diag['lo']:.4f}, {diag['hi']:.4f}]"
    )


def _maybe_sanity_warn(key: str, msg: str) -> None:
    """Rate-limited sanity warning. Main process only."""
    if not _is_main_process():
        return
    global _SANITY_WARN_COUNT
    c = _SANITY_WARN_COUNT.get(key, 0)
    if c >= _SANITY_WARN_LIMIT:
        return
    _SANITY_WARN_COUNT[key] = c + 1
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _stable_metric_seed(metric_name: str, base_seed: int = 42) -> int:
    """
    Compute a reproducible seed for a metric name.

    Uses SHA256 hash instead of Python's hash() which varies per process
    due to PYTHONHASHSEED randomization (default since Python 3.3).

    Returns a 31-bit positive integer suitable for np.random seeding.
    """
    h = hashlib.sha256(metric_name.encode("utf-8")).digest()
    # Use first 4 bytes as unsigned int, then combine with base_seed
    metric_hash = int.from_bytes(h[:4], byteorder="little", signed=False)
    return (base_seed + metric_hash) % (2**31)


@dataclass
class RegimeReference:
    """Frozen reference baseline from a known-good regime."""

    metrics: Dict[str, np.ndarray]
    counts: Dict[str, int]
    established_at: int
    n_samples_per_metric: Dict[str, int]
    # Audit: which quality gates were checked
    build_audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeConfig:
    """Configuration for regime-anchored detection.

    Reference Build Phase (CRITICAL for correctness):
    -------------------------------------------------
    Reference is ONLY established during [reference_build_min_iter, reference_build_max_iter).
    Outside this window, no reference can be established (but existing reference is used).

    This prevents the "bad but stationary" bug where reference gets established
    during a corrupted regime that has stabilized (past≈recent but both bad).

    Default values are SENTINELS (-1). At runtime, compute_build_window() derives
    actual values from gate config:
        min_iter = gate_warmup + 2 * gate_window  (when reference_build_min_iter == -1)
        max_iter = min(min_iter + build_span, pathology_start - gap)

    Where `gap` is determined by:
        - reference_build_gap_iters if >= 0 (explicit override)
        - default_gap_iters passed by caller (typically 2 * window_span_iters)
        - gate_window as final fallback

    Evidence Requirements:
    ---------------------
    min_evidence_per_metric: Each tracked metric must accumulate this many samples
    before reference can be established. This is a PER-METRIC minimum for reference
    establishment only.

    NOTE: The regime GateEngine's min_evidence is inherited from the base engine,
    NOT set to min_evidence_per_metric. These are separate concerns:
    - GateEngine.min_evidence: when the engine considers a check valid
    - min_evidence_per_metric: when we have enough data to establish reference

    Quality Gates:
    -------------
    Additional gates beyond "base check passed" for extra safety:
    - reference_build_max_dw: worst D_W must be well below epsilon
    - reference_build_min_gain: learning must be progressing (not stalled/diverging)
    - If worst_DW is missing/non-finite, quality gate FAILS (conservative)
    - If gain_bits is non-finite, quality gate FAILS (conservative)
    """

    # ---- Epsilon configuration ----
    # Epsilon for regime check: explicit value, or None to derive from base
    epsilon: Optional[float] = None
    # If epsilon is None, regime_epsilon = base_epsilon * epsilon_mult
    epsilon_mult: float = 1.5

    # ---- Reference build phase (CRITICAL) ----
    # Sentinel value -1 means "derive from gate config at runtime"
    # Only establish reference during [min_iter, max_iter)
    reference_build_min_iter: int = -1
    reference_build_max_iter: int = -1
    # Default span for build window when auto-computing max_iter
    reference_build_span: int = 1500
    # Explicit safety gap between pathology_start and reference build window end.
    # -1 => use default_gap_iters passed by caller (trainer), or gate_window as fallback.
    reference_build_gap_iters: int = -1

    # ---- Quality gates for reference establishment ----
    # Reference only established if base check worst_DW <= this threshold
    # Set conservatively below epsilon to ensure "truly healthy" baseline
    reference_build_max_dw: float = 0.08  # Should be << epsilon (default 0.12)
    # Reference only established if gain_bits >= this floor (learning health)
    # Slightly negative OK during warmup phase
    reference_build_min_gain: float = -0.01

    # ---- Evidence requirements ----
    # Min samples PER METRIC (not total) before establishing reference
    # NOTE: This is for reference establishment, NOT for GateEngine.min_evidence
    min_evidence_per_metric: int = 50

    # ---- Minimum windows for reference establishment ----
    # Require this many COMPLETE gate windows to be accumulated before reference
    # can be established. A window is "complete" if ALL tracked metrics have at
    # least one finite sample. This ensures temporal coverage even when
    # min_evidence_per_metric is reached quickly (e.g., large Wm).
    # Setting to 0 disables this check.
    # NOTE: Must be <= max_accumulator_windows or reference can never establish.
    min_windows_for_reference: int = 5

    # ---- eps_stat parameters for regime divergence calculation ----
    eps_stat_alpha: float = 0.05
    eps_stat_max_frac: float = 0.25

    # ---- Memory bounds ----
    max_reference_samples: int = 10000  # Per metric
    max_accumulator_windows: int = 20  # Total windows buffered

    # ---- Feature flag ----
    # Enable/disable regime detection entirely (for A/B testing)
    enabled: bool = True

    # Regime decision policy. "inherit" = use base GateEngine policy/persistence_k.
    policy: str = "inherit"
    persistence_k: int = 2


def _policy_to_str(x: Any) -> str:
    """Best-effort conversion of a GateEngine policy to a lowercase string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)


def _not_evaluated_from_reason(reason: str) -> bool:
    """Check if reason indicates gate was not evaluated."""
    r = str(reason or "").strip()
    return r.startswith("not_evaluated") or ("insufficient_evidence" in r)


def _as_opt_bool(x: Any) -> Optional[bool]:
    """
    Coerce bool/np.bool_ and integer-like 0/1 to Optional[bool].

    Returns None for anything else (including floats, strings, etc.)
    to avoid accidental truthiness traps.
    """
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        if x == 0:
            return False
        if x == 1:
            return True
        return None
    return None


def _needs_fill(x: Any) -> bool:
    """True if a field is missing/None/empty/'None' (string)."""
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip()
        return (s == "") or (s.lower() == "none")
    # Defensive: some pipelines serialize None as the literal string "None"
    try:
        s2 = str(x).strip().lower()
        return s2 == "none"
    except Exception:
        return False


def _coerce_float_maybe(v: Any) -> Optional[float]:
    """Best-effort float conversion; returns None if not finite/convertible."""
    try:
        fv = float(v)
    except Exception:
        return None
    return fv if np.isfinite(fv) else None


def _pick_worst_metric(per_metric: Any) -> Tuple[Optional[str], Optional[float]]:
    """Pick the metric with the largest finite TV/drift.

    Tolerates:
      - {metric: float}
      - {metric: np.floating}
      - {metric: {"tv": float}} (structured payload)
    """
    if not isinstance(per_metric, dict) or not per_metric:
        return None, None
    best_k: Optional[str] = None
    best_v: Optional[float] = None
    for k, v in per_metric.items():
        fv: Optional[float] = None
        if isinstance(v, dict) and "tv" in v:
            fv = _coerce_float_maybe(v.get("tv"))
        else:
            fv = _coerce_float_maybe(v)
        if fv is None:
            continue
        if best_v is None or fv > best_v:
            best_v = fv
            best_k = str(k)
    return best_k, best_v


def compute_build_window(
    config: RegimeConfig,
    gate_warmup: int,
    gate_window: int,
    pathology_start: Optional[int] = None,
    default_gap_iters: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute the reference build interval as a half-open range: [min_iter, max_iter).

    If max_iter <= min_iter, the window contains zero eligible checks; reference
    cannot be established. This is allowed but should be surfaced loudly by the
    caller/wrapper.

    Sentinel semantics for reference_build_min_iter:
        -1 (default): min_iter = gate_warmup + 2*gate_window
            This ensures at least 2 full windows of "past" data exist before
            the build phase begins.
        >= 0 (explicit): min_iter = max(gate_warmup, explicit_value)
            User override, floored at gate_warmup for safety.

    Gap semantics for max_iter when pathology_start is known:
        1. If config.reference_build_gap_iters >= 0, use that (explicit override)
        2. Else if default_gap_iters is provided, use that (caller's preference)
        3. Else use gate_window as final fallback

        max_iter = min(min_iter + build_span, pathology_start - gap)
    """
    if int(config.reference_build_min_iter) >= 0:
        # User explicitly set a value; respect it but floor at gate_warmup
        min_iter = int(max(int(gate_warmup), int(config.reference_build_min_iter)))
    else:
        # Sentinel case: derive from gate config per documented semantics
        min_iter = int(gate_warmup) + 2 * int(gate_window)

    if int(config.reference_build_max_iter) >= 0:
        return int(min_iter), int(config.reference_build_max_iter)

    default_max = int(min_iter) + int(config.reference_build_span)
    if pathology_start is not None and int(pathology_start) >= 0:
        gap = int(config.reference_build_gap_iters)
        if gap < 0:
            gap = int(default_gap_iters) if default_gap_iters is not None else int(gate_window)
        safe_max = int(pathology_start) - int(gap)
        max_iter = min(int(default_max), int(safe_max))
    else:
        max_iter = int(default_max)

    return int(min_iter), int(max_iter)


class RegimeAnchoredGateEngine:
    """
    Extended gate engine with reference-anchored regime detection.

    Combines two checks using the SAME GateEngine machinery:
    - Change: D_W(past, recent) via base engine (detects transitions)
    - Regime: D_W(ref, recent) via regime engine (detects sustained deviation)

    Decision: FAIL if EITHER check fails.

    Reference Establishment (all conditions must be met):
    1. Within configured reference build phase [min_iter, max_iter)
    2. Base check was actually evaluated (not skipped due to insufficient evidence)
    3. Base check passed (ok=True)
    4. Base check worst_DW is present, finite, and <= reference_build_max_dw
    5. Gain is finite and >= reference_build_min_gain (learning health gate)
    6. All tracked metrics have >= min_evidence_per_metric samples accumulated
    7. At least min_windows_for_reference COMPLETE windows accumulated

    A window is "complete" if ALL tracked metrics have at least one finite sample.
    Incomplete windows are skipped and do not count toward min_windows_for_reference.

    Once established, reference is frozen until manual reset_reference() call.

    Auto-Disable:
    If no metrics survive cal_ranges validation, regime detection is automatically
    disabled to avoid "enabled=True but never activates" confusion.
    """

    def __init__(
        self,
        base_engine: GateEngine,
        fr_win: FRWindow,
        config: Optional[RegimeConfig] = None,
        gate_warmup: int = 0,
        gate_window: int = 50,
        pathology_start: Optional[int] = None,
        default_gap_iters: Optional[int] = None,
    ):
        """
        Args:
            base_engine: Existing GateEngine for change detection
            fr_win: FRWindow (shared infrastructure for base)
            config: Regime detection configuration (uses defaults if None)
            gate_warmup: Warmup iterations (for computing build window)
            gate_window: Gate window size (for computing build window)
            pathology_start: If known, when pathology begins (for computing build window).
                            Caller should pass earliest among multiple pathologies.
            default_gap_iters: Default gap between build window end and pathology start.
                              If None, falls back to gate_window.
        """
        self.base = base_engine
        self.fr_win = fr_win
        self.config = config or RegimeConfig()

        # Reference-build starvation diagnostics (observability only)
        # Counts windows where one or more tracked metrics are missing/empty in `recent`.
        # These windows are SKIPPED (not added to accumulator) per the "complete windows" rule.
        self._ref_incomplete_windows_skipped: int = 0
        self._ref_last_incomplete_missing: List[str] = []

        # Compute build window (half-open: [min_iter, max_iter))
        self._build_min_iter, self._build_max_iter = compute_build_window(
            config=self.config,
            gate_warmup=gate_warmup,
            gate_window=gate_window,
            pathology_start=pathology_start,
            default_gap_iters=default_gap_iters,
        )

        # Stable forced-block for empty build windows (do not auto-bump).
        self._ref_build_forced_block: Optional[str] = None
        if self._build_max_iter <= self._build_min_iter:
            self._ref_build_forced_block = "empty_build_window"
            warnings.warn(
                f"Reference build window is empty: [{self._build_min_iter}, {self._build_max_iter}). "
                f"Reference cannot be established.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Canonicalize tracked metrics to weights.keys()
        # This matches what GateEngine actually iterates over during D_W computation
        weights_keys = set(fr_win.decl.weights.keys())
        decl_metrics = set(fr_win.decl.metrics)
        if weights_keys != decl_metrics:
            warnings.warn(
                f"WindowDecl.metrics ({sorted(decl_metrics)}) differs from "
                f"WindowDecl.weights.keys() ({sorted(weights_keys)}). "
                f"Using weights.keys() as canonical metric set.",
                RuntimeWarning,
            )

        # Validate cal_ranges: both key presence AND value validity
        cal_ranges = getattr(fr_win.decl, "cal_ranges", {}) or {}

        # First pass: drop metrics missing from cal_ranges
        missing_cal = weights_keys - set(cal_ranges.keys())
        if missing_cal:
            warnings.warn(
                f"Metrics {sorted(missing_cal)} are in weights but missing from cal_ranges. "
                f"Dropping these metrics from regime tracking to avoid undefined behavior.",
                RuntimeWarning,
            )
            weights_keys = weights_keys - missing_cal

        # Second pass: drop metrics with invalid cal_range values
        invalid_cal: Set[str] = set()
        for m in list(weights_keys):
            entry = cal_ranges.get(m)
            valid, _, _ = _validate_cal_range(entry)
            if not valid:
                invalid_cal.add(m)

        if invalid_cal:
            warnings.warn(
                f"Metrics {sorted(invalid_cal)} have invalid cal_ranges entries "
                f"(non-finite, inverted, or malformed). "
                f"Dropping these metrics from regime tracking.",
                RuntimeWarning,
            )
            weights_keys = weights_keys - invalid_cal

        self._metrics_tracked: Set[str] = weights_keys

        # Auto-disable if no valid metrics survived
        self._enabled_effective: bool = self.config.enabled
        if not self._metrics_tracked:
            warnings.warn(
                "No valid metrics to track after cal_ranges validation. Regime detection is automatically DISABLED.",
                RuntimeWarning,
            )
            self._enabled_effective = False

        # Guard against impossible configuration: min_windows_for_reference > max_accumulator_windows
        # If this holds, reference can never be established because we evict windows at capacity.
        try:
            min_win_req = int(getattr(self.config, "min_windows_for_reference", 0) or 0)
        except Exception:
            min_win_req = 0
        try:
            max_accum = int(getattr(self.config, "max_accumulator_windows", 20))
        except Exception:
            max_accum = 20

        if min_win_req > max_accum and self._enabled_effective:
            warnings.warn(
                f"min_windows_for_reference ({min_win_req}) > max_accumulator_windows ({max_accum}). "
                f"Reference can never be established. Clamping min_windows_for_reference to {max_accum}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._min_windows_for_reference_effective = int(max_accum)
        else:
            self._min_windows_for_reference_effective = int(min_win_req)

        # Derive regime epsilon from base if not explicitly set
        base_epsilon = float(fr_win.decl.epsilon)
        self._regime_epsilon = (
            float(self.config.epsilon)
            if self.config.epsilon is not None
            else base_epsilon * float(self.config.epsilon_mult)
        )

        # Clone WindowDecl safely for regime engine
        self._regime_decl = self._clone_decl_with_epsilon(
            fr_win.decl,
            self._regime_epsilon,
        )

        # Keep regime decl consistent with the tracked metric set.
        # We may drop metrics (e.g., missing cal_ranges) from self._metrics_tracked,
        # but GateEngine.check() typically iterates over decl.weights.keys().
        # If the decl still contains dropped metrics, it can trigger missing-key / no-data paths.
        self._regime_decl = self._prune_decl_to_metrics(self._regime_decl, self._metrics_tracked)

        # Create regime FRWindow with cloned decl
        regime_transport = DeclTransport(self._regime_decl)
        self._regime_decl.attach_transport(regime_transport)
        self._regime_fr_win = FRWindow(
            decl=self._regime_decl,
            transport=regime_transport,
            tests=(),
        )

        # Regime engine: same structure as base
        # IMPORTANT: min_evidence is inherited from base_engine, NOT min_evidence_per_metric
        # min_evidence_per_metric is only for reference establishment
        base_min_evidence = getattr(base_engine, "min_evidence", 16)

        # Extract eps_sens from base engine with fallback tracking
        self._eps_sens_from_base: bool = hasattr(base_engine, "eps_sens")
        if self._eps_sens_from_base:
            self._eps_sens_used = float(base_engine.eps_sens)
        else:
            self._eps_sens_used = 0.04  # Fallback default
            warnings.warn(
                f"base_engine has no 'eps_sens' attribute; using fallback value {self._eps_sens_used}. "
                f"If your GateEngine uses a different attribute name, update regime.py accordingly.",
                RuntimeWarning,
            )

        base_policy = _policy_to_str(getattr(base_engine, "policy", None)).strip().lower()
        base_pk = int(getattr(base_engine, "persistence_k", 2))
        cfg_policy = str(getattr(self.config, "policy", "inherit")).strip().lower()

        if cfg_policy == "inherit":
            regime_policy = base_policy or "either"
            regime_pk = base_pk
        else:
            regime_policy = cfg_policy
            regime_pk = int(getattr(self.config, "persistence_k", 2))

        self.regime_engine = GateEngine(
            frwin=self._regime_fr_win,
            gain_thresh=-1e9,  # Disable gain check - regime is drift-only
            eps_stat_alpha=float(self.config.eps_stat_alpha),
            eps_stat_max_frac=float(self.config.eps_stat_max_frac),
            eps_sens=self._eps_sens_used,
            min_evidence=int(base_min_evidence),
            policy=str(regime_policy),
            persistence_k=int(regime_pk),
        )
        self._regime_policy = str(regime_policy)
        self._regime_persistence_k = int(regime_pk)

        # Reference state
        self._ref: Optional[RegimeReference] = None
        self._accumulating: List[Dict[str, np.ndarray]] = []

        # Statistics for debugging
        self._n_checks: int = 0
        self._n_ref_attempts: int = 0

    @staticmethod
    def _clone_decl_with_epsilon(decl: WindowDecl, new_epsilon: float) -> WindowDecl:
        """
        Create a copy of WindowDecl with modified epsilon.

        Strategy (in order of preference):
        1. Use dataclasses.replace() if WindowDecl is a dataclass
        2. Use decl.copy_with(epsilon=...) if available
        3. Explicit constructor (with warning about potential field loss)
        """
        # Strategy 1: Use dataclasses.replace if it's a dataclass
        import dataclasses

        if dataclasses.is_dataclass(decl) and not isinstance(decl, type):
            try:
                return dataclasses.replace(decl, epsilon=float(new_epsilon))
            except Exception as e:
                warnings.warn(
                    f"dataclasses.replace failed ({e}); falling back to copy_with/constructor",
                    RuntimeWarning,
                )
                pass  # Fall through

        # Strategy 2: Use copy_with() if available
        if hasattr(decl, "copy_with") and callable(getattr(decl, "copy_with")):
            try:
                return decl.copy_with(epsilon=float(new_epsilon))
            except Exception as e:
                warnings.warn(
                    f"decl.copy_with() failed ({e}); falling back to explicit construction.",
                    RuntimeWarning,
                )

        # Strategy 3: Explicit construction (warn about potential field loss)
        warnings.warn(
            "Using explicit WindowDecl construction for cloning. "
            "If WindowDecl has additional fields beyond the standard set, they will be lost. "
            "Consider adding copy_with() method to WindowDecl for robustness.",
            RuntimeWarning,
        )
        return WindowDecl(
            epsilon=float(new_epsilon),
            metrics=list(decl.metrics),
            weights=copy.deepcopy(decl.weights),
            bins=int(decl.bins),
            interventions=decl.interventions,
            cal_ranges=copy.deepcopy(getattr(decl, "cal_ranges", {})),
        )

    @staticmethod
    def _prune_decl_to_metrics(decl: WindowDecl, metrics: Set[str]) -> WindowDecl:
        """Prune WindowDecl weights/metrics/cal_ranges to match the tracked metric set.

        This prevents a mismatch where we drop metrics from regime tracking (e.g., missing cal_ranges)
        but the cloned decl still includes them, while GateEngine iterates over decl.weights.keys().

        Prefer copy_with() when available; otherwise mutate in-place if possible; last resort is
        reconstruction via the known WindowDecl constructor args.
        """
        keep = set(metrics)
        weights2 = {k: decl.weights[k] for k in decl.weights.keys() if k in keep}
        metrics2 = [m for m in list(getattr(decl, "metrics", [])) if m in keep]
        cal_ranges = getattr(decl, "cal_ranges", None)
        cal_ranges2 = None
        if isinstance(cal_ranges, dict):
            cal_ranges2 = {k: cal_ranges[k] for k in cal_ranges.keys() if k in keep}

        # Prefer copy_with() for immutable decls / dataclass-style safety
        if hasattr(decl, "copy_with") and callable(getattr(decl, "copy_with")):
            kw = {"weights": weights2, "metrics": metrics2}
            if cal_ranges2 is not None:
                kw["cal_ranges"] = cal_ranges2
            try:
                return decl.copy_with(**kw)
            except Exception:
                # Fall through to in-place / reconstruction
                pass

        # Try in-place mutation for mutable decls
        try:
            decl.weights = weights2
            decl.metrics = metrics2
            if cal_ranges2 is not None:
                decl.cal_ranges = cal_ranges2
            return decl
        except Exception:
            # Last resort: reconstruct using the standard constructor args
            return WindowDecl(
                epsilon=float(getattr(decl, "epsilon")),
                metrics=metrics2,
                weights=copy.deepcopy(weights2),
                bins=int(getattr(decl, "bins")),
                interventions=getattr(decl, "interventions"),
                cal_ranges=copy.deepcopy(cal_ranges2 or {}),
            )

    # ---- Public properties ----

    @property
    def reference_established(self) -> bool:
        """True if a frozen reference has been established."""
        return self._ref is not None

    @property
    def regime_epsilon(self) -> float:
        """Epsilon threshold used for regime divergence check."""
        return self._regime_epsilon

    @property
    def regime_decl(self) -> WindowDecl:
        """Expose regime WindowDecl for external calculations if needed."""
        return self._regime_decl

    @property
    def enabled(self) -> bool:
        """Whether regime detection is effectively enabled (config AND valid metrics)."""
        return self._enabled_effective

    @property
    def metrics_tracked(self) -> Set[str]:
        """The canonical set of metrics being tracked."""
        return self._metrics_tracked.copy()

    @property
    def build_window(self) -> Tuple[int, int]:
        """The computed reference build window [min_iter, max_iter)."""
        return (self._build_min_iter, self._build_max_iter)

    # ---- Internal helpers ----

    def _get_per_metric_counts(self, data: Dict[str, np.ndarray]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for m in self._metrics_tracked:
            arr = data.get(m)
            if arr is None:
                counts[m] = 0
                continue
            try:
                x = np.asarray(arr, dtype=float).ravel()
            except Exception:
                counts[m] = 0
                continue
            counts[m] = int(np.isfinite(x).sum())
        return counts

    def _all_metrics_have_min_evidence(
        self,
        counts: Dict[str, int],
        min_evidence: int,
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if ALL tracked metrics have at least min_evidence samples.

        Returns (all_sufficient, per_metric_shortfall).
        """
        shortfall = {}
        for m in self._metrics_tracked:
            have = counts.get(m, 0)
            need = min_evidence
            if have < need:
                shortfall[m] = need - have
        return len(shortfall) == 0, shortfall

    def _count_accumulated_samples(self) -> Dict[str, int]:
        """Count total accumulated samples per metric across buffered windows."""
        counts: Dict[str, int] = {m: 0 for m in self._metrics_tracked}
        for snapshot in self._accumulating:
            for k, v in snapshot.items():
                if k in counts:
                    if isinstance(v, np.ndarray):
                        counts[k] += int(v.size)
                    else:
                        counts[k] += int(len(v))
        return counts

    def _merge_accumulated_samples(self) -> Dict[str, np.ndarray]:
        """Merge accumulated snapshots into single arrays per metric."""
        merged: Dict[str, List[np.ndarray]] = {m: [] for m in self._metrics_tracked}
        for snapshot in self._accumulating:
            for k, v in snapshot.items():
                if k in merged:
                    arr = np.asarray(v, dtype=np.float64)
                    if arr.size > 0:
                        merged[k].append(arr.ravel())
        return {k: np.concatenate(v) if v else np.array([], dtype=np.float64) for k, v in merged.items()}

    def _bound_array(
        self,
        arr: np.ndarray,
        max_samples: int,
        metric_name: str,
    ) -> np.ndarray:
        """
        Bound array to max_samples using reservoir-like random selection.

        Uses reproducible seeding based on metric name via SHA256.
        """
        if arr.size <= max_samples:
            return arr
        seed = _stable_metric_seed(metric_name, base_seed=42)
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(arr.size, size=max_samples, replace=False)
        return arr[indices]

    def _in_reference_build_phase(self, iter_num: int) -> Tuple[bool, str]:
        """
        Check if we're within the computed reference build phase.

        Returns (in_phase, reason_if_not).
        """
        iter_num = int(iter_num)
        if iter_num < self._build_min_iter:
            return False, f"iter={iter_num} < min={self._build_min_iter}"
        if iter_num >= self._build_max_iter:
            return False, f"iter={iter_num} >= max={self._build_max_iter}"
        return True, ""

    def _meets_quality_gates(
        self,
        base_result: GateResult,
        gain_bits: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if base result meets quality gates for reference establishment.
        CONSERVATIVE STANCE:
        - If worst_DW is missing or non-finite, the gate FAILS.
        - If gain_bits is non-finite, the gate FAILS.
        This prevents establishing reference during phases where divergence
        or learning health wasn't properly computed.
        Returns (passes, reason_if_failed, gate_values).
        """
        cfg = self.config
        audit = base_result.audit
        gate_values: Dict[str, Any] = {}

        # Extract worst_DW from audit
        worst_dw = audit.get("worst_DW")
        gate_values["worst_DW"] = worst_dw
        gate_values["max_dw_threshold"] = cfg.reference_build_max_dw

        # Gate 1: worst_DW must be present, finite, and below threshold
        if worst_dw is None:
            gate_values["worst_DW"] = None
            return (
                False,
                "worst_DW_missing: cannot establish reference without divergence measurement",
                gate_values,
            )
        try:
            dw_val = float(worst_dw)
            gate_values["worst_DW"] = dw_val
        except (ValueError, TypeError):
            return (
                False,
                f"worst_DW_invalid: cannot convert {worst_dw!r} to float",
                gate_values,
            )

        if not np.isfinite(dw_val):
            return (
                False,
                f"worst_DW_nonfinite: {dw_val} is not finite",
                gate_values,
            )

        if dw_val > cfg.reference_build_max_dw:
            return (
                False,
                f"worst_DW={dw_val:.4f} > threshold={cfg.reference_build_max_dw:.4f}",
                gate_values,
            )

        # Gate 2: gain must be finite and above floor
        # Prefer the base engine's audited gain_bits when present/finite; fall back to wrapper arg.
        gain_use = gain_bits
        try:
            g_audit = float(audit.get("gain_bits"))
            if np.isfinite(g_audit):
                gain_use = g_audit
        except Exception:
            pass

        gate_values["gain_bits"] = gain_use
        gate_values["min_gain_threshold"] = cfg.reference_build_min_gain

        # CONSERVATIVE: non-finite gain fails the quality gate
        if not np.isfinite(gain_use):
            return (
                False,
                f"gain_bits_nonfinite: {gain_use} is not finite",
                gate_values,
            )

        if gain_use < cfg.reference_build_min_gain:
            return (
                False,
                f"gain_bits={gain_use:.4f} < threshold={cfg.reference_build_min_gain:.4f}",
                gate_values,
            )

        return True, "", gate_values

    def _try_establish_reference(
        self,
        recent: Dict[str, np.ndarray],
        iter_num: int,
        quality_audit: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Accumulate healthy samples; establish reference once ALL metrics
        have sufficient evidence AND minimum COMPLETE windows accumulated.

        A window is "complete" if ALL tracked metrics have at least one finite sample.
        Incomplete windows are SKIPPED (not added to accumulator) and recorded
        in _ref_incomplete_windows_skipped for observability.

        Returns (established, reason_if_not).
        """
        self._n_ref_attempts += 1

        # Check for incomplete window: any tracked metric missing or with no finite samples
        missing: List[str] = []
        snapshot: Dict[str, np.ndarray] = {}

        for m in self._metrics_tracked:
            v = recent.get(m, None)
            arr = np.asarray(v if v is not None else [], dtype=np.float64).ravel()
            # Filter to finite values only
            if arr.size > 0:
                arr = arr[np.isfinite(arr)]

            if arr.size == 0:
                missing.append(m)
            else:
                snapshot[m] = arr.copy()

        # If any tracked metric is missing/empty, this is an INCOMPLETE window
        # Skip it entirely (do not add to accumulator) per the strict completeness rule
        if missing:
            self._ref_incomplete_windows_skipped += 1
            self._ref_last_incomplete_missing = sorted(missing)
            return False, f"incomplete_window_skipped: missing={sorted(missing)}"

        # Window is complete - proceed with accumulation

        # Bound accumulator growth (drop oldest window if at capacity)
        if len(self._accumulating) >= self.config.max_accumulator_windows:
            self._accumulating.pop(0)

        # Add current COMPLETE window to accumulator
        self._accumulating.append(snapshot)

        # Minimum temporal coverage requirement (COMPLETE windows, not just any windows).
        # This ensures temporal coverage across the tracked metric set.
        # Use the effective (possibly clamped) value from __init__.
        min_w = int(getattr(self, "_min_windows_for_reference_effective", 0) or 0)
        if min_w > 0 and len(self._accumulating) < min_w:
            return False, f"insufficient_complete_windows: have={len(self._accumulating)} need={min_w}"

        # Check if we have enough evidence per metric
        merged_counts = self._count_accumulated_samples()
        sufficient, shortfall = self._all_metrics_have_min_evidence(
            merged_counts,
            self.config.min_evidence_per_metric,
        )
        if not sufficient:
            return False, f"insufficient_evidence: shortfall={shortfall}"

        # Merge accumulated samples into reference
        merged = self._merge_accumulated_samples()

        # Bound reference size per metric (reproducible sampling)
        bounded_metrics: Dict[str, np.ndarray] = {}
        for k, v in merged.items():
            bounded_metrics[k] = self._bound_array(
                v,
                self.config.max_reference_samples,
                metric_name=k,
            )

        ref_counts = {k: int(v.size) for k, v in bounded_metrics.items()}

        self._ref = RegimeReference(
            metrics=bounded_metrics,
            counts=ref_counts,
            established_at=int(iter_num),
            n_samples_per_metric=ref_counts.copy(),
            build_audit={
                "quality_gates": quality_audit,
                "windows_used": len(self._accumulating),
                "min_windows_required": int(min_w),
                "max_accumulator_windows": int(self.config.max_accumulator_windows),
                "n_attempts": self._n_ref_attempts,
                "incomplete_windows_skipped": int(self._ref_incomplete_windows_skipped),
                "last_incomplete_missing": list(self._ref_last_incomplete_missing),
            },
        )

        # Clear accumulator
        self._accumulating = []
        return True, ""

    # ---- Main API ----

    def check(
        self,
        past: Dict[str, np.ndarray],
        recent: Dict[str, np.ndarray],
        counts_by_metric: Dict[str, int],
        gain_bits: float,
        kappa_sens: float,
        eps_stat_value: float,
        iter_num: int = 0,
    ) -> GateResult:
        """
        Combined check: change detection AND regime detection.

        Returns FAIL if either:
        - Change (past vs recent) exceeds threshold, OR
        - Regime (ref vs recent) exceeds threshold (once ref established)

        Args:
            past: Past window metrics {metric_name: values_array}
            recent: Recent window metrics
            counts_by_metric: Evidence counts for base check (past vs recent)
            gain_bits: Prequential gain for base check
            kappa_sens: Sensitivity parameter
            eps_stat_value: Pre-computed eps_stat for base check
            iter_num: Current iteration (for reference establishment timing)

        Returns:
            GateResult with combined decision and comprehensive audit trail
        """
        self._n_checks += 1

        # 1. Base change detection (existing behavior, unchanged)
        base_result = self.base.check(
            past=past,
            recent=recent,
            counts_by_metric=counts_by_metric,
            gain_bits=gain_bits,
            kappa_sens=kappa_sens,
            eps_stat_value=eps_stat_value,
        )

        # =========================================================================
        # WRAPPER AUDIT: shallow copy preserves all base keys for schema compat
        # =========================================================================
        audit_base = base_result.audit or {}
        audit: Dict[str, Any] = dict(audit_base)  # shallow copy, never mutate audit_base
        base_warn = getattr(base_result, "warn", False)

        # =========================================================================
        # DETERMINE EVALUATION STATUS
        # =========================================================================
        reason = str(audit_base.get("reason", "")).strip()
        not_eval = _not_evaluated_from_reason(reason)
        evaluated_flag = _as_opt_bool(audit_base.get("evaluated", None))
        base_evaluated = False if not_eval else (evaluated_flag if evaluated_flag is not None else True)

        # =========================================================================
        # DW STABILITY: prefer explicit flag, fallback to inference
        # Capture actual values for backfill
        # =========================================================================
        worst_dw_val: Optional[float] = None
        eps_eff_val: Optional[float] = None
        dw_ex = _as_opt_bool(audit_base.get("dw_exceeds_threshold", None))
        if dw_ex is None:
            try:
                worst_dw_val = float(audit_base.get("worst_DW"))
                eps_eff_val = float(audit_base.get("eps_eff"))
                if np.isfinite(worst_dw_val) and np.isfinite(eps_eff_val):
                    dw_ex = bool(worst_dw_val > eps_eff_val)
                else:
                    worst_dw_val = None
                    eps_eff_val = None
            except Exception:
                worst_dw_val = None
                eps_eff_val = None
                dw_ex = None

        if (not base_evaluated) or (dw_ex is None):
            change_dw_ok: Optional[bool] = None
        else:
            change_dw_ok = not dw_ex

        # =========================================================================
        # GAIN-EVALUATED INFERENCE (supports gain_evaluated if present)
        # =========================================================================
        gain_eval = _as_opt_bool(audit_base.get("gain_evaluated", None))
        if gain_eval is None:
            try:
                gb = float(audit_base.get("gain_bits"))
                gain_eval = bool(np.isfinite(gb))
            except Exception:
                gain_eval = False

        if (not base_evaluated) or (not gain_eval):
            change_gain_ok: Optional[bool] = None
        else:
            change_gain_ok = not bool(audit_base.get("gain_below_threshold", False))

        base_ok = bool(base_result.ok)

        # =========================================================================
        # BACKFILL LEGACY KEYS WITH ACTUAL VALUES
        # setdefault: never overwrites existing values from audit_base
        # =========================================================================
        audit.setdefault("reason", reason or None)
        if audit.get("evaluated", None) is None:
            audit["evaluated"] = bool(base_evaluated)
        audit.setdefault("dw_exceeds_threshold", dw_ex)
        audit.setdefault("worst_DW", worst_dw_val)
        audit.setdefault("eps_eff", eps_eff_val)

        # =========================================================================
        # WRAPPER SEMANTICS (tri-state + back-compat)
        # =========================================================================
        audit.update(
            {
                "base_ok": base_ok,
                "base_evaluated": bool(base_evaluated),
                "change_dw_ok": change_dw_ok,  # Optional[bool]
                "change_gain_ok": change_gain_ok,  # Optional[bool]
                # Back-compat only: unknown -> False (never use to regenerate tri-state)
                "change_ok": bool(change_dw_ok is True),
            }
        )

        # =========================================================================
        # NAMESPACED BASE DIAGNOSTICS (for unambiguous downstream analysis)
        # =========================================================================
        audit.update(
            {
                "base_reason": reason or None,
                "base_worst_DW": (
                    audit_base.get("worst_DW") if audit_base.get("worst_DW") is not None else worst_dw_val
                ),
                "base_dw_exceeds_threshold": dw_ex,
                "base_eps_eff": (audit_base.get("eps_eff") if audit_base.get("eps_eff") is not None else eps_eff_val),
                "base_eps_stat_used": audit_base.get("eps_stat", None),
                "base_eps": audit_base.get("eps", None),
                "base_gain_bits": audit_base.get("gain_bits", None),
                "base_gain_evaluated": audit_base.get("gain_evaluated", None),
                "base_gain_thr": audit_base.get("gain_thr", None),
            }
        )

        # =========================================================================
        # EXPLICIT CHANGE-SIDE FIELDS (ROBUST SOURCING)
        # Keep these stable regardless of headline disambiguation.
        # =========================================================================
        def _first_not_none(*vals: Any) -> Any:
            for v in vals:
                if v is not None:
                    return v
            return None

        audit["change_worst_DW"] = _first_not_none(audit.get("base_worst_DW"), audit_base.get("worst_DW"), worst_dw_val)
        audit["change_eps_eff"] = _first_not_none(audit.get("base_eps_eff"), audit_base.get("eps_eff"), eps_eff_val)
        audit["change_eps"] = _first_not_none(audit.get("base_eps"), audit_base.get("eps"))
        audit["change_eps_stat"] = _first_not_none(audit.get("base_eps_stat_used"), audit_base.get("eps_stat"))
        audit["change_gain_bits"] = _first_not_none(audit.get("base_gain_bits"), audit_base.get("gain_bits"))
        audit["change_gain_thr"] = _first_not_none(audit.get("base_gain_thr"), audit_base.get("gain_thr"))

        # Explicit change-side failure predicates (do NOT use base_result.ok here)
        _chg_dw_ok = _as_opt_bool(audit.get("change_dw_ok"))
        _chg_gain_ok = _as_opt_bool(audit.get("change_gain_ok"))
        change_stability_failed = _chg_dw_ok is False  # exceeded threshold
        change_gain_failed = _chg_gain_ok is False  # gain below threshold
        audit["change_stability_failed"] = bool(change_stability_failed)
        audit["change_gain_failed"] = bool(change_gain_failed)

        # If regime detection is disabled, return base result with consistent audit fields
        if not self._enabled_effective:
            audit.update(
                {
                    "regime_enabled": False,
                    "regime_auto_disabled": not self.config.enabled or not self._metrics_tracked,
                    "change_warn": base_warn,
                    "change_evaluated": bool(base_evaluated),
                    "regime_ok": True,
                    "regime_warn": False,
                    "regime_active": False,
                }
            )
            return GateResult(ok=base_result.ok, warn=base_warn, audit=audit)

        # 2. Regime detection (if reference established)
        regime_result: Optional[GateResult] = None
        regime_ok = True
        regime_warn = False
        regime_audit: Dict[str, Any] = {}

        if self._ref is not None:
            # Compute counts for ref vs recent comparison
            # IMPORTANT: Use ref and recent array sizes, NOT counts_by_metric
            recent_counts = self._get_per_metric_counts(recent)
            ref_counts = self._ref.counts
            regime_counts = {m: min(ref_counts.get(m, 0), recent_counts.get(m, 0)) for m in self._metrics_tracked}

            # Optional sanity checks (off by default)
            if os.environ.get("VERISCOPE_REGIME_SANITY", "0").strip() in ("1", "true", "True", "yes", "YES"):
                bad: List[str] = []
                for m in self._metrics_tracked:
                    rc = int(recent_counts.get(m, 0) or 0)
                    fc = int(ref_counts.get(m, 0) or 0)
                    gc = int(regime_counts.get(m, 0) or 0)
                    if gc < 0 or gc > rc or gc > fc:
                        bad.append(f"{m}: regime={gc}, recent={rc}, ref={fc}")
                if bad:
                    _maybe_sanity_warn(
                        "regime_counts",
                        "[regime][sanity] regime_counts invariant violated (expected 0 <= regime <= min(recent, ref)) for: "
                        + ", ".join(bad),
                    )

            # Compute eps_stat for regime check with correct counts
            regime_eps_stat = aggregate_epsilon_stat(
                self._regime_decl,
                regime_counts,
                alpha=float(self.config.eps_stat_alpha),
            )

            # Call regime engine: ref as "past", recent as "recent"
            # This reuses EXACTLY the same D_W calculation as base engine
            regime_result = self.regime_engine.check(
                past=self._ref.metrics,
                recent=recent,
                counts_by_metric=regime_counts,
                gain_bits=0.0,  # Ignored (gain_thresh=-1e9)
                kappa_sens=0.0,
                eps_stat_value=regime_eps_stat,
            )

            regime_ok = regime_result.ok
            regime_warn = getattr(regime_result, "warn", False)

            # Diagnostic: check for transport saturation when regime fails.
            # Use the EXACT gauge the regime engine uses (self._regime_decl, self._metrics_tracked).
            if self._ref is not None and (not regime_ok) and self.config.enabled:
                cal_ranges = getattr(self._regime_decl, "cal_ranges", {})
                for m in self._metrics_tracked:
                    if m not in recent:
                        continue
                    entry = cal_ranges.get(m)
                    if entry is None:
                        continue
                    valid, lo, hi = _validate_cal_range(entry)
                    if not valid:
                        continue
                    arr = np.asarray(recent.get(m, []), dtype=float)
                    diag = diagnose_transport_saturation(arr, lo, hi, name=m)
                    _maybe_warn_saturation(m, diag, threshold=0.10)

            # Optional: clipping diagnostics for cal_ranges mismatch.
            # Only compute when divergence is large to avoid bloating the audit.
            clip_diag_recent: Dict[str, Any] = {}
            clip_diag_ref: Dict[str, Any] = {}
            try:
                dw_val = float(regime_result.audit.get("worst_DW"))
            except Exception:
                dw_val = float("nan")

            if np.isfinite(dw_val) and (dw_val > 0.5):
                _tr = getattr(self._regime_fr_win, "transport", None)
                _clip = getattr(_tr, "clip_diagnostics", None)
                if callable(_clip):
                    for m in self._metrics_tracked:
                        try:
                            clip_diag_recent[m] = _clip(m, recent.get(m, np.array([], float)))
                        except Exception:
                            clip_diag_recent[m] = {"error": "clip_diag_failed"}
                        try:
                            clip_diag_ref[m] = _clip(m, self._ref.metrics.get(m, np.array([], float)))
                        except Exception:
                            clip_diag_ref[m] = {"error": "clip_diag_failed"}

            # Capture regime audit details
            # Prefer per_metric_tv (new), fall back to drifts (legacy), else empty
            _per_metric = regime_result.audit.get("per_metric_tv") or regime_result.audit.get("drifts") or {}
            regime_audit = {
                "regime_worst_DW": regime_result.audit.get("worst_DW"),
                "regime_eps_eff": regime_result.audit.get("eps_eff"),
                "regime_eps_stat": float(regime_eps_stat),
                "regime_per_metric": _per_metric,
                "regime_per_metric_n": regime_result.audit.get("per_metric_n", {}),
                "regime_counts": regime_counts,
            }

            # Attach clipping diagnostics only when computed.
            if clip_diag_recent:
                regime_audit["regime_clip_diag_threshold"] = 0.5
                regime_audit["regime_clip_diag_recent"] = clip_diag_recent
                regime_audit["regime_clip_diag_ref"] = clip_diag_ref

        # 3. Combined decision: fail if EITHER fails
        combined_ok = base_result.ok and regime_ok

        # 4. Reference establishment (only when all conditions met)
        ref_newly_established = False
        ref_build_status: Dict[str, Any] = {}

        if self._ref is None:
            # =========================================================================
            # REFERENCE BUILD GATING
            # =========================================================================
            if getattr(self, "_ref_build_forced_block", None) is not None:
                ref_build_status["blocked"] = str(self._ref_build_forced_block)
            else:
                in_phase, phase_reason = self._in_reference_build_phase(iter_num)
                ref_build_status["in_build_phase"] = in_phase

                if not in_phase:
                    ref_build_status["blocked"] = f"not_in_build_phase: {phase_reason}"
                elif change_dw_ok is not True:
                    ref_build_status["blocked"] = "base_stability_not_ok_or_not_evaluated"
                else:
                    # Preserve existing safety checks (conservative)
                    if not bool(base_evaluated):
                        ref_build_status["blocked"] = "base_not_evaluated"
                    elif not bool(base_ok):
                        ref_build_status["blocked"] = "base_check_failed"
                    else:
                        quality_ok, quality_reason, quality_audit = self._meets_quality_gates(base_result, gain_bits)
                        ref_build_status["quality_gates"] = quality_audit
                        ref_build_status["quality_ok"] = quality_ok
                        if not quality_ok:
                            ref_build_status["blocked"] = f"quality_gate: {quality_reason}"
                        else:
                            established, establish_reason = self._try_establish_reference(
                                recent, iter_num, quality_audit
                            )
                            ref_newly_established = established
                            if not established:
                                ref_build_status["blocked"] = establish_reason

            # Explicit window semantics in status
            ref_build_status.update(
                {
                    "build_window_min_iter": int(self._build_min_iter),
                    "build_window_max_iter": int(self._build_max_iter),
                    "build_window_semantics": "[min_iter, max_iter)",
                    "ref_incomplete_windows_skipped": int(self._ref_incomplete_windows_skipped),
                    "ref_last_incomplete_missing": list(self._ref_last_incomplete_missing),
                }
            )

        # 5. Build comprehensive audit trail
        audit.update(
            {
                # Add regime-specific fields (always present for consistency)
                "change_warn": base_warn,
                "change_evaluated": bool(base_evaluated),
                "regime_ok": regime_ok,
                "regime_warn": regime_warn,
                "regime_enabled": self._enabled_effective,
                "regime_active": self._ref is not None,
                "regime_epsilon": self._regime_epsilon,
                "regime_policy": getattr(self, "_regime_policy", ""),
                "regime_persistence_k": int(getattr(self, "_regime_persistence_k", 2)),
                "regime_eps_sens_used": self._eps_sens_used,
                "regime_eps_sens_from_base": self._eps_sens_from_base,
            }
        )

        # Add regime-specific audit
        audit.update(regime_audit)

        # Add reference state info
        if self._ref is not None:
            audit.update(
                {
                    "ref_established_at": self._ref.established_at,
                    "ref_n_samples": self._ref.n_samples_per_metric,
                }
            )
        else:
            # Report accumulation progress and build status
            accum_counts = self._count_accumulated_samples()
            audit["ref_accumulating"] = accum_counts
            audit["ref_windows_accumulated"] = len(self._accumulating)
            audit["ref_build_phase"] = [self._build_min_iter, self._build_max_iter]
            audit["ref_build_status"] = ref_build_status

        if ref_newly_established:
            audit["ref_just_established"] = True
            audit["ref_build_audit"] = self._ref.build_audit if self._ref else {}

        # =========================================================================
        # DECISION SOURCE + HEADLINE DISAMBIGUATION (MUST BE LAST)
        # - decision_source is based on STABILITY predicates, not base_result.ok
        # - only overwrite headline worst_DW/eps_eff when a STABILITY failure occurred
        # =========================================================================

        # Regime stability failure is only meaningful when regime is active.
        regime_active = bool(audit.get("regime_active", False))
        _reg_ok = _as_opt_bool(audit.get("regime_ok"))
        regime_stability_failed = (_reg_ok is False) and regime_active

        # Change stability failure predicate (read from audit for safety)
        _chg_dw_ok2 = _as_opt_bool(audit.get("change_dw_ok"))
        change_stability_failed = _chg_dw_ok2 is False

        if change_stability_failed and regime_stability_failed:
            decision_source = "both"
        elif regime_stability_failed:
            decision_source = "regime"
        elif change_stability_failed:
            decision_source = "change"
        elif not combined_ok:
            # combined_ok is False but no stability predicate failed => gain or other
            decision_source = "gain_or_other"
        else:
            decision_source = "none"

        audit["decision_source"] = decision_source

        # =========================================================================
        # REASON / PROVENANCE HYGIENE (metadata only; does NOT affect decisions)
        #
        # IMPORTANT:
        # - Do NOT use setdefault() here because artifacts can contain keys with value None.
        # - Only overwrite when the current value "needs fill".
        # =========================================================================
        evaluated = bool(audit.get("evaluated", True))
        ds = str(audit.get("decision_source") or "none").strip().lower() or "none"
        pfail = bool(audit.get("persistence_fail", False))

        # dw_exceeds_threshold: prefer explicit flag; fall back to worst_DW > eps_eff
        dw_ex = _as_opt_bool(audit.get("dw_exceeds_threshold", None))
        if dw_ex is None:
            dwv = _coerce_float_maybe(audit.get("worst_DW"))
            epsv = _coerce_float_maybe(audit.get("eps_eff"))
            if (dwv is not None) and (epsv is not None):
                dw_ex = bool(dwv > epsv)

        # warn_pending is strictly the "persistence warning" state
        warn_pending = bool(evaluated and (dw_ex is True) and (not pfail))
        if _needs_fill(audit.get("warn_pending")):
            audit["warn_pending"] = bool(warn_pending)

        # base_reason: prefer an existing informative string; else synthesize
        if _needs_fill(audit.get("base_reason")):
            # prefer base engine's audited reason if available
            br = audit_base.get("reason", None)
            if _needs_fill(br):
                br = "not_evaluated" if not evaluated else ("base_ok" if base_ok else "base_fail")
            audit["base_reason"] = br

        # change_reason: describe change detector state (not regime)
        if _needs_fill(audit.get("change_reason")):
            if not evaluated:
                cr = "not_evaluated_change"
            elif pfail:
                cr = "change_persistence_fail"
            elif dw_ex is True:
                cr = "change_warn_pending"  # (dw_ex True && not pfail)
            else:
                cr = "change_ok"
            audit["change_reason"] = cr

        # top-level reason: preserve if present; else synthesize using ds + persistence state
        if _needs_fill(audit.get("reason")):
            if not evaluated:
                rr = "not_evaluated"
            elif pfail:
                rr = f"{ds}_persistence_fail"
            elif warn_pending:
                rr = f"{ds}_warn_pending"
            elif not combined_ok:
                rr = f"{ds}_fail"
            else:
                rr = f"{ds}_ok"
            audit["reason"] = rr

        # worst_metric attribution (helps diagnose off-spike WARN/FAIL)
        per_metric = audit.get("per_metric_tv") or audit.get("drifts") or {}
        wk, wv = _pick_worst_metric(per_metric)
        if _needs_fill(audit.get("worst_metric")):
            audit["worst_metric"] = wk
        if _needs_fill(audit.get("worst_metric_tv")):
            audit["worst_metric_tv"] = wv

        # Optional sanity checks (off by default)
        if os.environ.get("VERISCOPE_REGIME_SANITY", "0").strip() in ("1", "true", "True", "yes", "YES"):
            allowed = {"none", "change", "regime", "both", "gain_or_other"}
            if decision_source not in allowed:
                _maybe_sanity_warn(
                    "decision_source",
                    f"[regime][sanity] unexpected decision_source={decision_source!r} (allowed={sorted(allowed)})",
                )
            # Guard against dtype/truthiness surprises in downstream consumers
            for k in ("change_dw_ok", "change_gain_ok", "regime_ok"):
                v = audit.get(k)
                if v is None:
                    continue
                if not isinstance(v, (bool, np.bool_, int, np.integer)):
                    _maybe_sanity_warn(
                        "audit_bool_types",
                        f"[regime][sanity] audit[{k!r}] has unexpected type {type(v).__name__}: {v!r}",
                    )

        def _as_finite_pair(dw: Any, eps: Any) -> Tuple[Optional[float], Optional[float]]:
            try:
                dwf = float(dw)
                epsf = float(eps)
                if np.isfinite(dwf) and np.isfinite(epsf) and epsf > 1e-12:
                    return dwf, epsf
            except Exception:
                pass
            return None, None

        def _exceedance_ratio(dw: Any, eps: Any) -> float:
            dwf, epsf = _as_finite_pair(dw, eps)
            if dwf is None or epsf is None:
                return float("-inf")
            return dwf / epsf

        # Only disambiguate headline for stability-driven failures
        if decision_source in ("change", "regime", "both"):
            if decision_source == "change":
                dw, eps = _as_finite_pair(audit.get("change_worst_DW"), audit.get("change_eps_eff"))
                if dw is not None and eps is not None:
                    audit["worst_DW"] = dw
                    audit["eps_eff"] = eps
            elif decision_source == "regime":
                dw, eps = _as_finite_pair(audit.get("regime_worst_DW"), audit.get("regime_eps_eff"))
                if dw is not None and eps is not None:
                    audit["worst_DW"] = dw
                    audit["eps_eff"] = eps
            else:
                # both: choose the larger normalized exceedance for headline
                r_change = _exceedance_ratio(audit.get("change_worst_DW"), audit.get("change_eps_eff"))
                r_regime = _exceedance_ratio(audit.get("regime_worst_DW"), audit.get("regime_eps_eff"))
                if r_regime >= r_change:
                    dw, eps = _as_finite_pair(audit.get("regime_worst_DW"), audit.get("regime_eps_eff"))
                else:
                    dw, eps = _as_finite_pair(audit.get("change_worst_DW"), audit.get("change_eps_eff"))
                if dw is not None and eps is not None:
                    audit["worst_DW"] = dw
                    audit["eps_eff"] = eps

        combined_warn = base_warn or regime_warn
        return GateResult(ok=combined_ok, warn=combined_warn, audit=audit)

    def reset_reference(self) -> Dict[str, Any]:
        """
        Manual reset - call after recovery intervention or intentional regime change.

        Returns summary of what was cleared.
        """
        summary = {
            "had_reference": self._ref is not None,
            "ref_established_at": self._ref.established_at if self._ref else None,
            "accumulated_windows_cleared": len(self._accumulating),
        }
        self._ref = None
        self._accumulating = []
        # Reset starvation diagnostics
        self._ref_incomplete_windows_skipped = 0
        self._ref_last_incomplete_missing = []
        return summary

    def get_reference_stats(self) -> Optional[Dict[str, Any]]:
        """Return summary stats about the current reference (if established)."""
        if self._ref is None:
            return None
        return {
            "established_at": self._ref.established_at,
            "n_samples_per_metric": dict(self._ref.n_samples_per_metric),
            "metrics_tracked": sorted(self._metrics_tracked),
            "regime_epsilon": self._regime_epsilon,
            "build_audit": self._ref.build_audit,
        }

    def get_accumulator_stats(self) -> Dict[str, Any]:
        """Return stats about the current accumulator state."""
        return {
            "windows_accumulated": len(self._accumulating),
            "max_windows": self.config.max_accumulator_windows,
            "samples_per_metric": self._count_accumulated_samples(),
            "min_evidence_per_metric": self.config.min_evidence_per_metric,
            "min_windows_for_reference": self._min_windows_for_reference_effective,
            "reference_established": self._ref is not None,
            "build_phase": [self._build_min_iter, self._build_max_iter],
            "ref_incomplete_windows_skipped": int(self._ref_incomplete_windows_skipped),
            "ref_last_incomplete_missing": list(self._ref_last_incomplete_missing),
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return comprehensive diagnostics for debugging."""
        return {
            "n_checks": self._n_checks,
            "n_ref_attempts": self._n_ref_attempts,
            "reference_established": self._ref is not None,
            "reference_stats": self.get_reference_stats(),
            "accumulator_stats": self.get_accumulator_stats(),
            "enabled_effective": self._enabled_effective,
            "enabled_config": self.config.enabled,
            "metrics_tracked": sorted(self._metrics_tracked),
            "eps_sens_used": self._eps_sens_used,
            "eps_sens_from_base": self._eps_sens_from_base,
            "min_windows_for_reference_config": int(getattr(self.config, "min_windows_for_reference", 0)),
            "min_windows_for_reference_effective": int(getattr(self, "_min_windows_for_reference_effective", 0)),
            "config": {
                "enabled": self.config.enabled,
                "epsilon": self._regime_epsilon,
                "build_phase": [self._build_min_iter, self._build_max_iter],
                "quality_gates": {
                    "max_dw": self.config.reference_build_max_dw,
                    "min_gain": self.config.reference_build_min_gain,
                },
                "min_evidence_per_metric": self.config.min_evidence_per_metric,
                "min_windows_for_reference": self._min_windows_for_reference_effective,
            },
        }
