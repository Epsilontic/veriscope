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
"""

from __future__ import annotations

import copy
import hashlib
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


def _is_main_process() -> bool:
    """Check if we're in the main process (stdlib, no torch dependency)."""
    try:
        return _mp.current_process().name == "MainProcess"
    except Exception:
        return True  # fail open


def _validate_cal_range(entry) -> tuple[bool, float, float]:
    """Validate a cal_ranges entry. Returns (valid, lo, hi)."""
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
    n = float(len(x))

    return {
        "name": name,
        "n": int(n),
        "lo": lo,
        "hi": hi,
        "actual_min": float(np.min(x)),
        "actual_max": float(np.max(x)),
        "actual_q01": float(np.quantile(x, 0.01)) if len(x) >= 10 else float(np.min(x)),
        "actual_q99": float(np.quantile(x, 0.99)) if len(x) >= 10 else float(np.max(x)),
        "clip_lo_frac": n_below / n,
        "clip_hi_frac": n_above / n,
        "clip_total_frac": (n_below + n_above) / n,
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
        min_iter = gate_warmup + 2 * gate_window
        max_iter = min(pathology_start - gate_window, min_iter + build_span)

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

    # ---- eps_stat parameters for regime divergence calculation ----
    eps_stat_alpha: float = 0.05
    eps_stat_max_frac: float = 0.25

    # ---- Memory bounds ----
    max_reference_samples: int = 10000  # Per metric
    max_accumulator_windows: int = 20  # Total windows buffered

    # ---- Feature flag ----
    # Enable/disable regime detection entirely (for A/B testing)
    enabled: bool = True


def compute_build_window(
    config: RegimeConfig,
    gate_warmup: int,
    gate_window: int,
    pathology_start: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute the reference build window from gate config.

    Called at runtime to resolve sentinel values (-1) in RegimeConfig.

    Args:
        config: RegimeConfig (may have sentinel values)
        gate_warmup: Warmup iterations before gating starts
        gate_window: Window size for gate checks
        pathology_start: If known, iteration where pathology begins
                        (caller should pass earliest among multiple pathologies)

    Returns:
        (min_iter, max_iter) for reference build window
    """
    # Derive min_iter
    if config.reference_build_min_iter >= 0:
        min_iter = int(config.reference_build_min_iter)
    else:
        # Default: after warmup + 2 windows (need history for gate evaluation)
        min_iter = int(gate_warmup) + 2 * int(gate_window)

    # Derive max_iter
    if config.reference_build_max_iter >= 0:
        max_iter = int(config.reference_build_max_iter)
    else:
        # Default: min_iter + build_span, but before any known pathology
        default_max = min_iter + int(config.reference_build_span)

        if pathology_start is not None and pathology_start >= 0:
            # Stop before pathology with safety margin
            safe_max = int(pathology_start) - int(gate_window)
            max_iter = max(min_iter + 100, min(default_max, safe_max))
        else:
            max_iter = default_max

    # Ensure valid window
    if max_iter <= min_iter:
        max_iter = min_iter + max(100, int(config.reference_build_span))
        warnings.warn(
            f"Reference build window was invalid; adjusted to [{min_iter}, {max_iter})",
            RuntimeWarning,
        )

    return min_iter, max_iter


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
    5. Gain >= reference_build_min_gain (learning health gate)
    6. All tracked metrics have >= min_evidence_per_metric samples accumulated

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
        """
        self.base = base_engine
        self.fr_win = fr_win
        self.config = config or RegimeConfig()

        # Compute build window from sentinels
        self._build_min_iter, self._build_max_iter = compute_build_window(
            self.config,
            gate_warmup=gate_warmup,
            gate_window=gate_window,
            pathology_start=pathology_start,
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

        # Validate cal_ranges exist for all tracked metrics
        cal_ranges = getattr(fr_win.decl, "cal_ranges", {}) or {}
        missing_cal = weights_keys - set(cal_ranges.keys())
        if missing_cal:
            warnings.warn(
                f"Metrics {sorted(missing_cal)} are in weights but missing from cal_ranges. "
                f"Dropping these metrics from regime tracking to avoid undefined behavior.",
                RuntimeWarning,
            )
            weights_keys = weights_keys - missing_cal

        self._metrics_tracked: Set[str] = weights_keys

        # Auto-disable if no valid metrics survived
        self._enabled_effective: bool = self.config.enabled
        if not self._metrics_tracked:
            warnings.warn(
                "No valid metrics to track after cal_ranges validation. Regime detection is automatically DISABLED.",
                RuntimeWarning,
            )
            self._enabled_effective = False

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

        self.regime_engine = GateEngine(
            frwin=self._regime_fr_win,
            gain_thresh=-1e9,  # Disable gain check - regime is drift-only
            eps_stat_alpha=float(self.config.eps_stat_alpha),
            eps_stat_max_frac=float(self.config.eps_stat_max_frac),
            eps_sens=self._eps_sens_used,
            min_evidence=int(base_min_evidence),
        )

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
        """Extract sample counts per metric from data arrays."""
        counts = {}
        for m in self._metrics_tracked:
            arr = data.get(m)
            if arr is None:
                counts[m] = 0
            elif isinstance(arr, np.ndarray):
                counts[m] = int(arr.size)
            else:
                try:
                    counts[m] = int(len(arr))
                except TypeError:
                    counts[m] = 0
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

        CONSERVATIVE: If worst_DW is missing or non-finite, the gate FAILS.
        This prevents establishing reference during phases where divergence
        wasn't properly computed.

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

        # Gate 2: gain must be above floor
        gate_values["gain_bits"] = gain_bits
        gate_values["min_gain_threshold"] = cfg.reference_build_min_gain

        if np.isfinite(gain_bits) and gain_bits < cfg.reference_build_min_gain:
            return (
                False,
                f"gain_bits={gain_bits:.4f} < threshold={cfg.reference_build_min_gain:.4f}",
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
        have sufficient evidence.

        Returns (established, reason_if_not).
        """
        self._n_ref_attempts += 1

        # Bound accumulator growth (drop oldest window if at capacity)
        if len(self._accumulating) >= self.config.max_accumulator_windows:
            self._accumulating.pop(0)

        # Add current window to accumulator (copy to avoid mutation)
        snapshot = {}
        for k, v in recent.items():
            if k in self._metrics_tracked:
                arr = np.asarray(v, dtype=np.float64).ravel()
                if arr.size > 0:
                    snapshot[k] = arr.copy()
        self._accumulating.append(snapshot)

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
                "n_attempts": self._n_ref_attempts,
            },
        )

        # Clear accumulator
        self._accumulating = []

        return True, ""

    def _base_check_was_evaluated(self, base_result: GateResult) -> Tuple[bool, str]:
        """
        Determine if the base check actually evaluated (not skipped
        due to insufficient history/evidence).

        Returns (was_evaluated, reason_if_not).
        """
        audit = base_result.audit

        # Check for known "skip" reasons that indicate non-evaluation
        reason = str(audit.get("reason", "")).lower()
        skip_reasons = (
            "insufficient_history",
            "insufficient_evidence",
            "not_evaluated",  # Generic catch-all
            "warmup",
            "skip",
            "no_data",
            "no_metrics",
        )
        for sr in skip_reasons:
            if sr in reason:
                return False, f"skip_reason={sr}"

        # Positive signal: presence of divergence results indicates evaluation
        if "worst_DW" in audit or "drifts" in audit or "per_metric" in audit:
            return True, ""

        # Check for minimum evidence if reported
        base_min_evidence = getattr(self.base, "min_evidence", 16)
        for key in ("min_evidence_seen", "min_count", "evidence", "n_samples"):
            val = audit.get(key)
            if val is not None:
                try:
                    if int(val) < base_min_evidence:
                        return False, f"{key}={val} < min_evidence={base_min_evidence}"
                except (ValueError, TypeError):
                    pass

        # Conservative default: if ok=True and no skip reason, assume evaluated
        return base_result.ok, "assumed_from_ok"

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

        # Extract warn and evaluated status from base result
        base_warn = getattr(base_result, "warn", False)
        base_evaluated = base_result.audit.get("evaluated", True)

        # If regime detection is disabled, return base result with consistent audit fields
        if not self._enabled_effective:
            audit = dict(base_result.audit)
            audit.update(
                {
                    "regime_enabled": False,
                    "regime_auto_disabled": not self.config.enabled or not self._metrics_tracked,
                    "change_ok": base_result.ok,
                    "change_warn": base_warn,
                    "change_evaluated": base_evaluated,
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
            # Check each establishment condition in order
            in_phase, phase_reason = self._in_reference_build_phase(iter_num)
            ref_build_status["in_build_phase"] = in_phase

            if not in_phase:
                ref_build_status["blocked"] = f"not_in_build_phase: {phase_reason}"
            else:
                base_evaluated, eval_reason = self._base_check_was_evaluated(base_result)
                ref_build_status["base_evaluated"] = base_evaluated

                if not base_evaluated:
                    ref_build_status["blocked"] = f"base_not_evaluated: {eval_reason}"
                elif not base_result.ok:
                    ref_build_status["blocked"] = "base_check_failed"
                else:
                    quality_ok, quality_reason, quality_audit = self._meets_quality_gates(base_result, gain_bits)
                    ref_build_status["quality_gates"] = quality_audit
                    ref_build_status["quality_ok"] = quality_ok

                    if not quality_ok:
                        ref_build_status["blocked"] = f"quality_gate: {quality_reason}"
                    else:
                        # All gates passed, try to establish
                        established, establish_reason = self._try_establish_reference(recent, iter_num, quality_audit)
                        ref_newly_established = established
                        if not established:
                            ref_build_status["blocked"] = establish_reason

        # 5. Build comprehensive audit trail
        audit: Dict[str, Any] = {
            # Inherit all base audit fields
            **base_result.audit,
            # Add regime-specific fields (always present for consistency)
            "change_ok": base_result.ok,
            "change_warn": base_warn,
            "change_evaluated": base_evaluated,
            "regime_ok": regime_ok,
            "regime_warn": regime_warn,
            "regime_enabled": self._enabled_effective,
            "regime_active": self._ref is not None,
            "regime_epsilon": self._regime_epsilon,
            "regime_eps_sens_used": self._eps_sens_used,
            "regime_eps_sens_from_base": self._eps_sens_from_base,
        }

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
            "reference_established": self._ref is not None,
            "build_phase": [self._build_min_iter, self._build_max_iter],
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
            "config": {
                "enabled": self.config.enabled,
                "epsilon": self._regime_epsilon,
                "build_phase": [self._build_min_iter, self._build_max_iter],
                "quality_gates": {
                    "max_dw": self.config.reference_build_max_dw,
                    "min_gain": self.config.reference_build_min_gain,
                },
                "min_evidence_per_metric": self.config.min_evidence_per_metric,
            },
        }
