# veriscope/runners/gpt/train_nanogpt.py
"""
nanoGPT training with veriscope FR gating.

Minimal modifications to the standard nanoGPT train.py.
"""

# -------------------------------------------------------------------------
# CANONICAL GPT SPIKE / CORRUPTION SMOKE CONFIG (empirically validated)
#
# This configuration is intended for *change-only* corruption detection
# experiments (e.g. data corruption between iters 2500–2900). Regime
# detection may remain enabled, but evaluation should read the
# "CORRUPTION DETECTION (change-only)" section from analyze_gates.py.
#
# Recommended CLI parameters:
#
#   --metric_interval 2
#   --gate_window 75
#   --gate_warmup 1500
#   --gate_epsilon 0.25
#   # (optional looser probe: --gate_epsilon 0.30)
#   --gate_eps_stat_max_frac 0.15
#   --gate_min_evidence 75
#   --gate_gain_thresh -0.002
#
# Example invocation:
#
#   python -m veriscope.runners.gpt.train_nanogpt \
#     --dataset shakespeare_char \
#     --nanogpt_dir /workspace/nanoGPT \
#     --device cuda \
#     --out_dir /workspace/out \
#     --out_json veriscope_gpt_datainject_perm15_gateE0p25_W75_$(date +%Y%m%d_%H%M%S).json \
#     --metric_interval 2 \
#     --gate_window 75 \
#     --gate_warmup 1500 \
#     --gate_epsilon 0.25 \
#     --gate_eps_stat_max_frac 0.15 \
#     --gate_min_evidence 75 \
#     --gate_gain_thresh -0.002 \
#     --data_corrupt_at 2500 \
#     --data_corrupt_len 400 \
#     --data_corrupt_frac 0.15 \
#     --data_corrupt_mode permute
#
# NOTE:
#   • Change detector = "is something happening now?"
#   • Regime detector = "has the model drifted from a known-good baseline?"
#   • Do NOT score spike experiments using the union gate when regime is active.
# -------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import random
from datetime import datetime, timezone
from pathlib import Path
import time
import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn


# Helper to ensure nanoGPT is importable
def _ensure_nanogpt_on_path(nanogpt_dir: str) -> None:
    """Ensure the nanoGPT checkout (containing model.py) is importable."""
    p = str(Path(nanogpt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_seed(cli_seed: Optional[int], env: Dict[str, str], default_seed: int) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    raw = (env.get("VERISCOPE_SEED") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            warnings.warn(f"Ignoring invalid VERISCOPE_SEED={raw!r}; using default seed {default_seed}.")
    return int(default_seed)


def _seed_all(seed: int) -> None:
    seed_int = int(seed)
    seed32 = seed_int % (2**32)
    random.seed(seed_int)
    np.random.seed(seed32)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_window_spans(
    gate_window_iters: int,
    metric_interval_iters: int,
) -> Tuple[int, int, int]:
    """Compute window spans with correct unit handling.

    All inputs and outputs are in iteration units unless noted.

    Args:
        gate_window_iters: The gate check cadence in iterations (config.gate_window).
            Gate checks run when iter_num % gate_window_iters == 0.
        metric_interval_iters: Stride between metric snapshots in iterations
            (config.metric_interval).

    Returns:
        Wm: Number of metric snapshots per half-window. This is what the gate
            actually uses for past/recent comparisons.
        window_span_iters: Actual iteration span covered by each half-window
            (= Wm * metric_interval_iters).
        stride_iters: Same as metric_interval_iters (returned for convenience).

    This is factored out to ensure consistency between:
    - VeriscopeGatedTrainer.__init__ (build window computation)
    - VeriscopeGatedTrainer._compute_gate_check (gate evaluation)
    """
    stride_iters = max(1, int(metric_interval_iters))
    Wm = max(1, int(gate_window_iters) // stride_iters)
    window_span_iters = Wm * stride_iters
    return Wm, window_span_iters, stride_iters


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _needs_fill(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        return (s == "") or (s.lower() == "none")
    try:
        return str(value).strip().lower() == "none"
    except Exception:
        return False


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _compute_metrics_exceeding(per_metric_tv: Dict[str, Any], eps_eff: float) -> Tuple[List[str], str]:
    if not isinstance(per_metric_tv, dict) or not per_metric_tv:
        return [], "combined_only_no_tv"
    try:
        eps_eff_f = float(eps_eff)
    except Exception:
        return [], "combined_only_no_tv"
    if not np.isfinite(eps_eff_f):
        return [], "combined_only_no_tv"
    exceeding: List[str] = []
    for name, val in per_metric_tv.items():
        try:
            if isinstance(val, dict) and "tv" in val:
                tv = float(val["tv"])
            else:
                tv = float(val)
        except Exception:
            continue
        if np.isfinite(tv) and tv > eps_eff_f:
            exceeding.append(str(name))
    if not exceeding:
        return [], "combined_only"
    return sorted(exceeding), "per_metric_threshold"


def _normalize_gate_audit(
    audit_in: Dict[str, Any],
    *,
    config: "TrainConfig",
    gain_bits: float,
    row_reason: str,
    row_base_reason: str,
    row_change_reason: str,
) -> Dict[str, Any]:
    audit: Dict[str, Any] = dict(audit_in or {})
    audit["evaluated"] = bool(audit.get("evaluated", True))
    audit["policy"] = str(audit.get("policy", getattr(config, "gate_policy", "either")) or "either")
    audit.setdefault("reason", row_reason)
    audit["base_reason"] = row_base_reason
    audit["change_reason"] = row_change_reason
    audit.setdefault("per_metric_tv", {})

    gain_thresh = float(getattr(config, "gate_gain_thresh", 0.0))
    audit["gain_thresh"] = gain_thresh
    if _needs_fill(audit.get("gain_thr")):
        audit["gain_thr"] = gain_thresh

    gain_bits_val = _coerce_float(audit.get("gain_bits", gain_bits))
    audit["gain_bits"] = gain_bits_val
    gain_ok_val = audit.get("gain_ok", audit.get("ok_gain", None))
    if gain_ok_val is None:
        if np.isfinite(gain_bits_val) and np.isfinite(gain_thresh):
            gain_ok_val = gain_bits_val >= gain_thresh
        else:
            gain_ok_val = True
    audit["gain_ok"] = bool(gain_ok_val)

    worst_dw = _coerce_float(audit.get("worst_DW", audit.get("worst_dw", float("nan"))))
    eps_eff = _coerce_float(audit.get("eps_eff", float("nan")))
    audit["worst_DW"] = worst_dw
    audit["eps_eff"] = eps_eff

    dw_exceeds = audit.get("dw_exceeds_threshold", None)
    if dw_exceeds is None:
        dw_exceeds = bool(np.isfinite(worst_dw) and np.isfinite(eps_eff) and worst_dw > eps_eff)
    audit["dw_exceeds_threshold"] = bool(dw_exceeds)

    per_metric_tv = audit.get("per_metric_tv")
    if not isinstance(per_metric_tv, dict):
        per_metric_tv = {}
        audit["per_metric_tv"] = per_metric_tv

    metrics_exceeding, exceedance_mode = _compute_metrics_exceeding(per_metric_tv, eps_eff)

    # If combined D_W is within tolerance, do not advertise a per-metric "exceedance" mode.
    # We may still keep metrics_exceeding as a diagnostic/probe signal, but it is not an exceedance.
    if not bool(audit.get("dw_exceeds_threshold", False)):
        if exceedance_mode == "per_metric_threshold":
            exceedance_mode = "combined_ok_metric_probe"
        elif exceedance_mode == "combined_only":
            exceedance_mode = "combined_ok"

    if audit["evaluated"]:
        audit["exceedance_mode"] = exceedance_mode
    else:
        audit.setdefault("exceedance_mode", "not_evaluated")

    # Keep the list (diagnostic). Decision logic should still key off dw_exceeds_threshold.
    if not metrics_exceeding:
        existing = audit.get("metrics_exceeding")
        if isinstance(existing, list) and existing:
            metrics_exceeding = sorted(str(m) for m in existing)
    audit["metrics_exceeding"] = metrics_exceeding

    min_metrics_exceeding = audit.get("min_metrics_exceeding", getattr(config, "gate_min_metrics_exceeding", 1))
    audit["min_metrics_exceeding"] = _coerce_int(min_metrics_exceeding, default=1)

    if "evidence_total" not in audit:
        audit["evidence_total"] = _coerce_int(audit.get("total_evidence", 0), default=0)
    audit["min_evidence"] = _coerce_int(audit.get("min_evidence", getattr(config, "gate_min_evidence", 0)), default=0)

    if audit["dw_exceeds_threshold"] and not audit["metrics_exceeding"]:
        items: List[Tuple[str, float]] = []
        for name, val in per_metric_tv.items():
            try:
                if isinstance(val, dict) and "tv" in val:
                    tv = float(val["tv"])
                else:
                    tv = float(val)
            except Exception:
                continue
            if np.isfinite(tv):
                items.append((str(name), float(tv)))
        items.sort(key=lambda kv: kv[1], reverse=True)
        audit["metrics_exceeding"] = [name for name, _ in items[:5]]
        audit["exceedance_mode"] = "topk_contributors" if items else exceedance_mode

    return audit


def _derive_gate_decision(evaluated: bool, ok: bool, warn: bool) -> str:
    return _derive_gate_decision_v1(evaluated=evaluated, ok=ok, warn=warn)


def _strip_regime_audit_fields(audit_in: Dict[str, Any]) -> Dict[str, Any]:
    audit = dict(audit_in or {})
    for key in list(audit.keys()):
        if str(key).startswith("regime_"):
            audit.pop(key, None)
    return audit


def _validate_gate_row(row: Dict[str, Any]) -> None:
    decision = row.get("decision")
    if decision not in {"pass", "warn", "fail", "skip"}:
        raise ValueError(f"invalid gate decision={decision!r}")
    audit = row.get("audit", {})
    if not isinstance(audit, dict):
        raise ValueError("gate row audit must be an object")
    if "reason" in audit and audit.get("reason") != row.get("reason"):
        row["reason"] = audit.get("reason")
    evaluated = bool(audit.get("evaluated", True))
    if not evaluated:
        if decision != "skip":
            raise ValueError("decision must be 'skip' when audit.evaluated=False")
    else:
        if decision == "warn":
            if row.get("ok") is not True:
                raise ValueError("decision='warn' requires ok=True")
        elif decision == "pass":
            if row.get("ok") is not True:
                raise ValueError("decision='pass' requires ok=True")
        elif decision == "fail":
            if row.get("ok") is not False:
                raise ValueError("decision='fail' requires ok=False")
    if row.get("ok") is False:
        if row.get("warn") is True:
            raise ValueError("warn cannot be True when ok=False")
    if row.get("warn"):
        if decision != "warn":
            raise ValueError("warn=True requires decision='warn'")
        if row.get("reason") in {"none_ok", "", None}:
            raise ValueError("warn rows require a non-empty non-'none_ok' reason")
    required = {
        "reason",
        "evaluated",
        "policy",
        "gain_bits",
        "gain_thresh",
        "gain_ok",
        "worst_DW",
        "eps_eff",
        "dw_exceeds_threshold",
        "metrics_exceeding",
        "min_metrics_exceeding",
        "evidence_total",
        "min_evidence",
    }
    missing = [k for k in required if k not in audit]
    if missing:
        raise ValueError(f"missing audit keys: {missing}")
    if audit.get("dw_exceeds_threshold"):
        mode = audit.get("exceedance_mode")
        if mode not in {"combined_only_no_tv", "not_evaluated"}:
            if not audit.get("metrics_exceeding"):
                raise ValueError("dw_exceeds_threshold rows must include metrics_exceeding for this exceedance_mode")


# Sanity check for emitted gate rows (example one-liner):
# python - <<'PY'
# import json, sys
# data = json.load(open(sys.argv[1]))
# gates = data.get("gate_history") or data.get("gates") or data.get("gate_checks") or []
# bad = [g for g in gates if g.get("audit", {}).get("dw_exceeds_threshold") and not g.get("audit", {}).get("metrics_exceeding")]
# modes = {g.get("audit", {}).get("exceedance_mode") for g in gates}
# print("bad_dw_exceeds_rows=", len(bad))
# print("exceedance_modes=", sorted(m for m in modes if m))
# PY


# Your veriscope imports
from veriscope.runners.gpt.adapter import (
    GPTMetricConfig,
    GPTFeatureExtractor,
    GPTMetricComputer,
    create_gpt_window_decl,
    create_gpt_gate_engine,
)
from veriscope.core.calibration import aggregate_epsilon_stat
from veriscope.runners.gpt.emit_artifacts import emit_gpt_artifacts_v1
from veriscope.core.artifacts import derive_gate_decision as _derive_gate_decision_v1
from veriscope.core.jsonutil import atomic_write_json, canonical_json_sha256
from veriscope.core.redaction_policy import POLICY_REV, prepare_env_capture, redact_argv

# Regime-anchored detection imports
from veriscope.core.regime import (
    RegimeAnchoredGateEngine,
    RegimeConfig,
    compute_build_window,
)


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    dataset: str = "openwebtext"
    batch_size: int = 12
    block_size: int = 1024

    # Model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    # Training
    max_iters: int = 600000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # Pathology injection (optional; for gate validation)
    lr_spike_at: int = -1  # iteration to start spike; <0 disables
    lr_spike_len: int = 0  # number of iterations to spike
    lr_spike_mult: float = 1.0  # multiplier during spike
    lr_spike_verify: bool = False  # record & print a verification ratio for debugging

    # Data pathology injection (token corruption)
    data_corrupt_at: int = -1  # iteration to start corruption; <0 disables
    data_corrupt_len: int = 0  # number of iterations to corrupt
    data_corrupt_frac: float = 0.0  # fraction of tokens to corrupt per sequence
    data_corrupt_mode: str = "permute"  # "permute", "random", or "mask"

    # Logging
    eval_interval: int = 1000
    log_interval: int = 10
    metric_interval: int = 5  # compute veriscope metrics every N iterations
    eval_iters: int = 200

    # --- Logit diagnostics (diagnostic-only; NOT gated) ---
    log_logit_norm: bool = False
    logit_norm_stride: int = 0  # 0 => use metric_interval
    logit_norm_max_tokens: int = 256  # 0 => use all tokens (B*T); order stats are guarded below
    logit_norm_order_stats_max_tokens: int = 4096  # compute median/p95 only when N <= this
    log_logit_std: bool = True  # std across vocab per token (optional)

    # Veriscope gating
    gate_enabled: bool = True
    gate_window: int = 50  # iterations, not epochs
    gate_warmup: int = 1000  # don't gate until model is warmed up
    gate_epsilon: float = 0.12
    gate_gain_thresh: float = 0.0  # stability-only by default; tune upward if you want "learning+stability"
    gate_min_evidence: int = 16
    # Multi-metric consensus: require >=K metrics with per-metric TV > eps_eff
    # before stability exceedance engages WARN/FAIL/persistence. 1 = legacy behavior.
    gate_min_metrics_exceeding: int = 1
    gate_eps_stat_max_frac: float = 0.25  # cap eps_stat as fraction of epsilon

    # Gate policy: controls when gate FAILs
    # - "either": FAIL if gain OR stability fails (original default)
    # - "conjunction": FAIL if gain AND stability both fail
    # - "persistence": FAIL if stability fails for K consecutive evaluated checks
    gate_policy: str = "either"
    gate_persistence_k: int = 2  # For persistence: consecutive exceedances to FAIL

    # Regime-anchored detection (reference-based drift)
    regime_enabled: bool = True
    regime_build_min_iter: int = -1  # Sentinel: auto-compute
    regime_build_max_iter: int = -1  # Sentinel: auto-compute
    regime_build_span: int = 1500
    regime_build_max_dw: float = 0.08
    regime_build_min_gain: float = -0.01
    regime_epsilon_mult: float = 1.5
    regime_min_evidence: int = 50
    regime_min_windows: int = 5
    freeze_metric_gauge_min_ref_updates: int = 25
    regime_build_gap_iters: int = -1  # explicit gap override (-1 = auto)

    # Optional: freeze GPT feature normalization once regime reference is established.
    # This makes anchored regime comparisons use a fixed metric gauge.
    freeze_metric_gauge_on_ref: bool = False

    # WindowDecl tuning
    cos_disp_max: float = 1.0

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False

    # Calibration recorder (optional)
    calibration_enabled: bool = False
    calibration_output_path: str = ""
    calibration_buffer_size: int = 100
    calibration_include_per_metric: bool = False
    calibration_only_evaluated: bool = True
    calibration_only_healthy: bool = False
    calibration_main_process_only: bool = True
    calibration_fail_on_schema_mismatch: bool = True
    base_seed: int = 42


class VeriscopeGatedTrainer:
    """
    GPT trainer with veriscope finite-window gating.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.ctx = self._get_autocast_context()

        # Initialize model UNCOMPILED first (hooks + torch.compile can be flaky otherwise)
        self.model = self._init_model()

        # Veriscope components (set up hooks BEFORE optional compilation)
        self.metric_config = GPTMetricConfig(
            probe_layers="last",
            max_tokens_per_batch=1024,
            geom_rp_dim=64,
        )
        self.extractor = GPTFeatureExtractor(self.model, self.metric_config, self.device)

        # Optionally compile AFTER hooks are registered.
        if config.compile:
            print("Compiling model...")
            self.model = torch.compile(self.model)
            # Ensure extractor forwards through the compiled wrapper
            self.extractor.model = self.model

        self.metric_computer = GPTMetricComputer(self.extractor, self.metric_config, self.device)

        # Optimizer/scaler AFTER potential compilation
        self.optimizer = self._init_optimizer()

        # Prefer torch.amp.GradScaler API; enable only for float16 on CUDA
        scaler_enabled = (config.dtype == "float16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        # Create window declaration and gate engine
        # Thread eff_dim_max from the JL projection dimension so eff_dim calibration
        # matches the feature space used by the metric computer.
        try:
            self.window_decl = create_gpt_window_decl(
                epsilon=config.gate_epsilon,
                bins=16,
                eff_dim_max=float(self.metric_config.geom_rp_dim),
                rankme_max=float(self.metric_config.geom_rp_dim),
                cos_disp_max=float(config.cos_disp_max),
            )
        except TypeError:
            # Backward-compat: older create_gpt_window_decl signatures
            self.window_decl = create_gpt_window_decl(
                epsilon=config.gate_epsilon,
                bins=16,
            )
        self.fr_win, base_gate_engine = create_gpt_gate_engine(
            self.window_decl,
            {
                "gate_gain_thresh": config.gate_gain_thresh,
                "gate_min_evidence": config.gate_min_evidence,
                "gate_min_metrics_exceeding": int(config.gate_min_metrics_exceeding),
                "gate_eps_stat_max_frac": float(config.gate_eps_stat_max_frac),
                "gate_epsilon_sens": 0.04,
                "gate_policy": config.gate_policy,
                "gate_persistence_k": config.gate_persistence_k,
            },
        )

        # Determine EARLIEST pathology start for auto build window computation
        pathology_start: Optional[int] = None
        if config.lr_spike_at >= 0:
            pathology_start = int(config.lr_spike_at)
        if config.data_corrupt_at >= 0:
            if pathology_start is None or int(config.data_corrupt_at) < pathology_start:
                pathology_start = int(config.data_corrupt_at)

        # --- Compute window spans with correct unit handling ---
        Wm, window_span_iters, stride_iters = compute_window_spans(config.gate_window, config.metric_interval)
        default_gap_iters = max(2 * int(window_span_iters), 2 * int(stride_iters))

        # Warn if Wm is dangerously small (noisy comparisons)
        if Wm < 5:
            print(
                f"[REGIME WARN] Wm={Wm} snapshots per half-window is small. "
                f"gate_window={config.gate_window}, metric_interval={config.metric_interval}. "
                f"Consider increasing gate_window or decreasing metric_interval."
            )

        # ---------------------------------------------------------------------
        # Configure regime detection
        # IMPORTANT: Do not recompute build windows here.
        # Let veriscope/core/regime.py::compute_build_window() be the canonical source.
        # We only validate/clamp explicit user inputs.
        # ---------------------------------------------------------------------
        regime_enabled = bool(config.regime_enabled)

        rb_min = int(config.regime_build_min_iter)
        rb_max = int(config.regime_build_max_iter)

        # Clamp explicit values to feasible iteration range (non-fatal).
        if rb_min >= 0:
            rb_min = max(0, min(int(config.max_iters), rb_min))
        if rb_max >= 0:
            rb_max = max(0, min(int(config.max_iters), rb_max))

        # If user explicitly provided both, ensure it's valid; fail closed.
        if rb_min >= 0 and rb_max >= 0 and rb_max <= rb_min:
            print(f"[REGIME WARN] Explicit build window invalid: [{rb_min}, {rb_max}). Disabling regime for this run.")
            regime_enabled = False

        # Configure regime detection. Build-window semantics are resolved inside RegimeAnchoredGateEngine.
        regime_kwargs: Dict[str, Any] = dict(
            enabled=regime_enabled,
            epsilon=None,  # Derive as epsilon_mult * base epsilon
            epsilon_mult=float(config.regime_epsilon_mult),
            reference_build_min_iter=int(rb_min),
            reference_build_max_iter=int(rb_max),
            reference_build_span=int(config.regime_build_span),
            reference_build_gap_iters=int(config.regime_build_gap_iters),
            reference_build_max_dw=float(config.regime_build_max_dw),
            reference_build_min_gain=float(config.regime_build_min_gain),
            min_evidence_per_metric=int(config.regime_min_evidence),
            eps_stat_alpha=0.05,
            eps_stat_max_frac=float(config.gate_eps_stat_max_frac),
            max_reference_samples=10000,
            max_accumulator_windows=20,
        )

        # Map CLI regime_min_windows to RegimeConfig.min_windows_for_reference.
        # IMPORTANT: Only override if user explicitly set a positive value.
        # Otherwise, preserve RegimeConfig's default (currently 5).
        # Guard handles both dataclass and non-dataclass RegimeConfig variants.
        if (
            hasattr(RegimeConfig, "__dataclass_fields__")
            and "min_windows_for_reference" in RegimeConfig.__dataclass_fields__
        ):
            rmw = int(getattr(config, "regime_min_windows", 0) or 0)
            if rmw > 0:
                regime_kwargs["min_windows_for_reference"] = rmw
            # If rmw <= 0, don't add to kwargs; RegimeConfig default applies

        regime_config = RegimeConfig(**regime_kwargs)
        self._regime_wrapper_enabled = bool(regime_enabled)

        if self._regime_wrapper_enabled:
            # Wrap with regime-anchored detection only when explicitly enabled.
            self.gate_engine = RegimeAnchoredGateEngine(
                base_engine=base_gate_engine,
                fr_win=self.fr_win,
                config=regime_config,
                gate_warmup=int(config.gate_warmup),
                gate_window=int(config.gate_window),
                pathology_start=pathology_start,
                default_gap_iters=int(default_gap_iters),
            )
            build_min, build_max = self.gate_engine.build_window
            regime_epsilon = float(self.gate_engine.regime_epsilon)
            regime_enabled_effective = bool(getattr(self.gate_engine, "enabled", False))
        else:
            self.gate_engine = base_gate_engine
            # Keep reference-build metadata deterministic even when regime wrapper is disabled.
            build_min, build_max = compute_build_window(
                regime_config,
                gate_warmup=int(config.gate_warmup),
                gate_window=int(config.gate_window),
                pathology_start=pathology_start,
                default_gap_iters=int(default_gap_iters),
            )
            regime_epsilon = float("nan")
            regime_enabled_effective = False

        self.build_window = (int(build_min), int(build_max))

        # ---------------------------------------------------------------------
        # Optional calibration recorder attachment
        # ---------------------------------------------------------------------
        from veriscope.core.calibration_recorder import CalibrationRecorder, CalibrationRecorderConfig

        self._calibration_recorder: Optional[CalibrationRecorder] = None
        if (
            bool(getattr(config, "calibration_enabled", False))
            and str(getattr(config, "calibration_output_path", "")).strip()
        ):
            rec_cfg = CalibrationRecorderConfig(
                enabled=True,
                output_path=str(config.calibration_output_path),
                buffer_size=int(getattr(config, "calibration_buffer_size", 100)),
                include_per_metric=bool(getattr(config, "calibration_include_per_metric", False)),
                only_evaluated=bool(getattr(config, "calibration_only_evaluated", True)),
                only_healthy=bool(getattr(config, "calibration_only_healthy", False)),
                main_process_only=bool(getattr(config, "calibration_main_process_only", True)),
                fail_on_schema_mismatch=bool(getattr(config, "calibration_fail_on_schema_mismatch", True)),
            )
            self._calibration_recorder = CalibrationRecorder(rec_cfg)
            if hasattr(self.gate_engine, "set_calibration_recorder"):
                self.gate_engine.set_calibration_recorder(self._calibration_recorder)

            if getattr(self._calibration_recorder, "is_active", False):
                print(f"[CAL] calibration recorder enabled -> {rec_cfg.output_path}")

        # Metric gauge freezing (optional): freeze before reference is built.
        self._freeze_gauge_at_iter: Optional[int] = None
        self._freeze_gauge_first_check: Optional[int] = None
        self._freeze_gauge_window_start_iter: Optional[int] = None
        # NOTE: GPTFeatureExtractor._ref_count increments once per metric snapshot (i.e., once per
        # compute_all() call), NOT once per training iteration. It scales with metric_interval.
        self._freeze_metric_gauge_min_ref_updates: int = int(getattr(config, "freeze_metric_gauge_min_ref_updates", 0))
        self._freeze_defer_log_every: int = 500

        # Log computed build window and effective status.
        if self._regime_wrapper_enabled:
            print(
                f"[REGIME] enabled={regime_enabled_effective}, "
                f"build_window=[{build_min}, {build_max}), "
                f"epsilon={regime_epsilon:.4f}"
            )
        else:
            print(f"[REGIME] enabled=False (wrapper not instantiated), build_window=[{build_min}, {build_max})")
        # Freeze gauge BEFORE the first eligible reference-building gate check uses "recent" samples.
        # Gate checks happen at iter multiples of gate_window. Metric snapshots happen every metric_interval.
        # We freeze at window_start so that all metric snapshots used to build the reference are in the frozen gauge.
        if bool(config.freeze_metric_gauge_on_ref) and bool(regime_enabled_effective):
            gw = max(1, int(config.gate_window))
            mi = max(1, int(config.metric_interval))
            first_check = ((int(build_min) + gw - 1) // gw) * gw
            last_snap = (int(first_check) // mi) * mi
            freeze_at = max(0, int(last_snap) - int(window_span_iters))
            self._freeze_gauge_first_check = int(first_check)
            self._freeze_gauge_window_start_iter = int(freeze_at)
            self._freeze_gauge_at_iter = int(freeze_at)
            print(
                f"[REGIME] freeze_metric_gauge_on_ref=True -> will freeze GPT metric gauge at iter={self._freeze_gauge_at_iter} "
                f"(first_check={first_check}, last_snap={last_snap}, window_start={freeze_at}, build_min={build_min}, "
                f"window_span_iters={window_span_iters}, min_ref_updates={self._freeze_metric_gauge_min_ref_updates})"
            )

        # If explicit override was requested, warn about tightness (few gate checks / insufficient evidence).
        explicit_override = (int(config.regime_build_min_iter) >= 0) or (int(config.regime_build_max_iter) >= 0)
        if explicit_override and bool(regime_enabled_effective):
            gw = max(1, int(config.gate_window))
            first_check = ((build_min + gw - 1) // gw) * gw
            last_check = ((build_max - 1) // gw) * gw
            n_checks = 0 if first_check >= build_max else (1 + (last_check - first_check) // gw)

            # Each check contributes ~Wm samples per metric; estimate checks needed to reach min_evidence_per_metric.
            need_checks = int((int(config.regime_min_evidence) + int(Wm) - 1) // int(Wm)) if int(Wm) > 0 else 2
            if n_checks < need_checks:
                print(
                    f"[REGIME WARN] Explicit build window contains only {n_checks} gate check(s); "
                    f"need ~{need_checks} to satisfy min_evidence_per_metric={config.regime_min_evidence} with Wm={Wm}. "
                    f"Reference establishment may never trigger."
                )

        # Hardening: independently compute build window and warn on mismatch.
        if self._regime_wrapper_enabled:
            try:
                bm2, bx2 = compute_build_window(
                    regime_config,
                    gate_warmup=int(config.gate_warmup),
                    gate_window=int(config.gate_window),
                    pathology_start=pathology_start,
                    default_gap_iters=int(default_gap_iters),
                )
                if (int(bm2), int(bx2)) != (int(build_min), int(build_max)):
                    warnings.warn(
                        f"[REGIME] build_window mismatch: engine=[{build_min},{build_max}) vs compute=[{bm2},{bx2})",
                        RuntimeWarning,
                    )
            except Exception:
                pass

        # Metric history for gating
        self.metric_history: List[Dict[str, Any]] = []
        self.loss_history: List[float] = []
        self.gate_history: List[Dict[str, Any]] = []

        # EWMA baseline for prequential gain
        self.ewma_loss: Optional[float] = None
        self.ewma_alpha = 0.1

        # Previous JL-projected features for SW2 (avoid re-projecting each metric step)
        self._prev_H_jl: Optional[torch.Tensor] = None

        # Pending extra diagnostics to merge into the next metric snapshot (keeps _log_metrics signature stable)
        self._pending_logit_diag: Optional[Dict[str, Any]] = None

        # Training state
        self.iter_num = 0
        self.best_val_loss = float("inf")

        # Optional LR trace for spike verification (disabled by default)
        self._lr_trace: Optional[List[tuple[int, float]]] = [] if config.lr_spike_verify else None

    def _get_autocast_context(self):
        """Get appropriate autocast context."""
        cfg = self.config
        use_cuda = "cuda" in str(cfg.device) and torch.cuda.is_available()
        if use_cuda and cfg.dtype == "bfloat16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif use_cuda and cfg.dtype == "float16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        else:
            return nullcontext()

    def _init_model(self, vocab_size: Optional[int] = None) -> nn.Module:
        """Initialize GPT model (uncompiled; compilation happens in __init__ after hooks)."""
        cfg = self.config

        # Ensure nanoGPT (model.py) is importable regardless of cwd.
        # The CLI sets NANOGPT_DIR from --nanogpt_dir.
        nanogpt_dir = os.environ.get("NANOGPT_DIR") or "nanoGPT"
        _ensure_nanogpt_on_path(nanogpt_dir)

        # Import nanoGPT model components after sys.path is prepared.
        from model import GPTConfig, GPT

        # Vocab size should come from data preparation, not hardcoded.
        # nanoGPT convention is: <nanogpt_root>/data/<dataset>/meta.pkl
        if vocab_size is None:
            import pickle

            meta_path: Optional[Path] = None

            # 1) If user provides NANOGPT_DIR, prefer it.
            nanogpt_dir_env = os.environ.get("NANOGPT_DIR")
            if nanogpt_dir_env:
                cand = Path(nanogpt_dir_env) / "data" / cfg.dataset / "meta.pkl"
                if cand.exists():
                    meta_path = cand

            # 2) If running from nanoGPT root, this works.
            if meta_path is None:
                cand = Path("data") / cfg.dataset / "meta.pkl"
                if cand.exists():
                    meta_path = cand

            # 3) If running from the veriscope repo root with a ./nanoGPT checkout.
            if meta_path is None:
                cand = Path("nanoGPT") / "data" / cfg.dataset / "meta.pkl"
                if cand.exists():
                    meta_path = cand

            # 4) Search upwards from this file for a nanoGPT checkout.
            if meta_path is None:
                here = Path(__file__).resolve()
                for parent in here.parents:
                    cand = parent / "nanoGPT" / "data" / cfg.dataset / "meta.pkl"
                    if cand.exists():
                        meta_path = cand
                        break

            if meta_path is not None:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                vocab_size = int(meta["vocab_size"])
                print(f"Found vocab_size={vocab_size} from {meta_path}")
            else:
                # Fallback: GPT-2 default (but warn). This is not correct for shakespeare_char.
                vocab_size = 50304
                print(
                    f"WARNING: meta.pkl not found for dataset='{cfg.dataset}'. "
                    f"Defaulting to vocab_size={vocab_size}. "
                    f"Set NANOGPT_DIR or run from nanoGPT root to pick up data/<dataset>/meta.pkl."
                )

        model_config = GPTConfig(
            block_size=cfg.block_size,
            vocab_size=int(vocab_size),
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
            bias=False,
        )
        model = GPT(model_config)
        model = model.to(self.device)
        # NOTE: Do NOT compile here - hooks must be registered first.
        # Compilation (if enabled) happens in __init__ after GPTFeatureExtractor setup.
        return model

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize AdamW optimizer with weight decay."""
        cfg = self.config

        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        use_fused = "cuda" in str(cfg.device) and torch.cuda.is_available()
        return torch.optim.AdamW(
            optim_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            fused=use_fused,  # H100 optimization (CUDA only)
        )

    def _get_lr(self, it: int) -> float:
        """Learning rate schedule with warmup and cosine decay."""
        cfg = self.config

        # Guard against zero warmup; avoid lr=0 at it=0 by using (it+1)
        if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
            return cfg.learning_rate * float(it + 1) / float(cfg.warmup_iters)

        # Clamp at/after decay end
        if it >= cfg.lr_decay_iters:
            return cfg.min_lr

        # Guard against bad ranges
        decay_range = int(cfg.lr_decay_iters) - int(cfg.warmup_iters)
        if decay_range <= 0:
            return cfg.min_lr

        decay_ratio = (float(it) - float(cfg.warmup_iters)) / float(decay_range)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    def _effective_lr(self, it: int) -> float:
        """Compute effective LR including any configured spike injection."""
        cfg = self.config
        lr = self._get_lr(it)

        if cfg.lr_spike_at >= 0 and cfg.lr_spike_len > 0:
            if cfg.lr_spike_at <= it < (cfg.lr_spike_at + cfg.lr_spike_len):
                lr = lr * float(cfg.lr_spike_mult)

        return lr

    def _maybe_corrupt_batch(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply token corruption within a configured window.

        Modes:
          - permute: permute a fraction of positions within each sequence
          - random: replace a fraction of positions with random token IDs
          - mask: replace a fraction of positions with token 0

        Returns (X_corrupt, Y). We keep Y unchanged to induce supervised inconsistency.
        """
        cfg = self.config

        if cfg.data_corrupt_at < 0 or cfg.data_corrupt_len <= 0 or cfg.data_corrupt_frac <= 0.0:
            return X, Y

        end = int(cfg.data_corrupt_at) + int(cfg.data_corrupt_len)
        if not (int(cfg.data_corrupt_at) <= int(self.iter_num) < end):
            return X, Y

        # expected input IDs shape: [B, T]
        if X.ndim != 2:
            return X, Y

        bsz, seq_len = int(X.shape[0]), int(X.shape[1])
        n_corrupt = int(round(float(seq_len) * float(cfg.data_corrupt_frac)))
        n_corrupt = max(0, min(seq_len, n_corrupt))
        if n_corrupt == 0:
            return X, Y

        Xc = X.clone()

        # deterministic per-iteration generator
        gen = torch.Generator(device=X.device)
        gen.manual_seed(int(self.config.base_seed) + int(self.iter_num) * 31337)

        mode = str(cfg.data_corrupt_mode).lower().strip()
        if mode not in ("permute", "random", "mask"):
            raise ValueError(f"Unknown data_corrupt_mode={cfg.data_corrupt_mode!r}")

        # determine vocab_size safely
        vocab_size = 50304
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
                vocab_size = int(self.model.config.vocab_size)
        except Exception:
            pass

        for b in range(bsz):
            pos = torch.randperm(seq_len, generator=gen, device=X.device)[:n_corrupt]

            if mode == "permute":
                shuf = torch.randperm(n_corrupt, generator=gen, device=X.device)
                Xc[b, pos] = X[b, pos[shuf]]

            elif mode == "random":
                rnd = torch.randint(0, vocab_size, (n_corrupt,), generator=gen, device=X.device)
                Xc[b, pos] = rnd

            else:  # mask
                Xc[b, pos] = 0

        return Xc, Y

    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        # Handle compiled model
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod

        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iter_num": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def _compute_gate_check(self) -> Dict[str, Any]:
        """
        Perform finite-window gate check using recent metric history.

        Returns audit dict with gate decision and diagnostics.
        """
        cfg = self.config

        # gate_window is specified in *iterations*, but metric_history is recorded every metric_interval.
        # Convert to a metric-snapshot window to keep evidence density consistent.
        Wm, _, _ = compute_window_spans(cfg.gate_window, cfg.metric_interval)

        if len(self.metric_history) < 2 * Wm:
            # Not enough history to form past/recent windows => not evaluated.
            # Include iter for downstream overlap analysis (analyze_gates.py).
            reason = "not_evaluated_insufficient_history"
            audit = _normalize_gate_audit(
                {
                    "evaluated": False,
                    "policy": str(getattr(cfg, "gate_policy", "either")),
                },
                config=cfg,
                gain_bits=float("nan"),
                row_reason=reason,
                row_base_reason=reason,
                row_change_reason=reason,
            )
            decision = _derive_gate_decision(evaluated=False, ok=True, warn=False)
            gate_row = {
                "ok": True,
                "warn": False,
                "decision": decision,
                "iter": int(self.iter_num),
                "reason": reason,
                "base_reason": reason,
                "change_reason": reason,
                "audit": audit,
                # keep schema stable
                "gain_bits": float("nan"),
                "change_ok": True,
                "regime_ok": True,
                "change_ok_tri": None,
                "regime_ok_tri": None,
                "change_dw_ok": None,
                "change_gain_ok": None,
                "change_warn": False,
                "change_evaluated": False,
                "regime_warn": False,
                "regime_active": False,
                "regime_enabled": bool(getattr(self.config, "regime_enabled", False)),
                "ref_established_at": None,
                "ref_just_established": False,
                # spike attribution fields (schema stability)
                "spike_active": False,
                "spike_overlap_past": False,
                "spike_overlap_recent": False,
                "spike_any_overlap": False,
            }
            _validate_gate_row(gate_row)
            return gate_row

        # Get past and recent windows (in metric snapshots)
        recent = self.metric_history[-(2 * Wm) :]
        past_slice = recent[:Wm]
        recent_slice = recent[Wm:]

        def _slice_window_identity(slice_data: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
            iters: List[int] = []
            for d in slice_data:
                it = d.get("iter", None)
                if it is None:
                    continue
                try:
                    iters.append(int(it))
                except Exception:
                    continue
            if not iters:
                return None, None
            start_iter = int(min(iters))
            end_iter = int(max(iters))
            meta = {
                "start_iter": start_iter,
                "end_iter": end_iter,
                "n_snapshots": int(len(iters)),
            }
            return meta, f"window:{start_iter}:{end_iter}"

        ref_window_range, ref_window_id = _slice_window_identity(past_slice)
        cur_window_range, cur_window_id = _slice_window_identity(recent_slice)

        # --- Spike attribution (overlap-based, not check-iter-based) ---
        spike_active = False
        spike_overlap_past = False
        spike_overlap_recent = False
        spike_any_overlap = False

        if cfg.lr_spike_at >= 0 and cfg.lr_spike_len > 0:
            s0 = int(cfg.lr_spike_at)
            s1 = int(cfg.lr_spike_at + cfg.lr_spike_len)

            def _overlaps_spike(slice_data: List[Dict[str, Any]]) -> bool:
                for d in slice_data:
                    it = d.get("iter", None)
                    if it is None:
                        continue
                    try:
                        it_i = int(it)
                    except Exception:
                        continue
                    if s0 <= it_i < s1:
                        return True
                return False

            spike_active = s0 <= int(self.iter_num) < s1
            spike_overlap_past = _overlaps_spike(past_slice)
            spike_overlap_recent = _overlaps_spike(recent_slice)
            spike_any_overlap = spike_overlap_past or spike_overlap_recent

        # Build metric arrays
        metrics = list(self.window_decl.weights.keys())

        def _extract(slice_data: List[Dict], key: str) -> np.ndarray:
            vals = [float(d.get(key, np.nan)) for d in slice_data]
            arr = np.array(vals, dtype=float)
            return arr[np.isfinite(arr)]

        past_dict = {m: _extract(past_slice, m) for m in metrics}
        recent_dict = {m: _extract(recent_slice, m) for m in metrics}

        # Count evidence
        counts = {}
        for m in metrics:
            counts[m] = min(len(past_dict[m]), len(recent_dict[m]))

        # Compute prequential gain (bits/sample)
        recent_losses = [d.get("loss", np.nan) for d in recent_slice]
        recent_baselines = [d.get("ewma_loss", np.nan) for d in recent_slice]

        gain_vals: List[float] = []
        for loss_val, baseline_val in zip(recent_losses, recent_baselines):
            if np.isfinite(loss_val) and np.isfinite(baseline_val):
                gain_vals.append((baseline_val - loss_val) / math.log(2))  # bits

        gain_bits = float(np.mean(gain_vals)) if gain_vals else float("nan")

        # Use gate engine
        result = self.gate_engine.check(
            past=past_dict,
            recent=recent_dict,
            counts_by_metric=counts,
            gain_bits=gain_bits,
            kappa_sens=0.0,  # Skip κ_sens for now
            eps_stat_value=aggregate_epsilon_stat(self.window_decl, counts, alpha=0.05),
            iter_num=self.iter_num,  # Pass iteration for regime reference timing
            window_ref_range=ref_window_range,
            window_cur_range=cur_window_range,
            ref_window_id=ref_window_id,
            cur_window_id=cur_window_id,
        )

        audit = dict(result.audit or {})
        ok = bool(result.ok)
        warn = bool(getattr(result, "warn", False))

        # Tri-state exports: represent "not evaluated" as None.
        # Regime checks may be enabled but not run on a given check.
        regime_check_ran = bool(audit.get("regime_check_ran", False))

        # Change tri-state: prefer the nullable per-check indicators.
        change_ok_tri = audit.get("change_dw_ok", None)  # Optional[bool]

        # Regime tri-state: only meaningful if the regime check actually ran.
        regime_ok_tri = audit.get("regime_ok", None) if regime_check_ran else None

        # Harden runner even if someone swaps gate_engine without regime wrapper.
        evaluated = bool(audit.get("evaluated", True))
        row_reason = audit.get("reason", None)
        if _needs_fill(row_reason):
            row_reason = "evaluated_unknown" if evaluated else "not_evaluated"
        rr = str(row_reason)
        if evaluated and ok and warn and rr in {"none_ok", "change_ok", "regime_ok", "evaluated_unknown"}:
            if audit.get("regime_warn", False):
                row_reason = "regime_warn_pending"
            elif audit.get("change_warn", False):
                row_reason = "change_warn_pending"
            elif audit.get("warn_pending", False):
                row_reason = "warn_pending"
            else:
                row_reason = "warn"
        row_base_reason = audit.get("base_reason", None)
        if _needs_fill(row_base_reason):
            row_base_reason = row_reason

        row_change_reason = audit.get("change_reason", None)
        if _needs_fill(row_change_reason):
            row_change_reason = row_reason

        # Legacy compatibility: treat explicit None as pass/OK (matches old "default True" semantics).
        v_change_ok = audit.get("change_ok", True)
        legacy_change_ok = True if v_change_ok is None else bool(v_change_ok)

        v_regime_ok = audit.get("regime_ok", True)
        legacy_regime_ok = True if v_regime_ok is None else bool(v_regime_ok)

        audit = _normalize_gate_audit(
            audit,
            config=cfg,
            gain_bits=gain_bits,
            row_reason=row_reason,
            row_base_reason=row_base_reason,
            row_change_reason=row_change_reason,
        )
        if not getattr(self, "_regime_wrapper_enabled", False):
            audit = _strip_regime_audit_fields(audit)
        evaluated = bool(audit.get("evaluated", True))
        policy = str(audit.get("policy", "either"))
        dw_exceeds = bool(audit.get("dw_exceeds_threshold", False))
        gain_ok = bool(audit.get("gain_ok", True))

        if evaluated and policy == "either" and (not dw_exceeds) and legacy_regime_ok and (not gain_ok):
            ok = True
            warn = True
            # Downgrade gain-only FAIL -> WARN.
            # In EITHER mode the core gate may not emit a reason; runner must canonicalize.
            _r = audit.get("reason", None)
            if (_r is None) or (_r == "") or (str(_r) == "evaluated_unknown"):
                row_reason = "gain_below_threshold"
            else:
                row_reason = str(_r)
            row_base_reason = row_reason
            row_change_reason = row_reason
            audit.setdefault("reason", row_reason)
            audit["base_reason"] = row_reason
            audit["change_reason"] = row_reason

        if evaluated and not ok:
            warn = False

        # Canonicalize runner sentinel: evaluated rows should never emit "evaluated_unknown".
        if evaluated and (row_reason is None or str(row_reason) == "" or str(row_reason) == "evaluated_unknown"):
            if not ok:
                # FAIL: choose the most specific invariant we have.
                if bool(audit.get("dw_exceeds_threshold", False)):
                    row_reason = "dw_exceeds_threshold"
                elif (not bool(audit.get("gain_ok", True))) or bool(audit.get("gain_below_threshold", False)):
                    row_reason = "gain_below_threshold"
                else:
                    row_reason = "fail"
            elif warn:
                # If we don't have a specific warn reason, at least mark it as warn.
                row_reason = "warn"
            else:
                # PASS, evaluated, no warn.
                row_reason = "none_ok"

            # Only overwrite base/change reasons if they are also unknown-ish.
            if row_base_reason is None or str(row_base_reason) in ("", "evaluated_unknown"):
                row_base_reason = row_reason
            if row_change_reason is None or str(row_change_reason) in ("", "evaluated_unknown"):
                row_change_reason = row_reason

            audit.setdefault("reason", row_reason)
            audit["base_reason"] = row_base_reason
            audit["change_reason"] = row_change_reason

        audit.setdefault("reason", row_reason)
        audit["base_reason"] = row_base_reason
        audit["change_reason"] = row_change_reason

        decision = _derive_gate_decision(evaluated=evaluated, ok=ok, warn=warn)

        gate_row = {
            "ok": ok,
            "warn": warn,
            "decision": decision,
            "audit": audit,
            "gain_bits": gain_bits,
            "iter": self.iter_num,
            "reason": row_reason,
            "base_reason": row_base_reason,
            "change_reason": row_change_reason,
            # Regime-specific fields (always present for consistency)
            # Tri-state fields must remain nullable; `change_ok` is back-compat only.
            "change_ok": legacy_change_ok,
            "change_dw_ok": audit.get("change_dw_ok", None),
            "change_gain_ok": audit.get("change_gain_ok", None),
            "change_warn": audit.get("change_warn", False),
            "change_evaluated": audit.get("change_evaluated", True),
            "regime_ok": legacy_regime_ok,
            # Tri-state exports for calibration/provenance (None => not evaluated)
            "change_ok_tri": change_ok_tri,
            "regime_ok_tri": regime_ok_tri,
            "regime_warn": audit.get("regime_warn", False),
            "regime_active": audit.get("regime_active", False),
            "regime_enabled": audit.get("regime_enabled", False),
            "ref_established_at": audit.get("ref_established_at"),
            "ref_just_established": audit.get("ref_just_established", False),
            # Spike attribution
            "spike_active": spike_active,
            "spike_overlap_past": spike_overlap_past,
            "spike_overlap_recent": spike_overlap_recent,
            "spike_any_overlap": spike_any_overlap,
        }
        _validate_gate_row(gate_row)
        return gate_row

    def _compute_logit_diagnostics(self, logits: torch.Tensor) -> Dict[str, Any]:
        """
        Diagnostics computed from *training-step pre-softmax logits*.

        Assumes logits L has shape (B, T, V).
        Per token:
          - l2 norm over vocab: ||L_{b,t}||_2
          - optional std over vocab: std(L_{b,t,:})

        Aggregation is over tokens (b,t). To bound overhead deterministically,
        optionally subsample tokens using evenly spaced indices (no RNG).
        Order stats (median/p95) are computed on CPU for determinism insurance.
        """
        cfg = self.config
        out: Dict[str, Any] = {}

        with torch.no_grad():
            if logits is None or logits.ndim != 3:
                return out

            # Keep native dtype until we've subsampled; avoids a full-size float32 copy.
            L = logits.detach()
            B, T, V = L.shape
            flat = L.reshape(B * T, V)  # (N_total, V)
            N_total = int(flat.shape[0])

            # Deterministic subsample (better coverage than flat[:K], no RNG).
            K = int(getattr(cfg, "logit_norm_max_tokens", 0) or 0)
            if K > 0 and N_total > K:
                if K == 1:
                    idx = torch.tensor([N_total // 2], device=flat.device, dtype=torch.long)
                else:
                    # Evenly spaced indices in [0, N_total-1], integer arithmetic.
                    idx = (torch.arange(K, device=flat.device, dtype=torch.long) * (N_total - 1)) // (K - 1)
                flat = flat.index_select(0, idx)

            # Now cast the (N_used, V) slice only.
            flat = flat.float()

            N = int(flat.shape[0])
            out["logit_diag_tokens_total"] = int(N_total)
            out["logit_diag_tokens_used"] = int(N)
            out["logit_has_nonfinite"] = 0
            out["logit_nonfinite_count"] = 0

            if N == 0:
                return out

            # Finiteness guard on the tensor we will reduce over.
            # Log count for extra signal; sanitize so aggregates remain usable.
            nonfinite_mask = ~torch.isfinite(flat)
            n_bad = int(nonfinite_mask.sum().item())
            out["logit_nonfinite_count"] = n_bad
            if n_bad > 0:
                out["logit_has_nonfinite"] = 1
                flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-token L2 norm over vocab
            token_l2 = torch.linalg.norm(flat, dim=-1)  # (N,)
            out["logit_l2_mean"] = float(token_l2.mean().item())

            # Variance over tokens of the per-token L2 (heteroskedasticity/spikiness)
            if N > 1:
                out["logit_l2_var_token"] = float(token_l2.var(unbiased=False).item())
            else:
                out["logit_l2_var_token"] = 0.0

            # Guard expensive order stats; compute them on CPU for determinism insurance.
            order_stats_limit = int(getattr(cfg, "logit_norm_order_stats_max_tokens", 4096) or 4096)
            if N <= order_stats_limit:
                t_cpu = token_l2.detach().cpu()
                out["logit_l2_median"] = float(t_cpu.median().item())
                out["logit_l2_p95"] = float(torch.quantile(t_cpu, 0.95).item()) if N > 1 else float(t_cpu.item())
            else:
                out["logit_l2_median"] = float("nan")
                out["logit_l2_p95"] = float("nan")

            # Optional: vocab-std per token
            if bool(getattr(cfg, "log_logit_std", True)):
                token_std = flat.std(dim=-1, unbiased=False)  # (N,)
                out["logit_std_mean"] = float(token_std.mean().item())

                if N > 1:
                    out["logit_std_var_token"] = float(token_std.var(unbiased=False).item())
                else:
                    out["logit_std_var_token"] = 0.0

                if N <= order_stats_limit:
                    s_cpu = token_std.detach().cpu()
                    out["logit_std_median"] = float(s_cpu.median().item())
                    out["logit_std_p95"] = float(torch.quantile(s_cpu, 0.95).item()) if N > 1 else float(s_cpu.item())
                else:
                    out["logit_std_median"] = float("nan")
                    out["logit_std_p95"] = float("nan")

        return out

    def _log_metrics(self, loss: float, input_ids: torch.Tensor):
        """Compute and log metrics for this iteration."""
        # Update EWMA baseline
        if self.ewma_loss is None:
            self.ewma_loss = loss
        else:
            self.ewma_loss = self.ewma_alpha * loss + (1 - self.ewma_alpha) * self.ewma_loss

        # Optional: freeze metric gauge once, at the planned pre-reference point.
        # We gate freezing on extractor._ref_count, which counts metric snapshots (compute_all calls),
        # not training iters. This avoids freezing a noisy normalization estimate.
        if self._freeze_gauge_at_iter is not None and int(self.iter_num) >= int(self._freeze_gauge_at_iter):
            ref_count = int(getattr(self.extractor, "_ref_count", 0) or 0)
            if ref_count < int(self._freeze_metric_gauge_min_ref_updates):
                # Defer freezing until we have enough reference-frame updates.
                if int(self.iter_num) % int(self._freeze_defer_log_every) == 0:
                    extra = ""
                    if self._freeze_gauge_window_start_iter is not None and int(self.iter_num) >= int(
                        self._freeze_gauge_window_start_iter
                    ):
                        extra = " (WARNING: past window_start; reference may be mixed-gauge)"
                    print(
                        f"[REGIME] freeze deferred at iter={self.iter_num}: ref_updates={ref_count} < "
                        f"min_ref_updates={self._freeze_metric_gauge_min_ref_updates}{extra}"
                    )
            else:
                try:
                    self.extractor.freeze_reference_frame()
                    print(
                        f"[REGIME] Froze GPT metric gauge at iter={self.iter_num} "
                        f"(scheduled {self._freeze_gauge_at_iter}, ref_updates={ref_count})."
                    )
                except Exception as e:
                    print(f"[REGIME WARN] Failed to freeze metric gauge at iter={self.iter_num}: {e}")
                self._freeze_gauge_at_iter = None

        # Compute veriscope metrics (every N iterations to save compute)
        if self.iter_num % max(1, int(self.config.metric_interval)) == 0:
            metrics = self.metric_computer.compute_all(
                input_ids,
                run_key=self.config.base_seed,
                epoch=self.iter_num,
                prev_H_jl=self._prev_H_jl,
            )

            # Store projected features for next SW2
            self._prev_H_jl = metrics.pop("_H_jl", None)
            # Drop raw features unless you explicitly want to persist them
            metrics.pop("_H_norm", None)

            metrics["loss"] = loss
            metrics["ewma_loss"] = self.ewma_loss
            metrics["iter"] = self.iter_num
            metrics["lr"] = self._effective_lr(self.iter_num)

            # Merge diagnostic-only extra metrics (e.g., logit norms) if present
            if isinstance(self._pending_logit_diag, dict) and self._pending_logit_diag:
                metrics.update(self._pending_logit_diag)
            self._pending_logit_diag = None

            cfg = self.config
            active = (
                cfg.data_corrupt_at >= 0
                and cfg.data_corrupt_len > 0
                and cfg.data_corrupt_at <= self.iter_num < (cfg.data_corrupt_at + cfg.data_corrupt_len)
                and cfg.data_corrupt_frac > 0.0
            )
            metrics["data_corrupt_active"] = int(active)

            # One-time metric naming invariant check (first snapshot only)
            if len(self.metric_history) == 0:
                expected = set(self.window_decl.weights.keys())
                got = set(metrics.keys())
                missing = sorted(expected - got)
                if missing:
                    import warnings

                    warnings.warn(
                        f"[VERISCOPE] Missing expected metrics for gate: {missing}",
                        RuntimeWarning,
                    )

            self.metric_history.append(metrics)
            self.loss_history.append(loss)

    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Execute one training step."""
        cfg = self.config

        # Optional token corruption pathology
        X, Y = self._maybe_corrupt_batch(X, Y)

        # Update learning rate (includes optional spike injection)
        lr = self._effective_lr(self.iter_num)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Optional: trace effective LR for spike verification
        if self._lr_trace is not None:
            try:
                self._lr_trace.append((int(self.iter_num), float(lr)))
            except Exception:
                pass

        # Forward pass
        with self.ctx:
            logits, loss = self.model(X, Y)

        # Logit diagnostics (diagnostic-only): compute from training-step logits
        # only on iterations where we will also take a metric snapshot.
        if bool(cfg.log_logit_norm):
            metric_stride = max(1, int(cfg.metric_interval))
            logit_stride = int(getattr(cfg, "logit_norm_stride", 0) or 0)
            logit_stride = metric_stride if logit_stride <= 0 else max(1, logit_stride)

            if (self.iter_num % metric_stride == 0) and (self.iter_num % logit_stride == 0):
                self._pending_logit_diag = self._compute_logit_diagnostics(logits)
            else:
                self._pending_logit_diag = None
        else:
            self._pending_logit_diag = None

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping
        if cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item()

        # Log metrics
        self._log_metrics(loss_val, X)

        # Belt-and-suspenders: prevent any stale diag leaking across iterations
        self._pending_logit_diag = None

        # Gate check (after warmup)
        if cfg.gate_enabled and self.iter_num >= cfg.gate_warmup:
            if self.iter_num % cfg.gate_window == 0:
                gate_result = self._compute_gate_check()
                self.gate_history.append(gate_result)

                audit = gate_result.get("audit", {})

                # Extract policy/persistence state for logging
                policy = audit.get("policy", "either")
                consec_before = audit.get("consecutive_exceedances_before", 0)
                consec_after = audit.get("consecutive_exceedances_after", 0)
                pers_k = audit.get("persistence_k", 2)
                was_evaluated = audit.get("evaluated", True)
                decision = str(gate_result["decision"])
                reason = str(gate_result.get("reason", audit.get("reason", "")))
                gain_bits = _coerce_float(audit.get("gain_bits", gate_result.get("gain_bits", float("nan"))))
                gain_thresh = _coerce_float(audit.get("gain_thresh", audit.get("gain_thr", cfg.gate_gain_thresh)))
                worst_dw = _coerce_float(audit.get("worst_DW", float("nan")))
                eps_eff = _coerce_float(audit.get("eps_eff", float("nan")))
                metrics_exceeding = audit.get("metrics_exceeding", []) or []
                metrics_exceeding_count = len(metrics_exceeding) if isinstance(metrics_exceeding, list) else 0
                min_metrics_exceeding = _coerce_int(
                    audit.get("min_metrics_exceeding", cfg.gate_min_metrics_exceeding),
                    default=int(cfg.gate_min_metrics_exceeding),
                )

                # Format regime status
                if gate_result.get("ref_just_established"):
                    regime_status = " [REF ESTABLISHED]"
                elif gate_result.get("regime_active"):
                    regime_dw = audit.get("regime_worst_DW")
                    regime_status = f", regime_D_W={regime_dw:.4f}" if regime_dw is not None else ""
                elif gate_result.get("regime_enabled"):
                    accum = audit.get("ref_windows_accumulated", 0)
                    regime_status = f" [REF building: {accum} windows]"
                else:
                    regime_status = " [regime disabled]"

                if not gate_result["ok"]:
                    eval_tag = "" if was_evaluated else " [NOT EVALUATED]"
                    print(
                        f"\n[GATE] iter={self.iter_num} {decision.upper()}: "
                        f"reason={reason}, "
                        f"gain={gain_bits:.4f}/{gain_thresh:.4f}, "
                        f"D_W={worst_dw:.4f}/eps_eff={eps_eff:.4f}, "
                        f"exceeding={metrics_exceeding_count}/{min_metrics_exceeding}, "
                        f"change_ok={gate_result.get('change_ok')}, "
                        f"regime_ok={gate_result.get('regime_ok')}{regime_status} "
                        f"[{policy}, {consec_before}->{consec_after}/{pers_k}]{eval_tag}"
                    )

                    # Print per-metric breakdown for debugging
                    _audit = gate_result.get("audit", {}) or {}
                    per_metric = _audit.get("per_metric_tv", {})
                    if per_metric:
                        pm_str = ", ".join(f"{m}={v:.4f}" for m, v in per_metric.items() if isinstance(v, (int, float)))
                        print(f"       per_metric_tv: {pm_str}")
                    regime_pm = _audit.get("regime_per_metric", {})
                    if regime_pm and gate_result.get("regime_active"):
                        rpm_str = ", ".join(f"{m}={v:.4f}" for m, v in regime_pm.items() if isinstance(v, (int, float)))
                        print(f"       regime_per_metric: {rpm_str}")
                elif gate_result.get("warn"):
                    # Log WARN: threshold exceeded but not yet FAIL under persistence
                    print(
                        f"[GATE] iter={self.iter_num} {decision.upper()}: "
                        f"reason={reason}, "
                        f"gain={gain_bits:.4f}/{gain_thresh:.4f}, "
                        f"D_W={worst_dw:.4f}/eps_eff={eps_eff:.4f}, "
                        f"exceeding={metrics_exceeding_count}/{min_metrics_exceeding}, "
                        f"consec={consec_before}->{consec_after}/{pers_k}{regime_status}"
                    )
                    # Attribution on WARN too (top-k so logs don't explode)
                    _audit = gate_result.get("audit", {}) or {}
                    per_metric = _audit.get("per_metric_tv", {}) or {}
                    if isinstance(per_metric, dict) and per_metric:
                        items = []
                        for m, v in per_metric.items():
                            try:
                                if isinstance(v, dict) and "tv" in v:
                                    fv = float(v["tv"])
                                else:
                                    fv = float(v)
                                if np.isfinite(fv):
                                    items.append((m, fv))
                            except Exception:
                                continue
                        items.sort(key=lambda kv: kv[1], reverse=True)
                        topk = items[:5]
                        pm_str = ", ".join(f"{m}={v:.4f}" for (m, v) in topk)
                        print(f"       per_metric_tv(top5): {pm_str}")

                    wm = _audit.get("worst_metric", None)
                    wmv = _audit.get("worst_metric_tv", None)
                    if wm is not None:
                        try:
                            fwmv = float(wmv)
                            if np.isfinite(fwmv):
                                print(f"       worst_metric: {wm} ({fwmv:.4f})")
                            else:
                                print(f"       worst_metric: {wm}")
                        except Exception:
                            print(f"       worst_metric: {wm}")
                elif self.iter_num % (cfg.gate_window * 10) == 0:
                    print(
                        f"[GATE] iter={self.iter_num} {decision.upper()}: "
                        f"reason={reason}, "
                        f"gain={gain_bits:.4f}/{gain_thresh:.4f}, "
                        f"D_W={worst_dw:.4f}/eps_eff={eps_eff:.4f}, "
                        f"exceeding={metrics_exceeding_count}/{min_metrics_exceeding}{regime_status} "
                        f"[{policy}]"
                    )

        self.iter_num += 1
        return loss_val

    def train(self, get_batch_fn):
        """Main training loop."""
        cfg = self.config

        t0 = time.time()
        running_loss = 0.0

        try:
            while self.iter_num < cfg.max_iters:
                # Get batch
                X, Y = get_batch_fn("train")
                X, Y = X.to(self.device), Y.to(self.device)

                # Train step
                loss = self.train_step(X, Y)
                running_loss += loss

                # Logging
                if self.iter_num % cfg.log_interval == 0:
                    dt = time.time() - t0
                    avg_loss = running_loss / cfg.log_interval
                    print(f"iter {self.iter_num}: loss {avg_loss:.4f}, time {dt * 1000:.2f}ms")
                    running_loss = 0.0
                    t0 = time.time()

                # Evaluation
                if self.iter_num % cfg.eval_interval == 0:
                    self._evaluate(get_batch_fn)

            # Optional LR spike verification summary
            if cfg.lr_spike_verify and cfg.lr_spike_at >= 0 and cfg.lr_spike_len > 0 and self._lr_trace:
                spike_start = int(cfg.lr_spike_at)
                spike_end = int(cfg.lr_spike_at + cfg.lr_spike_len)
                in_spike = [lr for (it, lr) in self._lr_trace if spike_start <= it < spike_end]
                nearby = [
                    lr
                    for (it, lr) in self._lr_trace
                    if (abs(it - spike_start) <= 100 and not (spike_start <= it < spike_end))
                ]
                if in_spike and nearby:
                    ratio = float(np.mean(in_spike) / np.mean(nearby))
                    print(f"[SPIKE VERIFY] effective_lr ratio={ratio:.2f}x (expected {cfg.lr_spike_mult:.2f}x)")
                else:
                    print("[SPIKE VERIFY] insufficient trace samples to compute ratio")
            print("Training complete!")
            return self.metric_history, self.gate_history
        finally:
            if getattr(self, "_calibration_recorder", None) is not None:
                try:
                    self._calibration_recorder.close()
                    print(
                        f"[CAL] calibration recorder closed (records_written={self._calibration_recorder.records_written})"
                    )
                except Exception:
                    pass

    @torch.no_grad()
    def _evaluate(self, get_batch_fn):
        """Evaluate on validation set."""
        self.model.eval()
        losses = []

        for _ in range(self.config.eval_iters):
            X, Y = get_batch_fn("val")
            X, Y = X.to(self.device), Y.to(self.device)

            with self.ctx:
                _, loss = self.model(X, Y)
            losses.append(loss.item())

        val_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"val loss: {val_loss:.4f}")

        if np.isfinite(val_loss) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint("best_ckpt.pt")

        self.model.train()


# -----------------
# Data loader helper
# -----------------


def get_batch_factory(data_dir: str, block_size: int, batch_size: int, device: str):
    """Create get_batch function compatible with nanoGPT .bin data format."""
    import numpy as _np
    from pathlib import Path

    dd = Path(data_dir)
    train_data = _np.memmap(dd / "train.bin", dtype=_np.uint16, mode="r")
    val_data = _np.memmap(dd / "val.bin", dtype=_np.uint16, mode="r")

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        # need room for x of length block_size and y shifted by 1
        max_start = int(len(data)) - int(block_size) - 1
        if max_start <= 0:
            raise ValueError(f"Dataset too small for block_size={block_size}: len(data)={len(data)}")
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(_np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(_np.int64)) for i in ix])

        if "cuda" in str(device):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    return get_batch


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--nanogpt_dir", default="./nanoGPT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5000,
        help="Max training iterations (default 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed (default: 42; env VERISCOPE_SEED, CLI wins).",
    )
    parser.add_argument(
        "--metric_interval",
        type=int,
        default=None,
        help="Compute veriscope metrics every N iterations (controls evidence density).",
    )

    # Logit diagnostics (diagnostic-only; NOT gated)
    parser.add_argument(
        "--log_logit_norm",
        action="store_true",
        help="Log diagnostics from training-step pre-softmax logits (L2 norm, etc.).",
    )
    parser.add_argument(
        "--logit_norm_stride",
        type=int,
        default=0,
        help="Compute logit diagnostics every N iters (0 => metric_interval).",
    )
    parser.add_argument(
        "--logit_norm_max_tokens",
        type=int,
        default=256,
        help="Max tokens (B*T) used per snapshot (0 => all tokens; median/p95 are guarded).",
    )
    parser.add_argument(
        "--logit_norm_order_stats_max_tokens",
        type=int,
        default=4096,
        help="Compute median/p95 only when the number of tokens used is <= this threshold (computed on CPU).",
    )
    parser.add_argument(
        "--no_logit_std",
        action="store_true",
        help="Disable vocab-std diagnostics from logits.",
    )

    # Output
    parser.add_argument(
        "--out_json",
        type=str,
        default="veriscope_gpt_run.json",
        help=(
            "Output JSON path (absolute or relative). If relative, it is resolved under --out_dir "
            "when provided; otherwise relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help=(
            "Optional output directory. If set and --out_json is a relative path, the output is written to "
            "<out_dir>/<out_json>. The directory is created if missing."
        ),
    )
    parser.add_argument(
        "--save_all_metrics",
        action="store_true",
        help=(
            "If set, write all metric snapshots to the output JSON. "
            "By default only the last 100 snapshots are written to keep files small."
        ),
    )

    # Calibration recorder (optional)
    parser.add_argument(
        "--calibration_csv",
        type=str,
        default="",
        help="If set, append per-gate calibration records to this CSV path.",
    )
    parser.add_argument(
        "--calibration_include_per_metric",
        action="store_true",
        help="Include per-metric TV JSON in calibration CSV (larger files).",
    )
    parser.add_argument(
        "--calibration_only_healthy",
        action="store_true",
        help="Only record calibration rows when combined_ok=True.",
    )

    # Gate configuration
    parser.add_argument(
        "--gate_preset",
        choices=["legacy", "tuned", "tuned_v0", "spike_v1"],
        default="legacy",
        help=(
            "Gate parameter preset. 'legacy' preserves existing defaults. "
            "'tuned' applies empirically safer defaults for GPT drift, without "
            "overriding any gate_* flags you explicitly pass on the CLI. "
            "'tuned_v0' is a backward-compatible alias for 'tuned' (same defaults). "
            "'spike_v1' applies the canonical spike/corruption config (change-focused) "
            "used for the 2500–2900 token-permutation experiments."
        ),
    )
    parser.add_argument("--gate_window", type=int, default=None)
    parser.add_argument("--gate_warmup", type=int, default=None)
    parser.add_argument("--gate_epsilon", type=float, default=None)
    parser.add_argument("--gate_gain_thresh", type=float, default=None)
    parser.add_argument("--gate_min_evidence", type=int, default=None)
    parser.add_argument(
        "--gate_min_metrics_exceeding",
        type=int,
        default=None,
        help="Require >=K metrics exceeding eps_eff before stability WARN/FAIL/persistence engages (1=legacy).",
    )
    parser.add_argument(
        "--regime_min_windows",
        type=int,
        default=5,
        help="Min gate windows to accumulate before establishing regime reference (variance reduction).",
    )
    parser.add_argument(
        "--freeze_metric_gauge_min_ref_updates",
        type=int,
        default=25,
        help="Min metric snapshots used to build gauge before freezing (counts compute_all calls).",
    )
    parser.add_argument(
        "--gate_eps_stat_max_frac",
        type=float,
        default=None,
        help="Cap eps_stat as a fraction of epsilon (effective eps = epsilon - eps_stat_capped).",
    )
    parser.add_argument(
        "--gate_policy",
        type=str,
        choices=["either", "conjunction", "persistence", "persistence_stability"],
        default=None,
        help=(
            "Gate failure policy. "
            "'either'=fail on gain OR stability (original), "
            "'conjunction'=fail on gain AND stability, "
            "'persistence'=legacy persistence on stability with gain/kappa immediate veto, "
            "'persistence_stability'=persistence on stability only with kappa veto; gain is audited only."
        ),
    )
    parser.add_argument(
        "--gate_persistence_k",
        type=int,
        default=2,
        help="For persistence policy: consecutive evaluated exceedances required to FAIL.",
    )
    parser.add_argument(
        "--lr_spike_at",
        type=int,
        default=-1,
        help="Iteration to start an LR spike (>=0 enables).",
    )
    parser.add_argument(
        "--lr_spike_len",
        type=int,
        default=0,
        help="Number of iterations to apply the LR spike.",
    )
    parser.add_argument(
        "--lr_spike_mult",
        type=float,
        default=1.0,
        help="LR multiplier during the spike window.",
    )
    parser.add_argument(
        "--lr_spike_verify",
        action="store_true",
        help="Record effective LR each iteration and print a spike verification ratio (debug).",
    )

    parser.add_argument(
        "--data_corrupt_at",
        type=int,
        default=-1,
        help="Iteration to start token corruption (>=0 enables).",
    )
    parser.add_argument(
        "--data_corrupt_len",
        type=int,
        default=0,
        help="Number of iterations to apply token corruption.",
    )
    parser.add_argument(
        "--data_corrupt_frac",
        type=float,
        default=0.0,
        help="Fraction of tokens to corrupt per sequence (e.g., 0.10).",
    )
    parser.add_argument(
        "--data_corrupt_mode",
        type=str,
        default="permute",
        choices=["permute", "random", "mask"],
        help="Token corruption mode.",
    )

    # Regime detection configuration
    parser.add_argument(
        "--no_regime",
        action="store_true",
        help="Disable regime-anchored detection (use only change detection).",
    )
    parser.add_argument(
        "--regime_build_min_iter",
        type=int,
        default=-1,
        help="Earliest iteration to establish reference. -1 = auto (gate_warmup + 2*gate_window).",
    )
    parser.add_argument(
        "--regime_build_max_iter",
        type=int,
        default=-1,
        help="Latest iteration to establish reference. -1 = auto (before pathology or min+span).",
    )
    parser.add_argument(
        "--regime_build_span",
        type=int,
        default=1500,
        help="Default build window span when auto-computing max_iter.",
    )
    parser.add_argument(
        "--regime_build_max_dw",
        type=float,
        default=0.08,
        help="Max D_W for reference establishment (should be << epsilon).",
    )
    parser.add_argument(
        "--regime_build_min_gain",
        type=float,
        default=-0.01,
        help="Min gain_bits for reference establishment (learning health gate).",
    )
    parser.add_argument(
        "--regime_epsilon_mult",
        type=float,
        default=1.5,
        help="Regime epsilon = base_epsilon * this multiplier.",
    )
    parser.add_argument(
        "--regime_min_evidence",
        type=int,
        default=50,
        help="Min samples per metric before reference can be established.",
    )

    parser.add_argument(
        "--regime_build_gap_iters",
        type=int,
        default=-1,
        help=(
            "Explicit gap (iterations) between build window end and pathology start. "
            "-1 = auto (2 * window_span_iters). Set to anchor reference closer to corruption."
        ),
    )
    parser.add_argument(
        "--cos_disp_max",
        type=float,
        default=1.0,
        help=("Upper bound for cos_disp cal_range. Default 1.0 (full range). Use 0.5 to test saturation diagnostic."),
    )

    parser.add_argument(
        "--freeze_metric_gauge_on_ref",
        action="store_true",
        help="Freeze GPT feature normalization when regime reference is established.",
    )

    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    resolved_seed = _resolve_seed(args.seed, os.environ, default_seed=42)
    _seed_all(resolved_seed)

    gate_preset_effective = "tuned" if args.gate_preset == "tuned_v0" else args.gate_preset

    # ----------------------------
    # Resolve output paths up-front
    # (needed so artifacts can be emitted even if training errors)
    # ----------------------------
    out_path = Path(str(args.out_json))
    out_dir = str(args.out_dir).strip()
    if out_dir and not out_path.is_absolute():
        out_path = Path(out_dir) / out_path
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical artifacts directory: colocate with legacy out_json
    artifacts_outdir = out_path.parent

    from datetime import timezone

    started_ts_utc_dt = datetime.now(timezone.utc)
    run_status = "success"
    _train_exc: BaseException | None = None

    PRESET_KEYS = [
        "metric_interval",
        "gate_window",
        "gate_warmup",
        "gate_epsilon",
        "gate_gain_thresh",
        "gate_min_evidence",
        "gate_min_metrics_exceeding",
        "gate_eps_stat_max_frac",
        "gate_policy",
    ]

    cli_gate_overrides = {k: getattr(args, k) for k in PRESET_KEYS if getattr(args, k, None) is not None}

    legacy = {
        "metric_interval": 5,
        "gate_window": 50,
        "gate_warmup": 500,
        "gate_epsilon": 0.12,
        "gate_gain_thresh": 0.0,
        "gate_min_evidence": 16,
        "gate_min_metrics_exceeding": 1,
        "gate_eps_stat_max_frac": 0.25,
        "gate_policy": "either",
    }

    tuned = {
        "gate_window": 100,
        "gate_warmup": 1000,
        "gate_epsilon": 0.15,
        "gate_gain_thresh": -0.003,
        "gate_min_evidence": 20,
        "gate_eps_stat_max_frac": 0.15,
    }
    spike_v1 = {
        "metric_interval": 2,
        "gate_window": 75,
        "gate_warmup": 1500,
        "gate_epsilon": 0.25,
        "gate_eps_stat_max_frac": 0.15,
        "gate_min_evidence": 75,
        "gate_gain_thresh": -0.002,
        "gate_policy": "persistence_stability",
        # Optional but recommended for your “var_out_k heavy-tail nuisance WARNs” case:
        "gate_min_metrics_exceeding": 2,
    }

    # Acceptance checks:
    # - legacy preset with no gate flags resolves to values in legacy dict above.
    # - resolved_gate_cfg/run_config_resolved reflect these values.
    # - spike_v1 preset sets gate_min_metrics_exceeding=2 unless CLI overrides it.
    resolved_gate_args = dict(legacy)
    if gate_preset_effective == "tuned":
        resolved_gate_args.update(tuned)
    elif gate_preset_effective == "spike_v1":
        resolved_gate_args.update(spike_v1)

    for k in PRESET_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            resolved_gate_args[k] = v

    # Set env for meta.pkl discovery
    os.environ["NANOGPT_DIR"] = args.nanogpt_dir

    # Handle regime enabled flag
    regime_enabled = not bool(args.no_regime)

    config = TrainConfig(
        dataset=args.dataset,
        # Small model for testing
        n_layer=6,
        n_head=6,
        n_embd=384,
        batch_size=64,
        block_size=256,
        max_iters=int(args.max_iters),
        eval_interval=500,
        log_interval=10,
        device=args.device,
        # metric_interval is set below
        # Logit diagnostics
        log_logit_norm=bool(args.log_logit_norm),
        logit_norm_stride=int(args.logit_norm_stride),
        logit_norm_max_tokens=int(args.logit_norm_max_tokens),
        logit_norm_order_stats_max_tokens=int(args.logit_norm_order_stats_max_tokens),
        log_logit_std=not bool(args.no_logit_std),
        lr_spike_at=args.lr_spike_at,
        lr_spike_len=args.lr_spike_len,
        lr_spike_mult=args.lr_spike_mult,
        lr_spike_verify=bool(args.lr_spike_verify),
        data_corrupt_at=args.data_corrupt_at,
        data_corrupt_len=args.data_corrupt_len,
        data_corrupt_frac=args.data_corrupt_frac,
        data_corrupt_mode=args.data_corrupt_mode,
        # Gate config
        gate_enabled=True,
        gate_persistence_k=args.gate_persistence_k,
        # Regime config
        regime_enabled=regime_enabled,
        regime_build_min_iter=args.regime_build_min_iter,
        regime_build_max_iter=args.regime_build_max_iter,
        regime_build_span=args.regime_build_span,
        regime_build_max_dw=args.regime_build_max_dw,
        regime_build_min_gain=args.regime_build_min_gain,
        regime_epsilon_mult=args.regime_epsilon_mult,
        regime_min_evidence=args.regime_min_evidence,
        regime_min_windows=args.regime_min_windows,
        freeze_metric_gauge_min_ref_updates=args.freeze_metric_gauge_min_ref_updates,
        regime_build_gap_iters=args.regime_build_gap_iters,
        cos_disp_max=args.cos_disp_max,
        freeze_metric_gauge_on_ref=bool(args.freeze_metric_gauge_on_ref),
        # Safer default for initial hook validation
        compile=False,
        # Calibration recorder
        calibration_enabled=bool(str(args.calibration_csv).strip()),
        calibration_output_path=str(args.calibration_csv).strip(),
        calibration_buffer_size=100,
        calibration_include_per_metric=bool(args.calibration_include_per_metric),
        calibration_only_evaluated=True,
        calibration_only_healthy=bool(args.calibration_only_healthy),
        calibration_main_process_only=True,
        calibration_fail_on_schema_mismatch=True,
        base_seed=int(resolved_seed),
    )

    missing_keys = [name for name in PRESET_KEYS if name not in resolved_gate_args]
    missing_vals = [name for name in PRESET_KEYS if resolved_gate_args.get(name, None) is None]
    if missing_keys or missing_vals:
        raise RuntimeError(f"Preset resolution bug: missing gate fields: keys={missing_keys} values={missing_vals}")
    for name in PRESET_KEYS:
        setattr(config, name, resolved_gate_args[name])
    metric_interval = int(config.metric_interval)
    if metric_interval < 1:
        raise ValueError(f"metric_interval must be >= 1 (got {metric_interval})")
    gate_window = int(config.gate_window)
    if gate_window < 1:
        raise ValueError(f"gate_window must be >= 1 (got {gate_window})")
    gate_warmup = int(config.gate_warmup)
    if gate_warmup < 0:
        raise ValueError(f"gate_warmup must be >= 0 (got {gate_warmup})")
    gate_min_evidence = int(config.gate_min_evidence)
    # gate_min_evidence=0 means "no minimum evidence" and is supported.
    if gate_min_evidence < 0:
        raise ValueError(f"gate_min_evidence must be >= 0 (got {gate_min_evidence})")
    gate_min_metrics_exceeding = int(config.gate_min_metrics_exceeding)
    if gate_min_metrics_exceeding < 1:
        raise ValueError(f"gate_min_metrics_exceeding must be >= 1 (got {gate_min_metrics_exceeding})")
    gate_epsilon = float(config.gate_epsilon)
    if not math.isfinite(gate_epsilon) or gate_epsilon <= 0:
        raise ValueError(f"gate_epsilon must be finite and > 0 (got {gate_epsilon})")
    gate_gain_thresh = float(config.gate_gain_thresh)
    if not math.isfinite(gate_gain_thresh):
        raise ValueError(f"gate_gain_thresh must be finite (got {gate_gain_thresh})")
    gate_eps_stat_max_frac = float(config.gate_eps_stat_max_frac)
    if not math.isfinite(gate_eps_stat_max_frac) or gate_eps_stat_max_frac < 0 or gate_eps_stat_max_frac > 1:
        raise ValueError(f"gate_eps_stat_max_frac must be finite and in [0, 1] (got {gate_eps_stat_max_frac})")

    data_dir = os.path.join(args.nanogpt_dir, "data", args.dataset)
    get_batch = get_batch_factory(
        data_dir=data_dir,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=config.device,
    )

    trainer: VeriscopeGatedTrainer | None = None
    metrics: List[Dict[str, Any]] = []
    gates: List[Dict[str, Any]] = []
    try:
        trainer = VeriscopeGatedTrainer(config)
        metrics, gates = trainer.train(get_batch)
    except BaseException as e:
        run_status = "user_code_failure"
        _train_exc = e
        # Preserve whatever we have so far, if any
        try:
            if trainer is not None:
                gates = list(getattr(trainer, "gate_history", []) or [])
                metrics = list(getattr(trainer, "metric_history", []) or [])
        except Exception:
            pass
    finally:
        ended_ts_utc_dt = datetime.now(timezone.utc)

        # Build a resolved gate config dict for the artifact signature.
        resolved_gate_cfg: Dict[str, Any] = {
            "metric_interval": int(config.metric_interval),
            "gate_window": int(config.gate_window),
            "gate_warmup": int(config.gate_warmup),
            "gate_epsilon": float(config.gate_epsilon),
            "gate_policy": str(config.gate_policy),
            "gate_persistence_k": int(config.gate_persistence_k),
            "gate_min_evidence": int(config.gate_min_evidence),
            "gate_min_metrics_exceeding": int(config.gate_min_metrics_exceeding),
            "gate_eps_stat_max_frac": float(config.gate_eps_stat_max_frac),
            "gate_gain_thresh": float(config.gate_gain_thresh),
            "regime_enabled": bool(config.regime_enabled),
        }

        gate_history_for_emit: List[Dict[str, Any]] = []
        if trainer is not None:
            try:
                gate_history_for_emit = list(getattr(trainer, "gate_history", []) or [])
            except Exception:
                gate_history_for_emit = []
        else:
            gate_history_for_emit = list(gates or [])

        # Emit canonical artifacts (V1). This is intentionally CPU-light.
        emitted = emit_gpt_artifacts_v1(
            outdir=artifacts_outdir,
            run_id=out_path.stem,
            started_ts_utc=started_ts_utc_dt,
            ended_ts_utc=ended_ts_utc_dt,
            gate_preset=str(args.gate_preset),
            overrides=dict(cli_gate_overrides),
            resolved_gate_cfg=resolved_gate_cfg,
            metric_interval=int(config.metric_interval),
            metric_pipeline={"transport": "nanoGPT"},
            gate_history=gate_history_for_emit,
            run_status=run_status,
        )

        # --- Resolved run config artifact (canonical, stable) ---
        env_safe, env_capture = prepare_env_capture(os.environ.copy())
        safe_argv, _argv_redacted = redact_argv(list(sys.argv))

        run_cfg_obj: Dict[str, Any] = {
            "schema_version": 1,
            "run_id": str(out_path.stem),
            "run_status": str(run_status),
            "started_ts_utc": _iso_utc(started_ts_utc_dt),
            "ended_ts_utc": _iso_utc(ended_ts_utc_dt),
            "resolved_seed": int(resolved_seed),
            "out_json": str(out_path),
            "artifacts_outdir": str(artifacts_outdir),
            "argv": safe_argv,
            "gate_preset": str(args.gate_preset),
            "gate_preset_effective": gate_preset_effective,
            "cli_gate_overrides": dict(cli_gate_overrides),
            "resolved_gate_args": dict(resolved_gate_args),
            "resolved_gate_cfg": dict(resolved_gate_cfg),
            "metric_pipeline": {"transport": "nanoGPT"},
            "window_signature_ref": {
                "hash": emitted.window_signature_hash,
                "path": "window_signature.json",
            },
            "env": env_safe,
            "env_capture": env_capture,
            "provenance": {"policy_rev": POLICY_REV, "resolved_seed": int(resolved_seed)},
        }

        if _train_exc is not None:
            run_cfg_obj["error"] = {
                "type": type(_train_exc).__name__,
                "message": str(_train_exc),
            }

        if getattr(config, "data_corrupt_at", None) is not None:
            run_cfg_obj["data_corrupt_at"] = int(config.data_corrupt_at)
        if getattr(config, "data_corrupt_len", None) is not None:
            run_cfg_obj["data_corrupt_len"] = int(config.data_corrupt_len)
        if getattr(config, "data_corrupt_frac", None) is not None:
            run_cfg_obj["data_corrupt_frac"] = float(config.data_corrupt_frac)
        if getattr(config, "data_corrupt_mode", None) is not None:
            run_cfg_obj["data_corrupt_mode"] = str(config.data_corrupt_mode)

        run_cfg_path = artifacts_outdir / "run_config_resolved.json"

        # Write once to normalize serialization (datetimes -> ISO strings, etc.), then hash what is on disk.
        # The stored hash authenticates the *content without the hash field itself*.
        atomic_write_json(run_cfg_path, run_cfg_obj)
        try:
            run_cfg_on_disk = json.loads(run_cfg_path.read_text(encoding="utf-8"))
            run_cfg_on_disk.pop("run_config_resolved_hash", None)
            run_cfg_obj["run_config_resolved_hash"] = canonical_json_sha256(run_cfg_on_disk)
            atomic_write_json(run_cfg_path, run_cfg_obj)
        except Exception:
            # Non-fatal: if hashing fails for any reason, keep the base artifact.
            pass

        if _train_exc is not None:
            raise _train_exc

    # --- select a compact but relevant metric subset for JSON output ---
    def _select_metrics_for_output(
        all_metrics: List[Dict[str, Any]],
        keep_last_n: int,
        extra_ranges: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        if not all_metrics:
            return []
        keep: Set[int] = set(range(max(0, len(all_metrics) - int(keep_last_n)), len(all_metrics)))
        for i, m in enumerate(all_metrics):
            it = m.get("iter", None)
            if it is None:
                continue
            try:
                it_i = int(it)
            except Exception:
                continue
            for a, b in extra_ranges:
                if int(a) <= it_i < int(b):
                    keep.add(i)
                    break
        return [all_metrics[i] for i in sorted(keep)]

    if bool(args.save_all_metrics):
        metrics_out = metrics
    else:
        # Keep last 100 + padded ranges around reference build and configured pathologies.
        # compute_window_spans() is defined in this module.
        Wm, window_span_iters, _ = compute_window_spans(config.gate_window, config.metric_interval)
        build_min, build_max = getattr(trainer, "build_window", (None, None))
        if build_min is None or build_max is None:
            build_min, build_max = (0, 0)
        pad = 2 * int(window_span_iters)
        ranges: List[Tuple[int, int]] = []
        ranges.append((max(0, int(build_min) - pad), min(int(config.max_iters), int(build_max) + pad)))
        if int(config.data_corrupt_at) >= 0 and int(config.data_corrupt_len) > 0:
            s0 = int(config.data_corrupt_at)
            s1 = int(config.data_corrupt_at + config.data_corrupt_len)
            ranges.append((max(0, s0 - pad), min(int(config.max_iters), s1 + pad)))
        if int(config.lr_spike_at) >= 0 and int(config.lr_spike_len) > 0:
            s0 = int(config.lr_spike_at)
            s1 = int(config.lr_spike_at + config.lr_spike_len)
            ranges.append((max(0, s0 - pad), min(int(config.max_iters), s1 + pad)))
        metrics_out = _select_metrics_for_output(metrics, keep_last_n=100, extra_ranges=ranges)

    # Persist reference + config metadata for analysis tooling.
    reference_info: Dict[str, Any] = {}
    try:
        if hasattr(trainer.gate_engine, "get_reference_stats"):
            ref_stats = trainer.gate_engine.get_reference_stats() or {}
            reference_info = {
                "established_at": ref_stats.get("established_at"),
                "n_samples_per_metric": ref_stats.get("n_samples_per_metric"),
                "metrics_tracked": ref_stats.get("metrics_tracked"),
                "regime_epsilon": ref_stats.get("regime_epsilon"),
            }
    except Exception:
        reference_info = {}

    # Fallback: derive established_at from gate audit stream if not provided.
    try:
        if reference_info.get("established_at") is None:
            for g in gates or []:
                a = (g or {}).get("audit", {}) or {}
                ra = a.get("ref_established_at", None)
                if ra is not None:
                    reference_info["established_at"] = ra
                    break
    except Exception:
        pass

    # Always persist the effective build window / epsilon from the constructed engine.
    try:
        reference_info["build_window"] = list(getattr(trainer, "build_window", (None, None)))
        reference_info["regime_epsilon"] = float(getattr(trainer.gate_engine, "regime_epsilon", float("nan")))
    except Exception:
        pass

    config_snapshot = {
        "seed": {"resolved_seed": int(resolved_seed)},
        "gate": {
            "metric_interval": config.metric_interval,
            "gate_window": config.gate_window,
            "gate_warmup": config.gate_warmup,
            "gate_epsilon": config.gate_epsilon,
            "gate_policy": config.gate_policy,
            "gate_persistence_k": config.gate_persistence_k,
            "gate_min_evidence": config.gate_min_evidence,
            "gate_min_metrics_exceeding": int(config.gate_min_metrics_exceeding),
            "gate_eps_stat_max_frac": config.gate_eps_stat_max_frac,
            "gate_gain_thresh": config.gate_gain_thresh,
        },
        "regime": {
            "regime_enabled": config.regime_enabled,
            "regime_build_min_iter_requested": config.regime_build_min_iter,
            "regime_build_max_iter_requested": config.regime_build_max_iter,
            "regime_build_span": config.regime_build_span,
            "regime_build_gap_iters": config.regime_build_gap_iters,
            "regime_epsilon_mult": config.regime_epsilon_mult,
            "regime_build_max_dw": config.regime_build_max_dw,
            "regime_build_min_gain": config.regime_build_min_gain,
            "regime_min_evidence": config.regime_min_evidence,
            "freeze_metric_gauge_on_ref": config.freeze_metric_gauge_on_ref,
            "regime_min_windows": config.regime_min_windows,
            "freeze_metric_gauge_min_ref_updates": int(config.freeze_metric_gauge_min_ref_updates),
        },
        "pathology": {
            "data_corrupt_at": config.data_corrupt_at,
            "data_corrupt_len": config.data_corrupt_len,
            "data_corrupt_frac": config.data_corrupt_frac,
            "data_corrupt_mode": config.data_corrupt_mode,
            "lr_spike_at": config.lr_spike_at,
            "lr_spike_len": config.lr_spike_len,
            "lr_spike_mult": config.lr_spike_mult,
        },
        "argv": list(sys.argv),
    }

    with out_path.open("w") as f:
        json.dump(
            {
                "metrics": metrics_out,
                "gates": gates,
                "reference": reference_info,
                "config": config_snapshot,
                "resolved_seed": int(resolved_seed),
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Saved {len(metrics)} metric snapshots, {len(gates)} gate checks")
    print(
        f"Wrote {len(metrics_out)} metric snapshots to JSON ("
        f"{'all' if bool(args.save_all_metrics) else 'selected subset'})"
    )
    print(f"Wrote results JSON to: {out_path}")
