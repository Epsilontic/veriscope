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
from pathlib import Path
import time
import math
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Helper to ensure nanoGPT is importable
def _ensure_nanogpt_on_path(nanogpt_dir: str) -> None:
    """Ensure the nanoGPT checkout (containing model.py) is importable."""
    p = str(Path(nanogpt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def compute_window_spans(
    gate_window_iters: int,
    metric_interval_iters: int,
) -> tuple[int, int, int]:
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


# Your veriscope imports
from veriscope.runners.gpt.adapter import (
    GPTMetricConfig,
    GPTFeatureExtractor,
    GPTMetricComputer,
    create_gpt_window_decl,
    create_gpt_gate_engine,
)
from veriscope.core.calibration import aggregate_epsilon_stat

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

    # Veriscope gating
    gate_enabled: bool = True
    gate_window: int = 50  # iterations, not epochs
    gate_warmup: int = 1000  # don't gate until model is warmed up
    gate_epsilon: float = 0.12
    gate_gain_thresh: float = 0.0  # stability-only by default; tune upward if you want "learning+stability"
    gate_min_evidence: int = 16
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
    regime_build_gap_iters: int = -1  # explicit gap override (-1 = auto)

    # WindowDecl tuning
    cos_disp_max: float = 1.0

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False


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
        Wm, window_span_iters, _stride_iters = compute_window_spans(config.gate_window, config.metric_interval)

        # Warn if Wm is dangerously small (noisy comparisons)
        if Wm < 5:
            print(
                f"[REGIME WARN] Wm={Wm} snapshots per half-window is small. "
                f"gate_window={config.gate_window}, metric_interval={config.metric_interval}. "
                f"Consider increasing gate_window or decreasing metric_interval."
            )

        # Gap ensures neither "past" nor "recent" half-windows overlap pathology
        if config.regime_build_gap_iters >= 0:
            gap_iters = int(config.regime_build_gap_iters)
        else:
            gap_iters = 2 * window_span_iters

        # --- Compute build window with correct semantics ---
        if pathology_start is not None and pathology_start > 0:
            # CORRUPTION/SPIKE RUN: anchor reference to pre-corruption window
            build_max = max(0, int(pathology_start) - gap_iters)
            min_start = int(config.gate_warmup) + gap_iters
            build_min = max(min_start, build_max - int(config.regime_build_span))

            if build_max <= build_min:
                print(
                    f"[REGIME WARN] Build window empty: [{build_min}, {build_max}). "
                    f"pathology_start={pathology_start}, warmup={config.gate_warmup}, "
                    f"gap={gap_iters}. Disabling regime for this run."
                )
                regime_enabled = False
                build_min = -1
                build_max = -1
            else:
                regime_enabled = bool(config.regime_enabled)
                print(
                    f"[REGIME] Anchored build window to pre-corruption: "
                    f"[{build_min}, {build_max}) (gap={gap_iters} iters, Wm={Wm})"
                )
        else:
            # CONTROL RUN: anchor reference to stable post-warmup window
            build_min = int(config.gate_warmup) + gap_iters
            build_max = build_min + int(config.regime_build_span)
            max_possible = int(config.max_iters) - gap_iters
            build_max = min(build_max, max_possible)

            if build_max <= build_min:
                print(
                    f"[REGIME WARN] Build window empty for control: [{build_min}, {build_max}). "
                    f"Disabling regime for this run."
                )
                regime_enabled = False
                build_min = -1
                build_max = -1
            else:
                regime_enabled = bool(config.regime_enabled)
                print(f"[REGIME] Control run: build window [{build_min}, {build_max})")

        # Configure regime detection
        regime_config = RegimeConfig(
            enabled=regime_enabled,
            epsilon=None,  # Derive as epsilon_mult * base epsilon
            epsilon_mult=float(config.regime_epsilon_mult),
            reference_build_min_iter=int(build_min),
            reference_build_max_iter=int(build_max),
            reference_build_span=int(config.regime_build_span),
            reference_build_max_dw=float(config.regime_build_max_dw),
            reference_build_min_gain=float(config.regime_build_min_gain),
            min_evidence_per_metric=int(config.regime_min_evidence),
            eps_stat_alpha=0.05,
            eps_stat_max_frac=float(config.gate_eps_stat_max_frac),
            max_reference_samples=10000,
            max_accumulator_windows=20,
        )

        # Wrap with regime-anchored detection
        self.gate_engine = RegimeAnchoredGateEngine(
            base_engine=base_gate_engine,
            fr_win=self.fr_win,
            config=regime_config,
            gate_warmup=int(config.gate_warmup),
            gate_window=int(config.gate_window),
            pathology_start=pathology_start,
        )

        # Log computed build window and effective status
        build_min, build_max = self.gate_engine.build_window
        print(
            f"[REGIME] enabled={self.gate_engine.enabled}, "
            f"build_window=[{build_min}, {build_max}), "
            f"epsilon={self.gate_engine.regime_epsilon:.4f}"
        )

        # Hardening: independently compute build window and warn on mismatch
        try:
            bm2, bx2 = compute_build_window(
                regime_config,
                gate_warmup=int(config.gate_warmup),
                gate_window=int(config.gate_window),
                pathology_start=pathology_start,
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

        # Training state
        self.iter_num = 0
        self.best_val_loss = float("inf")

        # Optional LR trace for spike verification (disabled by default)
        self._lr_trace: Optional[List[tuple[int, float]]] = [] if config.lr_spike_verify else None

    def _get_autocast_context(self):
        """Get appropriate autocast context."""
        cfg = self.config
        if cfg.dtype == "bfloat16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif cfg.dtype == "float16":
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

        return torch.optim.AdamW(
            optim_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            fused=True,  # H100 optimization
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
        gen.manual_seed(int(self.iter_num) * 31337 + 42)

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
            return {"ok": True, "reason": "insufficient_history"}

        # Get past and recent windows (in metric snapshots)
        recent = self.metric_history[-(2 * Wm) :]
        past_slice = recent[:Wm]
        recent_slice = recent[Wm:]

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
        )

        audit = result.audit

        return {
            "ok": result.ok,
            "warn": getattr(result, "warn", False),
            "audit": audit,
            "gain_bits": gain_bits,
            "iter": self.iter_num,
            # Regime-specific fields (always present for consistency)
            "change_ok": audit.get("change_ok", True),
            "change_warn": audit.get("change_warn", False),
            "change_evaluated": audit.get("change_evaluated", True),
            "regime_ok": audit.get("regime_ok", True),
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

    def _log_metrics(self, loss: float, input_ids: torch.Tensor):
        """Compute and log metrics for this iteration."""
        # Update EWMA baseline
        if self.ewma_loss is None:
            self.ewma_loss = loss
        else:
            self.ewma_loss = self.ewma_alpha * loss + (1 - self.ewma_alpha) * self.ewma_loss

        # Compute veriscope metrics (every N iterations to save compute)
        if self.iter_num % max(1, int(self.config.metric_interval)) == 0:
            metrics = self.metric_computer.compute_all(
                input_ids,
                run_key=42,
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
                    worst_dw = audit.get("worst_DW", float("nan"))
                    eps_eff = audit.get("eps_eff", float("nan"))
                    eval_tag = "" if was_evaluated else " [NOT EVALUATED]"
                    print(
                        f"\n[GATE] iter={self.iter_num} FAIL: "
                        f"change_ok={gate_result.get('change_ok')}, "
                        f"regime_ok={gate_result.get('regime_ok')}{regime_status} "
                        f"[{policy}, {consec_before}->{consec_after}/{pers_k}]{eval_tag}"
                    )
                    print(f"       D_W={worst_dw:.4f}, eps_eff={eps_eff:.4f}")

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
                    worst_dw = audit.get("worst_DW", float("nan"))
                    eps_eff = audit.get("eps_eff", float("nan"))
                    print(
                        f"[GATE] iter={self.iter_num} WARN: "
                        f"D_W={worst_dw:.4f} > eps_eff={eps_eff:.4f}, "
                        f"consec={consec_before}->{consec_after}/{pers_k}{regime_status}"
                    )
                elif self.iter_num % (cfg.gate_window * 10) == 0:
                    print(
                        f"[GATE] iter={self.iter_num} OK, "
                        f"gain={gate_result['gain_bits']:.4f} bits{regime_status} "
                        f"[{policy}]"
                    )

        self.iter_num += 1
        return loss_val

    def train(self, get_batch_fn):
        """Main training loop."""
        cfg = self.config

        t0 = time.time()
        running_loss = 0.0

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
        "--metric_interval",
        type=int,
        default=5,
        help="Compute veriscope metrics every N iterations (controls evidence density).",
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

    # Gate configuration
    parser.add_argument(
        "--gate_preset",
        choices=["legacy", "tuned", "spike_v1"],
        default="legacy",
        help=(
            "Gate parameter preset. 'legacy' preserves existing defaults. "
            "'tuned' applies empirically safer defaults for GPT drift, without "
            "overriding any gate_* flags you explicitly pass on the CLI. "
            "'spike_v1' applies the canonical spike/corruption config (change-focused) "
            "used for the 2500–2900 token-permutation experiments."
        ),
    )
    parser.add_argument("--gate_window", type=int, default=50)
    parser.add_argument("--gate_warmup", type=int, default=500)
    parser.add_argument("--gate_epsilon", type=float, default=0.12)
    parser.add_argument("--gate_gain_thresh", type=float, default=0.0)
    parser.add_argument("--gate_min_evidence", type=int, default=16)
    parser.add_argument(
        "--gate_eps_stat_max_frac",
        type=float,
        default=0.25,
        help="Cap eps_stat as a fraction of epsilon (effective eps = epsilon - eps_stat_capped).",
    )
    parser.add_argument(
        "--gate_policy",
        type=str,
        choices=["either", "conjunction", "persistence"],
        default="either",
        help=(
            "Gate failure policy. "
            "'either'=fail on gain OR stability (original), "
            "'conjunction'=fail on gain AND stability, "
            "'persistence'=fail on K consecutive evaluated stability exceedances."
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

    args = parser.parse_args()

    def _flag_present(name: str) -> bool:
        """Return True if a flag (or flag=value) is explicitly present in argv."""
        for a in sys.argv[1:]:
            if a == name or a.startswith(name + "="):
                return True
        return False

    if args.gate_preset == "tuned":
        tuned = {
            "gate_window": 100,
            "gate_warmup": 1000,
            "gate_epsilon": 0.15,
            "gate_gain_thresh": -0.003,
            "gate_min_evidence": 20,
            "gate_eps_stat_max_frac": 0.15,
        }
        for k, v in tuned.items():
            flag = "--" + k
            if not _flag_present(flag):
                setattr(args, k, v)

    elif args.gate_preset == "spike_v1":
        # Canonical, empirically validated spike/corruption smoke config.
        # Intended for change-focused corruption detection experiments.
        spike_v1 = {
            "metric_interval": 2,
            "gate_window": 75,
            "gate_warmup": 1500,
            "gate_epsilon": 0.25,
            "gate_eps_stat_max_frac": 0.15,
            "gate_min_evidence": 75,
            "gate_gain_thresh": -0.002,
        }
        for k, v in spike_v1.items():
            flag = "--" + k
            if not _flag_present(flag):
                setattr(args, k, v)

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
        max_iters=5000,
        eval_interval=500,
        log_interval=10,
        device=args.device,
        metric_interval=args.metric_interval,
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
        gate_window=args.gate_window,
        gate_warmup=args.gate_warmup,
        gate_epsilon=args.gate_epsilon,
        gate_gain_thresh=args.gate_gain_thresh,
        gate_min_evidence=args.gate_min_evidence,
        gate_eps_stat_max_frac=args.gate_eps_stat_max_frac,
        gate_policy=args.gate_policy,
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
        regime_build_gap_iters=args.regime_build_gap_iters,
        cos_disp_max=args.cos_disp_max,
        # Safer default for initial hook validation
        compile=False,
    )

    data_dir = os.path.join(args.nanogpt_dir, "data", args.dataset)
    get_batch = get_batch_factory(
        data_dir=data_dir,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=config.device,
    )

    trainer = VeriscopeGatedTrainer(config)
    metrics, gates = trainer.train(get_batch)

    # Resolve output path
    out_path = Path(str(args.out_json))
    out_dir = str(args.out_dir).strip()
    if out_dir and not out_path.is_absolute():
        out_path = Path(out_dir) / out_path
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_out = metrics if bool(args.save_all_metrics) else metrics[-100:]

    with out_path.open("w") as f:
        json.dump(
            {
                "metrics": metrics_out,
                "gates": gates,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Saved {len(metrics)} metric snapshots, {len(gates)} gate checks")
    print(f"Wrote {len(metrics_out)} metric snapshots to JSON ({'all' if bool(args.save_all_metrics) else 'last 100'})")
    print(f"Wrote results JSON to: {out_path}")
