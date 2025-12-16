# veriscope/runners/gpt/adapter.py
"""
Thin adapter layer: nanoGPT → veriscope FR gating system.

This adapter extracts hidden states from transformer layers and feeds them
to the existing metric/gate infrastructure with minimal changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---- Your existing core imports (unchanged) ----
from veriscope.core.window import WindowDecl, FRWindow
from veriscope.core.transport import DeclTransport
from veriscope.core.gate import GateEngine

# NOTE: Do NOT import veriscope.runners.legacy.features here.
# That module pulls in veriscope.runners.legacy.model which imports torchvision.
# The GPT runner should not depend on torchvision.


def _std0(x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Std-dev with finite guard. Matches the legacy intent: never return 0."""
    # unbiased=False for stability on small samples
    s = x.float().std(dim=0, unbiased=False, keepdim=keepdim)
    # clamp away zeros
    return torch.clamp(s, min=1e-8)


@torch.no_grad()
def cosine_dispersion(H: torch.Tensor, seed: int = 0, epoch: int = 0, n_pairs: int = 2048) -> float:
    """Return mean absolute cosine similarity over random pairs.

    This is bounded in [0, 1] and is typically small for well-spread features.
    """
    if H.numel() == 0:
        return float("nan")
    X = H.float()
    n = int(X.shape[0])
    if n < 2:
        return 0.0
    # normalize
    X = X / torch.clamp(torch.linalg.norm(X, dim=1, keepdim=True), min=1e-8)
    g = torch.Generator(device=X.device)
    g.manual_seed(int(seed) + 131 * int(epoch) + 17)
    m = min(int(n_pairs), n * (n - 1) // 2)
    i = torch.randint(0, n, (m,), generator=g, device=X.device)
    j = torch.randint(0, n, (m,), generator=g, device=X.device)
    # avoid i == j with a cheap fix
    j = (j + (i == j).to(j.dtype)) % n
    cos = (X[i] * X[j]).sum(dim=1)
    return float(torch.mean(torch.abs(cos)).item())


@torch.no_grad()
def variance_outside_k(H: torch.Tensor, k: Optional[int] = None) -> Tuple[float, float, int, float, int, int]:
    """Geometry summary metrics.

    Returns:
      (var_out_k, eff_dim, k_used, tail_mass, v_ok, neg_eigs)

    This is a lightweight replacement for the legacy implementation and is
    designed to be dependency-minimal (no torchvision).
    """
    X = H.float()
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 2 or d < 1:
        return float("nan"), float("nan"), 0, float("nan"), 0, 0

    X = X - X.mean(dim=0, keepdim=True)
    # Use SVD (stable) to obtain spectrum of covariance
    # cov eigenvalues are (s^2)/(n-1)
    try:
        # full_matrices=False keeps it efficient
        s = torch.linalg.svdvals(X)
    except Exception:
        s = torch.linalg.svdvals(X.cpu())
    eig = (s * s) / max(1.0, float(n - 1))

    total = float(eig.sum().item())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), float("nan"), 0, float("nan"), 0, 0

    # Effective dimension: (sum λ)^2 / sum λ^2
    denom = float((eig * eig).sum().item())
    eff_dim = (total * total) / denom if denom > 0.0 else float("nan")

    # Choose k if not provided: sqrt(d) clipped to [1, d]
    if k is None:
        k_used = int(max(1, min(d, round(math.sqrt(d)))))
    else:
        k_used = int(max(1, min(d, int(k))))

    # Sort descending
    eig_sorted, _ = torch.sort(eig, descending=True)
    top = float(eig_sorted[:k_used].sum().item())
    tail = max(0.0, total - top)
    var_out_k = float(tail / total)

    # tail_mass is kept for compatibility; here equal to variance fraction outside k
    tail_mass = var_out_k
    v_ok = 1
    neg_eigs = 0
    return var_out_k, float(eff_dim), int(k_used), float(tail_mass), int(v_ok), int(neg_eigs)


# Heavy metrics are optional; import lazily and fail closed if unavailable.
try:
    from veriscope.runners.legacy.metrics_heavy import (
        sliced_w2_gpu_budget,
        topo_h0_jl_agg,
    )
except Exception:
    sliced_w2_gpu_budget = None  # type: ignore[assignment]
    topo_h0_jl_agg = None  # type: ignore[assignment]


@dataclass
class GPTMetricConfig:
    """Configuration for GPT metric extraction."""

    # Which transformer layers to probe (indices or 'last', 'middle', 'all')
    probe_layers: List[int] | str = "last"
    # Max tokens to sample per batch for metric computation
    max_tokens_per_batch: int = 2048
    # JL projection dimension for geometry metrics
    geom_rp_dim: int = 64
    # Subsample for expensive metrics
    topo_sample_n: int = 256
    # Reference frame source: 'running' or 'fixed_init'
    ref_frame: str = "running"
    # Optional toggles (useful when heavy deps are slow/unavailable)
    compute_topo: bool = True
    compute_sw2: bool = True


class GPTFeatureExtractor:
    """
    Extract hidden states from GPT models for metric computation.

    Works with:
    - nanoGPT (model.transformer.h[i] layers)
    - HuggingFace GPT2 (model.transformer.h[i])
    - Any model with .transformer.h attribute
    """

    def __init__(
        self,
        model: nn.Module,
        config: GPTMetricConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device
        self._hooks = []
        self._activations: Dict[int, torch.Tensor] = {}

        # Reference frame (μ, σ) for normalization
        self._ref_mu: Optional[torch.Tensor] = None
        self._ref_sig: Optional[torch.Tensor] = None
        self._ref_count: int = 0

        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks on target layers."""
        # Clear existing hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

        # Get transformer blocks
        if hasattr(self.model, "transformer"):
            blocks = self.model.transformer.h
        elif hasattr(self.model, "blocks"):
            blocks = self.model.blocks
        else:
            raise ValueError("Cannot find transformer blocks in model")

        n_layers = len(blocks)

        # Determine which layers to probe
        if self.config.probe_layers == "last":
            layer_indices = [n_layers - 1]
        elif self.config.probe_layers == "middle":
            layer_indices = [n_layers // 2]
        elif self.config.probe_layers == "all":
            layer_indices = list(range(n_layers))
        else:
            layer_indices = self.config.probe_layers

        for idx in layer_indices:

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output shape: (batch, seq_len, hidden_dim)
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    self._activations[layer_idx] = h.detach()

                return hook

            self._hooks.append(blocks[idx].register_forward_hook(make_hook(idx)))

    @torch.no_grad()
    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract and flatten hidden states from probed layers.

        Returns: (N, D) tensor where N = batch * seq positions, D = hidden_dim
        """
        self._activations.clear()

        # Save and restore training mode (metric extraction must not change training semantics)
        was_training = bool(self.model.training)
        self.model.eval()

        try:
            # Forward pass to populate activations
            _ = self.model(input_ids)

            if not self._activations:
                raise RuntimeError("No activations captured - check hook setup")

            # Concatenate across probed layers and flatten
            # Shape: (batch, seq, hidden) → (batch * seq, hidden)
            all_h = []
            for idx in sorted(self._activations.keys()):
                h = self._activations[idx]  # (B, T, D)
                B, T, D = h.shape

                # Optional: subsample tokens (ensure indices are on the same device as h)
                if T > self.config.max_tokens_per_batch:
                    idx_sample = torch.randperm(T, device=h.device)[: self.config.max_tokens_per_batch]
                    h = h[:, idx_sample, :]

                # Flatten to (N, D)
                h_flat = h.reshape(-1, D)
                all_h.append(h_flat)

            # If multiple layers, concatenate along feature dim
            H = torch.cat(all_h, dim=-1) if len(all_h) > 1 else all_h[0]

            return H.float().cpu()
        finally:
            # Restore original mode
            if was_training:
                self.model.train()

    def update_reference_frame(self, H: torch.Tensor, ema_alpha: float = 0.1):
        """Update running reference (μ, σ) for normalization."""
        mu = H.mean(dim=0, keepdim=True)
        sig = _std0(H, keepdim=True) + 1e-8

        if self._ref_mu is None:
            self._ref_mu = mu
            self._ref_sig = sig
        else:
            self._ref_mu = ema_alpha * mu + (1 - ema_alpha) * self._ref_mu
            self._ref_sig = ema_alpha * sig + (1 - ema_alpha) * self._ref_sig

        self._ref_count += 1

    def normalize(self, H: torch.Tensor) -> torch.Tensor:
        """Normalize features using reference frame."""
        if self._ref_mu is None:
            return H
        return (H - self._ref_mu) / self._ref_sig

    def cleanup(self):
        """Remove hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


class GPTMetricComputer:
    """
    Compute veriscope metrics from GPT hidden states.

    Adapts the existing metric functions to work with transformer activations.
    """

    def __init__(
        self,
        extractor: GPTFeatureExtractor,
        config: GPTMetricConfig,
        device: torch.device,
    ):
        self.extractor = extractor
        self.config = config
        self.device = device

        # JL projection cache
        self._jl_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_jl_matrix(self, d_in: int, d_out: int, seed: int = 42) -> torch.Tensor:
        """Get or create JL projection matrix."""
        key = (d_in, d_out)
        if key not in self._jl_cache:
            g = torch.Generator().manual_seed(seed + d_in + 31 * d_out)
            A = torch.randn(d_in, d_out, generator=g) / math.sqrt(d_out)
            self._jl_cache[key] = A
        return self._jl_cache[key]

    @torch.no_grad()
    def compute_all(
        self,
        input_ids: torch.Tensor,
        run_key: int,
        epoch: int,
        prev_H_jl: Optional[torch.Tensor] = None,
        prev_H_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute all metrics for one step.

        Returns dict with:
            - var_out_k, eff_dim, cos_disp (geometry)
            - pers_H0 (topology)
            - sw2 (distribution shift, if prev_H_jl or prev_H_norm provided)
            - raw features for gating
        """
        # Extract hidden states
        H_raw = self.extractor.extract_features(input_ids)

        # Update and apply normalization
        self.extractor.update_reference_frame(H_raw)
        H_norm = self.extractor.normalize(H_raw)

        # JL projection for geometry metrics
        d_in = H_norm.shape[1]
        d_out = min(self.config.geom_rp_dim, d_in)
        A = self._get_jl_matrix(d_in, d_out)
        H_jl = H_norm @ A

        metrics: Dict[str, Any] = {}

        # ---- Geometry metrics (reuse your existing functions) ----
        try:
            var_out, eff_dim, k_used, tail_mass, v_ok, neg_eigs = variance_outside_k(H_jl)
            metrics.update(
                {
                    "var_out_k": float(var_out),
                    "eff_dim": float(eff_dim),
                    "k_used": int(k_used),
                    "var_out_k_valid": int(v_ok),
                }
            )
        except Exception:
            metrics.update(
                {
                    "var_out_k": float("nan"),
                    "eff_dim": float("nan"),
                    "k_used": 0,
                    "var_out_k_valid": 0,
                }
            )

        # Cosine dispersion
        try:
            cos = cosine_dispersion(H_jl, seed=run_key, epoch=epoch)
            metrics["cos_disp"] = float(cos)
        except Exception:
            metrics["cos_disp"] = float("nan")

        # ---- Topology (optional) ----
        if self.config.compute_topo and topo_h0_jl_agg is not None:
            try:
                pers_h0, topo_done, topo_ms, topo_n_used = topo_h0_jl_agg(
                    H_jl,
                    q=16,  # JL dim for topo
                    repeats=4,
                    run_key=run_key,
                    epoch=epoch,
                    agg="median",
                    sample_n=self.config.topo_sample_n,
                )
                metrics.update(
                    {
                        "pers_H0": float(pers_h0),
                        "topo_done": int(topo_done),
                        "topo_ms": float(topo_ms),
                    }
                )
            except Exception:
                metrics.update(
                    {
                        "pers_H0": float("nan"),
                        "topo_done": 0,
                        "topo_ms": 0.0,
                    }
                )
        else:
            metrics.update(
                {
                    "pers_H0": float("nan"),
                    "topo_done": 0,
                    "topo_ms": 0.0,
                }
            )

        # ---- Distribution shift (SW2) ----
        if (
            self.config.compute_sw2
            and sliced_w2_gpu_budget is not None
            and (prev_H_jl is not None or prev_H_norm is not None)
        ):
            try:
                prev_for_sw2 = prev_H_jl if prev_H_jl is not None else (prev_H_norm @ A)
                sw2, sw2_ms, n_proj, ok = sliced_w2_gpu_budget(
                    prev_for_sw2,
                    H_jl,
                    n_proj=64,
                    seed=run_key + epoch,
                    device=self.device,
                    budget_ms=200,
                )
                metrics.update(
                    {
                        "sw2": float(sw2) if ok else float("nan"),
                        "sw2_ms": float(sw2_ms),
                        "sw2_valid": int(ok),
                    }
                )
            except Exception:
                metrics.update(
                    {
                        "sw2": float("nan"),
                        "sw2_ms": 0.0,
                        "sw2_valid": 0,
                    }
                )
        else:
            metrics.update(
                {
                    "sw2": float("nan"),
                    "sw2_ms": 0.0,
                    "sw2_valid": 0,
                }
            )

        # Return features for next iteration / caller reuse
        metrics["_H_norm"] = H_norm
        metrics["_H_jl"] = H_jl

        return metrics


def create_gpt_window_decl(
    epsilon: float = 0.12,
    bins: int = 16,
    eff_dim_max: float = 64.0,
    cos_disp_max: float = 1.0,  # ← FIXED: was 0.5, caused transport saturation
) -> WindowDecl:
    """Create a WindowDecl appropriate for GPT training.

    IMPORTANT:
    - `eff_dim` is computed on the JL-projected representation (H_jl),
      so eff_dim_max should match the projection dimension (default 64).
    - `cos_disp` is bounded in [0, 1] by definition. The cal_range must
      cover the full range to avoid transport saturation artifacts.
    - `var_out_k` is bounded in [0, 1] by construction.

    The weights are intentionally set to downweight cos_disp (0.2) relative
    to var_out_k and eff_dim (0.4 each), since cos_disp can drift during
    healthy learning.
    """
    eff_dim_max = float(max(4.0, eff_dim_max))
    cos_disp_max = float(max(0.1, min(1.0, cos_disp_max)))

    # One-time warning for tight cos_disp range (common misconfiguration)
    if cos_disp_max < 0.9:
        try:
            import multiprocessing as mp

            if mp.current_process().name == "MainProcess":
                print(f"[WARN] cos_disp_max={cos_disp_max:.2f} is tight; consider 1.0 to avoid transport saturation")
        except Exception:
            pass

    return WindowDecl(
        epsilon=float(epsilon),
        metrics=["var_out_k", "eff_dim", "cos_disp"],
        weights={"var_out_k": 0.4, "eff_dim": 0.4, "cos_disp": 0.2},
        bins=int(bins),
        interventions=(lambda x: x,),
        cal_ranges={
            "var_out_k": (0.0, 1.0),
            "eff_dim": (0.0, eff_dim_max),
            "cos_disp": (0.0, cos_disp_max),  # ← THE FIX: full [0,1] range
        },
    )


def create_gpt_gate_engine(
    window_decl: WindowDecl,
    cfg: Dict[str, Any],
) -> Tuple[FRWindow, GateEngine]:
    """
    Create FR window and gate engine for GPT training.

    Reuses your existing core infrastructure.
    """
    # Attach transport
    transport = DeclTransport(window_decl)
    window_decl.attach_transport(transport)

    fr_win = FRWindow(
        decl=window_decl,
        transport=transport,
        tests=(),
    )

    ge = GateEngine(
        frwin=fr_win,
        gain_thresh=float(cfg.get("gate_gain_thresh", 0.05)),
        eps_stat_alpha=float(cfg.get("gate_eps_stat_alpha", 0.05)),
        eps_stat_max_frac=float(cfg.get("gate_eps_stat_max_frac", 0.25)),
        eps_sens=float(cfg.get("gate_epsilon_sens", 0.04)),
        min_evidence=int(cfg.get("gate_min_evidence", 16)),
    )

    return fr_win, ge
