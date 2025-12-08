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
import torch.nn.functional as F

# ---- Your existing core imports (unchanged) ----
from veriscope.core.window import WindowDecl, FRWindow
from veriscope.core.transport import DeclTransport
from veriscope.core.gate import GateEngine
from veriscope.core.calibration import aggregate_epsilon_stat
from veriscope.runners.legacy.features import (
    variance_outside_k,
    cosine_dispersion,
    _std0,
)
from veriscope.runners.legacy.metrics_heavy import (
    sliced_w2_gpu_budget,
    topo_h0_jl_agg,
)


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

        # ---- Topology (your existing implementation) ----
        if self.config.compute_topo:
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
        if self.config.compute_sw2 and (prev_H_jl is not None or prev_H_norm is not None):
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
) -> WindowDecl:
    """  
    Create a WindowDecl appropriate for GPT training.

    Note: cal_ranges should be calibrated from a warmup phase.
    These are reasonable defaults for transformer hidden state metrics.
    """
    return WindowDecl(
        epsilon=epsilon,
        metrics=["var_out_k", "eff_dim", "cos_disp"],
        weights={"var_out_k": 0.4, "eff_dim": 0.4, "cos_disp": 0.2},
        bins=bins,
        interventions=(lambda x: x,),
        cal_ranges={
            "var_out_k": (0.0, 1.0),
            "eff_dim": (0.0, 200.0),  # Adjust based on model size
            "cos_disp": (0.0, 0.5),
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
