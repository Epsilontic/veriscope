# veriscope/runners/gpt/adapter.py
"""
Thin adapter layer: nanoGPT → veriscope FR gating system.

This adapter extracts hidden states from transformer layers and feeds them
to the existing metric/gate infrastructure with minimal changes.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl

_RANKME_SANITY_WARNED: bool = False

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
    # Portable determinism: sample indices on CPU, then move to X.device.
    # (Avoid device-dependent RNG divergences across CUDA/MPS/CPU.)
    g = torch.Generator()  # CPU generator
    g.manual_seed(int(seed) + 131 * int(epoch) + 17)
    m = min(int(n_pairs), n * (n - 1) // 2)
    i = torch.randint(0, n, (m,), generator=g).to(X.device)
    j = torch.randint(0, n, (m,), generator=g).to(X.device)
    # avoid i == j with a cheap fix
    j = (j + (i == j).to(j.dtype)) % n
    cos = (X[i] * X[j]).sum(dim=1)
    return float(torch.mean(torch.abs(cos)).item())


# Helper: covariance eigenspectrum from activations
@torch.no_grad()
def _cov_eigs_sorted_from_activations(
    H: torch.Tensor,
    method: str = "auto",
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Compute covariance-spectrum eigenvalues from centered activations H (N,D).

    Returns:
      (eig_sorted_desc, total_variance)

    Notes:
      - For JL dims (d ~ 64), forming covariance and using eigvalsh is often faster
        than SVD when N is large.
      - method:
          * "auto": use covariance eigvalsh for d <= 256, else SVD
          * "cov":  force covariance eigvalsh
          * "svd":  force SVD->(s^2)/(n-1)
    """
    X = H.float()
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 2 or d < 1:
        return None, float("nan")

    X = X - X.mean(dim=0, keepdim=True)

    m = str(method).lower().strip()
    if m == "cov":
        use_cov = True
    elif m == "svd":
        use_cov = False
    else:
        use_cov = d <= 256

    try:
        if use_cov:
            C = (X.T @ X) / max(1.0, float(n - 1))
            # Numerical hygiene: enforce symmetry to reduce spurious negative eigenvalues.
            C = 0.5 * (C + C.T)
            w = torch.linalg.eigvalsh(C)  # ascending
            w = torch.clamp(w, min=0.0)
            eig_sorted = torch.flip(w, dims=[0])  # descending
        else:
            # SVD fallback: eig_i = s_i^2/(n-1)
            try:
                s = torch.linalg.svdvals(X)
            except Exception:
                s = torch.linalg.svdvals(X.cpu())
            eig = (s * s) / max(1.0, float(n - 1))
            eig_sorted, _ = torch.sort(eig, descending=True)

        # Guard before sum() to keep empty-spectrum semantics robust.
        if eig_sorted is None or eig_sorted.numel() == 0:
            return None, float("nan")
        total = float(eig_sorted.sum().item())
        if (not np.isfinite(total)) or total <= 0.0:
            return None, float("nan")
        return eig_sorted, total
    except Exception:
        return None, float("nan")


@torch.no_grad()
def rankme_cov_from_eigs(
    eig_sorted: Optional[torch.Tensor],
    total: float,
    eps: float = 1e-12,
) -> float:
    """
    RankMe / effective rank via spectral entropy of the covariance spectrum.

      p_i = λ_i / Σλ_i
      rankme = exp( - Σ p_i log(p_i) )

    This is the covariance-spectrum variant, matching eff_dim semantics here.
    """
    if eig_sorted is None or eig_sorted.numel() == 0:
        return float("nan")

    tot = float(total)
    if (not np.isfinite(tot)) or tot <= 0.0:
        return float("nan")

    p = eig_sorted / tot
    p = torch.clamp(p, min=float(eps))
    p = p / torch.clamp(p.sum(), min=float(eps))  # renormalize after clamp

    ent = -torch.sum(p * torch.log(p))
    r = torch.exp(ent)
    return float(r.item())


@torch.no_grad()
def _geom_metrics_from_eigs(
    eig_sorted: Optional[torch.Tensor],
    total: float,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Derive geometry scalar metrics from covariance eigenvalues.

    Returns dict containing:
      - var_out_k: fraction of variance outside top-k
      - eff_dim: participation ratio (sum λ)^2 / sum λ^2  (denom-clamped)
      - rankme: exp(entropy(p)), p = λ / sum λ
      - k_used: k actually used
      - var_out_k_valid: 1 if computed, else 0
    """
    if eig_sorted is None or eig_sorted.numel() == 0:
        return {
            "var_out_k": float("nan"),
            "eff_dim": float("nan"),
            "rankme": float("nan"),
            "k_used": 0,
            "var_out_k_valid": 0,
        }

    tot = float(total)
    if (not np.isfinite(tot)) or tot <= 0.0:
        return {
            "var_out_k": float("nan"),
            "eff_dim": float("nan"),
            "rankme": float("nan"),
            "k_used": 0,
            "var_out_k_valid": 0,
        }

    # eff_dim (participation ratio) with denom clamp to avoid numeric spikes without
    # turning rare issues into NaNs (important for consensus sensitivity).
    denom = float((eig_sorted * eig_sorted).sum().item())
    denom_floor = 1e-20  # float32-scale guard
    denom = denom if denom > denom_floor else denom_floor
    eff_dim = (tot * tot) / denom

    # k selection: sqrt(#eigs) is stable for JL dims; clamp to [1, #eigs]
    d_eigs = int(eig_sorted.numel())
    if k is None:
        k_used = int(max(1, min(d_eigs, round(math.sqrt(d_eigs)))))
    else:
        k_used = int(max(1, min(d_eigs, int(k))))

    top = float(eig_sorted[:k_used].sum().item())
    tail = max(0.0, tot - top)
    var_out_k = float(tail / tot)

    rme = rankme_cov_from_eigs(eig_sorted, tot)

    return {
        "var_out_k": float(var_out_k),
        "eff_dim": float(eff_dim),
        "rankme": float(rme),
        "k_used": int(k_used),
        "var_out_k_valid": 1,
    }


@torch.no_grad()
def variance_outside_k(H: torch.Tensor, k: Optional[int] = None) -> Tuple[float, float, int, float, int, int]:
    """
    Geometry summary metrics (covariance spectrum).

    Returns:
      (var_out_k, eff_dim, k_used, tail_frac, v_ok, neg_eigs)

    NOTE:
      - tail_frac is the FRACTION of variance outside top-k (same as var_out_k).
      - neg_eigs is always 0 here because covariance eigs are clamped nonnegative.
    """
    eig_sorted, total = _cov_eigs_sorted_from_activations(H, method="auto")
    out = _geom_metrics_from_eigs(eig_sorted, total, k=k)

    var_out_k = float(out["var_out_k"])
    eff_dim = float(out["eff_dim"])
    k_used = int(out["k_used"])
    v_ok = int(out["var_out_k_valid"])

    tail_frac = var_out_k
    neg_eigs = 0
    return var_out_k, eff_dim, k_used, float(tail_frac), v_ok, neg_eigs


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
    probe_layers: Union[List[int], str] = "last"
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
        self._ref_frozen: bool = False

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
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract and flatten hidden states from probed layers.

        Returns: (N, D) tensor where N = batch * seq positions, D = hidden_dim
        """
        # Kept for API compatibility (e.g., HF-style callers); intentionally unused.
        del attention_mask
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
                    K = int(self.config.max_tokens_per_batch)
                    if seed is None:
                        idx_sample = torch.randperm(T, device=h.device)[:K]
                    else:
                        # Portable determinism: CPU generator -> CPU randperm -> move indices to device.
                        # Mix layer index so different probed layers don't share the same subsample.
                        seed_eff = (int(seed) + 1009 * int(idx) + 17) % (2**31)
                        g = torch.Generator()  # CPU generator (no device=; portable across torch versions)
                        g.manual_seed(int(seed_eff))
                        idx_cpu = torch.randperm(T, generator=g)[:K]  # CPU by default
                        idx_sample = idx_cpu.to(h.device)
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
        if self._ref_frozen:
            return
        mu = H.mean(dim=0, keepdim=True)
        sig = _std0(H, keepdim=True) + 1e-8

        if self._ref_mu is None:
            self._ref_mu = mu
            self._ref_sig = sig
        else:
            self._ref_mu = ema_alpha * mu + (1 - ema_alpha) * self._ref_mu
            self._ref_sig = ema_alpha * sig + (1 - ema_alpha) * self._ref_sig

        self._ref_count += 1

    def freeze_reference_frame(self) -> None:
        """Freeze normalization statistics (anchored regime semantics)."""
        self._ref_frozen = True

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
        self._jl_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _get_jl_matrix(self, d_in: int, d_out: int, seed: int = 42) -> torch.Tensor:
        """Get or create JL projection matrix."""
        key = (d_in, d_out, int(seed))
        if key not in self._jl_cache:
            g = torch.Generator().manual_seed(seed + d_in + 31 * d_out)
            A = torch.randn(
                d_in,
                d_out,
                generator=g,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ) / math.sqrt(d_out)
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
        H_raw = self.extractor.extract_features(input_ids, seed=int(run_key))

        # Prequential semantics: normalize using the *previous* reference frame,
        # then update the reference from raw features for the *next* step.
        H_norm = self.extractor.normalize(H_raw)
        self.extractor.update_reference_frame(H_raw)

        # JL projection for geometry metrics
        d_in = H_norm.shape[1]
        d_out = min(self.config.geom_rp_dim, d_in)
        A = self._get_jl_matrix(d_in, d_out)
        H_jl = H_norm @ A

        metrics: Dict[str, Any] = {}

        # ---- Geometry metrics (single spectrum; derived metrics are centralized) ----
        try:
            eig_sorted, total = _cov_eigs_sorted_from_activations(H_jl, method="auto")
            geom = _geom_metrics_from_eigs(eig_sorted, total, k=None)

            metrics.update(
                {
                    "var_out_k": float(geom["var_out_k"]),
                    "eff_dim": float(geom["eff_dim"]),
                    "rankme": float(geom["rankme"]),
                    "k_used": int(geom["k_used"]),
                    "var_out_k_valid": int(geom["var_out_k_valid"]),
                }
            )

            # Dev-only sanity check (rate-limited): rankme should be finite if eff_dim is finite
            if os.environ.get("VERISCOPE_RANKME_SANITY", "0").strip().lower() in {"1", "true", "yes", "on"}:
                global _RANKME_SANITY_WARNED
                if not _RANKME_SANITY_WARNED:
                    ed = metrics["eff_dim"]
                    rm = metrics["rankme"]
                    if np.isfinite(ed) and (not np.isfinite(rm)):
                        _RANKME_SANITY_WARNED = True
                        warnings.warn(
                            f"[rankme_sanity] eff_dim finite but rankme non-finite. "
                            f"shape(H_jl)={tuple(H_jl.shape)}, total={float(total):.6g}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

        except Exception:
            metrics.update(
                {
                    "var_out_k": float("nan"),
                    "eff_dim": float("nan"),
                    "rankme": float("nan"),
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
        # Policy 1: metrics are CPU. This requires H_norm/H_jl/A (and prev_* tensors) to be CPU.
        # Do not partially migrate to GPU without switching fully to Policy 2.
        if (
            self.config.compute_sw2
            and sliced_w2_gpu_budget is not None
            and (prev_H_jl is not None or prev_H_norm is not None)
        ):
            try:
                prev_for_sw2 = prev_H_jl if prev_H_jl is not None else (prev_H_norm @ A)
                if prev_for_sw2.ndim != 2 or H_jl.ndim != 2 or prev_for_sw2.shape[1] != H_jl.shape[1]:
                    raise RuntimeError(f"SW2 dim mismatch: prev={tuple(prev_for_sw2.shape)} cur={tuple(H_jl.shape)}")

                # Low-cost hardening: ensure expected dtype/layout for SW2 implementations.
                prev_for_sw2 = prev_for_sw2.contiguous().float()
                cur_for_sw2 = H_jl.contiguous().float()

                sw2, sw2_ms, n_proj, ok = sliced_w2_gpu_budget(
                    prev_for_sw2,
                    cur_for_sw2,
                    n_proj=64,
                    seed=run_key + epoch,
                    device=torch.device("cpu"),
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
    rankme_max: Optional[float] = None,
    cos_disp_max: float = 1.0,
) -> WindowDecl:
    """
    Create a WindowDecl appropriate for GPT training.

    Notes on calibration bounds:
      - eff_dim is computed on JL-projected representation (H_jl), so eff_dim_max
        should typically be geom_rp_dim.
      - rankme (cov-spectrum) is in [1, min(d, n-1)] for centered data; calibrate
        on [1, rankme_max] for better histogram resolution.
      - cos_disp is bounded in [0, 1] by definition.
    """
    eff_dim_max = float(max(4.0, eff_dim_max))
    if rankme_max is None:
        rankme_max = eff_dim_max
    rankme_max = float(max(2.0, rankme_max))

    cos_disp_max = float(max(0.1, min(1.0, cos_disp_max)))

    return WindowDecl(
        epsilon=float(epsilon),
        metrics=["var_out_k", "eff_dim", "rankme", "cos_disp"],
        weights={
            "var_out_k": 0.35,
            "eff_dim": 0.35,
            "rankme": 0.20,
            "cos_disp": 0.10,
        },
        bins=int(bins),
        interventions=(lambda x: x,),
        cal_ranges={
            "var_out_k": (0.0, 1.0),
            "eff_dim": (0.0, eff_dim_max),
            "rankme": (1.0, rankme_max),
            "cos_disp": (0.0, cos_disp_max),
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
        policy=str(cfg.get("gate_policy", "either")),
        persistence_k=int(cfg.get("gate_persistence_k", 2)),
        min_metrics_exceeding=int(cfg.get("gate_min_metrics_exceeding", 1)),
    )

    return fr_win, ge
