# veriscope/runners/legacy/features.py
"""
Feature extraction and JL cache for legacy CIFAR/STL runners.

Pulled from runners/legacy_cli_refactor.py with identical behavior:
- _std0: dim-0 std with cross-version safety.
- JLCache + _JL: LRU cache for JL random projections.
- extract_features: the old _features_for_loader, but with cfg passed explicitly.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Optional, Tuple

import math
import torch
from torch import nn
from torch.utils.data import DataLoader

from veriscope.runners.legacy.model import penult

import os

import numpy as np
import torch.nn.functional as F

from veriscope.runners.legacy.data import _u01_from_hash

try:
    from veriscope.runners.legacy import runtime as _rt  # type: ignore[import]
    CFG = getattr(_rt, "CFG", {}) or {}
except Exception:
    CFG: dict = {}


def _std0(X: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """dim-0 std with cross-version safety."""
    try:
        # Newer PyTorch prefers 'correction'
        return X.std(dim=0, keepdim=keepdim, correction=0)
    except TypeError:
        # Older PyTorch uses 'unbiased'
        return X.std(0, keepdim=keepdim, unbiased=False)


class JLCache:
    """Simple LRU cache for JL projection matrices (CPU, float32)."""

    def __init__(self, capacity: int = 64):
        self.cap = int(capacity)
        self.store: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    def get(
        self,
        d_in: int,
        q: int,
        run_key: int,
        epoch: int,
        fixed: bool,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        q = int(min(q, d_in))
        # Cache on logical keys only (CPU, float32); move/cast on return to avoid fragmentation
        key = (int(d_in), int(q), int(run_key) if fixed else (int(run_key), int(epoch)))
        if key in self.store:
            self.store.move_to_end(key)
            A_cpu = self.store[key]
        else:
            base_seed = 1_234_567 + int(d_in) * 13 + int(q) * 29 + int(run_key) * 7 + (0 if fixed else int(epoch) * 17)
            g = torch.Generator(device="cpu").manual_seed(base_seed)
            A_cpu = torch.randn(d_in, q, generator=g, device="cpu", dtype=torch.float32) / math.sqrt(float(q))
            self.store[key] = A_cpu
            if len(self.store) > self.cap:
                self.store.popitem(last=False)
        return A_cpu.to(device=device, dtype=dtype)


_JL = JLCache(capacity=128)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int,
    cap: int,
    ref_mu_sig: Optional[Tuple[torch.Tensor, torch.Tensor]],
    run_key: int,
    epoch: int,
    cfg: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Formerly `_features_for_loader` in legacy_cli_refactor.

    Returns:
        Z_geom      – JL-projected, normalized features (if geom_rp_dim > 0)
        Z_geom_full – native normalized features (no JL projection)
    """
    model.eval()
    feats = []
    cnt = 0
    it = iter(loader)
    for _ in range(int(n_batches)):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            try:
                xb, _ = next(it)
            except Exception as e:
                print(f"[WARN] features fetch failed: {e}")
                break
        except Exception as e:
            print(f"[WARN] features fetch failed: {e}")
            break
        xb = xb.to(device)
        h = penult(model, xb).detach().cpu().float()
        feats.append(h)
        cnt += h.shape[0]
        if cnt >= int(cap):
            break

    if feats:
        H = torch.cat(feats, dim=0)
    else:
        # Guard: if no features and no reference frame, avoid width=1 footgun.
        # Use a sensible penultimate feature width when ref_mu_sig is absent.
        if ref_mu_sig is not None:
            d = int(ref_mu_sig[0].shape[1])
        else:
            d = int(cfg.get("penult_dim", 512))
        H = torch.zeros((0, d), dtype=torch.float32)

    if H.numel() == 0:
        if ref_mu_sig is not None:
            mu, sig_ref = ref_mu_sig
        else:
            d = H.shape[1]
            mu = torch.zeros((1, d), dtype=torch.float32)
            sig_ref = torch.ones((1, d), dtype=torch.float32)
    else:
        if ref_mu_sig is None:
            mu = H.mean(dim=0, keepdim=True)
            sig_ref = _std0(H, keepdim=True) + 1e-8
        else:
            mu, sig_ref = ref_mu_sig

    std_frame = sig_ref  # freeze to reference across epochs
    Z_geom_native = ((H - mu) / std_frame).to(torch.float32)

    Z_geom = Z_geom_native
    geom_dim = int(cfg.get("geom_rp_dim", 0) or 0)
    if geom_dim:
        d_in = Z_geom.shape[1]
        q = int(min(geom_dim, d_in))
        if q < d_in:
            fixed = bool(cfg.get("rp_fixed", True))
            A = _JL.get(d_in, q, int(run_key), int(epoch), fixed, device="cpu", dtype=torch.float32)
            Z_geom = (Z_geom_native @ A).to(torch.float32)

    return Z_geom, Z_geom_native


# Backwards-compatible alias name so imports can use the old symbol
_features_for_loader = extract_features

def cov_eigs(Z: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, int]:
    """Eigenvalues of covariance with small diagonal jitter; returns (eigs, n_negative)."""
    n = Z.shape[0]
    COV = (Z.T @ Z) / max(1, (n - 1))
    COV = COV + eps * torch.eye(COV.shape[0], device=Z.device, dtype=Z.dtype)
    try:
        w = torch.linalg.eigvalsh(COV)
    except Exception:
        # Fallback for older torch: use generic eigvals and take real part
        w = torch.linalg.eigvals(COV).real
    neg = int((w < -1e-8).sum().item())
    return w.clamp_min(0), neg


def choose_k_by_energy(eigs: torch.Tensor, energy: float, kmax: int) -> Tuple[int, float, bool]:
    """Choose k so top-k eigenvalues capture a given energy fraction.

    Returns (k, tail_mass, good) where tail_mass is the remaining fraction
    and good indicates whether a valid k was found.
    """
    if not (0.0 < energy <= 0.999999):
        raise AssertionError("var_k_energy must be in (0,1).")
    s = eigs.sum()
    if s <= 0 or eigs.numel() == 0:
        return 1, float("nan"), False
    w_desc = torch.flip(eigs, dims=[0])
    cum_top = torch.cumsum(w_desc, dim=0)
    idxs = (cum_top >= energy * s).nonzero(as_tuple=False)
    if idxs.numel() == 0:
        return 1, float("nan"), False
    k = int(idxs[0].item()) + 1
    k = max(1, min(k, int(min(eigs.shape[0], kmax))))
    tail_frac = float(((s - cum_top[k - 1]) / (s + 1e-12)).cpu().item())
    return k, tail_frac, True


def variance_outside_k(Z: torch.Tensor) -> Tuple[float, float, int, float, int, int]:
    """Variance outside top-k eigenspace and effective dimension.

    Uses CFG["var_k_energy"] and CFG["var_k_max"] to choose k.
    Returns (var_out_k, eff_dim, k, tail_mass, ok, n_negative).
    """
    w, neg = cov_eigs(Z)
    ok = 1
    k_energy = float(CFG.get("var_k_energy", 0.90))
    k_max = int(CFG.get("var_k_max", 32))
    k, tail_mass, good = choose_k_by_energy(w, k_energy, k_max)
    if not good:
        ok = 0
        k = 1
        tail_mass = float("nan")
    s = w.sum().clamp_min(1e-12)
    topk = w[-k:].sum()
    var_out_k = float(1.0 - (topk / s))
    eff_dim = float((s**2 / (w.pow(2).sum().clamp_min(1e-12))).cpu())
    return var_out_k, eff_dim, int(k), float(tail_mass), ok, neg


def spectral_r2(Z: torch.Tensor) -> float:
    """Top-2 energy ratio (λ1+λ2)/Σλ using covariance eigenvalues; nan if d<2 or degenerate."""
    try:
        w, _ = cov_eigs(Z)
        if w.numel() < 2:
            return float("nan")
        s = float(w.sum().cpu().item())
        if not np.isfinite(s) or s <= 0:
            return float("nan")
        r2 = float(((w[-1] + w[-2]) / w.sum()).cpu().item())
        return r2
    except Exception:
        return float("nan")


@torch.no_grad()
def cosine_dispersion(Z: torch.Tensor, seed: int, epoch: int, sample: int = 800) -> float:
    """Cosine-similarity dispersion of JL/geom features (observability only)."""
    if os.environ.get("SCAR_SMOKE", "0") == "1" and CFG.get("skip_cos_disp_in_smoke", False):
        return float("nan")
    n = Z.shape[0]
    if n <= 2:
        return float("nan")
    sample_n = int(min(max(2, sample), n))
    _seed = int(1e6 * _u01_from_hash("cos", seed, epoch))
    # Device-safe generator
    if Z.device.type == "cuda":
        g = torch.Generator(device=Z.device).manual_seed(_seed)
        idx = torch.randperm(n, generator=g, device=Z.device)[:sample_n]
    else:
        g = torch.Generator().manual_seed(_seed)
        idx = torch.randperm(n, generator=g)[:sample_n]
    X = F.normalize(Z[idx], dim=1)
    G = X @ X.T
    off = G - torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    var = off.pow(2).sum() / max(1, off.numel() - G.shape[0])
    return float(var.sqrt().item())
