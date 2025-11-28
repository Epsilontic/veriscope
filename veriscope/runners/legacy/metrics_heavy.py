# veriscope/runners/legacy/metrics_heavy.py
"""
Heavy, budgeted metrics for the legacy runner.

This module owns:
- Sliced Wasserstein-2 on GPU (sliced_w2_gpu, sliced_w2_gpu_budget).
- H0 total persistence via ripser (h0_total_persistence_np, topo_h0_jl_agg).
- A local JL projection cache and a guarded ripser import.

It reads CFG/BUDGET from the shared legacy runtime and keeps all heavy,
finite-window metric logic out of the CLI to avoid cycles and runner bloat.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# Budget ledger is installed by the runner via runtime.install_runtime().
# IMPORTANT: do not import BUDGET at module import-time (it may not be installed yet).
# Fetch lazily via runtime.get_budget() to avoid cycles and stale globals.


def _get_budget():
    try:
        from veriscope.runners.legacy.runtime import get_budget  # local import

        return get_budget()
    except Exception:
        # Fallback: attempt attribute access if older runtime versions exist.
        try:
            from veriscope.runners.legacy import runtime as _rt  # type: ignore

            return getattr(_rt, "BUDGET", None)
        except Exception:
            return None


# Lightweight numeric helpers
try:
    from veriscope.runners.legacy.utils import as_int, as_float  # type: ignore
except Exception:

    def as_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def as_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default


# Stable per-feature std helper used by topo_h0_jl_agg
try:
    from veriscope.runners.legacy.features import _std0  # type: ignore
except Exception:

    def _std0(X: torch.Tensor, keepdim: bool = False) -> torch.Tensor:  # minimal fallback
        x = X.float()
        std = x.std(dim=0, keepdim=keepdim, unbiased=False)
        return std.clamp_min(1e-12)

# Config access: prefer runtime-installed CFG, fall back to packaged defaults, else {}.
# IMPORTANT: do not snapshot CFG at module import-time (runner may install it later).
# Fetch lazily via runtime.get_cfg() to avoid cycles and stale globals.


def _get_cfg() -> Dict[str, Any]:
    try:
        from veriscope.runners.legacy.runtime import get_cfg  # local import

        c = get_cfg()
        # get_cfg() returns a live Mapping; return a dict-like object with .get
        return dict(c) if not isinstance(c, dict) else c
    except Exception:
        # Fallback: attribute access for older runtime versions.
        try:
            from veriscope.runners.legacy import runtime as _rt  # type: ignore

            c = getattr(_rt, "CFG", None)
            if isinstance(c, dict):
                return c
            if c is not None:
                return dict(c)
        except Exception:
            pass

    try:
        from veriscope.config import CFG as _CFG_CENTER  # type: ignore

        return dict(_CFG_CENTER)
    except Exception:
        return {}


# --- Simple JL projection cache (local dedicated copy for heavy metrics) ---
class _JLCache:
    """Minimal JL projection cache keyed by (d, k).
    Keeps behavior deterministic and avoids repeated allocation.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def get(self, d: int, k: int, *args, **kwargs):
        """Return a d×k random Gaussian matrix, cached by (d, k).
        Extra args/kwargs accepted for compatibility (e.g., device=..., dtype=...).
        """
        d_int = int(d)
        k_int = int(k)
        key = (d_int, k_int)
        A = self._cache.get(key)
        if A is not None:
            return A

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32

        g = torch.Generator(device=device).manual_seed(17_4242 + d_int + 31 * k_int)
        A = torch.randn(d_int, k_int, generator=g, device=device, dtype=dtype)
        try:
            A = F.normalize(A, dim=0)
        except Exception:
            # if normalization fails, keep raw Gaussian
            pass
        self._cache[key] = A
        return A


# Process-global JL cache within this module
_JL = _JLCache()

# ---- ripser import guard (H0 persistence) ----
try:
    from ripser import ripser as _ripser  # type: ignore

    def _ripser_safe(X: np.ndarray):
        return _ripser(X, maxdim=0)
except Exception as e:
    try:
        if mp.current_process().name == "MainProcess":
            print(f"[WARN] ripser unavailable ({e!r}) — pers_H0 will be NaN.")
    except Exception:
        pass

    def _ripser_safe(X: np.ndarray):
        raise RuntimeError("ripser_unavailable")

# ======================
# Heavy metric functions
# ======================


@torch.no_grad()
def sliced_w2_gpu(Zt: torch.Tensor, Zt1: torch.Tensor, n_proj: int, seed: int, device) -> Tuple[float, float, int]:
    """
    Compute sliced W2 with deterministic generators on CPU/CUDA.
    Respects CFG['sw2_budget_ms'] and charges BUDGET. Returns (w2, elapsed_ms, n_proj_used).
    """
    budget = _get_budget()
    cfg = _get_cfg()
    # --- finite-window budget guard (no-op if BUDGET is None or allow("sw2") passes) ---
    try:
        if budget is not None and not budget.allow("sw2"):  # type: ignore[attr-defined]
            # Hard skip: budget exhausted for SW2; return a neutral "no-compute" capsule.
            return float("nan"), 0.0, 0
    except Exception:
        # Keep metric robust even if the budget ledger is misconfigured.
        pass

    t0 = time.time()
    try:
        dev = device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Zt = Zt.to(dev)
    Zt1 = Zt1.to(dev)
    d = int(Zt.shape[1])
    n_proj = int(max(1, n_proj))
    used = 0
    acc = 0.0
    budget_ms = int(cfg.get("sw2_budget_ms", 200))
    gen_dev = dev if getattr(dev, "type", "cpu") == "cuda" else torch.device("cpu")
    g = torch.Generator(device=gen_dev).manual_seed(17_4242 + int(seed))
    chunk = min(64, n_proj)

    while used < n_proj:
        # Per-call wall-clock guard (local to this invocation)
        if (time.time() - t0) * 1000.0 > budget_ms:
            break
        k = min(chunk, n_proj - used)
        if getattr(gen_dev, "type", "cpu") == "cuda":
            U = torch.randn(d, k, generator=g, device=gen_dev, dtype=Zt.dtype)
        else:
            U = torch.randn(d, k, generator=g, device=gen_dev, dtype=Zt.dtype).to(dev)
        U = F.normalize(U, dim=0)
        Xt = Zt @ U
        Xt1 = Zt1 @ U
        Xt_sorted, _ = Xt.sort(dim=0)
        Xt1_sorted, _ = Xt1.sort(dim=0)
        acc += float(((Xt_sorted - Xt1_sorted) ** 2).mean().item()) * k
        used += k

    elapsed = float((time.time() - t0) * 1000.0)
    try:
        if budget is not None:
            budget.charge("sw2", elapsed)  # type: ignore[attr-defined]
    except Exception:
        pass
    if used == 0:
        return float("nan"), elapsed, 0
    return float(acc / used), elapsed, int(used)


def sliced_w2_gpu_budget(Zt, Zt1, n_proj, seed, device, budget_ms):
    budget = _get_budget()
    try:
        if (budget is not None) and (not budget.allow("sw2")):  # type: ignore[attr-defined]
            # Hard skip: budget exhausted for SW2; return a neutral "no-compute" capsule.
            return float("nan"), 0.0, 0, False
        val, ms, nproj_done = sliced_w2_gpu(Zt, Zt1, n_proj, seed, device)
    except Exception:
        return float("nan"), 0.0, 0, False
    ok = (ms <= budget_ms) and np.isfinite(val) and (nproj_done > 0)
    # NOTE: do NOT charge here; sliced_w2_gpu already charged the SW2 ledger.
    return (
        float(val) if ok else float("nan"),
        float(ms),
        int(nproj_done if ok else 0),
        bool(ok),
    )


def h0_total_persistence_np(X: np.ndarray) -> float:
    try:
        dgm0 = _ripser_safe(X)["dgms"][0]
    except Exception:
        return float("nan")
    if dgm0.size == 0:
        return 0.0
    lifetimes = dgm0[:, 1] - dgm0[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return float(lifetimes.sum())


def topo_h0_jl_agg(
    Z: torch.Tensor,
    q: int,
    repeats: int,
    run_key: int,
    epoch: int,
    agg: str = "median",
    sample_n: int = 192,
) -> Tuple[float, int, float, int]:
    """
    Returns: (value, n_successful_repeats, elapsed_ms, sampled_n_each_repeat)
    On any exception inside a repeat, that repeat is skipped. If none succeed, value=nan.
    """
    budget = _get_budget()
    cfg = _get_cfg()
    # --- finite-window budget guard (global ripser budget) ---
    try:
        if budget is not None and not budget.allow("ripser"):  # type: ignore[attr-defined]
            # Budget already exhausted for ripser; hard skip.
            return float("nan"), 0, 0.0, 0
    except Exception:
        # Keep metric robust even if ledger is misconfigured.
        pass

    t0 = time.time()
    vals: list[float] = []
    min_used_n = 10**9
    for r in range(repeats):
        # Enforce both per-call and global finite-window budgets
        if (time.time() - t0) * 1000.0 > as_float(cfg.get("ripser_budget_ms", 250), default=250.0):
            break
        if (budget is not None) and (not budget.allow("ripser")):  # type: ignore[attr-defined]
            break
        try:
            t_rep = time.time()
            A = _JL.get(
                Z.shape[1],
                min(q, Z.shape[1]),
                run_key + 1009 * (r + 1),
                epoch,
                cfg.get("rp_fixed", True),
                device=Z.device,
                dtype=Z.dtype,
            )
            Zr = Z @ A
            X = (Zr - Zr.mean(dim=0, keepdim=True)) / (_std0(Zr, keepdim=True) + 1e-8)
            n = X.shape[0]
            if n > sample_n:
                _seed = 17_000_000 + run_key + 31 * epoch + 101 * r
                if X.device.type == "cuda":
                    g = torch.Generator(device=X.device).manual_seed(_seed)
                    idx = torch.randperm(n, generator=g, device=X.device)[:sample_n]
                else:
                    g = torch.Generator().manual_seed(_seed)
                    idx = torch.randperm(n, generator=g)[:sample_n]
                X = X[idx]
                used_n_this = int(sample_n)
                min_used_n = min(min_used_n, used_n_this)
            else:
                used_n_this = int(n)
                min_used_n = min(min_used_n, used_n_this)

            vals.append(h0_total_persistence_np(X.cpu().numpy()))
            ms_rep = (time.time() - t_rep) * 1000.0
            try:
                if budget is not None:
                    budget.charge("ripser", ms_rep)  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            continue

    elapsed = (time.time() - t0) * 1000.0
    if not vals:
        return float("nan"), 0, elapsed, 0
    val = float(np.median(vals) if agg == "median" else np.mean(vals))
    return val, len(vals), elapsed, int(min_used_n if min_used_n != 10**9 else 0)


__all__ = [
    "sliced_w2_gpu",
    "sliced_w2_gpu_budget",
    "h0_total_persistence_np",
    "topo_h0_jl_agg",
]
