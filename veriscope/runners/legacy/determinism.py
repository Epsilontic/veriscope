# veriscope/runners/legacy/determinism.py
"""
Determinism helpers for legacy runners: seeding and strict-det fallback.
These are imported by legacy_cli_refactor and any future runner modules.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def _enable_strict_det_or_fallback() -> None:
    """Enable strict deterministic algorithms; if cuBLAS determinism is unavailable,
    fall back to warn_only=True so the run proceeds without device-side asserts.
    """
    if not torch.cuda.is_available():
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception:
            pass
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        # Force one cuBLAS GEMM under strict determinism to validate workspace config
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        (a @ b).sum().item()
        print("[det] strict determinism OK (cuBLAS probe passed)")
    except Exception as e:
        print(f"[WARN] strict determinism failed on cuBLAS probe: {e!r}; falling back to warn_only")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_all(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy, and torch (CPU/GPU), and optionally enable strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        _enable_strict_det_or_fallback()


def new_gen(master: int, off: int) -> torch.Generator:
    """Return a derived torch.Generator from a master seed and offset."""
    g = torch.Generator()
    g.manual_seed((master * 7919 + off * 104729) % (2**63 - 1))
    return g


def seed_worker(worker_id: int) -> None:  # worker_id kept for DataLoader hook signature
    """DataLoader worker_init_fn: propagates torch initial seed to NumPy/random."""
    worker_seed = torch.initial_seed() % (2**32)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
