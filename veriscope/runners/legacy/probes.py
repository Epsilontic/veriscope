# veriscope/runners/legacy/probes.py
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# CFG from shared runtime if installed
try:
    from veriscope.runners.legacy import runtime as _rt  # type: ignore[import]

    CFG: Dict[str, Any] = getattr(_rt, "CFG", {}) or {}
except Exception:
    CFG = {}

# Feature extractors and geometry metrics
from veriscope.runners.legacy.features import (
    _features_for_loader,  # keep legacy behavior
    variance_outside_k,
    cosine_dispersion,
)

# Common TV under decl transport (canonical implementation in core)
from veriscope.core.ipm import dPi_product_tv as _dPi_product_tv


def _apply_for_window(window) -> Callable[[str, np.ndarray], np.ndarray]:
    """Return an apply(ctx, x) callable for a WindowDecl-like object.

    Priority:
      1) window._DECL_TRANSPORT (if present)
      2) identity
    """
    adapter = getattr(window, "_DECL_TRANSPORT", None)
    if adapter is not None and hasattr(adapter, "apply"):
        return lambda ctx, x: adapter.apply(ctx, np.asarray(x, float))  # type: ignore[attr-defined]
    return lambda ctx, x: np.asarray(x, float)


def collect_feature_snapshot(
    model: torch.nn.Module,
    loader,
    device,
    metrics: List[str],
    ref_mu_sig,
) -> Dict[str, np.ndarray]:
    """
    Minimal snapshot for κ_sens probes.
    Returns a dict mapping metric name -> 1D numpy array.
    Unknown metrics return an empty array to be ignored downstream.
    """
    Z_geom, Z_native = _features_for_loader(
        model=model,
        loader=loader,
        device=device,
        n_batches=int(CFG.get("metric_batches", 3)),
        cap=int(CFG.get("metric_total_cap", 512)),
        ref_mu_sig=ref_mu_sig,
        run_key=0,
        epoch=0,
    )
    out: Dict[str, np.ndarray] = {}
    try:
        var_out, eff_dim, *_ = variance_outside_k(Z_native)
    except Exception:
        var_out, eff_dim = float("nan"), float("nan")
    try:
        cos = cosine_dispersion(Z_geom, seed=0, epoch=0)
    except Exception:
        cos = float("nan")

    for m in metrics:
        if m == "var_out_k":
            out[m] = np.array([var_out], dtype=float)
        elif m == "eff_dim":
            out[m] = np.array([eff_dim], dtype=float)
        elif m == "cos_disp":
            out[m] = np.array([cos], dtype=float)
        else:
            out[m] = np.array([], dtype=float)
    return out


def kappa_sens_probe(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    pool_loader,
    device,
    window,
    ref_mu_sig,
    probe_cfg: Dict,
) -> float:
    """Compute κ_sens using predeclared micro-probes. Always restores model params and mode."""
    was_training = model.training  # snapshot current train/eval mode
    try:
        base = collect_feature_snapshot(model, pool_loader, device, list(window.metrics), ref_mu_sig)
        kappa = 0.0

        apply = _apply_for_window(window)

        # Augmentation-based probe
        if probe_cfg.get("aug_probe", True):
            try:
                temp_loader = probe_cfg.get("aug_loader_factory", lambda: pool_loader)()
                interv = collect_feature_snapshot(model, temp_loader, device, list(window.metrics), ref_mu_sig)
                tv = _dPi_product_tv(window, base, interv, apply=apply)
                if np.isfinite(tv):
                    kappa = max(kappa, float(tv))
            except Exception:
                pass

        # Micro learning-rate probe
        if probe_cfg.get("lr_probe", False):
            state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            try:
                pg0 = opt.param_groups[0] if hasattr(opt, "param_groups") else {}
                lr0 = float(pg0.get("lr", CFG.get("base_lr", 0.1)))
                micro_lr = float(probe_cfg.get("lr_factor", 1.1)) * lr0
                micro_opt = torch.optim.SGD(model.parameters(), lr=micro_lr, momentum=0.0, weight_decay=0.0)
                it = iter(pool_loader)
                for _ in range(2):
                    try:
                        xb, yb = next(it)
                    except StopIteration:
                        break
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    micro_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    micro_opt.step()

                interv2 = collect_feature_snapshot(model, pool_loader, device, list(window.metrics), ref_mu_sig)
                tv2 = _dPi_product_tv(window, base, interv2, apply=apply)
                if np.isfinite(tv2):
                    kappa = max(kappa, float(tv2))
            except Exception:
                pass
            finally:
                # restore parameters; mode is restored in the outer finally
                model.load_state_dict({k: v.to(device) for k, v in state.items()})

        return float(kappa)
    finally:
        # ensure we leave the model in the same mode it started
        model.train(was_training)


__all__ = ["collect_feature_snapshot", "kappa_sens_probe"]
