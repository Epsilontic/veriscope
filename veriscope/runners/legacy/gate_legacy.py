# veriscope/runners/legacy/gate_legacy.py
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from veriscope.core.ipm import dPi_product_tv as _core_dPi_product_tv


def _apply_for_window(window: Any) -> Callable[[str, np.ndarray], np.ndarray]:
    """Return an apply(ctx, x) callable for a WindowDecl-like object.

    Priority:
      1) window._DECL_TRANSPORT (if present)
      2) identity

    This keeps the legacy 3-arg dPi_product_tv signature while delegating the
    actual product-TV computation to the canonical core implementation.
    """
    adapter = getattr(window, "_DECL_TRANSPORT", None)
    if adapter is not None and hasattr(adapter, "apply"):
        return lambda ctx, x: adapter.apply(ctx, np.asarray(x, float))  # type: ignore[attr-defined]
    return lambda ctx, x: np.asarray(x, float)


def dPi_product_tv(
    window: Any,
    past_by_metric: Dict[str, np.ndarray],
    recent_by_metric: Dict[str, np.ndarray],
) -> float:
    apply = _apply_for_window(window)
    return _core_dPi_product_tv(window, past_by_metric, recent_by_metric, apply=apply)


__all__ = ["dPi_product_tv"]
