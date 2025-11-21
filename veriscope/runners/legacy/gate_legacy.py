# veriscope/runners/legacy/gate_legacy.py
from __future__ import annotations

from typing import Dict, Mapping, Union, cast

import numpy as np

from veriscope.runners.legacy.utils import as_int  # numeric guard
from veriscope.core.ipm import tv_hist_fixed  # canonical TV kernel

# Optional FR transport adapter (legacy fallback if unavailable)
try:
    from veriscope.fr_integration import DeclTransport as _DECL_TRANSPORT  # type: ignore[attr-defined]
except Exception:
    _DECL_TRANSPORT = None  # type: ignore[assignment]


def dPi_product_tv(
    window,
    past_by_metric: Dict[str, np.ndarray],
    recent_by_metric: Dict[str, np.ndarray],
) -> float:
    tv = 0.0

    # Choose apply: instance adapter if available, else FR global, else identity
    _adapter = None
    try:
        _adapter = getattr(window, "_DECL_TRANSPORT", None)
    except Exception:
        _adapter = None

    if _adapter is not None:
        _apply = _adapter.apply  # type: ignore[assignment]
    elif _DECL_TRANSPORT is not None:
        _apply = _DECL_TRANSPORT.apply  # type: ignore[assignment]
    else:
        _apply = lambda name, arr: np.asarray(arr, float)

    weights = cast(Mapping[str, float], getattr(window, "weights", {}) or {})
    bins = as_int(getattr(window, "bins", 16), default=16)

    for m, w in weights.items():
        if m not in past_by_metric or m not in recent_by_metric:
            continue
        a = np.asarray(_apply(m, past_by_metric[m]), dtype=float)
        b = np.asarray(_apply(m, recent_by_metric[m]), dtype=float)
        tv += float(cast(Union[int, float], w)) * tv_hist_fixed(a, b, bins)
    return float(tv)


__all__ = ["dPi_product_tv"]
