# veriscope/core/window.py
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

import hashlib
import json

import numpy as np


class Transport(Protocol):
    def apply(self, ctx: str, x: np.ndarray) -> np.ndarray: ...
    def natural_with(self, restrict: Callable[..., np.ndarray]) -> bool: ...


class Intervention(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class WindowDecl:
    epsilon: float
    metrics: Sequence[str]
    weights: Dict[str, float]
    bins: int
    interventions: Sequence[Callable[[Any], Any]] = ()
    cal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    _DECL_TRANSPORT: Optional[Any] = field(default=None, repr=False, compare=False)

    def normalize_weights(self) -> None:
        s = sum(abs(v) for v in self.weights.values()) or 1.0
        self.weights = {k: float(v) / s for k, v in self.weights.items()}

    def attach_transport(self, transport: Any) -> None:
        self._DECL_TRANSPORT = transport


def window_decl_identity_hash(wd: "WindowDecl") -> str:
    """Stable identity hash for a WindowDecl configuration.

    Captures: epsilon, metrics, weights, bins, cal_ranges.
    Does NOT capture interventions (lambdas aren't reliably hashable).

    Returns a short 16-hex prefix of SHA-256 over a canonical JSON payload.
    """

    def _float_pair(v: Any) -> Tuple[float, float]:
        try:
            a, b = v  # type: ignore[misc]
        except Exception:
            return (float("nan"), float("nan"))
        try:
            return (float(a), float(b))
        except Exception:
            return (float("nan"), float("nan"))

    metrics = sorted([str(m) for m in (getattr(wd, "metrics", []) or [])])

    weights_raw = getattr(wd, "weights", {}) or {}
    try:
        weights_items = sorted((str(k), float(v)) for k, v in weights_raw.items())
    except Exception:
        # Fallback if values are not float-castable
        weights_items = sorted((str(k), float("nan")) for k in weights_raw.keys())
    weights = {k: v for k, v in weights_items}

    cal_ranges_raw = getattr(wd, "cal_ranges", {}) or {}
    cal_ranges_items = []
    for k, v in cal_ranges_raw.items():
        a, b = _float_pair(v)
        cal_ranges_items.append((str(k), [a, b]))
    cal_ranges = {k: v for k, v in sorted(cal_ranges_items)}

    payload = {
        "epsilon": float(getattr(wd, "epsilon", 0.0)),
        "metrics": metrics,
        "weights": weights,
        "bins": int(getattr(wd, "bins", 0)),
        "cal_ranges": cal_ranges,
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class FRWindow:
    decl: WindowDecl
    transport: Transport  # common G_T for both arguments
    tests: Sequence[Callable]  # Φ_W, closed under post-proc/mixtures/pullbacks

    def normalized_weights(self) -> Dict[str, float]:
        s = sum(abs(v) for v in self.decl.weights.values()) or 1.0
        return {k: float(v) / s for k, v in self.decl.weights.items()}
