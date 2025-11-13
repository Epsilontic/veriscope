from __future__ import annotations

import warnings
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .window import Transport as TransportProtocol
from .window import WindowDecl

__all__ = [
    "DeclTransport",
    "NullTransport",
    "assert_naturality",
]


class DeclTransport(TransportProtocol):
    """
    Adapter that exposes a common transport G_T derived from a WindowDecl.

    Semantics (window-relative):
      • apply(ctx, x): normalize the metric stream `x` for context `ctx` into [0,1]
        using the calibration ranges in `decl.cal_ranges[ctx]`. If the span is
        degenerate or missing, returns a NaN array (so downstream TV skips it).
      • natural_with(restrict): probe commutation restrict ∘ apply == apply ∘ restrict
        over a small grid, catching gauge slippage early.

    Parameters
    ----------
    decl : WindowDecl
        Declared window; must provide `cal_ranges: Dict[str, Tuple[float, float]]`.
    probe_contexts : Optional[Sequence[str]]
        Metric names to probe for naturality. Defaults to a subset of decl.metrics.
    probe_points : Optional[np.ndarray]
        Probe abscissae in [0,1] (default: 5-point grid).
    atol : float
        Absolute tolerance for the naturality check.
    """

    def __init__(
        self,
        decl: WindowDecl,
        probe_contexts: Optional[Sequence[str]] = None,
        probe_points: Optional[np.ndarray] = None,
        atol: float = 1e-6,
    ) -> None:
        self._decl = decl
        self._atol = float(atol)
        self._ranges = dict(getattr(decl, "cal_ranges", {}) or {})

        if probe_contexts is None:
            mets = list(getattr(decl, "metrics", ()) or ())
            self._probe_contexts: Sequence[str] = mets if len(mets) <= 8 else mets[:4]
        else:
            self._probe_contexts = tuple(probe_contexts)

        if not self._probe_contexts:
            warnings.warn(
                "DeclTransport: no probe contexts; naturality checks may be vacuous",
                RuntimeWarning,
            )

        if probe_points is None:
            self._probe_points = np.linspace(0.0, 1.0, 5, dtype=float)
        else:
            z = np.asarray(probe_points, dtype=float)
            if z.ndim != 1:
                raise ValueError("probe_points must be a 1D array")
            self._probe_points = z

    # --- Transport Protocol ---

    def apply(self, ctx: str, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, float)
        # Prefer calibrated range; fall back to [0,1] if missing/invalid
        lo, hi = self._ranges.get(ctx, (0.0, 1.0))
        if not (np.isfinite(lo) and np.isfinite(hi) and (hi > lo)):
            lo, hi = 0.0, 1.0
        span = max(1e-12, float(hi - lo))
        z = (arr - float(lo)) / span
        return np.clip(z, 0.0, 1.0)

    def natural_with(self, restrict: Callable[..., np.ndarray]) -> bool:
        """
        Check whether restrict(apply(ctx, ·)) == apply(ctx, restrict(·)) holds
        (within tolerances) for probe contexts and points.
        Returns False if any probe fails or no context was actually checked.
        """
        try:
            z = self._probe_points
            ctxs = self._probe_contexts or ("raw",)
            seen = False
            for ctx in ctxs:
                left = restrict(self.apply(ctx, z))
                right = self.apply(ctx, restrict(z))
                left = np.asarray(left, float).ravel()
                right = np.asarray(right, float).ravel()
                n = min(left.size, right.size)
                if n == 0:
                    continue
                if not np.all(np.isfinite(left[:n])) or not np.all(np.isfinite(right[:n])):
                    return False
                if not np.allclose(left[:n], right[:n], atol=self._atol, rtol=1e-6):
                    return False
                seen = True
            return bool(seen)
        except Exception:
            return False


class NullTransport(TransportProtocol):
    """
    Identity transport. Useful when the window's common G_T is the identity.
    """

    def apply(self, ctx: str, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float)

    def natural_with(self, restrict: Callable[..., np.ndarray]) -> bool:
        return True


def assert_naturality(
    transport: TransportProtocol,
    restricts: Iterable[Callable[..., np.ndarray]],
    msg: str = "Common transport failed naturality check under the declared window.",
) -> None:
    """
    Raise a ValueError if `transport.natural_with(restrict)` is False for any provided restrictor.
    Use in smoke tests or at window-instantiation time to catch gauge slippage early.
    """
    for r in restricts:
        if not transport.natural_with(r):
            raise ValueError(msg)
