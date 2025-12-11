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

    def copy_with(self, **overrides) -> "WindowDecl":
        """Create a copy of this WindowDecl with specified fields overridden.

        This is the preferred way to create variant WindowDecls (e.g., with
        different epsilon) without risking field loss from manual construction.

        If WindowDecl is a dataclass, consider using dataclasses.replace() instead,
        which this method will delegate to if available.

        Example:
            regime_decl = base_decl.copy_with(epsilon=0.18)

        Args:
            **overrides: Field names and new values to override

        Returns:
            New WindowDecl with overridden fields

        Raises:
            ValueError: If an unknown field name is provided in overrides

        Note:
            If WindowDecl gains new fields in the future, this method must be
            updated to include them. This is a maintenance point.
        """
        import copy as _copy
        import dataclasses as _dc

        # If this is a dataclass, use replace() for robustness
        if _dc.is_dataclass(self) and not isinstance(self, type):
            try:
                return _dc.replace(self, **overrides)
            except TypeError as e:
                # Unknown field in overrides
                raise ValueError(f"Invalid field in overrides: {e}") from e

        # Manual approach for non-dataclass WindowDecl
        # Get current values for all known constructor args
        known_fields = {
            "epsilon",
            "metrics",
            "weights",
            "bins",
            "interventions",
            "cal_ranges",
        }

        # Validate overrides
        unknown = set(overrides.keys()) - known_fields
        if unknown:
            raise ValueError(f"Unknown WindowDecl field(s): {sorted(unknown)}. Valid fields: {sorted(known_fields)}")

        kwargs = {
            "epsilon": self.epsilon,
            "metrics": list(self.metrics),
            "weights": _copy.deepcopy(self.weights),
            "bins": self.bins,
            "interventions": self.interventions,
            "cal_ranges": _copy.deepcopy(getattr(self, "cal_ranges", {})),
            "_DECL_TRANSPORT": self._DECL_TRANSPORT,
        }

        # Apply overrides
        kwargs.update(overrides)

        return WindowDecl(**kwargs)


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
