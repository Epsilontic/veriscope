from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

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

@dataclass
class FRWindow:
    decl: WindowDecl
    transport: Transport         # common G_T for both arguments
    tests: Sequence[Callable]     # Φ_W, closed under post-proc/mixtures/pullbacks

    def normalized_weights(self) -> Dict[str,float]:
        s = sum(abs(v) for v in self.decl.weights.values()) or 1.0
        return {k: float(v)/s for k, v in self.decl.weights.items()}
