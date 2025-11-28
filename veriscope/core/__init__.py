# veriscope/core/__init__.py
from __future__ import annotations

"""
Public re-exports for the Finite Realism core.
Downstream code should import from `veriscope.core` to avoid shadowing internals.
"""

from .calibration import aggregate_epsilon_stat, epsilon_statistic_bhc
from .gate import GateEngine, GateResult
from .ipm import tv_hist_fixed
from .transport import DeclTransport, assert_naturality
from .window import WindowDecl

__all__ = [
    "WindowDecl",
    "DeclTransport",
    "assert_naturality",
    "tv_hist_fixed",
    "epsilon_statistic_bhc",
    "aggregate_epsilon_stat",
    "GateEngine",
    "GateResult",
]
