# veriscope/core/__init__.py
from __future__ import annotations

"""
Public re-exports for the Finite Realism core.
Downstream code should import from `veriscope.core` to avoid shadowing internals.
"""

from importlib import import_module

_EXPORT_MAP = {
    "WindowDecl": ("veriscope.core.window", "WindowDecl"),
    "DeclTransport": ("veriscope.core.transport", "DeclTransport"),
    "assert_naturality": ("veriscope.core.transport", "assert_naturality"),
    "tv_hist_fixed": ("veriscope.core.ipm", "tv_hist_fixed"),
    "epsilon_statistic_bhc": ("veriscope.core.calibration", "epsilon_statistic_bhc"),
    "aggregate_epsilon_stat": ("veriscope.core.calibration", "aggregate_epsilon_stat"),
    "GateEngine": ("veriscope.core.gate", "GateEngine"),
    "GateResult": ("veriscope.core.gate", "GateResult"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
