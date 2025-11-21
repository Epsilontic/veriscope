# veriscope/__init__.py
from importlib.metadata import PackageNotFoundError as _PNF
from importlib.metadata import version as _v

try:
    __version__ = _v("veriscope")
except _PNF:
    __version__ = "0.1.0+dev"

from .core import (
    DeclTransport,
    GateEngine,
    GateResult,
    WindowDecl,
    aggregate_epsilon_stat,
    assert_naturality,
    epsilon_statistic_bhc,
    tv_hist_fixed,
)

__all__ = [
    "WindowDecl",
    "DeclTransport", "assert_naturality",
    "tv_hist_fixed",
    "epsilon_statistic_bhc", "aggregate_epsilon_stat",
    "GateEngine", "GateResult",
    "__version__",
]