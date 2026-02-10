# veriscope/__init__.py
from importlib.metadata import PackageNotFoundError as _PNF
from importlib.metadata import version as _v

try:
    __version__ = _v("veriscope")
except _PNF:
    __version__ = "0.1.0+dev"

_CORE_EXPORTS = (
    "WindowDecl",
    "DeclTransport",
    "assert_naturality",
    "tv_hist_fixed",
    "epsilon_statistic_bhc",
    "aggregate_epsilon_stat",
    "GateEngine",
    "GateResult",
)

__all__ = [*_CORE_EXPORTS, "__version__"]


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        from . import core as _core

        return getattr(_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
