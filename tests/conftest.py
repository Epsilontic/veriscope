# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

# De-risked fallback for non-editable installs: append only (no shadowing).
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def make_window_decl() -> Callable[..., object]:
    """
    Minimal-valid WindowDecl factory pinned to the explicit dataclass signature.

    Expected:
      WindowDecl(
        epsilon: float,
        metrics: Sequence[str],
        weights: Dict[str, float],
        bins: int,
        interventions: Sequence[Callable] = (),
        cal_ranges: Dict[str, Tuple[float, float]] = {},
      )
    """
    from veriscope.core.window import WindowDecl

    def _make(
        metrics: List[str],
        *,
        epsilon: float = 0.12,
        weights: Optional[Dict[str, float]] = None,
        bins: int = 10,
        cal_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        interventions=(),
    ):
        metrics = list(metrics)
        if not metrics:
            raise AssertionError("make_window_decl: metrics must be non-empty")

        if weights is None:
            weights = {m: 1.0 for m in metrics}
        if cal_ranges is None:
            cal_ranges = {m: (0.0, 1.0) for m in metrics}

        wd = WindowDecl(
            epsilon=float(epsilon),
            metrics=metrics,
            weights=dict(weights),
            bins=int(bins),
            interventions=tuple(interventions),
            cal_ranges=dict(cal_ranges),
        )

        # Fail loudly / early if constructor contract drifts.
        assert hasattr(wd, "metrics") and wd.metrics, "WindowDecl missing/nonempty metrics"
        assert hasattr(wd, "weights") and isinstance(wd.weights, dict) and wd.weights, (
            "WindowDecl missing/nonempty weights"
        )
        assert hasattr(wd, "bins") and isinstance(wd.bins, int), "WindowDecl missing/invalid bins"
        assert hasattr(wd, "epsilon"), "WindowDecl missing epsilon"
        assert hasattr(wd, "cal_ranges") and isinstance(wd.cal_ranges, dict), "WindowDecl missing cal_ranges"
        return wd

    return _make


@pytest.fixture
def make_fr_window() -> Callable[..., object]:
    """
    Minimal-valid FRWindow factory pinned to the explicit dataclass signature:
      FRWindow(decl, transport, tests)
    """
    from veriscope.core.transport import DeclTransport
    from veriscope.core.window import FRWindow

    def _make(decl, *, tests=()):
        transport = DeclTransport(decl)
        decl.attach_transport(transport)
        fr = FRWindow(decl=decl, transport=transport, tests=tuple(tests))

        assert hasattr(fr, "decl") and fr.decl is not None, "FRWindow missing decl"
        assert hasattr(fr, "transport") and fr.transport is not None, "FRWindow missing transport"
        assert hasattr(fr, "tests"), "FRWindow missing tests"
        return fr

    return _make


@pytest.fixture
def make_gate_engine() -> Callable[..., object]:
    """
    GateEngine factory pinned to explicit __init__ signature in provided gate.py.
    """
    from veriscope.core.gate import GateEngine

    def _make(
        frwin,
        *,
        gain_thresh: float = 0.05,
        eps_stat_alpha: float = 0.05,
        eps_stat_max_frac: float = 0.25,
        eps_sens: float = 0.04,
        min_evidence: int = 16,
        policy: str = "either",
        persistence_k: int = 2,
        min_metrics_exceeding: int = 1,
        trend_n: int = 8,
    ):
        ge = GateEngine(
            frwin=frwin,
            gain_thresh=gain_thresh,
            eps_stat_alpha=eps_stat_alpha,
            eps_stat_max_frac=eps_stat_max_frac,
            eps_sens=eps_sens,
            min_evidence=min_evidence,
            policy=policy,
            persistence_k=persistence_k,
            min_metrics_exceeding=min_metrics_exceeding,
            trend_n=trend_n,
        )
        assert hasattr(ge, "check") and callable(ge.check), "GateEngine missing callable check()"
        return ge

    return _make


@pytest.fixture
def window_decl_simple(make_window_decl):
    return make_window_decl(
        ["test_metric"],
        weights={"test_metric": 1.0},
        bins=10,
        epsilon=0.12,
        cal_ranges={"test_metric": (0.0, 1.0)},
    )


@pytest.fixture
def window_decl_multi(make_window_decl):
    return make_window_decl(
        ["m1", "m2"],
        weights={"m1": 0.6, "m2": 0.4},
        bins=16,
        epsilon=0.12,
        cal_ranges={"m1": (0.0, 1.0), "m2": (0.0, 64.0)},
    )


@pytest.fixture
def decl_transport(window_decl_simple):
    from veriscope.core.transport import DeclTransport

    transport = DeclTransport(window_decl_simple)
    window_decl_simple.attach_transport(transport)
    return transport


@pytest.fixture
def fr_window(window_decl_simple, make_fr_window):
    return make_fr_window(window_decl_simple)


@pytest.fixture
def gate_engine(fr_window, make_gate_engine):
    return make_gate_engine(fr_window)
