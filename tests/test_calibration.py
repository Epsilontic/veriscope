# tests/test_calibration.py
from __future__ import annotations

import numpy as np
import pytest

from veriscope.core.calibration import (
    aggregate_epsilon_stat,
    epsilon_statistic_bhc,
    resolve_epsilon_from_controls,
)

pytestmark = pytest.mark.unit


def _np_quantile_linear(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=float)
    try:
        return float(np.quantile(a, q, method="linear"))
    except TypeError:
        return float(np.quantile(a, q))  # numpy<1.22 fallback


class TestCalibrationMath:
    def test_resolve_epsilon_from_controls_empty_invalid(self):
        eps, n = resolve_epsilon_from_controls([], 0.95, 0.1)
        assert eps == 0.1
        assert n == 0

        eps, n = resolve_epsilon_from_controls([np.nan, np.inf, -np.inf], 0.95, 0.1)
        assert eps == 0.1
        assert n == 0

    def test_resolve_epsilon_from_controls_filters_non_numeric_and_nonfinite(self):
        vals = [0.1, "0.9", None, np.nan, np.inf]
        eps, n = resolve_epsilon_from_controls(vals, 0.5, 0.2)
        assert n == 1
        assert np.isclose(eps, 0.1)

    def test_resolve_epsilon_from_controls_bool_inclusion_current_behavior(self):
        vals = [True, False, 0.3]  # -> [1.0, 0.0, 0.3]
        expected = _np_quantile_linear(np.array([1.0, 0.0, 0.3]), 0.5)
        eps, n = resolve_epsilon_from_controls(vals, 0.5, 0.1)
        assert n == 3
        np.testing.assert_allclose(eps, expected, atol=1e-12)

    def test_resolve_epsilon_from_controls_matches_numpy_linear_quantile(self):
        vals = np.array([0.01, 0.02, 0.05, 0.10, 0.20], dtype=float)
        expected = _np_quantile_linear(vals, 0.95)
        eps, n = resolve_epsilon_from_controls(vals, 0.95, 0.1)
        assert n == 5
        np.testing.assert_allclose(eps, expected, atol=1e-12)

    def test_resolve_epsilon_from_controls_filters_nans_and_matches_quantile(self):
        vals_nan = np.array([0.01, np.nan, 0.05], dtype=float)
        finite = vals_nan[np.isfinite(vals_nan)]
        expected = _np_quantile_linear(finite, 0.5)

        eps, n = resolve_epsilon_from_controls(vals_nan, 0.5, 0.1)
        assert n == finite.size
        np.testing.assert_allclose(eps, expected, atol=1e-12)

    def test_epsilon_statistic_bhc_edges(self):
        assert epsilon_statistic_bhc(0, 10) == 1.0
        assert epsilon_statistic_bhc(-1, 10) == 1.0
        assert epsilon_statistic_bhc(1, 1) == 1.0

    def test_epsilon_statistic_bhc_monotone_in_n(self):
        eps_small = epsilon_statistic_bhc(1000, 16, alpha=0.05)
        eps_large = epsilon_statistic_bhc(10000, 16, alpha=0.05)
        assert 0.0 <= eps_small <= 1.0
        assert 0.0 <= eps_large <= 1.0
        assert np.isfinite(eps_small) and np.isfinite(eps_large)
        assert eps_large < eps_small

    def test_aggregate_epsilon_stat_bounds_and_monotonicity(self, make_window_decl):
        wd = make_window_decl(["m1", "m2"], weights={"m1": 0.6, "m2": 0.4}, bins=16)
        if hasattr(wd, "normalize_weights"):
            wd.normalize_weights()

        eps0 = aggregate_epsilon_stat(wd, {"m1": 0, "m2": 0})
        assert eps0 == 1.0

        eps_small = aggregate_epsilon_stat(wd, {"m1": 10, "m2": 10})
        eps_large = aggregate_epsilon_stat(wd, {"m1": 10_000, "m2": 10_000})

        for eps in (eps_small, eps_large):
            assert 0.0 <= eps <= 1.0
            assert np.isfinite(eps)

        assert eps_large <= eps_small
