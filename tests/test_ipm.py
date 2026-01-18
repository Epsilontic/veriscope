# tests/test_ipm.py  
from __future__ import annotations  
  
import numpy as np  
import pytest  
  
from veriscope.core.ipm import D_W, d_Pi, dPi_product_tv_robust, tv_hist_fixed  
  
pytestmark = pytest.mark.unit  
  
  
class TestIPMTransportMath:  
    def test_tv_hist_fixed_empty_semantics(self):  
        assert tv_hist_fixed([], [], bins=10) == 0.0  
        assert np.isnan(tv_hist_fixed([], [0.1, 0.2], bins=10))  
        assert np.isnan(tv_hist_fixed([0.1, 0.2], [], bins=10))  
  
    def test_tv_hist_fixed_filters_nonfinite_then_empty_rules_apply(self):  
        assert tv_hist_fixed([np.nan, np.inf], [np.nan], bins=10) == 0.0  
        assert np.isnan(tv_hist_fixed([np.nan, np.inf], [0.2], bins=10))  
  
    def test_tv_hist_fixed_symmetry_and_bounds(self):  
        a = np.full(200, 0.05, dtype=float)  
        b = np.full(200, 0.95, dtype=float)  
        tv_ab = tv_hist_fixed(a, b, bins=10)  
        tv_ba = tv_hist_fixed(b, a, bins=10)  
        assert np.isfinite(tv_ab) and np.isfinite(tv_ba)  
        assert 0.0 <= tv_ab <= 1.0  
        assert 0.0 <= tv_ba <= 1.0  
        np.testing.assert_allclose(tv_ab, tv_ba, atol=1e-12)  
  
    def test_d_Pi_and_robust_near_zero_on_identical(self, make_window_decl):  
        wd = make_window_decl(["m1"], weights={"m1": 1.0}, bins=10)  
  
        def identity(name, arr):  
            return np.asarray(arr, float)  
  
        P = {"m1": np.array([0.1, 0.2, 0.3], dtype=float)}  
        Q = {"m1": P["m1"].copy()}  
  
        d = d_Pi(wd, P, Q, apply=identity)  
        d_r = dPi_product_tv_robust(wd, P, Q, apply=identity)  
  
        assert np.isfinite(d) and np.isfinite(d_r)  
        assert np.isclose(d, 0.0, atol=1e-12)  
        assert np.isclose(d_r, 0.0, atol=1e-12)  
  
    def test_dPi_product_tv_robust_no_finite_metrics_returns_nan(self, make_window_decl):  
        wd = make_window_decl(["m1"], weights={"m1": 1.0}, bins=10)  
  
        def nan_if_small(name, arr):  
            arr = np.asarray(arr, float)  
            # P has mean ~0.15; Q has mean ~0.95. Make P empty-after-filter only.  
            if np.nanmean(arr) < 0.5:  
                return np.full_like(arr, np.nan)  
            return arr  
  
        P = {"m1": np.array([0.1, 0.2], dtype=float)}  
        Q = {"m1": np.array([0.9, 1.0], dtype=float)}  
        d_r = dPi_product_tv_robust(wd, P, Q, apply=nan_if_small)  
        assert np.isnan(d_r)  
  
    def test_D_W_nonnegative_and_zero_on_identical(self, make_window_decl, make_fr_window):  
        wd = make_window_decl(  
            ["m1", "m2"],  
            weights={"m1": 0.5, "m2": 0.5},  
            bins=10,  
            cal_ranges={"m1": (0.0, 1.0), "m2": (0.0, 1.0)},  
        )  
        if hasattr(wd, "normalize_weights"):  
            wd.normalize_weights()  
  
        frwin = make_fr_window(wd)  
  
        P = {"m1": np.full(200, 0.2), "m2": np.full(200, 0.7)}  
        Q = {"m1": P["m1"].copy(), "m2": P["m2"].copy()}  
        dw0 = D_W(frwin, P, Q)  
        assert np.isfinite(dw0)  
        assert dw0 >= 0.0  
        assert np.isclose(dw0, 0.0, atol=1e-12)  
