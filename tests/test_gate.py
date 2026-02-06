# tests/test_gate.py
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _is_floatlike(x) -> bool:
    return isinstance(x, (float, int, np.floating, np.integer))


def _is_intlike(x) -> bool:
    return isinstance(x, (int, np.integer))


class TestGateEngineSemantics:
    def test_min_evidence_neutral_not_evaluated(self, fr_window, make_gate_engine):
        ge = make_gate_engine(fr_window, min_evidence=16)

        past = {"test_metric": np.full(10, 0.5, dtype=float)}
        recent = {"test_metric": np.full(5, 0.5, dtype=float)}
        counts = {"test_metric": 5}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.01)
        assert r.ok is True
        assert r.warn is False
        assert r.audit["evaluated"] is False
        assert r.audit["reason"] == "not_evaluated_insufficient_evidence"

    def test_no_finite_metrics_triggers_not_evaluated(self, fr_window, make_gate_engine):
        ge = make_gate_engine(fr_window, min_evidence=0, policy="either")

        past = {"test_metric": np.array([np.nan], dtype=float)}
        recent = {"test_metric": np.array([0.2, 0.3], dtype=float)}
        counts = {"test_metric": 100}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        a = r.audit
        assert r.ok is True
        assert r.warn is False
        assert a["evaluated"] is False
        assert a["reason"] == "not_evaluated_no_finite_metrics"
        assert np.isnan(a["worst_DW"])
        assert a["per_metric_tv"] == {}
        assert a["per_metric_n"] == {}

    def test_identical_distributions_pass(self, fr_window, make_gate_engine):
        ge = make_gate_engine(fr_window, min_evidence=0, policy="either", gain_thresh=0.0, eps_sens=0.0)

        past = {"test_metric": np.full(200, 0.5, dtype=float)}
        recent = {"test_metric": np.full(200, 0.5, dtype=float)}
        counts = {"test_metric": 200}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=np.nan, eps_stat_value=0.0)
        assert r.audit["evaluated"] is True
        assert r.ok is True
        assert np.isfinite(r.audit["worst_DW"])
        assert np.isclose(r.audit["worst_DW"], 0.0, atol=1e-12)

    def test_zero_tv_nonidentical_windows_fail_loudly(self, fr_window, make_gate_engine, force_zero_tv):
        ge = make_gate_engine(fr_window, min_evidence=0, policy="either", gain_thresh=0.0, eps_sens=0.0)

        # Raw windows differ, but TV is collapsed to 0.0 by the stubbed comparator.
        past = {"test_metric": np.array([0.10, 0.20, 0.30], dtype=float)}
        recent = {"test_metric": np.array([0.70, 0.80, 0.90], dtype=float)}
        counts = {"test_metric": 3}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=np.nan, eps_stat_value=0.0)
        assert r.ok is False
        assert r.warn is False
        assert r.audit["evaluated"] is True
        assert r.audit["reason"] == "zero_tv_nonidentical_windows"
        assert "test_metric" in r.audit.get("window_debug_nonidentical_metrics", [])
        assert "test_metric" not in r.audit.get("window_debug_empty_metrics", [])

    def test_empty_window_fail_loudly(self, fr_window, make_gate_engine, force_zero_tv):
        ge = make_gate_engine(fr_window, min_evidence=0, policy="either", gain_thresh=0.0, eps_sens=0.0)

        past = {"test_metric": np.array([], dtype=float)}
        recent = {"test_metric": np.array([], dtype=float)}
        counts = {"test_metric": 10}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=np.nan, eps_stat_value=0.0)
        assert r.ok is False
        assert r.warn is False
        assert r.audit["evaluated"] is True
        assert r.audit["reason"] == "empty_window"
        npts = r.audit.get("window_debug_n_points", {}).get("test_metric", {})
        assert npts.get("past") == 0
        assert npts.get("recent") == 0
        assert "test_metric" in r.audit.get("window_debug_empty_metrics", [])

    def test_degenerate_window_persistence_counter_monotone(self, fr_window, make_gate_engine, force_zero_tv):
        ge = make_gate_engine(
            fr_window,
            min_evidence=0,
            policy="persistence",
            persistence_k=2,
            gain_thresh=0.0,
            eps_sens=0.0,
        )

        past = {"test_metric": np.array([0.1, 0.2], dtype=float)}
        recent = {"test_metric": np.array([0.8, 0.9], dtype=float)}
        counts = {"test_metric": 2}

        r1 = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=np.nan, eps_stat_value=0.0)
        assert r1.audit["consecutive_exceedances_after"] == 1
        assert r1.audit["persistence_fail"] is False
        assert r1.audit["reason"] == "zero_tv_nonidentical_windows"
        assert r1.audit["degenerate_window"] is True

        r2 = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=np.nan, eps_stat_value=0.0)
        assert r2.audit["consecutive_exceedances_after"] == 2
        assert r2.audit["persistence_fail"] is True
        assert r2.audit["reason"] == "zero_tv_nonidentical_windows"
        assert r2.audit["degenerate_window"] is True

    def test_single_metric_exceedance_populates_metrics_exceeding(
        self, make_window_decl, make_fr_window, make_gate_engine
    ):
        wd = make_window_decl(
            ["test_metric"],
            epsilon=0.13,
            weights={"test_metric": 1.0},
            bins=10,
            cal_ranges={"test_metric": (0.0, 1.0)},
        )
        fr = make_fr_window(wd)
        ge = make_gate_engine(
            fr,
            min_evidence=0,
            policy="either",
            gain_thresh=0.0,
            eps_sens=0.0,
            min_metrics_exceeding=1,
        )

        past = {"test_metric": np.full(200, 0.1, dtype=float)}
        recent = {"test_metric": np.full(200, 0.9, dtype=float)}
        counts = {"test_metric": 200}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        a = r.audit
        assert a["evaluated"] is True
        assert a["metrics_exceeding"] == ["test_metric"]
        assert a["n_metrics_exceeding"] == 1

        per_metric_tv = a["per_metric_tv"]
        values = []
        for v in per_metric_tv.values():
            if isinstance(v, dict) and "tv" in v:
                tv = float(v["tv"])
            else:
                tv = float(v)
            if np.isfinite(tv):
                values.append(tv)
        expected = max(values, default=0.0)
        assert np.isclose(a["worst_DW"], expected, atol=1e-12)
        assert np.isfinite(float(a["eps_eff"]))
        assert a["worst_DW"] > float(a["eps_eff"])

    def test_worst_dw_uses_weighted_aggregate_not_per_metric_max(
        self, make_window_decl, make_fr_window, make_gate_engine
    ):
        wd = make_window_decl(
            ["m1", "m2"],
            epsilon=0.15,
            weights={"m1": 0.9, "m2": 0.1},
            bins=10,
            cal_ranges={"m1": (0.0, 1.0), "m2": (0.0, 1.0)},
        )
        fr = make_fr_window(wd)
        ge = make_gate_engine(
            fr,
            min_evidence=0,
            policy="either",
            gain_thresh=0.0,
            eps_sens=0.0,
            min_metrics_exceeding=1,
        )

        past = {"m1": np.full(400, 0.1, dtype=float), "m2": np.full(400, 0.1, dtype=float)}
        recent = {"m1": np.full(400, 0.1, dtype=float), "m2": np.full(400, 0.9, dtype=float)}
        counts = {"m1": 400, "m2": 400}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=np.nan, eps_stat_value=0.0)
        a = r.audit

        m1_tv = float(a["per_metric_tv"]["m1"])
        m2_tv = float(a["per_metric_tv"]["m2"])
        expected_weighted = 0.9 * m1_tv + 0.1 * m2_tv

        assert np.isclose(float(a["worst_DW"]), expected_weighted, atol=1e-12)
        assert float(a["worst_DW"]) < float(max(m1_tv, m2_tv))
        assert a["dw_exceeds_threshold"] is False

    def test_persistence_k2_warn_then_fail(self, fr_window, make_gate_engine):
        ge = make_gate_engine(
            fr_window,
            min_evidence=0,
            policy="persistence",
            persistence_k=2,
            gain_thresh=0.0,
            eps_sens=0.0,
        )

        past = {"test_metric": np.full(200, 0.1, dtype=float)}
        counts = {"test_metric": 200}

        r1 = ge.check(
            past,
            {"test_metric": np.full(200, 0.9, dtype=float)},
            counts,
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.0,
        )
        assert r1.audit["evaluated"] is True
        assert r1.audit["dw_exceeds_threshold"] is True
        assert r1.warn is True
        assert r1.ok is True
        assert r1.audit["consecutive_exceedances_after"] == 1

        r2 = ge.check(
            past,
            {"test_metric": np.full(200, 0.95, dtype=float)},
            counts,
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.0,
        )
        assert r2.audit["evaluated"] is True
        assert r2.audit["dw_exceeds_threshold"] is True
        assert r2.warn is False
        assert r2.ok is False
        assert r2.audit["persistence_fail"] is True
        assert r2.audit["consecutive_exceedances_after"] == 2

    def test_gain_below_threshold_is_warn_not_fail(self, fr_window, make_gate_engine):
        ge = make_gate_engine(
            fr_window,
            min_evidence=0,
            policy="persistence",
            persistence_k=1,
            gain_thresh=0.0,  # gain warning if gain_bits < 0.0
            eps_sens=0.0,
        )

        past = {"test_metric": np.full(200, 0.5, dtype=float)}
        recent = {"test_metric": np.full(200, 0.5, dtype=float)}
        counts = {"test_metric": 200}

        r = ge.check(past, recent, counts, gain_bits=-0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r.audit["evaluated"] is True
        assert r.audit["dw_exceeds_threshold"] is False
        assert r.audit["persistence_fail"] is False
        assert r.audit["gain_below_threshold"] is True
        assert r.ok is True
        assert r.warn is True
        assert r.audit.get("reason") == "gain_below_threshold"

    def test_drift_fail_still_fails_even_if_gain_below_threshold(self, fr_window, make_gate_engine):
        ge = make_gate_engine(
            fr_window,
            min_evidence=0,
            policy="persistence",
            persistence_k=1,
            gain_thresh=0.0,  # gain warning if gain_bits < 0.0
            eps_sens=0.0,
        )

        past = {"test_metric": np.full(200, 0.1, dtype=float)}
        recent = {"test_metric": np.full(200, 0.9, dtype=float)}
        counts = {"test_metric": 200}

        r = ge.check(past, recent, counts, gain_bits=-0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r.audit["evaluated"] is True
        assert r.audit["dw_exceeds_threshold"] is True
        assert r.audit["persistence_fail"] is True
        assert r.audit["gain_below_threshold"] is True
        assert r.ok is False
        assert r.warn is False

    def test_gain_below_threshold_does_not_reset_persistence_counter(self, fr_window, make_gate_engine):
        ge = make_gate_engine(
            fr_window,
            min_evidence=0,
            policy="persistence",
            persistence_k=3,
            gain_thresh=0.0,  # gain warning if gain_bits < 0.0
            eps_sens=0.0,
        )

        past = {"test_metric": np.full(200, 0.1, dtype=float)}
        same_recent = {"test_metric": np.full(200, 0.1, dtype=float)}
        counts = {"test_metric": 200}
        drift_recent = {"test_metric": np.full(200, 0.9, dtype=float)}

        r1 = ge.check(past, drift_recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r1.audit["evaluated"] is True
        assert r1.audit["dw_exceeds_threshold"] is True
        assert r1.audit["consecutive_exceedances_after"] == 1

        # Gain-only warning (no drift) must NOT reset persistence counter.
        r2 = ge.check(past, same_recent, counts, gain_bits=-0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r2.audit["evaluated"] is True
        assert r2.audit["dw_exceeds_threshold"] is False
        assert r2.audit["gain_below_threshold"] is True
        assert r2.ok is True
        assert r2.warn is True
        assert r2.audit["persistence_fail"] is False
        assert r2.audit["consecutive_exceedances_after"] == 1

        r3 = ge.check(past, drift_recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r3.audit["evaluated"] is True
        assert r3.audit["dw_exceeds_threshold"] is True
        assert r3.audit["persistence_fail"] is False
        assert r3.ok is True
        assert r3.warn is True
        assert r3.audit["consecutive_exceedances_after"] == 2

        r4 = ge.check(past, drift_recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r4.audit["evaluated"] is True
        assert r4.audit["dw_exceeds_threshold"] is True
        assert r4.audit["persistence_fail"] is True
        assert r4.ok is False
        assert r4.warn is False
        assert r4.audit["consecutive_exceedances_after"] == 3

    def test_gain_nan_does_not_warn(self, fr_window, make_gate_engine):
        ge = make_gate_engine(
            fr_window,
            min_evidence=0,
            policy="persistence",
            persistence_k=2,
            gain_thresh=0.0,
            eps_sens=0.0,
        )

        past = {"test_metric": np.full(200, 0.5, dtype=float)}
        recent = {"test_metric": np.full(200, 0.5, dtype=float)}
        counts = {"test_metric": 200}

        r = ge.check(past, recent, counts, gain_bits=np.nan, kappa_sens=0.0, eps_stat_value=0.0)
        assert r.audit["evaluated"] is True
        assert r.audit["dw_exceeds_threshold"] is False
        assert r.audit["gain_below_threshold"] is False
        assert r.ok is True
        assert r.warn is False

    def test_gain_below_threshold_not_evaluated_does_not_warn(self, fr_window, make_gate_engine):
        ge = make_gate_engine(
            fr_window,
            min_evidence=999,
            policy="persistence",
            persistence_k=2,
            gain_thresh=0.0,
            eps_sens=0.0,
        )

        past = {"test_metric": np.full(200, 0.5, dtype=float)}
        recent = {"test_metric": np.full(200, 0.5, dtype=float)}
        counts = {"test_metric": 200}

        r = ge.check(past, recent, counts, gain_bits=-0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r.audit["evaluated"] is False
        assert r.audit["gain_below_threshold"] is True
        assert r.ok is True
        assert r.warn is False
        assert r.audit.get("reason") != "gain_below_threshold"

    def test_either_and_conjunction_gain_semantics_unchanged(self, fr_window, make_gate_engine):
        past = {"test_metric": np.full(200, 0.5, dtype=float)}
        recent = {"test_metric": np.full(200, 0.5, dtype=float)}
        counts = {"test_metric": 200}

        ge_either = make_gate_engine(fr_window, min_evidence=0, policy="either", gain_thresh=0.0, eps_sens=0.0)
        r_either = ge_either.check(past, recent, counts, gain_bits=-0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r_either.audit["dw_exceeds_threshold"] is False
        assert r_either.ok is False
        assert r_either.warn is False

        ge_conj = make_gate_engine(fr_window, min_evidence=0, policy="conjunction", gain_thresh=0.0, eps_sens=0.0)
        r_conj = ge_conj.check(past, recent, counts, gain_bits=-0.1, kappa_sens=0.0, eps_stat_value=0.0)
        assert r_conj.audit["dw_exceeds_threshold"] is False
        assert r_conj.ok is True
        assert r_conj.warn is False

    def test_multi_metric_consensus_filter_blocks_single_metric_drift(
        self, make_window_decl, make_fr_window, make_gate_engine
    ):
        wd = make_window_decl(
            ["m1", "m2"],
            epsilon=0.01,
            weights={"m1": 0.5, "m2": 0.5},
            bins=10,
            cal_ranges={"m1": (0.0, 1.0), "m2": (0.0, 1.0)},
        )
        if hasattr(wd, "normalize_weights"):
            wd.normalize_weights()

        fr = make_fr_window(wd)
        ge = make_gate_engine(
            fr,
            min_evidence=0,
            policy="either",
            gain_thresh=0.0,
            eps_sens=0.0,
            min_metrics_exceeding=2,
        )

        past = {"m1": np.full(300, 0.1), "m2": np.full(300, 0.5)}
        recent = {"m1": np.full(300, 0.9), "m2": np.full(300, 0.5)}
        counts = {"m1": 300, "m2": 300}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        a = r.audit
        assert a["evaluated"] is True

        assert a["dw_exceeds_threshold_raw"] is True
        assert a["ok_stab_raw"] is False
        assert a["min_metrics_exceeding_effective"] == 2
        assert a["n_metrics_exceeding"] == 1
        assert a["metrics_exceeding"] == ["m1"]

        assert a["multi_metric_filtered"] is True
        assert a["dw_exceeds_threshold"] is False
        assert a["ok_stab"] is True

        assert np.isfinite(a["margin_change_eff"])
        assert np.isclose(a["margin_change_eff"], 0.0, atol=1e-12)

    def test_audit_schema_insufficient_evidence_hard_contract(self, fr_window, make_gate_engine):
        min_evidence = 16
        ge = make_gate_engine(fr_window, min_evidence=min_evidence)

        past = {"test_metric": np.full(10, 0.5, dtype=float)}
        recent = {"test_metric": np.full(10, 0.5, dtype=float)}
        counts = {"test_metric": 0}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        a = r.audit
        assert a["evaluated"] is False
        assert a["reason"] == "not_evaluated_insufficient_evidence"
        assert a["min_evidence"] == min_evidence

        required = [
            "evaluated",
            "reason",
            "evidence_total",
            "total_evidence",
            "min_evidence",
            "policy",
            "consecutive_exceedances",
            "consecutive_exceedances_before",
            "consecutive_exceedances_after",
            "persistence_k",
            "persistence_fail",
            "gain_bits",
            "gain_thr",
            "gain_evaluated",
            "ok_gain",
            "gain_below_threshold",
            "eps",
            "eps_sens",
            "kappa_sens",
            "kappa_checked",
            "counts_by_metric",
            "worst_DW",
            "eps_stat",
            "eps_eff",
            "per_metric_tv",
            "per_metric_n",
            "drifts",
            "dw_exceeds_threshold",
            "ok_stab",
            "min_metrics_exceeding",
            "min_metrics_exceeding_effective",
            "n_metrics_exceeding",
            "metrics_exceeding",
            "multi_metric_filtered",
            "dw_exceeds_threshold_raw",
            "ok_stab_raw",
            "trend_x",
            "trend_x_source",
            "trend_n",
            "check_idx",
            "margin_change_raw",
            "margin_change_eff",
            "margin_change_slope_eff",
            "margin_change",
            "margin_change_slope",
            "margin_change_rel_raw",
            "margin_change_rel_eff",
            "regime_state",
        ]
        for k in required:
            assert k in a, f"missing audit key: {k}"
            assert a[k] is not None, f"audit key is None: {k}"

        assert isinstance(a["evaluated"], bool)
        assert isinstance(a["reason"], str)
        assert isinstance(a["policy"], str)
        assert isinstance(a["counts_by_metric"], dict)
        assert _is_intlike(a["evidence_total"])
        assert _is_intlike(a["min_evidence"])
        assert _is_intlike(a["persistence_k"])
        assert isinstance(a["metrics_exceeding"], list)
        assert _is_floatlike(a["gain_bits"])
        assert _is_floatlike(a["gain_thr"])
        assert _is_floatlike(a["eps"])
        assert _is_floatlike(a["worst_DW"])
        assert isinstance(a["regime_state"], str)

    def test_audit_schema_no_finite_metrics_hard_contract(self, fr_window, make_gate_engine):
        ge = make_gate_engine(fr_window, min_evidence=0)

        past = {"test_metric": np.array([np.nan], dtype=float)}
        recent = {"test_metric": np.array([0.2, 0.3], dtype=float)}
        counts = {"test_metric": 100}

        r = ge.check(past, recent, counts, gain_bits=0.1, kappa_sens=0.0, eps_stat_value=0.0)
        a = r.audit
        assert a["evaluated"] is False
        assert a["reason"] == "not_evaluated_no_finite_metrics"
        assert np.isnan(a["worst_DW"])

        required = [
            "evaluated",
            "reason",
            "policy",
            "consecutive_exceedances",
            "consecutive_exceedances_before",
            "consecutive_exceedances_after",
            "persistence_k",
            "persistence_fail",
            "gain_bits",
            "gain_thr",
            "gain_evaluated",
            "ok_gain",
            "gain_below_threshold",
            "worst_DW",
            "eps",
            "eps_sens",
            "eps_stat",
            "eps_eff",
            "kappa_sens",
            "kappa_checked",
            "counts_by_metric",
            "evidence_total",
            "total_evidence",
            "min_evidence",
            "eps_aggregation",
            "per_metric_tv",
            "per_metric_n",
            "drifts",
            "dw_exceeds_threshold",
            "dw_exceeds_threshold_raw",
            "ok_stab",
            "ok_stab_raw",
            "min_metrics_exceeding",
            "min_metrics_exceeding_effective",
            "n_metrics_exceeding",
            "metrics_exceeding",
            "multi_metric_filtered",
            "trend_x",
            "trend_x_source",
            "trend_n",
            "check_idx",
            "margin_change_raw",
            "margin_change_eff",
            "margin_change_slope_eff",
            "margin_change",
            "margin_change_slope",
            "margin_change_rel_raw",
            "margin_change_rel_eff",
            "regime_state",
        ]
        for k in required:
            assert k in a, f"missing audit key: {k}"
            assert a[k] is not None, f"audit key is None: {k}"

        assert isinstance(a["evaluated"], bool)
        assert isinstance(a["reason"], str)
        assert isinstance(a["policy"], str)
        assert isinstance(a["counts_by_metric"], dict)
        assert isinstance(a["eps_aggregation"], str)
        assert isinstance(a["metrics_exceeding"], list)
        assert isinstance(a["regime_state"], str)

    def test_audit_schema_evaluated_core_subset(self, fr_window, make_gate_engine):
        # Non-brittle evaluated-path schema: small stable subset + types.
        ge = make_gate_engine(fr_window, min_evidence=0, policy="either", gain_thresh=0.0, eps_sens=0.0)

        past = {"test_metric": np.full(50, 0.5, dtype=float)}
        recent = {"test_metric": np.full(50, 0.5, dtype=float)}
        counts = {"test_metric": 50}

        r = ge.check(past, recent, counts, gain_bits=0.0, kappa_sens=np.nan, eps_stat_value=0.0)
        a = r.audit
        assert a["evaluated"] is True

        core = [
            "evaluated",
            "policy",
            "worst_DW",
            "eps_eff",
            "counts_by_metric",
            "evidence_total",
            "min_evidence",
            "dw_exceeds_threshold",
            "persistence_fail",
            "trend_x",
            "trend_x_source",
            "trend_n",
            "check_idx",
            "margin_change_eff",
            "margin_change_slope_eff",
            "regime_state",
        ]
        for k in core:
            assert k in a, f"missing evaluated-core audit key: {k}"
            assert a[k] is not None, f"evaluated-core audit key is None: {k}"

        assert isinstance(a["evaluated"], bool)
        assert isinstance(a["policy"], str)
        assert isinstance(a["counts_by_metric"], dict)
        assert _is_intlike(a["evidence_total"])
        assert _is_intlike(a["min_evidence"])
        assert _is_floatlike(a["worst_DW"])
        assert _is_floatlike(a["eps_eff"])
        assert np.isfinite(float(a["eps_eff"]))  # expected finite for this test input
        assert isinstance(a["regime_state"], str)
