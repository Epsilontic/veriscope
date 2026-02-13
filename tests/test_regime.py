# tests/test_regime.py
from __future__ import annotations

import numpy as np
import pytest

from veriscope.core.regime import RegimeAnchoredGateEngine, RegimeConfig, RegimeReference

pytestmark = pytest.mark.unit


class TestRegimeGatingSemantics:
    def test_regime_disabled_functional_fields(self, gate_engine, fr_window):
        rag = RegimeAnchoredGateEngine(
            base_engine=gate_engine,
            fr_win=fr_window,
            config=RegimeConfig(enabled=False),
            gate_warmup=0,
            gate_window=10,
        )
        r = rag.check(
            past={"test_metric": np.full(200, 0.1)},
            recent={"test_metric": np.full(200, 0.2)},
            counts_by_metric={"test_metric": 200},
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=100,
        )
        assert r.audit["regime_enabled"] is False
        assert r.audit["regime_has_reference"] is False
        # Schema stability: keep placeholder regime diagnostics present when disabled.
        for key in (
            "regime_margin_raw",
            "regime_margin_eff",
            "regime_margin_slope_eff",
            "regime_margin_rel_raw",
            "regime_margin_rel_eff",
            "regime_trend_x",
            "regime_trend_x_source",
            "regime_trend_n",
            "regime_check_idx",
            "regime_worst_DW",
            "regime_eps_eff",
            "regime_eps_stat",
            "regime_per_metric",
            "regime_per_metric_n",
            "regime_counts",
            "ref_ready",
            "ref_windows_built",
            "regime_ref_ready",
            "regime_ref_windows_built",
        ):
            assert key in r.audit
        assert r.audit.get("ref_ready") is False
        assert isinstance(r.audit.get("ref_windows_built"), int)
        assert r.audit.get("regime_ref_ready") is None
        assert r.audit.get("regime_ref_windows_built") is None
        if "regime_state" in r.audit:
            assert isinstance(r.audit["regime_state"], str)

    def test_regime_enabled_no_reference_outside_build_window(self, gate_engine, fr_window):
        cfg = RegimeConfig(enabled=True, reference_build_min_iter=50, reference_build_max_iter=100)
        rag = RegimeAnchoredGateEngine(
            base_engine=gate_engine,
            fr_win=fr_window,
            config=cfg,
            gate_warmup=0,
            gate_window=10,
        )
        r = rag.check(
            past={"test_metric": np.full(200, 0.1)},
            recent={"test_metric": np.full(200, 0.2)},
            counts_by_metric={"test_metric": 200},
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=120,
        )
        assert r.audit["regime_enabled"] is True
        assert r.audit["regime_has_reference"] is False
        if "regime_state" in r.audit:
            assert isinstance(r.audit["regime_state"], str)

    def test_regime_bootstrap_warn_and_builds_reference(self, gate_engine, fr_window):
        cfg = RegimeConfig(
            enabled=True,
            reference_build_min_iter=0,
            reference_build_max_iter=100,
            reference_build_max_dw=2.0,
            reference_build_min_gain=-1.0,
            min_evidence_per_metric=1,
            min_windows_for_reference=2,
        )
        rag = RegimeAnchoredGateEngine(
            base_engine=gate_engine,
            fr_win=fr_window,
            config=cfg,
            gate_warmup=0,
            gate_window=10,
        )

        past = {"test_metric": np.full(50, 0.0)}
        recent = {"test_metric": np.full(50, 1.0)}
        counts = {"test_metric": 50}

        r1 = rag.check(
            past=past,
            recent=recent,
            counts_by_metric=counts,
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=10,
        )
        assert r1.warn is True
        assert r1.ok is True
        assert r1.audit.get("reason") == "change_warn_pending"
        assert r1.audit.get("regime_ref_candidate_windows_seen") == 1
        assert r1.audit.get("regime_ref_windows_built", 0) >= 1
        assert r1.audit.get("regime_has_reference") is False
        assert "regime_build_window" in r1.audit
        assert r1.audit.get("regime_ref_last_reject_reason", "").startswith("insufficient_complete_windows")

        r2 = rag.check(
            past=past,
            recent=recent,
            counts_by_metric=counts,
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=20,
        )
        assert r2.warn is True
        assert r2.ok is True
        assert r2.audit.get("regime_ref_candidate_windows_seen") == 2
        assert r2.audit.get("regime_ref_windows_built", 0) >= 2
        assert r2.audit.get("regime_has_reference") is True
        assert r2.audit.get("regime_ref_last_reject_reason") in (None, "")

    def test_regime_emits_clip_diagnostics_on_saturation_even_when_dw_small(self, gate_engine, fr_window):
        cfg = RegimeConfig(
            enabled=True,
            reference_build_min_iter=0,
            reference_build_max_iter=100,
            min_evidence_per_metric=1,
            min_windows_for_reference=1,
        )
        rag = RegimeAnchoredGateEngine(
            base_engine=gate_engine,
            fr_win=fr_window,
            config=cfg,
            gate_warmup=0,
            gate_window=10,
        )

        rag._ref = RegimeReference(  # type: ignore[attr-defined]
            metrics={"test_metric": np.full(200, 2.0)},
            counts={"test_metric": 200},
            established_at=0,
            n_samples_per_metric={"test_metric": 200},
        )

        r = rag.check(
            past={"test_metric": np.full(200, 2.0)},
            recent={"test_metric": np.full(200, 2.5)},
            counts_by_metric={"test_metric": 200},
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=200,
        )

        assert r.audit.get("regime_check_ran") is True
        assert r.audit.get("regime_clip_diag_threshold") == 0.10
        clip_recent = r.audit.get("regime_clip_diag_recent") or {}
        assert "test_metric" in clip_recent

    def test_persistence_stability_treats_regime_only_fail_as_warn(self, make_gate_engine, fr_window):
        base_engine = make_gate_engine(
            fr_window,
            min_evidence=1,
            policy="persistence_stability",
            persistence_k=1,
        )
        cfg = RegimeConfig(
            enabled=True,
            reference_build_min_iter=0,
            reference_build_max_iter=0,
            min_evidence_per_metric=1,
            min_windows_for_reference=1,
        )
        rag = RegimeAnchoredGateEngine(
            base_engine=base_engine,
            fr_win=fr_window,
            config=cfg,
            gate_warmup=0,
            gate_window=10,
        )

        rag._ref = RegimeReference(  # type: ignore[attr-defined]
            metrics={"test_metric": np.full(64, 1.0)},
            counts={"test_metric": 64},
            established_at=0,
            n_samples_per_metric={"test_metric": 64},
        )

        result = rag.check(
            past={"test_metric": np.full(64, 0.0)},
            recent={"test_metric": np.full(64, 0.0)},
            counts_by_metric={"test_metric": 64},
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=200,
        )

        assert result.ok is True
        assert result.warn is True
        assert result.audit.get("change_ok") is True
        assert result.audit.get("regime_ok") is False
        assert result.audit.get("reason") == "regime_fail_suppressed"
        assert result.audit.get("regime_fail_suppressed") is True
        assert result.audit.get("ref_ready") is True
        assert result.audit.get("regime_ref_ready") is True

    def test_persistence_stability_keeps_fail_when_change_channel_fails(self, make_gate_engine, fr_window):
        base_engine = make_gate_engine(
            fr_window,
            min_evidence=1,
            policy="persistence_stability",
            persistence_k=1,
        )
        rag = RegimeAnchoredGateEngine(
            base_engine=base_engine,
            fr_win=fr_window,
            config=RegimeConfig(enabled=True, reference_build_min_iter=0, reference_build_max_iter=0),
            gate_warmup=0,
            gate_window=10,
        )

        rag._ref = RegimeReference(  # type: ignore[attr-defined]
            metrics={"test_metric": np.full(64, 0.0)},
            counts={"test_metric": 64},
            established_at=0,
            n_samples_per_metric={"test_metric": 64},
        )

        result = rag.check(
            past={"test_metric": np.full(64, 0.0)},
            recent={"test_metric": np.full(64, 1.0)},
            counts_by_metric={"test_metric": 64},
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.01,
            iter_num=200,
        )

        assert result.ok is False
        assert result.warn is False
        assert result.audit.get("change_ok") is False
        assert result.audit.get("regime_ok") is False
        assert result.audit.get("regime_fail_suppressed") is not True

    def test_shadow_mode_freezes_regime_persistence_counter(self, make_gate_engine, fr_window):
        base_engine = make_gate_engine(
            fr_window,
            min_evidence=1,
            policy="persistence",
            persistence_k=2,
            gain_thresh=0.0,
            eps_sens=0.0,
        )
        rag = RegimeAnchoredGateEngine(
            base_engine=base_engine,
            fr_win=fr_window,
            config=RegimeConfig(
                enabled=True,
                shadow_mode=True,
                reference_build_min_iter=0,
                reference_build_max_iter=0,
                min_evidence_per_metric=1,
                min_windows_for_reference=1,
            ),
            gate_warmup=0,
            gate_window=10,
        )

        rag._ref = RegimeReference(  # type: ignore[attr-defined]
            metrics={"test_metric": np.full(64, 0.0)},
            counts={"test_metric": 64},
            established_at=0,
            n_samples_per_metric={"test_metric": 64},
        )

        baseline = rag.regime_engine.save_persistence_state()["consecutive_exceedances"]

        for i in range(3):
            r_shadow = rag.check(
                past={"test_metric": np.full(64, 1.0)},
                recent={"test_metric": np.full(64, 1.0)},
                counts_by_metric={"test_metric": 64},
                gain_bits=0.1,
                kappa_sens=0.0,
                eps_stat_value=0.0,
                iter_num=200 + i,
            )
            assert r_shadow.audit.get("regime_check_ran") is True
            assert r_shadow.audit.get("regime_shadow_mode") is True

        after_shadow = rag.regime_engine.save_persistence_state()["consecutive_exceedances"]
        assert after_shadow == baseline

        rag.config.shadow_mode = False
        r_live = rag.check(
            past={"test_metric": np.full(64, 1.0)},
            recent={"test_metric": np.full(64, 1.0)},
            counts_by_metric={"test_metric": 64},
            gain_bits=0.1,
            kappa_sens=0.0,
            eps_stat_value=0.0,
            iter_num=300,
        )
        after_live = rag.regime_engine.save_persistence_state()["consecutive_exceedances"]
        assert r_live.ok is True
        assert r_live.audit.get("regime_ok") is True
        assert r_live.audit.get("regime_shadow_mode") is False
        assert after_live == baseline + 1
