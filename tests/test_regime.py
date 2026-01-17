# tests/test_regime.py  
from __future__ import annotations  
  
import numpy as np  
import pytest  
  
from veriscope.core.regime import RegimeAnchoredGateEngine, RegimeConfig  
  
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
