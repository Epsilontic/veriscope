# tests/test_schema.py
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestSchemaResultIntegrity:
    def test_nan_arrays_both_empty_after_filter_are_unevaluated(self, fr_window, make_gate_engine):
        ge = make_gate_engine(fr_window, min_evidence=0, gain_thresh=0.0, eps_sens=0.0)

        r = ge.check(
            past={"test_metric": np.full(10, np.nan)},
            recent={"test_metric": np.full(10, np.nan)},
            counts_by_metric={"test_metric": 0},
            gain_bits=0.1,
            kappa_sens=np.nan,
            eps_stat_value=0.0,
        )
        assert r.audit["evaluated"] is False
        assert np.isnan(r.audit["worst_DW"])
        assert r.ok is True
