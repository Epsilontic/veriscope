# tests/test_window_transport.py  
from __future__ import annotations  
  
import numpy as np  
import pytest  
  
from veriscope.core.transport import assert_naturality  
  
pytestmark = pytest.mark.unit  
  
  
class TestWindowDeclDeclTransport:  
    def test_window_decl_normalization(self, make_window_decl):  
        wd = make_window_decl(["m1", "m2"], weights={"m1": 10.0, "m2": 20.0}, bins=16)  
        wd.normalize_weights()  
        s = sum(abs(wd.weights[m]) for m in wd.metrics)  
        assert np.isclose(s, 1.0, atol=1e-12)  
  
    def test_decl_transport_apply(self, decl_transport):  
        x = np.array([0.3, 0.5, 0.7])  
        y = decl_transport.apply("test_metric", x)  
        np.testing.assert_allclose(y, x, atol=1e-12)  
  
        x_out = np.array([-0.1, 0.5, 1.1])  
        y_out = decl_transport.apply("test_metric", x_out)  
        np.testing.assert_allclose(y_out, [0.0, 0.5, 1.0], atol=1e-12)  
  
        z = np.array([-5.0, 10.0])  
        w = decl_transport.apply("missing_metric", z)  
        np.testing.assert_allclose(w, [0.0, 1.0], atol=1e-12)  
  
    def test_decl_transport_naturality(self, decl_transport):  
        def identity(x):  
            return x  
  
        def center(x):  
            return x[1:-1] if len(x) > 2 else x  
  
        assert decl_transport.natural_with(identity)  
        assert decl_transport.natural_with(center)  
        assert_naturality(decl_transport, [identity, center])  
