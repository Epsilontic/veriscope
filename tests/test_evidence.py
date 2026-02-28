from __future__ import annotations

import numpy as np
import pytest

from veriscope.core.evidence import extract_metric_array

pytestmark = pytest.mark.unit


def test_extract_metric_array_strict_raises_on_non_coercible_value() -> None:
    rows = [{"iter": 7, "loss_delta_z": "not-a-number"}]
    with pytest.raises(ValueError, match="Non-coercible metric value"):
        extract_metric_array(rows, "loss_delta_z", strict=True)


def test_extract_metric_array_default_mode_drops_non_finite_or_invalid() -> None:
    rows = [
        {"iter": 1, "loss_delta_z": "not-a-number"},
        {"iter": 2, "loss_delta_z": float("nan")},
        {"iter": 3, "loss_delta_z": 1.5},
    ]
    arr = extract_metric_array(rows, "loss_delta_z")
    assert np.array_equal(arr, np.array([1.5], dtype=float))
