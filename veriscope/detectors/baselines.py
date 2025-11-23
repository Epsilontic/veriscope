# veriscope/detectors/baselines.py
from __future__ import annotations

"""
Public baseline detector facade.

Phase-1 boundary:
- Never import from legacy_cli_refactor.py.
- Re-export stable APIs from legacy implementations.
"""

from veriscope.runners.legacy.detectors.baselines import (  # noqa: F401
    SCHEDULED_METRICS,
    _prep_series_for_ph,
    robust_z_series,
    ph_window_sparse,
    cusum_one_sided,
    newma_warn_epoch,
    calibrate_ph_directions,
)

__all__ = [
    "SCHEDULED_METRICS",
    "_prep_series_for_ph",
    "robust_z_series",
    "ph_window_sparse",
    "cusum_one_sided",
    "newma_warn_epoch",
    "calibrate_ph_directions",
]