# veriscope/detectors/learned.py
from __future__ import annotations

"""Public learned detector facade.

Phase-1 boundary:
- Never import from legacy_cli_refactor.py.
- Re-export only stable APIs from the legacy learned-detector implementation.

Note: The learned-detector *implementation* lives in:
  - veriscope.runners.legacy.detectors.learned
This facade intentionally keeps a small surface area to reduce refactor drift.
"""

from typing import Any, Dict, List, Optional

# Default learned-detector feature set used by legacy runs.
DET_FEATURES: List[str] = [
    "cos_disp",
    "var_out_k",
    "ftle",
    "ftle_lowent",
    "slope_eff_dim",
    "slope_var_out_k",
    "reg_eff_dim",
    "reg_var_out_k",
    "corr_effdim_varoutk",
]


def default_features(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the default list of learned-detector features.

    `cfg` is accepted for forward compatibility but is currently unused.
    """
    _ = cfg
    return list(DET_FEATURES)


# Stable entrypoints only.
from veriscope.runners.legacy.detectors.learned import (  # noqa: F401,E402
    map_threshold_to_gated_fp,
)

__all__ = [
    "DET_FEATURES",
    "default_features",
    "map_threshold_to_gated_fp",
]
