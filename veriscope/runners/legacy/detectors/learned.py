# veriscope/detectors/learned.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Public feature list (same as legacy)
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
    _ = cfg
    return list(DET_FEATURES)

# Facade imports from legacy. If legacy changes location later, only this file updates.
from veriscope.runners.legacy_cli_refactor import (  # type: ignore
    _metrics_matrix_with_missing as metrics_matrix_with_missing,
    _fit_global_robust_norm_precollapse as fit_global_robust_norm_precollapse,
    _apply_global_norm_impute as apply_global_norm_impute,
    _train_logistic_ridge_balanced as train_logistic_ridge_balanced,
    _cv_grouped_fit as cv_grouped_fit,
    _oof_probs_for_params as oof_probs_for_params,
    map_threshold_to_gated_fp as map_threshold_to_gated_fp,
)

__all__ = [
    "DET_FEATURES",
    "default_features",
    "metrics_matrix_with_missing",
    "fit_global_robust_norm_precollapse",
    "apply_global_norm_impute",
    "train_logistic_ridge_balanced",
    "cv_grouped_fit",
    "oof_probs_for_params",
    "map_threshold_to_gated_fp",
]