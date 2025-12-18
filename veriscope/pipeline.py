# veriscope/pipeline.py
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def _legacy_mod():
    """Lazy-load legacy CLI runner without a direct import statement.

    This avoids top-level dependency/circular-import issues during refactor.
    """
    return importlib.import_module("veriscope.runners.legacy_cli_refactor")


def run_one(seed: int, tag: str, monitor_ds: Any, factor: Dict[str, Any]) -> pd.DataFrame:
    legacy = _legacy_mod()
    return legacy.run_one(seed=seed, tag=tag, monitor_ds=monitor_ds, factor=factor)


def run_sweep(tag: str) -> Optional[pd.DataFrame]:
    legacy = _legacy_mod()

    # Coverage preflight: if the pipeline path is used, ensure we still surface
    # gate evaluability warnings (and fail-fast when policy requires) even if
    # the legacy runner is later refactored.
    cfg = getattr(legacy, "CFG", None)
    if isinstance(cfg, dict):
        # Determine policy once (and use it to control whether we swallow reconcile failures).
        require_gate = False
        if hasattr(legacy, "require_gate_eval"):
            try:
                require_gate = bool(legacy.require_gate_eval(cfg))
            except Exception:
                require_gate = False

        # Reconcile/install preflight. If fail-fast is enabled, do NOT swallow failures,
        # to avoid validating/enforcing against a stale CFG.
        if require_gate:
            if hasattr(legacy, "reconcile_cfg_inplace"):
                legacy.reconcile_cfg_inplace(cfg, stage="pipeline_preflight")
            if hasattr(legacy, "runtime") and hasattr(legacy, "OUTDIR") and hasattr(legacy, "BUDGET"):
                legacy.runtime.install_runtime(cfg=cfg, outdir=legacy.OUTDIR, budget=legacy.BUDGET)
        else:
            try:
                if hasattr(legacy, "reconcile_cfg_inplace"):
                    legacy.reconcile_cfg_inplace(cfg, stage="pipeline_preflight")
                if hasattr(legacy, "runtime") and hasattr(legacy, "OUTDIR") and hasattr(legacy, "BUDGET"):
                    legacy.runtime.install_runtime(cfg=cfg, outdir=legacy.OUTDIR, budget=legacy.BUDGET)
            except Exception:
                pass

        # Gate evaluability coverage (warn always; fail-fast when policy requires)
        if hasattr(legacy, "require_gate_eval_or_die") and hasattr(legacy, "validate_gate_evaluability"):
            if require_gate:
                legacy.require_gate_eval_or_die(cfg, context="pipeline:run_sweep:post_reconcile")
            else:
                legacy.validate_gate_evaluability(cfg, context="pipeline:run_sweep:post_reconcile")

    return legacy.run_sweep(tag=tag)


def evaluate(df_all: pd.DataFrame, tag: str) -> None:
    legacy = _legacy_mod()
    legacy.evaluate(df_all, tag=tag)


__all__ = ["run_one", "run_sweep", "evaluate"]
