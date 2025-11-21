# veriscope/pipeline.py
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from veriscope.runners.legacy_cli_refactor import (  # type: ignore
    run_one as _legacy_run_one,
    run_sweep as _legacy_run_sweep,
    evaluate as _legacy_evaluate,
)

def run_one(seed: int, tag: str, monitor_ds: Any, factor: Dict[str, Any]) -> pd.DataFrame:
    return _legacy_run_one(seed=seed, tag=tag, monitor_ds=monitor_ds, factor=factor)

def run_sweep(tag: str) -> Optional[pd.DataFrame]:
    return _legacy_run_sweep(tag=tag)

def evaluate(df_all: pd.DataFrame, tag: str) -> None:
    _legacy_evaluate(df_all, tag=tag)

__all__ = ["run_one", "run_sweep", "evaluate"]