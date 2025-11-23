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
    return legacy.run_sweep(tag=tag)


def evaluate(df_all: pd.DataFrame, tag: str) -> None:
    legacy = _legacy_mod()
    legacy.evaluate(df_all, tag=tag)

__all__ = ["run_one", "run_sweep", "evaluate"]
