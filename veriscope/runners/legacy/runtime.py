# veriscope/runners/legacy/runtime.py
"""
Single source of truth for global configuration and runtime state.
Provides install_runtime() to set CFG, OUTDIR, and BUDGET, and getters to access them.

This module centralizes globals to reduce cyclic-import and multi-CFG bugs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type checking; avoids runtime cycles.
    from .budget import BudgetLedger

# Global holders (private; access via getters)
_CFG: Optional[Mapping[str, Any]] = None
_OUTDIR: Optional[Path] = None
_BUDGET: Optional["BudgetLedger"] = None


def install_runtime(
    cfg: Mapping[str, Any],
    outdir: Optional[Path] = None,
    budget: Optional["BudgetLedger"] = None,
) -> None:
    """
    Install the global CFG, OUTDIR, and BUDGET.

    - CFG is stored as a shallow copy of the provided mapping.
    - OUTDIR defaults to $SCAR_OUTDIR or "." if not provided; it is always created.
    - BUDGET may be left as None if no budget ledger is in use.
    """
    global _CFG, _OUTDIR, _BUDGET

    # Shallow copy to avoid accidental external mutation.
    _CFG = dict(cfg) if not isinstance(cfg, dict) else cfg.copy()

    if outdir is None:
        outdir = Path(os.environ.get("SCAR_OUTDIR", "."))
    outdir.mkdir(parents=True, exist_ok=True)
    _OUTDIR = outdir

    _BUDGET = budget


def get_cfg() -> Mapping[str, Any]:
    """Return the installed CFG; raises if install_runtime() has not been called."""
    if _CFG is None:
        raise RuntimeError("CFG not installed. Call install_runtime() first.")
    return _CFG


def get_outdir() -> Path:
    """Return the installed OUTDIR; raises if install_runtime() has not been called."""
    if _OUTDIR is None:
        raise RuntimeError("OUTDIR not installed. Call install_runtime() first.")
    return _OUTDIR


def get_budget() -> Optional["BudgetLedger"]:
    """
    Return the installed BUDGET (BudgetLedger instance) or None if not installed.

    Call sites should explicitly handle the None case if running without a budget.
    """
    return _BUDGET