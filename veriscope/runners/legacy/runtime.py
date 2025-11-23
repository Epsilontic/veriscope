# veriscope/runners/legacy/runtime.py
"""
Single source of truth for global configuration and runtime state.
Provides install_runtime() to set CFG, OUTDIR, and BUDGET, and getters to access them.

This module centralizes globals to reduce cyclic-import and multi-CFG bugs.

Refactor note (Phase 1):
- CFG must be a single live dict exported at module scope.
- install_runtime() must MUTATE that dict in place (no rebinding / copying),
  so any `from runtime import CFG` import sees updates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type checking; avoids runtime cycles.
    from .budget import BudgetLedger

# ---------------------------------------------------------------------
# Public live globals (modules may import these by reference)
# ---------------------------------------------------------------------

CFG: Dict[str, Any] = {}
OUTDIR: Path = Path(os.environ.get("SCAR_OUTDIR", "."))
BUDGET: Optional["BudgetLedger"] = None

# ---------------------------------------------------------------------
# Back-compat private aliases (some legacy code may still touch these)
# ---------------------------------------------------------------------

_CFG: Optional[Mapping[str, Any]] = CFG
_OUTDIR: Optional[Path] = OUTDIR
_BUDGET: Optional["BudgetLedger"] = BUDGET


def install_runtime(
    cfg: Mapping[str, Any],
    outdir: Optional[Path] = None,
    budget: Optional["BudgetLedger"] = None,
) -> None:
    """
    Install the global CFG, OUTDIR, and BUDGET.

    IMPORTANT:
    - CFG is mutated in place (clear/update). Do NOT rebind it.
      This preserves references for `from runtime import CFG`.

    - OUTDIR defaults to $SCAR_OUTDIR or "." if not provided; it is always created.
    - BUDGET may be left as None if no budget ledger is in use.
    """
    global CFG, OUTDIR, BUDGET, _CFG, _OUTDIR, _BUDGET

    # Mutate CFG in place so all importers see updates
    try:
        CFG.clear()
        # Mapping -> dict to avoid surprises from non-dict mappings
        CFG.update(dict(cfg) if cfg is not None else {})
    except Exception:
        # Last-resort fallback: rebind, but keep alias consistent
        CFG = dict(cfg) if cfg is not None else {}

    if outdir is None:
        outdir = Path(os.environ.get("SCAR_OUTDIR", "."))
    outdir.mkdir(parents=True, exist_ok=True)
    OUTDIR = outdir

    BUDGET = budget

    # Keep private aliases pointing at live globals
    _CFG = CFG
    _OUTDIR = OUTDIR
    _BUDGET = BUDGET


def get_cfg() -> Mapping[str, Any]:
    """Return the installed CFG (live mapping)."""
    return CFG


def get_outdir() -> Path:
    """Return the installed OUTDIR; defaults to env/ '.' if not installed yet."""
    return OUTDIR


def get_budget() -> Optional["BudgetLedger"]:
    """
    Return the installed BUDGET (BudgetLedger instance) or None if not installed.

    Call sites should explicitly handle the None case if running without a budget.
    """
    return BUDGET


__all__ = ["CFG", "OUTDIR", "BUDGET", "install_runtime", "get_cfg", "get_outdir", "get_budget"]
