# veriscope/runners/legacy/budget.py
"""
Finite-window heavy-metric budget primitives for legacy runners.

This module centralizes the definition of:
- WindowBudget: declarative per-window limits, in ms and call counts.
- BudgetLedger: runtime accumulator/guard for heavy-metric usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class WindowBudget:
    """
    Per-window heavy-metric budgets in milliseconds and call counts.

    Note:
      - sw2_ms / ripser_ms are cumulative per-run budgets enforced via BudgetLedger.allow().
      - Per-call ceilings are enforced locally inside the metric implementations
        (e.g., CFG['sw2_budget_ms'], CFG['ripser_budget_ms']).
    """
    sw2_ms: int = 200                # per-run cumulative budget for sliced W2 (ms)
    ripser_ms: int = 250             # per-run cumulative budget for H0 persistence (ms)
    total_heavy_ms: int = 180_000    # total wall-clock budget for all heavy metrics in a run
    sw2_calls: int = 1_000_000       # defensive ceiling; tuned by CFG
    ripser_calls: int = 1_000_000    # defensive ceiling; tuned by CFG


class BudgetLedger:
    """
    Simple accumulator for heavy-metric usage against a WindowBudget.

    Tracks:
      - used_ms["sw2"], used_ms["ripser"], used_ms["total"]
      - calls["sw2"], calls["ripser"]

    Public API:
      - reset()
      - allow(kind: str) -> bool
      - charge(kind: str, elapsed_ms: float) -> None
      - within_limits() -> bool
      - spent() -> Dict[str, float]
      - counts() -> Dict[str, int]
    """

    def __init__(self, lim: WindowBudget) -> None:
        self.lim = lim
        self.used_ms: Dict[str, float] = {
            "sw2": 0.0,
            "ripser": 0.0,
            "total": 0.0,
        }
        self.calls: Dict[str, int] = {
            "sw2": 0,
            "ripser": 0,
        }

    # --- lifecycle ---

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.used_ms.update({"sw2": 0.0, "ripser": 0.0, "total": 0.0})
        self.calls.update({"sw2": 0, "ripser": 0})

    # --- policy ---

    def allow(self, kind: str) -> bool:
        """
        Return True if another call of `kind` ('sw2' or 'ripser') is allowed,
        under both per-metric and total heavy-metric budgets.
        """
        # Global heavy-metric wall-clock guard
        if self.used_ms["total"] >= float(self.lim.total_heavy_ms):
            return False

        if kind == "sw2":
            return (
                self.used_ms["sw2"] < float(self.lim.sw2_ms)
                and self.calls["sw2"] < int(self.lim.sw2_calls)
            )

        if kind == "ripser":
            return (
                self.used_ms["ripser"] < float(self.lim.ripser_ms)
                and self.calls["ripser"] < int(self.lim.ripser_calls)
            )

        # Unknown kinds: only the total budget applies
        return True

    def charge(self, kind: str, elapsed_ms: float) -> None:
        """
        Charge elapsed_ms against the per-metric and total budgets.

        Safe to call with any float; negative values are clamped to 0.
        """
        ms = float(max(0.0, elapsed_ms))
        # total heavy budget
        self.used_ms["total"] += ms
        # per-metric bucket + call count if we recognize the kind
        if kind in self.used_ms:
            self.used_ms[kind] += ms
        if kind in self.calls:
            self.calls[kind] += 1

    def within_limits(self) -> bool:
        """
        Backwards-compatible summary: True iff nothing has breached its soft limits.
        """
        # Check total + both known metric types
        if self.used_ms["total"] > float(self.lim.total_heavy_ms):
            return False
        if self.used_ms["sw2"] > float(self.lim.sw2_ms):
            return False
        if self.used_ms["ripser"] > float(self.lim.ripser_ms):
            return False
        if self.calls["sw2"] > int(self.lim.sw2_calls):
            return False
        if self.calls["ripser"] > int(self.lim.ripser_calls):
            return False
        return True

    # --- audit helpers ---

    def spent(self) -> Dict[str, float]:
        """Return a shallow copy of the ms counters."""
        return dict(self.used_ms)

    def counts(self) -> Dict[str, int]:
        """Return a shallow copy of the call counters."""
        return dict(self.calls)
