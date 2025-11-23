# veriscope/runners/legacy/eval/eval.py
from __future__ import annotations

"""
Legacy eval facade.

Phase-1 boundary:
- Must not import from legacy_cli_refactor.py.
- Implementations will be moved into a dedicated module in Phase-1/2,
  then re-exported from here.

Right now no code imports this package (rg confirmed), so keeping this
as a placeholder avoids dependency back-edges while preserving the plan.
"""

__all__: list[str] = []
