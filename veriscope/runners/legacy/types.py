# veriscope/runners/legacy/types.py
from __future__ import annotations

from typing import TypedDict


class DropCfg(TypedDict, total=False):
    drop_classes: set[int]
    drop_frac: float
    start: int
    end: int


__all__ = ["DropCfg"]
