# veriscope/runners/seed_policy.py
from __future__ import annotations
import os
from typing import Any, Mapping, List


def env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def _unique_ints_preserve_order(xs: Any) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for x in xs or []:
        try:
            i = int(x)
        except Exception:
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def seeds_for_eval_from_env(CFG_dict: Mapping[str, Any]) -> List[int]:
    raw = os.environ.get("SCAR_EVAL_SPLIT")
    if raw is None and env_truthy("SCAR_SMOKE"):
        mode = "eval"
    else:
        mode = (raw or "eval").lower().strip()

    if mode == "both":
        seeds = list(CFG_dict.get("seeds_calib", [])) + list(CFG_dict.get("seeds_eval", []))
        return _unique_ints_preserve_order(seeds)
    if mode == "calib":
        return _unique_ints_preserve_order(CFG_dict.get("seeds_calib", []))
    return _unique_ints_preserve_order(CFG_dict.get("seeds_eval", []))
