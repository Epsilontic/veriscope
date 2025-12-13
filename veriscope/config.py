# veriscope/config.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

# Project defaults live here; keep these minimal and window-agnostic.
DEFAULTS: Dict[str, Any] = {
    "epochs": 16,  # smoke default
    "warmup": 8,
    "ph_burn": 0,
    "gate_window": 16,
    "gate_bins": 16,
    "gate_min_evidence": 16,
    "gate_gain_thresh": 0.10,  # bits/sample
    "gate_epsilon": 0.08,
    "gate_eps_stat_max_frac": 0.25,
    "eps_sens": 0.04,
    "warn_consec": 3,
    "family_window": 1,
    "detector_horizon": 8,
    "var_k_max": 32,
    "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") else "cpu",
}


def _coerce_env(v: str) -> Any:
    s = v.strip()
    try:
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        f = float(s)
        return f
    except Exception:
        return s


def _env_truthy(name: str, default: str = "0") -> bool:
    """Return True if env var is set to a truthy value."""
    v = str(os.getenv(name, default)).strip().lower()
    return v in ("1", "true", "yes", "on")


def load_cfg(path: str | Path | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    p = Path(path) if path else Path(os.getenv("SCAR_CFG", ""))
    if p and p.exists():
        try:
            cfg.update(json.loads(p.read_text()))
        except Exception:
            pass
    if _env_truthy("SCAR_SMOKE"):
        # minimal smoke seeds if the runner doesn’t set them
        cfg.setdefault("seeds_calib", [101, 102])
        cfg.setdefault("seeds_eval", [201, 202])
        cfg.setdefault("epochs", 16)
    # apply SCAR_* env overrides in a conservative way
    for k, v in list(cfg.items()):
        env = os.getenv(f"SCAR_{k.upper()}", None)
        if env is not None:
            cfg[k] = _coerce_env(env)
    return cfg


# Public export used by runners
CFG: Dict[str, Any] = load_cfg()
