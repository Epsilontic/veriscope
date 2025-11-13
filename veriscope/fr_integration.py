# veriscope/fr_integration.py
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Callable, Mapping, Tuple

import numpy as np

from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport, assert_naturality
from veriscope.core.window import FRWindow, WindowDecl

from veriscope.core.calibration import aggregate_epsilon_stat  # the real one

# Optional hook that higher layers may replace
AggFn = Callable[[WindowDecl, dict[str, int], float], float]
AGGREGATE_EPSILON: Optional[AggFn] = None

# Treat env as a Mapping, not a dict|None
ENV: Mapping[str, str] = os.environ

# Normalize env to a plain dict[str, str] for typing sanity
def _env_to_dict(e: Optional[Mapping[str, str]]) -> dict[str, str]:
    if e is None:
        # Copy to freeze current view; keeps mypy happy
        return dict(os.environ)
    # If it's a MutableMapping or Mapping, materialize as dict
    return dict(e)

# ... later, when you use the hook:
def resolve_eps(window: WindowDecl, counts: dict[str, int], alpha: float = 0.05) -> float:
    fn = AGGREGATE_EPSILON or aggregate_epsilon_stat
    return fn(window, counts, alpha)

# and when you read env:
def env_truthy(name: str, default: str = "0") -> bool:
    v = ENV.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def init_fr(window_decl: Optional[WindowDecl], cfg: Dict[str, Any], env: Optional[Dict[str, str]] = None) -> Tuple[bool, Optional[FRWindow], Optional[GateEngine]]:
    """
    Initialize the FR licensor and gate after WINDOW_DECL is resolved.
    Returns (use_fr, FR_WIN, GE) where use_fr is the SCAR_FR toggle.
    """
    env_map: dict[str, str] = _env_to_dict(env)
    use_fr = str(env_map.get("SCAR_FR", "0")).strip().lower() in ("1", "true", "yes", "on")

    if window_decl is None:
        return False, None, None

    # Normalize weights on the declaration and attach a common transport adapter
    window_decl.normalize_weights()
    tp = DeclTransport(window_decl)
    window_decl.attach_transport(tp)
    # Probe naturality (commutation with simple restrictions) non-fatally
    try:
        assert_naturality(tp, [lambda z: z,
                               lambda z: z[1:-1] if getattr(z, "ndim", 1) == 1 and z.size > 2 else z])
    except Exception as _e:
        try:
            outdir = Path(os.environ.get("SCAR_OUTDIR", "."))
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "naturality.log").write_text(f"naturality probe failed: {type(_e).__name__}: {_e}\n")
        except Exception:
            pass

    fr_win = FRWindow(decl=window_decl, transport=tp, tests=())

    # Resolve a window-relative evidence floor from the environment
    try:
        _min_evidence = int(os.getenv("SCAR_GATE_MIN_EVIDENCE", "0"))
    except Exception:
        _min_evidence = 0

    ge = GateEngine(
        frwin=fr_win,
        gain_thresh=float(cfg.get("gate_gain_thresh", 0.1)),
        eps_stat_alpha=float(cfg.get("gate_eps_stat_alpha", 0.05)),
        eps_stat_max_frac=float(cfg.get("gate_eps_stat_max_frac", 0.25)),
        eps_sens=float(cfg.get("gate_epsilon_sens", cfg.get("eps_sens", 0.04))),
        min_evidence=int(_min_evidence),
    )
    return use_fr, fr_win, ge


def gate_step_fr(
    fr_win: FRWindow,
    ge: GateEngine,
    logs_window: Iterable[Dict[str, Any]],
    gate_window: int,
    metrics_cols: Iterable[str],
    window_decl: WindowDecl,
    aggregate_epsilon_stat: Callable[[WindowDecl, dict[str, int], float], float],
    gate_gain: float,
    gate_kappa: float,
) -> Dict[str, float]:
    """
    Perform one FR gate check given the most recent 2*gate_window logs and return fields to log.

    Returns a dict with keys:
      gate_warn, gate_worst_tv, gate_eps_stat, gate_gain
    """
    W = int(gate_window)
    seg = list(logs_window)
    if len(seg) < 2 * W:
        return {}

    mets = [m for m in getattr(window_decl, "metrics", []) if m in metrics_cols]

    def _series(slice_rows: Iterable[Dict[str, Any]], key: str) -> np.ndarray:
        v = np.array([float(r.get(key, np.nan)) for r in slice_rows], dtype=float)
        return v[np.isfinite(v)]

    past_dict   = {m: _series(seg[:W], m) for m in mets}
    recent_dict = {m: _series(seg[W:],  m) for m in mets}

    # common transport before counts/distances
    # Prefer the declaration-attached adapter; fall back to FRWindow.transport
    _adapter = getattr(window_decl, "_DECL_TRANSPORT", None)
    _apply = _adapter.apply if _adapter is not None else fr_win.transport.apply
    tpast   = {m: _apply(m, v) for m, v in past_dict.items()   if v.size > 0}
    trecent = {m: _apply(m, v) for m, v in recent_dict.items() if v.size > 0}

    counts_by_metric = {
        m: int(min(np.isfinite(tpast[m]).sum(), np.isfinite(trecent[m]).sum()))
        for m in (tpast.keys() & trecent.keys())
    }

    # resolve epsilon from decl; cap eps_stat by a fraction of ε
    eps_resolved = float(getattr(window_decl, "epsilon", 0.16))
    fr_win.decl.epsilon = eps_resolved

    eps_stat_value = float(aggregate_epsilon_stat(
        window_decl,
        counts_by_metric,
        float(getattr(ge, "eps_stat_alpha", 0.05)),
    ))
    eps_stat_value = min(max(0.0, eps_stat_value),
                         float(getattr(ge, "eps_stat_max_frac", 0.25)) * eps_resolved)

    gr = ge.check(
        past=tpast,
        recent=trecent,
        counts_by_metric=counts_by_metric,
        gain_bits=float(gate_gain) if np.isfinite(gate_gain) else float("nan"),
        kappa_sens=(float(gate_kappa) if np.isfinite(gate_kappa) else float("inf")),
        eps_stat_value=eps_stat_value,
    )

    return {
        "gate_warn":     float(int(gr.ok)),
        "gate_worst_tv": float(gr.audit.get("worst_DW", np.nan)),
        "gate_eps_stat": float(gr.audit.get("eps_stat", np.nan)),
        "gate_gain":     float(gr.audit.get("gain_bits", np.nan)),
    }

# --- Minimal helpers expected by legacy_cli ---
WINDOW_DECL: Optional[WindowDecl] = None

essential_attrs = ("epsilon", "metrics", "weights", "bins", "cal_ranges", "interventions")


def install_window_decl(win: WindowDecl) -> None:
    """
    Install the declaration, attach a DeclTransport, and run a light naturality probe.
    """
    global WINDOW_DECL
    WINDOW_DECL = win
    try:
        tp = DeclTransport(win)
        try:
            setattr(win, "_DECL_TRANSPORT", tp)
        except Exception:
            pass
        try:
            assert_naturality(tp, [lambda z: z,
                                   lambda z: z[1:-1] if getattr(z, "ndim", 1) == 1 and z.size > 2 else z])
        except Exception:
            pass
    except Exception:
        pass


def write_window_audit(outdir: Path, window_decl: WindowDecl, note: str = "", controls_used: int = 0) -> None:
    """
    Minimal JSON writer describing the window declaration.
    Safe and typed; no-op on failure.
    """
    try:
        payload = {
            "epsilon": float(getattr(window_decl, "epsilon", float("nan"))),
            "metrics": list(getattr(window_decl, "metrics", [])),
            "weights": {str(k): float(v) for k, v in (getattr(window_decl, "weights", {}) or {}).items()},
            "bins": int(getattr(window_decl, "bins", 0)),
            "cal_ranges": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in (getattr(window_decl, "cal_ranges", {}) or {}).items()
                if isinstance(v, tuple) and len(v) == 2
            },
            "interventions": [f"predeclared_T_{i}" for i, _ in enumerate(tuple(getattr(window_decl, "interventions", ()) or ()))],
            "controls_used": int(controls_used),
            "notes": note or "Fixed partitions and transport calibrated from factor=='none' after warm",
        }
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "window_audit.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def write_window_provenance_from_decl(outdir: Path, wd: WindowDecl) -> None:
    """
    Emit a provenance capsule for the deployed WindowDecl.
    """
    try:
        capsule = {
            "epsilon": float(getattr(wd, "epsilon", float("nan"))),
            "metrics": list(getattr(wd, "metrics", [])),
            "weights": {str(k): float(v) for k, v in (getattr(wd, "weights", {}) or {}).items()},
            "bins": int(getattr(wd, "bins", 0)),
            "n_interventions": int(len(tuple(getattr(wd, "interventions", ()) or ()))),
            "cal_ranges": {
                str(k): [float(v[0]), float(v[1])]
                for k, v in (getattr(wd, "cal_ranges", {}) or {}).items()
                if isinstance(v, tuple) and len(v) == 2
            },
        }
        s = json.dumps(capsule, sort_keys=True)
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "window_provenance_decl.json").write_text(json.dumps({"capsule": capsule, "sha256": h}, indent=2))
        try:
            (outdir / "window_provenance_decl.json.sha256").write_text(h + "\n")
        except Exception:
            pass
    except Exception:
        pass
