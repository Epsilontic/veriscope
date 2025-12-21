# veriscope/fr_integration.py
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Callable, Mapping, Tuple, List

import numpy as np

from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport, assert_naturality
from veriscope.core.window import FRWindow, WindowDecl
from veriscope.core.calibration import aggregate_epsilon_stat


# ---- Epsilon aggregation hook ------------------------------------------------

# NOTE: resolve_eps is the contract boundary and may accept keyword-only args;
# keep this alias permissive to avoid false mypy failures across adapters.
AggFn = Callable[..., float]
AGGREGATE_EPSILON: Optional[AggFn] = None


def _env_to_dict(e: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Materialize an env mapping as a plain dict (for typing sanity)."""
    if e is None:
        return dict(os.environ)
    return dict(e)


def resolve_eps(
    window: WindowDecl,
    counts: Optional[Dict[str, int]] = None,
    *,
    counts_by_metric: Optional[Dict[str, int]] = None,
    alpha: float = 0.05,
    **_ignored: Any,
) -> float:
    """
    Resolve an effective epsilon_stat value from Φ_W and per-metric counts.

    Contract boundary:
      - Callers may pass counts either positionally (legacy) or via the
        keyword `counts_by_metric` (newer callers).
      - Uses a module-level hook AGGREGATE_EPSILON if set; otherwise falls back
        to veriscope.core.calibration.aggregate_epsilon_stat.

    Returns 0.0 on failure.
    """
    c = counts_by_metric if counts_by_metric is not None else (counts or {})
    fn = AGGREGATE_EPSILON or aggregate_epsilon_stat

    try:
        out = fn(window, c, float(alpha))
        out_f = float(out)
        return out_f if np.isfinite(out_f) and out_f >= 0.0 else 0.0
    except TypeError:
        # Tolerate hook/core signature drift (kw-only implementations).
        try:
            out = fn(window, counts_by_metric=c, alpha=float(alpha))
            out_f = float(out)
            return out_f if np.isfinite(out_f) and out_f >= 0.0 else 0.0
        except Exception:
            return 0.0
    except Exception:
        return 0.0


def env_truthy(name: str, default: str = "0", env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True iff env[name] parses as a truthy flag (env overrides os.environ)."""
    # Deterministic fallback—if env passed use it, else use os.environ directly.
    # Avoids brittle globals that may not be a Mapping or may not exist.
    src = env if env is not None else os.environ
    v = src.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# ---- FR initialization -------------------------------------------------------


def init_fr(
    window_decl: Optional[WindowDecl],
    cfg: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, Optional[FRWindow], Optional[GateEngine]]:
    """
    Initialize the FR licensor and gate after WINDOW_DECL is resolved.

    Returns (use_fr, FR_WIN, GE) where use_fr is the SCAR_FR toggle coming
    from the environment, and FR_WIN/GE are None if no declaration is given.
    """
    env_map: Dict[str, str] = dict(os.environ) if env is None else dict(env)

    # Enable FR license path?
    use_fr = env_truthy("SCAR_FR", default="0", env=env_map)
    if not use_fr:
        return False, None, None
    if window_decl is None:
        return False, None, None

    # Normalize weights on the declaration
    try:
        window_decl.normalize_weights()
    except Exception:
        # Older declarations may not have normalize_weights; fall through.
        pass

    # Attach a common transport adapter
    tp = DeclTransport(window_decl)
    window_decl.attach_transport(tp)

    # Light naturality probe (non-fatal; logs to naturality.log on failure)
    try:

        def _id(z):
            return z

        def _center(z):
            if getattr(z, "ndim", 1) == 1 and getattr(z, "size", 0) > 2:
                return z[1:-1]
            return z

        assert_naturality(tp, [_id, _center])
    except Exception as _e:
        try:
            outdir = Path(os.environ.get("SCAR_OUTDIR", "."))
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "naturality.log").write_text(f"naturality probe failed: {type(_e).__name__}: {_e}\n")
        except Exception:
            pass

    fr_win = FRWindow(decl=window_decl, transport=tp, tests=())

    # Evidence floor for gating
    try:
        _min_evidence = int(env_map.get("SCAR_GATE_MIN_EVIDENCE", "0"))
    except Exception:
        _min_evidence = 0

    # Gate policy/persistence (env wins over cfg).
    # Core policies: either|conjunction|persistence|persistence_stability
    # Legacy-only "stability_only" maps to persistence_stability with k=1.
    policy_raw = str(env_map.get("SCAR_GATE_POLICY", cfg.get("gate_policy", "either"))).strip().lower()
    try:
        persistence_k = int(env_map.get("SCAR_GATE_PERSISTENCE_K", cfg.get("gate_persistence_k", 2)))
    except Exception:
        persistence_k = 2
    persistence_k = int(max(1, persistence_k))

    if policy_raw == "stability_only":
        policy = "persistence_stability"
        persistence_k = 1
    else:
        policy = policy_raw

    ge = GateEngine(
        frwin=fr_win,
        gain_thresh=float(cfg.get("gate_gain_thresh", 0.1)),
        eps_stat_alpha=float(cfg.get("gate_eps_stat_alpha", 0.05)),
        eps_stat_max_frac=float(cfg.get("gate_eps_stat_max_frac", 0.25)),
        eps_sens=float(cfg.get("gate_epsilon_sens", cfg.get("eps_sens", 0.04))),
        min_evidence=int(_min_evidence),
        policy=policy,
        persistence_k=persistence_k,
    )

    # Requested vs effective policy transparency (GateEngine may normalize/fallback in its parser).
    # Use underscore-prefixed attributes to avoid collision with future GateEngine properties.
    try:
        ge._vs_policy_requested = str(policy_raw)
        ge._vs_policy_effective = str(getattr(getattr(ge, "policy", None), "value", getattr(ge, "policy", policy)))
    except Exception:
        pass

    return use_fr, fr_win, ge


# ---- Single-step FR gate over a sliding window -------------------------------


def gate_step_fr(
    fr_win: FRWindow,
    ge: GateEngine,
    logs_window: Iterable[Dict[str, Any]],
    gate_window: int,
    metrics_cols: Iterable[str],
    window_decl: WindowDecl,
    gate_gain: float,
    gate_kappa: float,
) -> Dict[str, float]:
    """
    Perform one FR gate check given the most recent 2*gate_window logs and return fields to log.

    Returns a dict with keys:
      gate_warn, gate_worst_tv, gate_eps_stat, gate_gain
    or {} if there is not yet enough history (len(logs) < 2W).
    """
    W = int(gate_window)
    seg = list(logs_window)
    if len(seg) < 2 * W:
        return {}

    mets = [m for m in getattr(window_decl, "metrics", []) if m in set(metrics_cols)]

    def _series(slice_rows: Iterable[Dict[str, Any]], key: str) -> np.ndarray:
        v = np.array([float(r.get(key, np.nan)) for r in slice_rows], dtype=float)
        return v[np.isfinite(v)]

    past_dict = {m: _series(seg[:W], m) for m in mets}
    recent_dict = {m: _series(seg[W:], m) for m in mets}

    # Common transport before counts/distances
    adapter = getattr(window_decl, "_DECL_TRANSPORT", None)
    apply_tp = adapter.apply if adapter is not None else fr_win.transport.apply

    tpast = {m: apply_tp(m, v) for m, v in past_dict.items() if v.size > 0}
    trecent = {m: apply_tp(m, v) for m, v in recent_dict.items() if v.size > 0}

    keys = set(tpast.keys()) & set(trecent.keys())
    counts_by_metric: Dict[str, int] = {
        m: int(min(np.isfinite(tpast[m]).sum(), np.isfinite(trecent[m]).sum())) for m in keys
    }

    # Window epsilon and eps_stat budget
    eps_resolved = float(getattr(window_decl, "epsilon", 0.16))
    fr_win.decl.epsilon = eps_resolved

    alpha = float(getattr(ge, "eps_stat_alpha", 0.05))
    max_frac = float(getattr(ge, "eps_stat_max_frac", 0.25))

    try:
        # Route ε_stat through the stable adapter (contract boundary).
        eps_stat_value = float(resolve_eps(window_decl, counts_by_metric=counts_by_metric, alpha=alpha))
    except Exception:
        eps_stat_value = 0.0

    eps_stat_value = max(0.0, eps_stat_value)
    eps_stat_value = min(eps_stat_value, max_frac * eps_resolved)

    gain_bits = float(gate_gain) if np.isfinite(gate_gain) else float("nan")
    kappa_val = float(gate_kappa) if np.isfinite(gate_kappa) else float("inf")

    gr = ge.check(
        past=tpast,
        recent=trecent,
        counts_by_metric=counts_by_metric,
        gain_bits=gain_bits,
        kappa_sens=kappa_val,
        eps_stat_value=eps_stat_value,
    )

    return {
        "gate_warn": float(int(bool(getattr(gr, "ok", False)))),
        "gate_worst_tv": float(gr.audit.get("worst_DW", np.nan)),
        "gate_eps_stat": float(gr.audit.get("eps_stat", np.nan)),
        "gate_gain": float(gr.audit.get("gain_bits", np.nan)),
    }


# ---- Minimal window install + audit/provenance -------------------------------

WINDOW_DECL: Optional[WindowDecl] = None


def install_window_decl(win: WindowDecl) -> None:
    """
    Install Φ_W, attach a DeclTransport adapter, and run a light naturality probe.

    This does NOT touch any global CFG/OUTDIR state; callers decide where to log.
    """
    global WINDOW_DECL
    WINDOW_DECL = win

    try:
        tp = DeclTransport(win)
        try:
            setattr(win, "_DECL_TRANSPORT", tp)
        except Exception:
            pass

        def _id(z):
            return z

        def _center(z):
            if getattr(z, "ndim", 1) == 1 and getattr(z, "size", 0) > 2:
                return z[1:-1]
            return z

        try:
            assert_naturality(tp, [_id, _center])
        except Exception:
            # Non-fatal here; caller may have its own logging
            pass
    except Exception:
        # Keep runner robust even if FR core is partially missing
        pass


def write_window_audit(
    outdir: Path,
    window_decl: WindowDecl,
    note: str = "",
    controls_used: int = 0,
) -> None:
    """
    Persist a small JSON audit for Φ_W (fixed partitions, transports, epsilon).

    This is intentionally self-contained: no dependence on runner CFG/OUTDIR.
    """
    try:
        weights = dict(getattr(window_decl, "weights", {}) or {})
        metrics = list(getattr(window_decl, "metrics", []) or [])
        bins = int(getattr(window_decl, "bins", 0))
        epsilon = float(getattr(window_decl, "epsilon", float("nan")))
        cal_ranges = getattr(window_decl, "cal_ranges", {}) or {}
        intervs = tuple(getattr(window_decl, "interventions", ()) or ())

        cal_ranges_norm: Dict[str, list[float]] = {}
        for k, v in cal_ranges.items():
            try:
                a, b = v  # assume 2-tuple or list-like
                cal_ranges_norm[str(k)] = [float(a), float(b)]
            except Exception:
                continue

        payload = {
            "epsilon": epsilon,
            "metrics": metrics,
            "weights": {str(k): float(v) for k, v in weights.items()},
            "bins": bins,
            "cal_ranges": cal_ranges_norm,
            "interventions": [f"predeclared_T_{i}" for i, _ in enumerate(intervs)],
            "controls_used": int(controls_used),
            "notes": note or "Fixed partitions and transport calibrated from factor=='none' after warm",
        }

        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "window_audit.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        # Silent failure is acceptable for audit; runner continues.
        pass


def write_window_provenance_from_decl(outdir: Path, wd: WindowDecl) -> None:
    """
    Emit a provenance capsule for the deployed WindowDecl, with a SHA-256 hash.

    Writes:
      - window_provenance_decl.json
      - window_provenance_decl.json.sha256
    """
    try:
        weights = dict(getattr(wd, "weights", {}) or {})
        metrics = list(getattr(wd, "metrics", []) or [])
        bins = int(getattr(wd, "bins", 0))
        epsilon = float(getattr(wd, "epsilon", float("nan")))
        intervs = tuple(getattr(wd, "interventions", ()) or ())
        cal_ranges = getattr(wd, "cal_ranges", {}) or {}

        cal_ranges_norm: Dict[str, list[float]] = {}
        for k, v in cal_ranges.items():
            try:
                a, b = v
                cal_ranges_norm[str(k)] = [float(a), float(b)]
            except Exception:
                continue

        capsule = {
            "epsilon": epsilon,
            "metrics": metrics,
            "weights": {str(k): float(v) for k, v in weights.items()},
            "bins": bins,
            "n_interventions": int(len(intervs)),
            "cal_ranges": cal_ranges_norm,
            # breadcrumbs
            "gain_units": "bits/sample",
        }

        # Hash the capsule in a stable, canonical form (object-level fingerprint)
        capsule_canon = json.dumps(
            capsule,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        capsule_sha = hashlib.sha256(capsule_canon.encode("utf-8")).hexdigest()

        outdir.mkdir(parents=True, exist_ok=True)

        # Write provenance payload first
        prov_path = outdir / "window_provenance_decl.json"
        prov_text = (
            json.dumps(
                {"capsule": capsule, "capsule_sha256": capsule_sha},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
        prov_path.write_text(prov_text, encoding="utf-8")

        # Then hash the exact bytes written to disk and emit a sha256sum -c compatible manifest
        file_sha = hashlib.sha256(prov_path.read_bytes()).hexdigest()
        try:
            (outdir / "window_provenance_decl.json.sha256").write_text(
                f"{file_sha}  {prov_path.name}\n",
                encoding="utf-8",
            )

            # Back-compat: also emit a digest-only file (older tooling may expect just the hex digest).
            (outdir / "window_provenance_decl.json.sha256.raw").write_text(
                file_sha + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
    except Exception:
        try:
            print("[WARN] write_window_provenance_from_decl failed", flush=True)
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    """
    Console entrypoint for the `veriscope` CLI.

    Thin shim that delegates to the legacy CLI runner so that the
    `veriscope` console script keeps working during phase-1.

    IMPORTANT: this function must not contain a direct import statement
    from legacy_cli_refactor; use lazy import to avoid circular deps.
    """
    import sys
    import importlib

    if argv is None:
        argv = sys.argv[1:]

    # Lazy import to avoid circular imports and keep phase-1 boundaries.
    legacy_mod = importlib.import_module("veriscope.runners.legacy_cli_refactor")
    legacy_main = getattr(legacy_mod, "main")

    # legacy_main currently reads sys.argv itself and does not use argv,
    # but we keep argv for future-proofing.
    _ = argv
    rv = legacy_main()
    return 0 if rv is None else int(rv)
