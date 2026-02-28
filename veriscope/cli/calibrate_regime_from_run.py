from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from veriscope.cli.validate import validate_outdir
from veriscope.core.calibration import resolve_epsilon_from_controls


def _read_json_obj(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid capsule: failed to read {path.name}: {exc}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid capsule: {path.name} must be a JSON object")
    return obj


def _as_opt_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return bool(value)
    return None


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _is_regime_active_for_sampling(audit: Dict[str, Any]) -> bool:
    ready = _as_opt_bool(audit.get("regime_ref_ready"))
    active = _as_opt_bool(audit.get("regime_active"))
    ran = _as_opt_bool(audit.get("regime_check_ran"))

    if ready is True or active is True:
        return True

    # Back-compat fallback for older capsules that may not emit regime_ref_ready/regime_active.
    if ready is None and active is None and ran is True:
        return True
    return False


def _extract_regime_dw(gate: Dict[str, Any], audit: Dict[str, Any]) -> Optional[float]:
    for key in ("regime_D_W", "regime_worst_DW"):
        fv = _as_finite_float(audit.get(key))
        if fv is not None:
            return fv
    # Legacy fallback: occasionally emitted at gate root.
    for key in ("regime_D_W", "regime_worst_DW"):
        fv = _as_finite_float(gate.get(key))
        if fv is not None:
            return fv
    return None


def _finite_regime_dw_samples_from_results(results_obj: Dict[str, Any]) -> List[float]:
    gates = results_obj.get("gates")
    if not isinstance(gates, list):
        return []

    samples: List[float] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        audit = gate.get("audit")
        if not isinstance(audit, dict):
            continue
        if not _is_regime_active_for_sampling(audit):
            continue
        value = _extract_regime_dw(gate, audit)
        if value is not None:
            samples.append(value)
    return samples


def _fallback_regime_epsilon(
    results_obj: Dict[str, Any], window_signature_obj: Dict[str, Any], default: float = 0.18
) -> float:
    gates = results_obj.get("gates")
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            audit = gate.get("audit")
            if not isinstance(audit, dict):
                continue
            for key in ("eps_regime_eff", "regime_eps_eff", "eps_regime", "regime_epsilon"):
                f = _as_finite_float(audit.get(key))
                if f is not None and f > 0:
                    return f

    gate_controls = window_signature_obj.get("gate_controls")
    if isinstance(gate_controls, dict):
        for key in ("regime_epsilon", "eps_regime"):
            f = _as_finite_float(gate_controls.get(key))
            if f is not None and f > 0:
                return f

        base_eps = _as_finite_float(gate_controls.get("gate_epsilon"))
        mult = _as_finite_float(gate_controls.get("regime_epsilon_mult"))
        if base_eps is not None and base_eps > 0:
            if mult is None or mult <= 0:
                mult = 1.5
            return float(base_eps) * float(mult)

    return float(default)


def calibrate_regime_from_run(capsule_dir: Path, *, quantile: float = 0.95) -> Dict[str, Any]:
    outdir = Path(capsule_dir).expanduser()
    if not outdir.exists():
        raise ValueError(f"Invalid capsule: missing directory {outdir}")

    validation = validate_outdir(
        outdir,
        strict_identity=True,
        allow_partial=False,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    if not validation.ok:
        raise ValueError(f"Invalid capsule: {validation.message}")

    results_obj = _read_json_obj(outdir / "results.json")
    ws_obj = _read_json_obj(outdir / "window_signature.json")

    samples = _finite_regime_dw_samples_from_results(results_obj)
    if not samples:
        raise ValueError("No regime-active gates with finite regime_D_W values found")

    fallback = _fallback_regime_epsilon(results_obj, ws_obj)
    epsilon, n_samples = resolve_epsilon_from_controls(samples, float(quantile), fallback)
    if not math.isfinite(epsilon):
        raise ValueError("No regime-active gates with finite regime_D_W values found")

    return {
        "schema_version": 1,
        "capsule_dir": str(outdir),
        "source": "regime_D_W",
        "quantile": float(quantile),
        "epsilon": float(epsilon),
        "n_samples": int(n_samples),
        "window_signature_hash": validation.window_signature_hash,
    }
