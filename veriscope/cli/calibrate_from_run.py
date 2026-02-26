from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

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


def _finite_tv_samples_from_results(results_obj: Dict[str, Any]) -> List[float]:
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
        if not bool(audit.get("evaluated", False)):
            continue
        per_metric_tv = audit.get("per_metric_tv")
        if not isinstance(per_metric_tv, dict):
            continue
        for value in per_metric_tv.values():
            try:
                f = float(value)
            except Exception:
                continue
            if math.isfinite(f):
                samples.append(f)
    return samples


def _fallback_gate_epsilon(window_signature_obj: Dict[str, Any], default: float = 0.12) -> float:
    gate_controls = window_signature_obj.get("gate_controls")
    if isinstance(gate_controls, dict):
        raw = gate_controls.get("gate_epsilon")
        try:
            value = float(raw)
        except Exception:
            value = float(default)
        if math.isfinite(value) and value > 0:
            return value
    return float(default)


def calibrate_from_run(capsule_dir: Path, *, quantile: float = 0.95) -> Dict[str, Any]:
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

    samples = _finite_tv_samples_from_results(results_obj)
    if not samples:
        raise ValueError("No evaluated gates with finite per_metric_tv values found")

    fallback = _fallback_gate_epsilon(ws_obj)
    epsilon, n_samples = resolve_epsilon_from_controls(samples, float(quantile), fallback)
    if not math.isfinite(epsilon):
        raise ValueError("No evaluated gates with finite per_metric_tv values found")
    # Single-run calibration should not loosen the declared gate threshold.
    # Treat the window signature's gate_epsilon as an upper bound.
    if math.isfinite(fallback) and fallback > 0:
        epsilon = min(float(epsilon), float(fallback))

    return {
        "schema_version": 1,
        "capsule_dir": str(outdir),
        "source": "gate_audit_per_metric_tv",
        "quantile": float(quantile),
        "epsilon": float(epsilon),
        "n_samples": int(n_samples),
        "window_signature_hash": validation.window_signature_hash,
    }
