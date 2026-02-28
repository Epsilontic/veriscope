from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from veriscope.cli.governance import append_run_started
from veriscope.core.governance import append_gate_decision
from veriscope.core.jsonutil import canonical_json_sha256

pytestmark = pytest.mark.unit


T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_dict(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return obj


def _minimal_window_signature() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "code_identity": {"package_version": "test"},
        "gate_controls": {"gate_window": 16, "gate_epsilon": 0.12, "min_evidence": 16},
        "metric_interval": 16,
        "metric_pipeline": {"transport": "test"},
    }


def _make_capsule(outdir: Path, *, regime_samples: list[float]) -> Path:
    run_id = "test_run_regime_calibration"
    outdir.mkdir(parents=True, exist_ok=True)

    ws_path = outdir / "window_signature.json"
    _write_json(ws_path, _minimal_window_signature())
    ws_hash = canonical_json_sha256(_read_json_dict(ws_path))

    common = {
        "schema_version": 1,
        "run_id": run_id,
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(T0),
        "ended_ts_utc": _iso_z(T1),
    }

    gates: list[dict[str, Any]] = []
    for idx, value in enumerate(regime_samples):
        gates.append(
            {
                "iter": idx,
                "decision": "pass",
                "ok": True,
                "warn": False,
                "audit": {
                    "evaluated": True,
                    "reason": "evaluated_pass",
                    "policy": "test_policy",
                    "per_metric_tv": {"m": 0.01},
                    "evidence_total": 16,
                    "min_evidence": 16,
                    "regime_ref_ready": True,
                    "regime_active": True,
                    "regime_check_ran": True,
                    "regime_D_W": float(value),
                    "regime_worst_DW": float(value),
                },
            }
        )

    gates.append(
        {
            "iter": len(gates),
            "decision": "pass",
            "ok": True,
            "warn": False,
            "audit": {
                "evaluated": True,
                "reason": "evaluated_pass",
                "policy": "test_policy",
                "per_metric_tv": {"m": 0.01},
                "evidence_total": 16,
                "min_evidence": 16,
                "regime_ref_ready": False,
                "regime_active": False,
                "regime_check_ran": False,
            },
        }
    )

    _write_json(outdir / "results.json", {**common, "gates": gates, "metrics": []})
    _write_json(
        outdir / "results_summary.json",
        {
            **common,
            "counts": {"evaluated": len(gates), "skip": 0, "pass": len(gates), "warn": 0, "fail": 0},
            "final_decision": "pass",
            "first_fail_iter": None,
        },
    )

    append_run_started(
        outdir,
        run_id=run_id,
        outdir_path=outdir,
        argv=["pytest", "fixture"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.fixture"},
        ts_utc=_iso_z(T0),
    )
    for gate in gates:
        append_gate_decision(
            outdir,
            run_id=run_id,
            iter_num=int(gate["iter"]),
            decision=str(gate["decision"]),
            ok=bool(gate["ok"]),
            warn=bool(gate["warn"]),
            audit=dict(gate["audit"]),
        )
    return outdir


def test_calibrate_regime_from_run_uses_regime_dw_quantile(tmp_path: Path) -> None:
    from veriscope.cli.calibrate_regime_from_run import calibrate_regime_from_run

    outdir = _make_capsule(tmp_path / "run", regime_samples=[0.20, 0.40, 0.80])
    result = calibrate_regime_from_run(outdir, quantile=0.50)
    assert result["source"] == "regime_D_W"
    assert result["n_samples"] == 3
    assert result["epsilon"] == pytest.approx(0.40)
    assert isinstance(result["window_signature_hash"], str)
    assert len(result["window_signature_hash"]) == 64


def test_calibrate_regime_from_run_raises_without_regime_samples(tmp_path: Path) -> None:
    from veriscope.cli.calibrate_regime_from_run import calibrate_regime_from_run

    outdir = _make_capsule(tmp_path / "run_no_regime", regime_samples=[])
    with pytest.raises(ValueError, match="No regime-active gates"):
        calibrate_regime_from_run(outdir, quantile=0.95)
