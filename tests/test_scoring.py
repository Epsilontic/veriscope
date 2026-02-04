# tests/test_scoring.py
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Optional

import pytest

from veriscope.core.pilot_calibration import CalibrationError, calibrate_pilot


def _load_score_module() -> object:
    score_path = Path(__file__).resolve().parents[1] / "scripts" / "pilot" / "score.py"
    spec = importlib.util.spec_from_file_location("pilot_score", score_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def _write_run(
    outdir: Path,
    *,
    gates: list[dict[str, object]],
    gate_window: int,
    gate_warmup: int,
    data_corrupt_at: Optional[int] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    window_signature = {"schema_version": 1, "gate_controls": {"gate_window": gate_window}}
    (outdir / "window_signature.json").write_text(json.dumps(window_signature), encoding="utf-8")

    results = {"schema_version": 1, "gates": gates}
    (outdir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    evaluated = sum(1 for g in gates if g.get("decision") in {"pass", "warn", "fail"})
    warn = sum(1 for g in gates if g.get("decision") == "warn")
    fail = sum(1 for g in gates if g.get("decision") == "fail")
    skip = sum(1 for g in gates if g.get("decision") == "skip")
    counts = {"evaluated": evaluated, "warn": warn, "fail": fail, "skip": skip, "pass": evaluated - warn - fail}
    summary = {"schema_version": 1, "run_status": "success", "counts": counts}
    (outdir / "results_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    run_cfg = {"schema_version": 1, "resolved_gate_cfg": {"gate_warmup": gate_warmup}}
    if data_corrupt_at is not None:
        run_cfg["data_corrupt_at"] = int(data_corrupt_at)
    (outdir / "run_config_resolved.json").write_text(json.dumps(run_cfg), encoding="utf-8")

    gov_lines = []
    for idx, gate in enumerate(gates):
        payload = {
            "iter": gate.get("iter"),
            "decision": gate.get("decision"),
            "audit": gate.get("audit", {}),
        }
        entry = {
            "schema_version": 1,
            "rev": idx + 1,
            "ts_utc": "2026-01-01T00:00:00Z",
            "event_type": "gate_decision_v1",
            "payload": payload,
        }
        gov_lines.append(json.dumps(entry))
    (outdir / "governance_log.jsonl").write_text("\n".join(gov_lines) + "\n", encoding="utf-8")


def test_scoring_far_and_delay(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    control_gates = [
        {"iter": 50, "decision": "pass"},
        {"iter": 100, "decision": "pass"},
        {"iter": 150, "decision": "warn"},
        {"iter": 200, "decision": "pass"},
        {"iter": 250, "decision": "fail"},
    ]
    injected_gates = [
        {"iter": 80, "decision": "pass"},
        {"iter": 120, "decision": "pass"},
        {"iter": 180, "decision": "pass"},
        {"iter": 200, "decision": "warn"},
        {"iter": 250, "decision": "fail"},
    ]

    _write_run(control_dir, gates=control_gates, gate_window=10, gate_warmup=100)
    _write_run(injected_dir, gates=injected_gates, gate_window=10, gate_warmup=100, data_corrupt_at=160)

    score = _load_score_module()
    out_json = tmp_path / "calibration.json"
    out_md = tmp_path / "calibration.md"

    exit_code = score.main(
        [
            "--control-dir",
            str(control_dir),
            "--injected-dir",
            str(injected_dir),
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert exit_code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["FAR"] == 0.5
    assert payload["FAR_fail"] == 0.25
    assert payload["Delay_W"] == 4.0
    assert payload["control_gate_events_first"][0]["iter"] == 50


def test_scoring_delay_normalized_by_window(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    control_gates = [
        {"iter": 0, "decision": "pass"},
        {"iter": 20, "decision": "pass"},
        {"iter": 40, "decision": "pass"},
    ]
    injected_gates = [
        {"iter": 0, "decision": "pass"},
        {"iter": 30, "decision": "warn"},
    ]

    _write_run(control_dir, gates=control_gates, gate_window=20, gate_warmup=0)
    _write_run(injected_dir, gates=injected_gates, gate_window=20, gate_warmup=0, data_corrupt_at=10)

    score = _load_score_module()
    out_json = tmp_path / "calibration.json"
    out_md = tmp_path / "calibration.md"

    exit_code = score.main(
        [
            "--control-dir",
            str(control_dir),
            "--injected-dir",
            str(injected_dir),
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert exit_code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["Delay_W"] == 1.0


def test_scoring_delay_uses_warmup_threshold(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    control_gates = [
        {"iter": 0, "decision": "pass"},
        {"iter": 50, "decision": "pass"},
        {"iter": 100, "decision": "pass"},
    ]
    injected_gates = [
        {"iter": 40, "decision": "pass"},
        {"iter": 120, "decision": "warn"},
    ]

    _write_run(control_dir, gates=control_gates, gate_window=10, gate_warmup=100, data_corrupt_at=0)
    _write_run(injected_dir, gates=injected_gates, gate_window=10, gate_warmup=100, data_corrupt_at=50)

    score = _load_score_module()
    out_json = tmp_path / "calibration.json"
    out_md = tmp_path / "calibration.md"

    exit_code = score.main(
        [
            "--control-dir",
            str(control_dir),
            "--injected-dir",
            str(injected_dir),
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert exit_code == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["Delay_warn_iters"] == 20
    assert payload["Delay_warn_W"] == 2.0


def test_scoring_missing_iter_raises(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    _write_run(control_dir, gates=[{"iter": 10, "decision": "pass"}], gate_window=10, gate_warmup=0)
    _write_run(
        injected_dir,
        gates=[{"iter": 10, "decision": "pass"}],
        gate_window=10,
        gate_warmup=0,
        data_corrupt_at=0,
    )

    bad_entry = {"event_type": "gate_decision_v1", "payload": {"decision": "pass"}}
    (control_dir / "governance_log.jsonl").write_text(json.dumps(bad_entry) + "\n", encoding="utf-8")

    with pytest.raises(CalibrationError, match="MISSING_GATE_ITER"):
        calibrate_pilot(control_dir, injected_dir)
