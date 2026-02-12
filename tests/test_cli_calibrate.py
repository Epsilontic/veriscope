# tests/test_cli_calibrate.py
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from veriscope.cli.calibrate import run_calibrate
from veriscope.core.jsonutil import window_signature_sha256
from veriscope.core.pilot_calibration import CalibrationError, calibrate_pilot

pytestmark = pytest.mark.unit


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_capsule(
    outdir: Path,
    *,
    run_id: str,
    gate_window: int = 2,
    gate_warmup: Optional[int] = None,
    gates: Optional[List[Dict[str, Any]]] = None,
    with_results: bool = True,
    with_run_config: bool = True,
    with_governance: bool = True,
    data_corrupt_at: Optional[int] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    window_signature = {
        "schema_version": 1,
        "gate_controls": {"gate_window": gate_window},
    }
    ws_hash = window_signature_sha256(window_signature)
    _write_json(outdir / "window_signature.json", window_signature)

    results_summary = {
        "schema_version": 1,
        "run_id": run_id,
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test"},
        "run_status": "success",
        "runner_exit_code": 0,
        "runner_signal": None,
        "started_ts_utc": "2026-01-01T00:00:00Z",
        "ended_ts_utc": "2026-01-01T00:01:00Z",
        "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
        "final_decision": "pass",
    }
    _write_json(outdir / "results_summary.json", results_summary)

    if with_results:
        gates_out: List[Dict[str, Any]] = []
        for idx, gate in enumerate(gates or []):
            gate_out = dict(gate)
            if "iter" not in gate_out:
                gate_out["iter"] = idx
            gates_out.append(gate_out)
        results = {
            "schema_version": 1,
            "run_id": run_id,
            "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
            "profile": {"gate_preset": "test"},
            "run_status": "success",
            "runner_exit_code": 0,
            "runner_signal": None,
            "started_ts_utc": "2026-01-01T00:00:00Z",
            "ended_ts_utc": "2026-01-01T00:01:00Z",
            "gates": gates_out,
            "metrics": [],
        }
        _write_json(outdir / "results.json", results)

    if with_governance:
        gov_lines: List[str] = []
        for idx, gate in enumerate(gates or []):
            payload = {
                "iter": gate.get("iter", idx),
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

    if with_run_config:
        resolved_gate_cfg: Dict[str, Any] = {}
        if gate_warmup is not None:
            resolved_gate_cfg["gate_warmup"] = gate_warmup
        run_config = {
            "schema_version": 1,
            "resolved_gate_cfg": resolved_gate_cfg,
            "timing": {"runner_wall_s": 100.0, "veriscope_wall_s": 110.0},
            "env_capture": {"redactions_applied": True},
            "provenance": {"policy_rev": "rev-test"},
        }
        if data_corrupt_at is not None:
            run_config["data_corrupt_at"] = int(data_corrupt_at)
        _write_json(outdir / "run_config_resolved.json", run_config)


def test_calibrate_pilot_happy_path(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    warmup = 1
    gate_window = 2

    control_gates = [
        {"iter": 0, "decision": "pass"},
        {"iter": 1, "decision": "pass"},
        {"iter": 2, "decision": "pass"},
        {"iter": 3, "decision": "pass"},
    ]
    injected_gates = [
        {"iter": 0, "decision": "pass"},
        {"iter": 1, "decision": "pass"},
        {"iter": 2, "decision": "pass"},
        {"iter": 3, "decision": "warn"},
    ]

    _write_capsule(
        control_dir,
        run_id="control",
        gate_window=gate_window,
        gate_warmup=warmup,
        gates=control_gates,
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_window=gate_window,
        gate_warmup=warmup,
        gates=injected_gates,
        data_corrupt_at=1,
    )

    output = calibrate_pilot(control_dir, injected_dir)

    assert output["FAR"] == 0.0
    assert output["Delay_W"] == 1.0
    assert output["overhead"] == pytest.approx(1.1)
    assert output["calibration_status"] == "complete"


def test_calibrate_missing_injected_results(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    _write_capsule(
        control_dir,
        run_id="control",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_warmup=0,
        with_results=False,
        with_governance=False,
    )

    output = calibrate_pilot(control_dir, injected_dir)

    assert output["Delay_W"] is None
    assert output["calibration_status"] == "incomplete"
    assert "injected.gate_events_missing" in output["missing_fields"]


def test_calibrate_invalid_injected_results(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    _write_capsule(
        control_dir,
        run_id="control",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_warmup=0,
        with_governance=False,
    )
    (injected_dir / "results.json").write_text("{not-json", encoding="utf-8")

    output = calibrate_pilot(control_dir, injected_dir)

    assert output["Delay_W"] is None
    assert output["calibration_status"] == "incomplete"
    assert "injected.delay_warn_missing" in output["missing_fields"]
    assert "injected.injection_onset_missing" in output["missing_fields"]


def test_calibrate_governance_log_empty_marks_missing(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    _write_capsule(
        control_dir,
        run_id="control",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
        with_governance=False,
        data_corrupt_at=0,
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        with_governance=False,
        data_corrupt_at=0,
    )
    empty_entry_control = {
        "schema_version": 1,
        "rev": 1,
        "ts_utc": "2026-01-01T00:00:00Z",
        "event_type": "run_started_v1",
        "payload": {"run_id": "control"},
    }
    empty_entry_injected = {
        "schema_version": 1,
        "rev": 1,
        "ts_utc": "2026-01-01T00:00:00Z",
        "event_type": "run_started_v1",
        "payload": {"run_id": "injected"},
    }
    (control_dir / "governance_log.jsonl").write_text(json.dumps(empty_entry_control) + "\n", encoding="utf-8")
    (injected_dir / "governance_log.jsonl").write_text(json.dumps(empty_entry_injected) + "\n", encoding="utf-8")

    output = calibrate_pilot(control_dir, injected_dir)

    assert output["control_gate_events_source"] == "governance_log_empty"
    assert output["injected_gate_events_source"] == "governance_log_empty"
    assert "control.gate_events_missing" in output["missing_fields"]
    assert "injected.gate_events_missing" in output["missing_fields"]


def test_calibrate_rejects_window_signature_hash_mismatch(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(
        control_dir,
        run_id="control",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_window=3,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
    )

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "WINDOW_SIGNATURE_HASH_MISMATCH"


def test_calibrate_rejects_summary_hash_tampered(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(
        control_dir,
        run_id="control",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
    )

    control_summary_path = control_dir / "results_summary.json"
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
    control_summary["window_signature_ref"]["hash"] = "bad"
    _write_json(control_summary_path, control_summary)

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "WINDOW_SIGNATURE_HASH_MISMATCH"


def test_calibrate_rejects_missing_window_signature_ref(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(
        control_dir,
        run_id="control",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
    )

    control_summary_path = control_dir / "results_summary.json"
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
    control_summary.pop("window_signature_ref", None)
    _write_json(control_summary_path, control_summary)

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "MISSING_WINDOW_SIGNATURE_REF"


def test_calibrate_rejects_empty_window_signature_ref_hash(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(
        control_dir,
        run_id="control",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
    )

    control_summary_path = control_dir / "results_summary.json"
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
    control_summary["window_signature_ref"] = {"hash": "", "path": "window_signature.json"}
    _write_json(control_summary_path, control_summary)

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "MISSING_WINDOW_SIGNATURE_REF_HASH"


@pytest.mark.xfail(
    reason=(
        "calibrate_pilot gate-event parsing currently rejects 'skip' decisions; "
        "see docs/audit_core_20260212.md"
    )
)
def test_calibrate_skip_gate_events(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(
        control_dir,
        run_id="control",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "skip"}],
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_window=2,
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
    )

    calibrate_pilot(control_dir, injected_dir)


def test_cli_calibrate_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(control_dir, run_id="control", gate_warmup=0, gates=[{"decision": "pass"}])
    _write_capsule(injected_dir, run_id="injected", gate_warmup=0, gates=[{"decision": "warn"}])

    args = Namespace(
        control_dir=str(control_dir),
        injected_dir=str(injected_dir),
        out=str(tmp_path / "calibration.json"),
        out_md=str(tmp_path / "calibration.md"),
    )
    exit_code = run_calibrate(args)
    assert exit_code == 0

    missing_dir = tmp_path / "missing"
    args_missing = Namespace(
        control_dir=str(control_dir),
        injected_dir=str(missing_dir),
        out=str(tmp_path / "calibration_missing.json"),
        out_md=str(tmp_path / "calibration_missing.md"),
    )
    exit_code_missing = run_calibrate(args_missing)
    assert exit_code_missing == 2

    captured = capsys.readouterr()
    assert "MISSING_RESULTS_SUMMARY" in captured.err


def test_cli_calibrate_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(control_dir, run_id="control", gate_warmup=0, gates=[{"decision": "pass"}])
    _write_capsule(injected_dir, run_id="injected", gate_warmup=0, gates=[{"decision": "warn"}])

    def _boom(_: Dict[str, Any]) -> str:
        raise RuntimeError("render failure")

    import veriscope.cli.calibrate as cli_calibrate

    monkeypatch.setattr(cli_calibrate, "render_calibration_md", _boom)

    args = Namespace(
        control_dir=str(control_dir),
        injected_dir=str(injected_dir),
        out=str(tmp_path / "calibration.json"),
        out_md=str(tmp_path / "calibration.md"),
    )
    exit_code = run_calibrate(args)
    assert exit_code == 3


def test_pilot_score_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"
    _write_capsule(control_dir, run_id="control", gate_warmup=0, gates=[{"decision": "pass"}])
    _write_capsule(injected_dir, run_id="injected", gate_warmup=0, gates=[{"decision": "warn"}])

    def _boom(_: Dict[str, Any]) -> str:
        raise RuntimeError("render failure")

    from scripts.pilot import score as pilot_score

    monkeypatch.setattr(pilot_score, "render_calibration_md", _boom)

    exit_code = pilot_score.main(
        [
            "--control-dir",
            str(control_dir),
            "--injected-dir",
            str(injected_dir),
            "--out",
            str(tmp_path / "calibration.json"),
            "--out-md",
            str(tmp_path / "calibration.md"),
        ]
    )
    assert exit_code == 3
