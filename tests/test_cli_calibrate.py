# tests/test_cli_calibrate.py
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from veriscope.cli.calibrate import run_calibrate
from veriscope.cli.governance import append_run_started
from veriscope.core.governance import append_gate_decision
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
    gate_preset: str = "test",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    window_signature = {
        "schema_version": 1,
        "gate_controls": {"gate_window": gate_window},
    }
    ws_hash = window_signature_sha256(window_signature)
    _write_json(outdir / "window_signature.json", window_signature)

    gates_out: List[Dict[str, Any]] = []
    for idx, gate in enumerate(gates or []):
        gate_out = dict(gate)
        if "iter" not in gate_out:
            gate_out["iter"] = idx
        decision = str(gate_out.get("decision", "skip")).strip().lower()
        if "ok" not in gate_out:
            gate_out["ok"] = decision != "fail"
        if "warn" not in gate_out:
            gate_out["warn"] = decision == "warn"
        if "audit" not in gate_out:
            gate_out["audit"] = {
                "evaluated": decision != "skip",
                "reason": "not_evaluated" if decision == "skip" else f"evaluated_{decision}",
                "policy": "test_policy",
                "per_metric_tv": {},
                "evidence_total": 0 if decision == "skip" else 1,
                "min_evidence": 1,
            }
        gates_out.append(gate_out)

    counts = {
        "evaluated": sum(1 for gate in gates_out if gate.get("decision") in {"pass", "warn", "fail"}),
        "skip": sum(1 for gate in gates_out if gate.get("decision") == "skip"),
        "pass": sum(1 for gate in gates_out if gate.get("decision") == "pass"),
        "warn": sum(1 for gate in gates_out if gate.get("decision") == "warn"),
        "fail": sum(1 for gate in gates_out if gate.get("decision") == "fail"),
    }
    if counts["fail"] > 0:
        final_decision = "fail"
    elif counts["warn"] > 0:
        final_decision = "warn"
    elif counts["pass"] > 0:
        final_decision = "pass"
    else:
        final_decision = "skip"

    results_summary = {
        "schema_version": 1,
        "run_id": run_id,
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": gate_preset},
        "run_status": "success",
        "runner_exit_code": 0,
        "runner_signal": None,
        "started_ts_utc": "2026-01-01T00:00:00Z",
        "ended_ts_utc": "2026-01-01T00:01:00Z",
        "counts": counts,
        "final_decision": final_decision,
    }
    if counts["fail"] > 0:
        results_summary["first_fail_iter"] = min(
            int(gate["iter"]) for gate in gates_out if gate.get("decision") == "fail"
        )
    _write_json(outdir / "results_summary.json", results_summary)
    if counts["fail"] > 0:
        (outdir / "first_fail_iter.txt").write_text(f"{results_summary['first_fail_iter']}\n", encoding="utf-8")

    if with_results:
        results = {
            "schema_version": 1,
            "run_id": run_id,
            "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
            "profile": {"gate_preset": gate_preset},
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
        append_run_started(
            outdir,
            run_id=run_id,
            outdir_path=outdir,
            argv=["pytest", "calibration_fixture"],
            code_identity={"package_version": "test"},
            window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
            entrypoint={"kind": "runner", "name": "tests.calibration_fixture"},
            ts_utc="2026-01-01T00:00:00Z",
        )
        for gate in gates_out:
            decision = str(gate.get("decision"))
            append_gate_decision(
                outdir,
                run_id=run_id,
                iter_num=int(gate["iter"]),
                decision=decision,
                ok=bool(gate["ok"]),
                warn=bool(gate["warn"]),
                audit=dict(gate.get("audit", {})),
                ts_utc="2026-01-01T00:00:00Z",
            )

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
        with_governance=True,
    )

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "INVALID_INJECTED_CAPSULE"


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
        with_governance=True,
    )
    (injected_dir / "results.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "INVALID_INJECTED_CAPSULE"


def test_calibrate_rejects_governance_results_divergence(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    _write_capsule(
        control_dir,
        run_id="control",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
        data_corrupt_at=0,
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
    )
    stale_fail = {
        "evaluated": True,
        "reason": "evaluated_fail",
        "policy": "test_policy",
        "per_metric_tv": {},
        "evidence_total": 1,
        "min_evidence": 1,
    }
    append_gate_decision(
        control_dir,
        run_id="control",
        iter_num=0,
        decision="fail",
        ok=False,
        warn=False,
        audit=stale_fail,
        ts_utc="2026-01-01T00:00:00Z",
    )

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "INVALID_CONTROL_CAPSULE"
    assert "GOVERNANCE_RESULTS_DIVERGENCE" in exc_info.value.message


def test_calibrate_rejects_gate_preset_mismatch_by_default(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    _write_capsule(
        control_dir,
        run_id="control",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "pass"}],
        gate_preset="alpha",
    )
    _write_capsule(
        injected_dir,
        run_id="injected",
        gate_warmup=0,
        gates=[{"iter": 0, "decision": "warn"}],
        data_corrupt_at=0,
        gate_preset="beta",
    )

    with pytest.raises(CalibrationError) as exc_info:
        calibrate_pilot(control_dir, injected_dir)

    assert exc_info.value.token == "GATE_PRESET_MISMATCH"


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

    assert exc_info.value.token == "INVALID_CONTROL_CAPSULE"
    assert "window_signature_ref" in exc_info.value.message


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

    assert exc_info.value.token == "INVALID_CONTROL_CAPSULE"
    assert "window_signature_ref" in exc_info.value.message


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

    assert exc_info.value.token == "INVALID_CONTROL_CAPSULE"
    assert "window_signature_ref" in exc_info.value.message


@pytest.mark.xfail(
    reason=("calibrate_pilot gate-event parsing currently rejects 'skip' decisions; see docs/audit_core_20260212.md")
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
