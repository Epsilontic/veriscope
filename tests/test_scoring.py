# tests/test_scoring.py
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Optional

import pytest

from veriscope.cli.governance import append_run_started
from veriscope.core.governance import append_gate_decision
from veriscope.core.jsonutil import window_signature_sha256
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
    ws_hash = window_signature_sha256(window_signature)

    gates_out: list[dict[str, object]] = []
    for idx, gate in enumerate(gates):
        gate_out = dict(gate)
        gate_out.setdefault("iter", idx)
        decision = str(gate_out.get("decision", "skip")).strip().lower()
        gate_out.setdefault("ok", decision != "fail")
        gate_out.setdefault("warn", decision == "warn")
        gate_out.setdefault(
            "audit",
            {
                "evaluated": decision != "skip",
                "reason": "not_evaluated" if decision == "skip" else f"evaluated_{decision}",
                "policy": "test_policy",
                "per_metric_tv": {},
                "evidence_total": 0 if decision == "skip" else 1,
                "min_evidence": 1,
            },
        )
        gates_out.append(gate_out)

    results = {
        "schema_version": 1,
        "run_id": outdir.name,
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
    (outdir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    evaluated = sum(1 for g in gates_out if g.get("decision") in {"pass", "warn", "fail"})
    warn = sum(1 for g in gates_out if g.get("decision") == "warn")
    fail = sum(1 for g in gates_out if g.get("decision") == "fail")
    skip = sum(1 for g in gates_out if g.get("decision") == "skip")
    counts = {"evaluated": evaluated, "warn": warn, "fail": fail, "skip": skip, "pass": evaluated - warn - fail}
    if fail > 0:
        final_decision = "fail"
    elif warn > 0:
        final_decision = "warn"
    elif counts["pass"] > 0:
        final_decision = "pass"
    else:
        final_decision = "skip"
    summary = {
        "schema_version": 1,
        "run_id": outdir.name,
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test"},
        "run_status": "success",
        "runner_exit_code": 0,
        "runner_signal": None,
        "started_ts_utc": "2026-01-01T00:00:00Z",
        "ended_ts_utc": "2026-01-01T00:01:00Z",
        "counts": counts,
        "final_decision": final_decision,
    }
    if fail > 0:
        summary["first_fail_iter"] = min(int(g["iter"]) for g in gates_out if g.get("decision") == "fail")
    (outdir / "results_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if fail > 0:
        (outdir / "first_fail_iter.txt").write_text(f"{summary['first_fail_iter']}\n", encoding="utf-8")

    run_cfg = {"schema_version": 1, "resolved_gate_cfg": {"gate_warmup": gate_warmup}}
    if data_corrupt_at is not None:
        run_cfg["data_corrupt_at"] = int(data_corrupt_at)
    (outdir / "run_config_resolved.json").write_text(json.dumps(run_cfg), encoding="utf-8")

    append_run_started(
        outdir,
        run_id=outdir.name,
        outdir_path=outdir,
        argv=["pytest", "score_fixture"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.score_fixture"},
        ts_utc="2026-01-01T00:00:00Z",
    )
    for gate in gates_out:
        append_gate_decision(
            outdir,
            run_id=outdir.name,
            iter_num=int(gate["iter"]),
            decision=str(gate["decision"]),
            ok=bool(gate["ok"]),
            warn=bool(gate["warn"]),
            audit=dict(gate["audit"]),
            ts_utc="2026-01-01T00:00:00Z",
        )


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

    with pytest.raises(CalibrationError, match="INVALID_CONTROL_CAPSULE"):
        calibrate_pilot(control_dir, injected_dir)
