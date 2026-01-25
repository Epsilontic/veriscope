# tests/test_scoring.py
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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
    (outdir / "run_config_resolved.json").write_text(json.dumps(run_cfg), encoding="utf-8")


def test_scoring_far_and_delay(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    control_gates = [
        {"decision": "pass"},
        {"decision": "warn"},
        {"decision": "pass"},
        {"decision": "fail"},
        {"decision": "pass"},
    ]
    injected_gates = [
        {"decision": "pass"},
        {"decision": "pass"},
        {"decision": "warn"},
    ]

    _write_run(control_dir, gates=control_gates, gate_window=10, gate_warmup=2)
    _write_run(injected_dir, gates=injected_gates, gate_window=10, gate_warmup=2)

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
    assert payload["FAR"] == 1 / 3
    assert payload["Delay_W"] == 0 / 10


def test_scoring_delay_normalized_by_window(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    injected_dir = tmp_path / "injected"

    control_gates = [{"decision": "pass"}, {"decision": "pass"}, {"decision": "pass"}]
    injected_gates = [{"decision": "pass"}, {"decision": "pass"}, {"decision": "warn"}]

    _write_run(control_dir, gates=control_gates, gate_window=20, gate_warmup=1)
    _write_run(injected_dir, gates=injected_gates, gate_window=20, gate_warmup=1)

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
    assert payload["Delay_W"] == 1 / 20
