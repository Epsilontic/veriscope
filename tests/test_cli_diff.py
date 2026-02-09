# tests/test_cli_diff.py
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from veriscope.cli.comparability import comparable, comparable_explain, load_run_metadata
from veriscope.cli.governance import append_run_started
from veriscope.cli.diff import diff_outdirs
from veriscope.cli.report import render_report_compare
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ManualJudgementV1, ResultsSummaryV1, ResultsV1
from veriscope.core.governance import append_gate_decision
from veriscope.core.jsonutil import canonical_json_sha256

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_dict(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return obj


def _tamper_governance_log(outdir: Path) -> None:
    log_path = outdir / "governance_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError("governance_log.jsonl unexpectedly empty")
    first = json.loads(lines[0])
    payload = first.get("payload")
    if isinstance(payload, dict):
        payload["run_id"] = "tampered"
    lines[0] = json.dumps(first, sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _minimal_window_signature() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "code_identity": {"package_version": "test"},
        "gate_controls": {"gate_window": 16, "gate_epsilon": 0.08, "min_evidence": 16},
        "metric_interval": 16,
        "metric_pipeline": {"transport": "test"},
    }


def _make_minimal_artifacts(
    outdir: Path,
    *,
    run_id: str,
    gate_preset: str = "test",
    distributed: Optional[Dict[str, Any]] = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ws_path = outdir / "window_signature.json"
    _write_json(ws_path, _minimal_window_signature())
    ws_hash = canonical_json_sha256(_read_json_dict(ws_path))

    common = {
        "schema_version": 1,
        "run_id": run_id,
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": gate_preset, "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(T0),
        "ended_ts_utc": _iso_z(T1),
    }

    gates_obj = [
        {
            "iter": 0,
            "decision": "pass",
            "ok": True,
            "warn": False,
            "audit": {
                "evaluated": True,
                "reason": "evaluated_pass",
                "policy": "test_policy",
                "per_metric_tv": {},
                "evidence_total": 1,
                "min_evidence": 1,
            },
        }
    ]
    res_obj = {**common, "gates": gates_obj, "metrics": []}
    summ_obj = {
        **common,
        "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
        "final_decision": "pass",
    }

    _write_json(outdir / "results.json", res_obj)
    _write_json(outdir / "results_summary.json", summ_obj)

    append_run_started(
        outdir,
        run_id=run_id,
        outdir_path=outdir,
        argv=["pytest", "diff_fixture"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.diff_fixture"},
        distributed=distributed,
        ts_utc=_iso_z(T0),
    )
    for gate in gates_obj:
        append_gate_decision(
            outdir,
            run_id=run_id,
            iter_num=gate["iter"],
            decision=gate["decision"],
            ok=gate["ok"],
            warn=gate["warn"],
            audit=gate["audit"],
        )

    ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text(encoding="utf-8"))


def _rewrite_window_signature(outdir: Path, *, extra: str = "") -> None:
    ws_path = outdir / "window_signature.json"
    ws_obj = _minimal_window_signature()
    if extra:
        ws_obj["extra"] = extra
    _write_json(ws_path, ws_obj)
    ws_hash = canonical_json_sha256(_read_json_dict(ws_path))
    for name in ("results.json", "results_summary.json"):
        path = outdir / name
        obj = _read_json_dict(path)
        obj["window_signature_ref"]["hash"] = ws_hash
        _write_json(path, obj)


def _rewrite_profile_gate_preset(outdir: Path, *, gate_preset: str) -> None:
    for name in ("results.json", "results_summary.json"):
        path = outdir / name
        obj = _read_json_dict(path)
        obj["profile"]["gate_preset"] = gate_preset
        _write_json(path, obj)


def _write_manual_json(outdir: Path, *, status: str) -> None:
    run_id = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8")).run_id
    judgement = ManualJudgementV1(
        run_id=run_id,
        status=status,
        reason="manual override",
        reviewer="tester",
        ts_utc=T1,
    )
    _write_json(outdir / "manual_judgement.json", judgement.model_dump(by_alias=True, mode="json"))


def _write_manual_jsonl(outdir: Path, entries: list[ManualJudgementV1]) -> None:
    lines = []
    for idx, judgement in enumerate(entries, start=1):
        payload = judgement.model_dump(by_alias=True, mode="json")
        payload["rev"] = idx
        lines.append(json.dumps(payload, sort_keys=True))
    (outdir / "manual_judgement.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_diff_rejects_window_hash_mismatch(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")
    _rewrite_window_signature(outdir_b, extra="different")

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 2
    assert "WINDOW_HASH_MISMATCH" in result.stderr


def test_diff_rejects_gate_preset_mismatch_without_flag(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a", gate_preset="alpha")
    _make_minimal_artifacts(outdir_b, run_id="run_b", gate_preset="alpha")
    _rewrite_profile_gate_preset(outdir_b, gate_preset="beta")

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 2
    assert "GATE_PRESET_MISMATCH" in result.stderr

    allowed = diff_outdirs(outdir_a, outdir_b, allow_gate_preset_mismatch=True)
    assert allowed.exit_code == 0


def test_diff_rejects_intra_capsule_identity_mismatch(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    summary_obj = _read_json_dict(outdir_b / "results_summary.json")
    summary_obj["run_id"] = "run_b_mismatched"
    _write_json(outdir_b / "results_summary.json", summary_obj)

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 2
    assert "ARTIFACT_IDENTITY_MISMATCH" in result.stderr


def test_report_compare_rejects_intra_capsule_identity_mismatch(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    summary_obj = _read_json_dict(outdir_b / "results_summary.json")
    summary_obj["run_id"] = "run_b_mismatched"
    _write_json(outdir_b / "results_summary.json", summary_obj)

    result = render_report_compare([outdir_a, outdir_b], fmt="text")
    assert result.exit_code == 2
    assert "ARTIFACT_IDENTITY_MISMATCH" in result.stderr


def test_report_compare_rejects_invalid_governance(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")
    _tamper_governance_log(outdir_b)

    result = render_report_compare([outdir_a, outdir_b], fmt="text")
    assert result.exit_code == 2
    assert "governance_log.jsonl invalid" in result.stderr


def test_comparable_rejects_schema_mismatch(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    v_a = validate_outdir(outdir_a, allow_partial=True)
    v_b = validate_outdir(outdir_b, allow_partial=True)
    run_a = load_run_metadata(outdir_a, v_a, prefer_jsonl=True)
    run_b = load_run_metadata(outdir_b, v_b, prefer_jsonl=True)
    run_b = replace(run_b, schema_version=2)

    ok, reason = comparable(run_a, run_b)
    assert not ok
    assert reason == "SCHEMA_MISMATCH"


def test_comparable_explain_window_hash_mismatch_details(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")
    _rewrite_window_signature(outdir_b, extra="different")

    v_a = validate_outdir(outdir_a, allow_partial=True)
    v_b = validate_outdir(outdir_b, allow_partial=True)
    run_a = load_run_metadata(outdir_a, v_a, prefer_jsonl=True)
    run_b = load_run_metadata(outdir_b, v_b, prefer_jsonl=True)

    result = comparable_explain(run_a, run_b)
    assert not result.ok
    assert result.reason == "WINDOW_HASH_MISMATCH"
    assert result.details["window_signature_hash"]["expected"] == run_a.window_signature_hash
    assert result.details["window_signature_hash"]["got"] == run_b.window_signature_hash


def test_comparable_explain_gate_preset_policy(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a", gate_preset="alpha")
    _make_minimal_artifacts(outdir_b, run_id="run_b", gate_preset="alpha")
    _rewrite_profile_gate_preset(outdir_b, gate_preset="beta")

    v_a = validate_outdir(outdir_a, allow_partial=True)
    v_b = validate_outdir(outdir_b, allow_partial=True)
    run_a = load_run_metadata(outdir_a, v_a, prefer_jsonl=True)
    run_b = load_run_metadata(outdir_b, v_b, prefer_jsonl=True)

    result = comparable_explain(run_a, run_b)
    assert not result.ok
    assert result.reason == "GATE_PRESET_MISMATCH"
    assert result.details["gate_preset"]["expected"] == "alpha"
    assert result.details["gate_preset"]["got"] == "beta"
    assert result.policy == {"allow_gate_preset_mismatch": False}


def test_comparable_rejects_missing_window_hash(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    v_a = validate_outdir(outdir_a, allow_partial=True)
    v_b = validate_outdir(outdir_b, allow_partial=True)
    run_a = load_run_metadata(outdir_a, v_a, prefer_jsonl=True)
    run_b = load_run_metadata(outdir_b, v_b, prefer_jsonl=True)
    run_a = replace(run_a, validation=replace(run_a.validation, window_signature_hash=None))

    ok, reason = comparable(run_a, run_b)
    assert not ok
    assert reason == "WINDOW_HASH_MISSING"


def test_comparable_rejects_partial_capsules(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    (outdir_b / "results.json").unlink()
    marker_path = outdir_b / "first_fail_iter.txt"
    if marker_path.exists():
        marker_path.unlink()
    summ_path = outdir_b / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["partial"] = True
    summ_obj["counts"] = {"evaluated": 0, "skip": 0, "pass": 0, "warn": 0, "fail": 0}
    summ_obj["final_decision"] = "skip"
    summ_obj["first_fail_iter"] = None
    _write_json(summ_path, summ_obj)

    v_a = validate_outdir(outdir_a, allow_partial=True)
    v_b = validate_outdir(outdir_b, allow_partial=True)
    assert v_b.ok, v_b.message
    run_a = load_run_metadata(outdir_a, v_a, prefer_jsonl=True)
    run_b = load_run_metadata(outdir_b, v_b, prefer_jsonl=True)

    result = comparable_explain(run_a, run_b)
    assert not result.ok
    assert result.reason == "PARTIAL_CAPSULE"


def test_diff_rejects_partial_capsule(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    (outdir_b / "results.json").unlink()
    summ_path = outdir_b / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["partial"] = True
    summ_obj["counts"] = {"evaluated": 0, "skip": 0, "pass": 0, "warn": 0, "fail": 0}
    summ_obj["final_decision"] = "skip"
    summ_obj["first_fail_iter"] = None
    _write_json(summ_path, summ_obj)

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 2
    assert "PARTIAL_CAPSULE" in result.stderr


def test_diff_prefers_jsonl_manual_judgement(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    _write_manual_json(outdir_a, status="fail")
    _write_manual_json(outdir_b, status="pass")

    _write_manual_jsonl(
        outdir_a,
        [ManualJudgementV1(run_id="run_a", status="pass", reason="log", reviewer=None, ts_utc=T0)],
    )
    _write_manual_jsonl(
        outdir_b,
        [ManualJudgementV1(run_id="run_b", status="fail", reason="log", reviewer=None, ts_utc=T0)],
    )

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 0
    assert "manual_status: pass" in result.stdout
    assert "manual_status: fail" in result.stdout
    assert "manual_status_differs: yes" in result.stdout


def test_diff_json_includes_headers(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    result = diff_outdirs(outdir_a, outdir_b, fmt="json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["comparability"]["ok"] is True
    assert payload["run_a"]["run_id"] == "run_a"
    assert payload["run_b"]["run_id"] == "run_b"
    assert payload["partial_mode"] is False


def test_diff_skips_invalid_jsonl_lines(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    log_path = outdir_a / "manual_judgement.jsonl"
    valid = ManualJudgementV1(run_id="run_a", status="pass", reason="ok", reviewer=None, ts_utc=T1)
    payload = valid.model_dump(by_alias=True, mode="json")
    payload["rev"] = 1
    log_path.write_text("not json\n" + json.dumps(payload) + "\n", encoding="utf-8")

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 0
    assert "RUN_A:WARNING:MANUAL_JUDGEMENT_LOG_INVALID" in result.stderr
    assert "manual_status: pass" in result.stdout


def test_diff_falls_back_to_json_on_empty_jsonl(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    _write_manual_json(outdir_a, status="fail")
    (outdir_a / "manual_judgement.jsonl").write_text("not json\n", encoding="utf-8")

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 0
    assert "manual_status: fail" in result.stdout
    assert "RUN_A:WARNING:MANUAL_JUDGEMENT_LOG_INVALID" in result.stderr


def test_diff_warns_on_manual_run_id_mismatch(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    _make_minimal_artifacts(outdir_a, run_id="run_a")
    _make_minimal_artifacts(outdir_b, run_id="run_b")

    _write_manual_jsonl(
        outdir_a,
        [ManualJudgementV1(run_id="other_run", status="pass", reason="log", reviewer=None, ts_utc=T0)],
    )

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 0
    assert "manual_status: -" in result.stdout
    assert "RUN_A:WARNING:MANUAL_JUDGEMENT_RUN_ID_MISMATCH" in result.stderr


def test_diff_warns_on_distributed_metadata_mismatch(tmp_path: Path) -> None:
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"
    distributed_a = {
        "distributed_mode": "single_process",
        "world_size_observed": 1,
        "rank_observed": 0,
        "local_rank_observed": None,
        "ddp_backend": None,
        "ddp_active": False,
    }
    distributed_b = {
        "distributed_mode": "ddp_wrapped",
        "world_size_observed": 2,
        "rank_observed": 1,
        "local_rank_observed": 1,
        "ddp_backend": "nccl",
        "ddp_active": True,
    }

    _make_minimal_artifacts(outdir_a, run_id="run_a", distributed=distributed_a)
    _make_minimal_artifacts(outdir_b, run_id="run_b", distributed=distributed_b)

    result = diff_outdirs(outdir_a, outdir_b)
    assert result.exit_code == 0
    assert "WARNING:DISTRIBUTED_METADATA_MISMATCH" in result.stderr
