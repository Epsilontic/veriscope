# tests/test_cli_validate_report.py
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from veriscope.cli.governance import GOVERNANCE_LOG_SCHEMA_VERSION, append_governance_log, append_run_started
from veriscope.cli.override import write_manual_judgement
from veriscope.cli.report import render_report_md
from veriscope.core.jsonutil import canonical_dumps
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ManualJudgementV1, ResultsSummaryV1, ResultsV1
from veriscope.core.governance import append_gate_decision
from veriscope.core.jsonutil import canonical_json_sha256

pytestmark = pytest.mark.unit

# Fixed timestamps for deterministic testing
T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_obj(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_dict(path: Path) -> dict[str, Any]:
    obj = _read_json_obj(path)
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object, got {type(obj).__name__}")
    return obj


def _minimal_window_signature() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "code_identity": {"package_version": "test"},
        "gate_controls": {"gate_window": 16, "gate_epsilon": 0.08, "min_evidence": 16},
        "metric_interval": 16,
        "metric_pipeline": {"transport": "test"},
    }


def _make_minimal_artifacts(outdir: Path, *, run_id: str = "test_run_min") -> None:
    """
    Write the minimal canonical artifact set needed to satisfy validate_outdir()
    and render_report_md(), without depending on the emitter.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) window_signature.json
    ws_path = outdir / "window_signature.json"
    _write_json(ws_path, _minimal_window_signature())

    # Hash what is actually on disk (parsed back), to mirror consumer behavior
    ws_hash = canonical_json_sha256(_read_json_dict(ws_path))

    # Optional: validate the window signature if a model exists.
    # IMPORTANT: only swallow "model not available" errors, NOT validation errors.
    try:
        from veriscope.core.artifacts import WindowSignatureV1  # type: ignore
    except (ImportError, AttributeError):
        WindowSignatureV1 = None  # type: ignore[assignment]

    if WindowSignatureV1 is not None:
        WindowSignatureV1.model_validate(_read_json_dict(ws_path))  # type: ignore[attr-defined]

    common = {
        "schema_version": 1,
        "run_id": run_id,
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
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
                "per_metric_tv": {"m": 0.01},
                "evidence_total": 16,
                "min_evidence": 16,
            },
        },
        {
            "iter": 1,
            "decision": "warn",
            "ok": True,
            "warn": True,
            "audit": {
                "evaluated": True,
                "reason": "evaluated_warn",
                "policy": "test_policy",
                "per_metric_tv": {"m": 0.02},
                "evidence_total": 16,
                "min_evidence": 16,
            },
        },
        {
            "iter": 2,
            "decision": "fail",
            "ok": False,
            "warn": False,
            "audit": {
                "evaluated": True,
                "reason": "evaluated_fail",
                "policy": "test_policy",
                "per_metric_tv": {"m": 0.12},
                "evidence_total": 16,
                "min_evidence": 16,
            },
        },
        {
            "iter": 3,
            "decision": "skip",
            "ok": True,
            "warn": False,
            "audit": {
                "evaluated": False,
                "reason": "not_evaluated",
                "policy": "test_policy",
                "per_metric_tv": {},
                "evidence_total": 0,
                "min_evidence": 16,
            },
        },
    ]

    res_obj = {**common, "gates": gates_obj, "metrics": []}

    # Counts correspond to: total=4 derived from evaluated+skip
    summ_obj = {
        **common,
        "counts": {"evaluated": 3, "skip": 1, "pass": 1, "warn": 1, "fail": 1},
        "final_decision": "fail",
        "first_fail_iter": 2,
    }

    _write_json(outdir / "results.json", res_obj)
    _write_json(outdir / "results_summary.json", summ_obj)
    (outdir / "first_fail_iter.txt").write_text("2\n", encoding="utf-8")

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

    # Sanity: fixtures must conform to schemas
    ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text(encoding="utf-8"))


def _rewrite_run_status(outdir: Path, run_status: str) -> None:
    res_path = outdir / "results.json"
    summ_path = outdir / "results_summary.json"
    res_obj = _read_json_dict(res_path)
    summ_obj = _read_json_dict(summ_path)
    res_obj["run_status"] = run_status
    summ_obj["run_status"] = run_status
    _write_json(res_path, res_obj)
    _write_json(summ_path, summ_obj)


def _write_manual_judgement(outdir: Path, *, status: str = "fail") -> None:
    run_id = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8")).run_id
    judgement = ManualJudgementV1(
        run_id=run_id,
        status=status,
        reason="Model collapsed despite automated PASS.",
        reviewer="alice",
        ts_utc=T1,
    )
    _write_json(outdir / "manual_judgement.json", judgement.model_dump(by_alias=True, mode="json"))


def _write_valid_governance_log(outdir: Path) -> None:
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "test_run_fixture", "note": "initial governance entry"},
        ts_utc=_iso_z(T0),
        actor="tester",
    )


def _write_legacy_governance_log(outdir: Path) -> None:
    entry = {
        "schema_version": GOVERNANCE_LOG_SCHEMA_VERSION,
        "rev": 1,
        "ts_utc": _iso_z(T0),
        "actor": "tester",
        "event_type": "artifact_note",
        "payload": {"run_id": "test_run_fixture", "note": "legacy entry"},
        "prev_hash": None,
    }
    (outdir / "governance_log.jsonl").write_text(canonical_dumps(entry) + "\n", encoding="utf-8")


def _tamper_governance_log(outdir: Path) -> None:
    path = outdir / "governance_log.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    obj = json.loads(lines[0])
    payload = obj.get("payload")
    if isinstance(payload, dict):
        payload["note"] = "tampered"
    elif "run_id" in obj:
        obj["run_id"] = f"{obj['run_id']}-tampered"
    else:
        obj["event"] = "tampered"
    lines[0] = json.dumps(obj, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def minimal_artifact_dir(tmp_path: Path) -> Path:
    outdir = tmp_path / "out_minimal"
    _make_minimal_artifacts(outdir, run_id="test_run_fixture")
    return outdir


def _mutate_delete_results_json(d: Path) -> None:
    (d / "results.json").unlink()


def _mutate_corrupt_window_signature_json(d: Path) -> None:
    (d / "window_signature.json").write_text("{bad json", encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "expected_token", "require_single_line"),
    [
        (_mutate_delete_results_json, "results.json", False),
        (_mutate_corrupt_window_signature_json, "window_signature.json", True),
    ],
)
def test_validate_failure_modes_minimal(
    minimal_artifact_dir: Path,
    mutate: Callable[[Path], None],
    expected_token: str,
    require_single_line: bool,
) -> None:
    mutate(minimal_artifact_dir)

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert expected_token in v.message
    assert "Traceback" not in v.message
    if require_single_line:
        assert "\n" not in v.message


def test_validate_happy_path_minimal(minimal_artifact_dir: Path) -> None:
    v = validate_outdir(minimal_artifact_dir)
    assert v.ok, v.message
    assert v.window_signature_hash is not None


def test_validate_allows_user_code_failure_status(minimal_artifact_dir: Path) -> None:
    _rewrite_run_status(minimal_artifact_dir, "user_code_failure")
    ResultsV1.model_validate_json((minimal_artifact_dir / "results.json").read_text(encoding="utf-8"))
    ResultsSummaryV1.model_validate_json((minimal_artifact_dir / "results_summary.json").read_text(encoding="utf-8"))
    v = validate_outdir(minimal_artifact_dir)
    assert v.ok, v.message


def test_counts_v1_canonical_json_key_is_pass() -> None:
    """Lock the public JSON contract: CountsV1 uses `pass` as the canonical JSON key."""
    from veriscope.core.artifacts import CountsV1

    c = CountsV1.model_validate({"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0})
    dumped = c.model_dump(by_alias=True, mode="json")

    assert "pass" in dumped
    assert "pass_" not in dumped
    assert dumped["pass"] == 1


def test_validate_detects_tampering_minimal(minimal_artifact_dir: Path) -> None:
    ws_path = minimal_artifact_dir / "window_signature.json"
    ws_obj = _read_json_dict(ws_path)
    ws_obj["tampered_field"] = "malicious_value"
    _write_json(ws_path, ws_obj)

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert re.search(r"window_signature_ref\.hash", v.message, flags=re.IGNORECASE)
    assert re.search(r"mismatch", v.message, flags=re.IGNORECASE)


def test_validate_rejects_non_string_created_ts_utc(minimal_artifact_dir: Path) -> None:
    ws_path = minimal_artifact_dir / "window_signature.json"
    ws_obj = _read_json_dict(ws_path)
    ws_obj["created_ts_utc"] = {"tampered": {"gate_controls": {"gate_epsilon": 999}}}
    _write_json(ws_path, ws_obj)

    v = validate_outdir(minimal_artifact_dir, allow_partial=True)
    assert not v.ok
    assert "created_ts_utc" in v.message


def test_validate_detects_run_id_mismatch_minimal(minimal_artifact_dir: Path) -> None:
    p = minimal_artifact_dir / "results_summary.json"
    obj = _read_json_dict(p)
    obj["run_id"] = "different_run_id"
    _write_json(p, obj)

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "run_id" in v.message
    assert re.search(r"mismatch", v.message, flags=re.IGNORECASE)


def test_validate_identity_mismatch_fails_default(minimal_artifact_dir: Path) -> None:
    p = minimal_artifact_dir / "results_summary.json"
    obj = _read_json_dict(p)
    obj["profile"]["gate_preset"] = "different"
    _write_json(p, obj)

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "ARTIFACT_IDENTITY_MISMATCH" in v.message


def test_report_rejects_identity_mismatch(minimal_artifact_dir: Path) -> None:
    p = minimal_artifact_dir / "results_summary.json"
    obj = _read_json_dict(p)
    obj["run_id"] = "different_run_id"
    _write_json(p, obj)

    with pytest.raises(ValueError, match=r"Cannot report: .*ARTIFACT_IDENTITY_MISMATCH"):
        render_report_md(minimal_artifact_dir, fmt="text")


def test_validate_detects_summary_counts_mismatch_results(minimal_artifact_dir: Path) -> None:
    summ_path = minimal_artifact_dir / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["counts"] = {"evaluated": 2, "skip": 2, "pass": 1, "warn": 1, "fail": 0}
    summ_obj["final_decision"] = "warn"
    summ_obj["first_fail_iter"] = None
    _write_json(summ_path, summ_obj)

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "summary counts mismatch results gates" in v.message


def test_validate_requires_first_fail_marker_when_fails_present(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "first_fail_iter.txt").unlink()

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "counts.fail > 0 but first_fail_iter.txt is missing" in v.message


def test_validate_partial_requires_explicit_partial_marker(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results.json").unlink()

    v = validate_outdir(minimal_artifact_dir, allow_partial=True)
    assert not v.ok
    assert "not marked partial=true" in v.message


def test_report_smoke_and_integrity_minimal(minimal_artifact_dir: Path) -> None:
    """
    Truth is asserted via Pydantic; report checks are presentation-layer smoke checks.
    """
    summ = ResultsSummaryV1.model_validate_json(
        (minimal_artifact_dir / "results_summary.json").read_text(encoding="utf-8")
    )
    assert summ.counts.evaluated == 3
    assert summ.counts.skip == 1
    assert summ.counts.pass_ == 1
    assert summ.counts.warn == 1
    assert summ.counts.fail == 1

    derived_total = int(summ.counts.evaluated) + int(summ.counts.skip)
    assert derived_total == 4  # explicit fixture contract

    md = render_report_md(minimal_artifact_dir, fmt="md")
    assert "# Veriscope Report" in md
    assert "test_run_fixture" in md
    assert "## Gate Summary" in md
    assert "## Artifacts" in md

    # Prefer stable text output for numeric assertions
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert f"Total: {derived_total}" in txt


def test_report_governance_absent(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "governance_log.jsonl").unlink()
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "GOVERNANCE:" in txt
    assert "Log: NO" in txt


def test_report_governance_valid(minimal_artifact_dir: Path) -> None:
    _write_valid_governance_log(minimal_artifact_dir)
    gov_lines = (minimal_artifact_dir / "governance_log.jsonl").read_text(encoding="utf-8").splitlines()
    expected_last_rev = json.loads(gov_lines[-1])["rev"]
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "Log: YES" in txt
    assert "Valid: YES" in txt
    assert f"Last Rev: {expected_last_rev}" in txt


def test_report_governance_invalid_tampered(minimal_artifact_dir: Path) -> None:
    _write_valid_governance_log(minimal_artifact_dir)
    _tamper_governance_log(minimal_artifact_dir)
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "Log: YES" in txt
    assert "Valid: NO" in txt
    assert "GOVERNANCE_LOG_HASH_MISMATCH" in txt


def test_report_governance_legacy_missing_entry_hash(minimal_artifact_dir: Path) -> None:
    _write_legacy_governance_log(minimal_artifact_dir)
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "Legacy (missing entry_hash): YES" in txt
    assert "Valid: YES" in txt
    assert "Evaluated: 3" in txt
    assert "PASS: 1" in txt
    assert "SKIP: 1" in txt
    assert "WARN: 1" in txt
    assert "FAIL: 1" in txt


def test_report_manual_judgement_overlay(minimal_artifact_dir: Path) -> None:
    _write_manual_judgement(minimal_artifact_dir, status="fail")

    md = render_report_md(minimal_artifact_dir, fmt="md")
    assert "## ⚠ Manual Judgement" in md
    assert "Final Status" in md
    assert "Model collapsed despite automated PASS." in md
    assert "MANUAL FAIL" in md

    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "MANUAL JUDGEMENT:" in txt
    assert "Final Status: MANUAL FAIL" in txt


def test_validate_fails_on_malformed_governance_log(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "governance_log.jsonl").write_text("{bad json\n", encoding="utf-8")
    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "governance_log.jsonl invalid" in v.message


def test_validate_rejects_invalid_manual_judgement(minimal_artifact_dir: Path) -> None:
    _write_json(
        minimal_artifact_dir / "manual_judgement.json",
        {
            "schema_version": 1,
            "run_id": "test_run_fixture",
            "status": "invalid",
            "reason": "bad status",
            "ts_utc": _iso_z(T1),
        },
    )

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "Schema validation failed" in v.message


def test_validate_rejects_manual_judgement_run_id_mismatch(minimal_artifact_dir: Path) -> None:
    _write_json(
        minimal_artifact_dir / "manual_judgement.json",
        {
            "schema_version": 1,
            "run_id": "wrong_run_id",
            "status": "fail",
            "reason": "mismatched run id",
            "ts_utc": _iso_z(T1),
        },
    )

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "run_id mismatch" in v.message


def test_override_accepts_naive_ts_and_serializes_utc_z(minimal_artifact_dir: Path) -> None:
    path, warnings = write_manual_judgement(
        minimal_artifact_dir,
        status="pass",
        reason="naive timestamp accepted",
        ts_utc="2026-01-01T00:00:00",
        force=True,
    )
    assert path.exists()
    assert warnings == ()
    obj = _read_json_dict(path)
    assert obj["ts_utc"] == "2026-01-01T00:00:00Z"

    gov_path = minimal_artifact_dir / "governance_log.jsonl"
    assert gov_path.exists()
    last_entry = json.loads(gov_path.read_text(encoding="utf-8").splitlines()[-1])
    event = last_entry.get("event") or last_entry.get("event_type")
    assert event == "manual_judgement_set"
    assert last_entry["ts_utc"] == "2026-01-01T00:00:00Z"


def test_report_raises_on_invalid_dir_minimal(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results_summary.json").unlink()
    with pytest.raises(ValueError, match=r"Cannot report"):
        render_report_md(minimal_artifact_dir)


def test_partial_summary_allows_report_without_results(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results.json").unlink()
    summ_path = minimal_artifact_dir / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["partial"] = True
    summ_obj["counts"] = {"evaluated": 0, "skip": 0, "pass": 0, "warn": 0, "fail": 0}
    summ_obj["final_decision"] = "skip"
    summ_obj["first_fail_iter"] = None
    _write_json(summ_path, summ_obj)
    (minimal_artifact_dir / "first_fail_iter.txt").unlink(missing_ok=True)

    v = validate_outdir(minimal_artifact_dir, allow_partial=True)
    assert v.ok, v.message
    assert v.partial is True

    md = render_report_md(minimal_artifact_dir, fmt="md")
    assert "# Veriscope Report" in md
    assert "test_run_fixture" in md

    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "Gate Preset: test" in txt
    assert "Started:" in txt


def test_partial_summary_rejects_non_neutral_counts_without_results(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results.json").unlink()
    summ_path = minimal_artifact_dir / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["partial"] = True
    _write_json(summ_path, summ_obj)

    v = validate_outdir(minimal_artifact_dir, allow_partial=True)
    assert not v.ok
    assert "requires neutral summary counts" in v.message


def test_validate_rejects_stale_first_fail_marker_when_fail_zero(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results.json").unlink()
    summ_path = minimal_artifact_dir / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["partial"] = True
    summ_obj["counts"] = {"evaluated": 0, "skip": 0, "pass": 0, "warn": 0, "fail": 0}
    summ_obj["final_decision"] = "skip"
    summ_obj["first_fail_iter"] = None
    _write_json(summ_path, summ_obj)
    (minimal_artifact_dir / "first_fail_iter.txt").write_text("950\n", encoding="utf-8")

    v = validate_outdir(minimal_artifact_dir, allow_partial=True)
    assert not v.ok
    assert "counts.fail == 0 but first_fail_iter.txt must be absent" in v.message
