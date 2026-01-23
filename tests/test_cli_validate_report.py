# tests/test_cli_validate_report.py
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from veriscope.cli.governance import GOVERNANCE_LOG_SCHEMA_VERSION, append_governance_log
from veriscope.cli.report import render_report_md
from veriscope.core.jsonutil import canonical_dumps
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ManualJudgementV1, ResultsSummaryV1, ResultsV1
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

    res_obj = {**common, "gates": [], "metrics": []}

    # Counts correspond to: total=4 derived from evaluated+skip
    summ_obj = {
        **common,
        "counts": {"evaluated": 3, "skip": 1, "pass": 1, "warn": 1, "fail": 1},
        "final_decision": "fail",
    }

    _write_json(outdir / "results.json", res_obj)
    _write_json(outdir / "results_summary.json", summ_obj)

    # Sanity: fixtures must conform to schemas
    ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text(encoding="utf-8"))


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
    obj["payload"]["note"] = "tampered"
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


def test_validate_detects_run_id_mismatch_minimal(minimal_artifact_dir: Path) -> None:
    p = minimal_artifact_dir / "results_summary.json"
    obj = _read_json_dict(p)
    obj["run_id"] = "different_run_id"
    _write_json(p, obj)

    v = validate_outdir(minimal_artifact_dir)
    assert not v.ok
    assert "run_id" in v.message
    assert re.search(r"mismatch", v.message, flags=re.IGNORECASE)


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
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "GOVERNANCE:" in txt
    assert "Log: NO" in txt


def test_report_governance_valid(minimal_artifact_dir: Path) -> None:
    _write_valid_governance_log(minimal_artifact_dir)
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "Log: YES" in txt
    assert "Valid: YES" in txt
    assert "Last Rev: 1" in txt


def test_report_governance_invalid_tampered(minimal_artifact_dir: Path) -> None:
    _write_valid_governance_log(minimal_artifact_dir)
    _tamper_governance_log(minimal_artifact_dir)
    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "⚠ Governance log invalid" in txt
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


def test_report_raises_on_invalid_dir_minimal(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results_summary.json").unlink()
    with pytest.raises(ValueError, match=r"Cannot report"):
        render_report_md(minimal_artifact_dir)


def test_partial_summary_allows_report_without_results(minimal_artifact_dir: Path) -> None:
    (minimal_artifact_dir / "results.json").unlink()
    summ_path = minimal_artifact_dir / "results_summary.json"
    summ_obj = _read_json_dict(summ_path)
    summ_obj["partial"] = True
    _write_json(summ_path, summ_obj)

    v = validate_outdir(minimal_artifact_dir, allow_partial=True)
    assert v.ok, v.message
    assert v.partial is True

    md = render_report_md(minimal_artifact_dir, fmt="md")
    assert "# Veriscope Report" in md
    assert "test_run_fixture" in md

    txt = render_report_md(minimal_artifact_dir, fmt="text")
    assert "Gate Preset: test" in txt
    assert "Started:" in txt
