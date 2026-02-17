from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from veriscope.cli.main import main
from veriscope.cli.report import render_report_md
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ResultsSummaryV1, ResultsV1
from veriscope.core.assemble import assemble_capsule_from_jsonl
from veriscope.step_logging import UniversalStepLogger

pytestmark = pytest.mark.unit

_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def _read_json(path: Path) -> dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must be a JSON object")
    return obj


def _sha256_bytes(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def test_universal_logs_roundtrip_to_valid_capsule(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logger = UniversalStepLogger(logs_dir, run_id="universal_roundtrip")
    logger.log(
        step=0,
        loss=1.10,
        lr=1e-3,
        step_time=0.09,
        grad_norm=0.41,
        update_norm=0.03,
        overflow=0,
        nan_count=0,
    )
    logger.log(
        step=1,
        loss=1.31,
        lr=9.5e-4,
        step_time=0.10,
        grad_norm=0.48,
        update_norm=0.04,
        overflow=1,
        nan_count=0,
    )
    logger.finalize()

    outdir = tmp_path / "assembled_capsule"
    assembled = assemble_capsule_from_jsonl(
        outdir=outdir,
        logs_path=logger.jsonl_path,
        manifest_path=logger.manifest_path,
    )
    assert assembled.records == 2
    assert assembled.run_id == "universal_roundtrip"

    validate_result = validate_outdir(outdir, strict_identity=True)
    assert validate_result.ok, validate_result.message

    results = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    summary = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary.counts.evaluated == 2
    assert summary.counts.fail == 1
    assert summary.counts.warn == 0
    assert summary.counts.pass_ == 1
    assert summary.final_decision == "fail"
    assert summary.first_fail_iter == 1
    assert len(results.metrics) == 0
    assert results.metrics_ref is not None
    assert not results.metrics_ref.path.startswith("/")
    assert re.match(r"^[A-Za-z]:", results.metrics_ref.path) is None
    assert ".." not in Path(results.metrics_ref.path).parts
    assert not Path(results.metrics_ref.path).is_absolute()
    assert results.metrics_ref.format == "universal_jsonl_v1"
    assert results.metrics_ref.count == 2
    assert results.metrics_ref.sha256 is not None
    resolved_log_path = (outdir / results.metrics_ref.path).resolve()
    assert resolved_log_path.exists()
    resolved_log_path.parent.relative_to(outdir.resolve())
    assert results.metrics_ref.sha256 == _sha256_bytes(resolved_log_path)

    report_text = render_report_md(outdir, fmt="text")
    assert "Final Status: FAIL" in report_text
    assert "plumbing/safety gate only" in report_text


def test_universal_assemble_window_hash_stable_across_repeated_assembly(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logger = UniversalStepLogger(
        logs_dir,
        run_id="universal_hash_stability",
        gate_preset="universal_v0",
        calibration_hooks={"window_scope": {"profile": "default"}},
    )
    logger.log(step=0, loss=0.5, lr=1e-3, step_time=0.08, overflow=0, nan_count=0)
    logger.log(step=1, loss=0.6, lr=9e-4, step_time=0.09, overflow=0, nan_count=0)
    logger.finalize()

    manifest_obj = _read_json(logger.manifest_path)
    manifest_a = dict(manifest_obj)
    manifest_b = dict(manifest_obj)
    manifest_a["created_ts_utc"] = "2026-01-01T00:00:00Z"
    manifest_a["ended_ts_utc"] = "2026-01-01T00:10:00Z"
    manifest_b["created_ts_utc"] = "2026-01-02T00:00:00Z"
    manifest_b["ended_ts_utc"] = "2026-01-02T00:10:00Z"

    manifest_a_path = tmp_path / "manifest_a.json"
    manifest_b_path = tmp_path / "manifest_b.json"
    manifest_a_path.write_text(json.dumps(manifest_a), encoding="utf-8")
    manifest_b_path.write_text(json.dumps(manifest_b), encoding="utf-8")

    outdir_a = tmp_path / "assembled_a"
    outdir_b = tmp_path / "assembled_b"
    assembled_a = assemble_capsule_from_jsonl(
        outdir=outdir_a,
        logs_path=logger.jsonl_path,
        manifest_path=manifest_a_path,
    )
    assembled_b = assemble_capsule_from_jsonl(
        outdir=outdir_b,
        logs_path=logger.jsonl_path,
        manifest_path=manifest_b_path,
    )

    assert assembled_a.window_signature_hash == assembled_b.window_signature_hash
    ws_a = _read_json(outdir_a / "window_signature.json")
    ws_b = _read_json(outdir_b / "window_signature.json")
    assert ws_a == ws_b
    assert "created_ts_utc" not in ws_a
    assert "code_identity" not in ws_a

    v_a = validate_outdir(outdir_a, strict_identity=True)
    v_b = validate_outdir(outdir_b, strict_identity=True)
    assert v_a.ok, v_a.message
    assert v_b.ok, v_b.message


def test_universal_assemble_parses_string_overflow_values(tmp_path: Path) -> None:
    logs_path = tmp_path / "overflow_strings.jsonl"
    row_a = {
        "schema_version": 1,
        "record_type": "step",
        "run_id": "overflow_strings",
        "step": 0,
        "ts_utc": "2026-01-01T00:00:00Z",
        "loss": 0.5,
        "lr": 1e-3,
        "step_time": 0.08,
        "overflow": "0",
        "nan_count": 0,
    }
    row_b = {
        "schema_version": 1,
        "record_type": "step",
        "run_id": "overflow_strings",
        "step": 1,
        "ts_utc": "2026-01-01T00:00:01Z",
        "loss": 0.6,
        "lr": 9e-4,
        "step_time": 0.09,
        "overflow": "1",
        "nan_count": 0,
    }
    logs_path.write_text(f"{json.dumps(row_a)}\n{json.dumps(row_b)}\n", encoding="utf-8")

    outdir = tmp_path / "overflow_out"
    assemble_capsule_from_jsonl(outdir=outdir, logs_path=logs_path, run_id="overflow_strings")

    results = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    assert [gate.decision for gate in results.gates] == ["pass", "fail"]


def test_report_disclaimer_uses_universal_v0_prefix_match(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs_prefix"
    logger = UniversalStepLogger(logs_dir, run_id="prefix_disclaimer")
    logger.log(step=0, loss=0.4, lr=1e-3, step_time=0.1, overflow=0, nan_count=0)
    logger.finalize()

    outdir = tmp_path / "prefix_out"
    assemble_capsule_from_jsonl(
        outdir=outdir,
        logs_path=logger.jsonl_path,
        manifest_path=logger.manifest_path,
        gate_preset="universal_v0.experimental",
    )
    report_text = render_report_md(outdir, fmt="text")
    assert "plumbing/safety gate only" in report_text


def test_cli_assemble_supports_no_governance_mode(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logger = UniversalStepLogger(logs_dir, run_id="universal_cli")
    logger.log(step=0, loss=0.90, lr=1e-3, step_time=0.11, overflow=0, nan_count=0)
    logger.log(step=1, loss=0.87, lr=9.0e-4, step_time=0.10, overflow=0, nan_count=0)
    logger.finalize()

    outdir = tmp_path / "assembled_cli"
    exit_code = main(
        [
            "assemble",
            str(outdir),
            "--from",
            str(logger.jsonl_path),
            "--manifest",
            str(logger.manifest_path),
            "--no-governance",
        ]
    )
    assert exit_code == 0
    assert not (outdir / "governance_log.jsonl").exists()

    validate_result = validate_outdir(outdir, strict_identity=True, allow_missing_governance=True)
    assert validate_result.ok, validate_result.message
    summary_obj = _read_json(outdir / "results_summary.json")
    assert summary_obj.get("final_decision") == "pass"
