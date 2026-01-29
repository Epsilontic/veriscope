# tests/test_cli_inspect.py
from __future__ import annotations

import json
from datetime import datetime, timezone
import argparse
from pathlib import Path
from typing import Any

import pytest

from veriscope.cli.governance import append_run_started
from veriscope.cli.main import _cmd_inspect
from veriscope.core.artifacts import ResultsSummaryV1, ResultsV1
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


def _minimal_window_signature() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "code_identity": {"package_version": "test"},
        "gate_controls": {"gate_window": 16, "gate_epsilon": 0.08, "min_evidence": 16},
        "metric_interval": 16,
        "metric_pipeline": {"transport": "test"},
    }


def _make_minimal_artifacts(outdir: Path, *, run_id: str) -> None:
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

    res_obj = {**common, "gates": [], "metrics": []}
    summ_obj = {
        **common,
        "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
        "final_decision": "pass",
    }

    _write_json(outdir / "results.json", res_obj)
    _write_json(outdir / "results_summary.json", summ_obj)

    ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text(encoding="utf-8"))


def test_inspect_strict_governance_requires_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outdir = tmp_path / "run_a"
    _make_minimal_artifacts(outdir, run_id="run_a")

    args = argparse.Namespace(
        outdir=str(outdir),
        format="text",
        no_report=True,
        strict_governance=True,
        allow_legacy_governance=False,
        require_governance=True,
    )
    exit_code = _cmd_inspect(args)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Missing governance_log.jsonl for results.json" in captured.err


def test_inspect_strict_governance_warns_when_optional(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outdir = tmp_path / "run_b"
    _make_minimal_artifacts(outdir, run_id="run_b")

    args = argparse.Namespace(
        outdir=str(outdir),
        format="text",
        no_report=True,
        strict_governance=True,
        allow_legacy_governance=False,
        require_governance=False,
    )
    exit_code = _cmd_inspect(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WARNING:GOVERNANCE_LOG_MISSING" in captured.err


def _mutate_summary_run_id(outdir: Path, *, run_id: str) -> None:
    path = outdir / "results_summary.json"
    obj = _read_json_dict(path)
    obj["run_id"] = run_id
    _write_json(path, obj)


def test_inspect_allow_partial_warns_on_identity_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outdir = tmp_path / "run_identity_warn"
    _make_minimal_artifacts(outdir, run_id="run_identity_warn")
    append_run_started(
        outdir,
        run_id="run_identity_warn",
        outdir_path=outdir,
        argv=["pytest", "inspect_fixture"],
        code_identity={"package_version": "test"},
        window_signature_ref={
            "hash": canonical_json_sha256(_read_json_dict(outdir / "window_signature.json")),
            "path": "window_signature.json",
        },
        entrypoint={"kind": "runner", "name": "tests.inspect_fixture"},
        ts_utc=_iso_z(T0),
    )
    _mutate_summary_run_id(outdir, run_id="mismatched_run")

    args = argparse.Namespace(
        outdir=str(outdir),
        format="text",
        no_report=True,
        strict_governance=False,
        allow_legacy_governance=False,
        require_governance=False,
        strict_identity=False,
    )
    exit_code = _cmd_inspect(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WARNING:ARTIFACT_IDENTITY_MISMATCH" in captured.err


def test_inspect_strict_identity_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outdir = tmp_path / "run_identity_strict"
    _make_minimal_artifacts(outdir, run_id="run_identity_strict")
    _mutate_summary_run_id(outdir, run_id="mismatched_run")

    args = argparse.Namespace(
        outdir=str(outdir),
        format="text",
        no_report=True,
        strict_governance=False,
        allow_legacy_governance=False,
        require_governance=False,
        strict_identity=True,
    )
    exit_code = _cmd_inspect(args)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "ERROR:ARTIFACT_IDENTITY_MISMATCH" in captured.err
