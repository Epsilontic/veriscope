from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from veriscope.cli.governance import append_run_started
from veriscope.cli.validate import validate_outdir
from veriscope.core.governance import append_gate_decision
from veriscope.core.jsonutil import canonical_json_sha256

pytestmark = pytest.mark.unit


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_validate_fails_when_world_size_gt_1_and_distributed_mode_missing(tmp_path: Path) -> None:
    outdir = tmp_path / "capsule"
    outdir.mkdir(parents=True, exist_ok=True)

    ws_obj = {
        "schema_version": 1,
        "transport": {"name": "test_transport"},
        "evidence": {"metrics": ["m"]},
        "gates": {"preset": "test"},
    }
    _write_json(outdir / "window_signature.json", ws_obj)
    ws_hash = canonical_json_sha256(ws_obj)

    gate_audit = {
        "evaluated": True,
        "reason": "ok",
        "policy": "test_policy",
        "per_metric_tv": {"m": 0.01},
        "evidence_total": 16,
        "min_evidence": 16,
    }
    common = {
        "schema_version": 1,
        "run_id": "run_distributed_missing_mode",
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
        "ended_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)),
    }
    _write_json(
        outdir / "results.json",
        {
            **common,
            "gates": [{"iter": 0, "decision": "pass", "ok": True, "warn": False, "audit": gate_audit}],
            "metrics": [],
        },
    )
    _write_json(
        outdir / "results_summary.json",
        {
            **common,
            "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
            "final_decision": "pass",
        },
    )

    append_run_started(
        outdir,
        run_id="run_distributed_missing_mode",
        outdir_path=outdir,
        argv=["pytest", "distributed_mode_contract"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.distributed_mode_contract"},
        distributed={
            "world_size_observed": 2,
            "backend": "nccl",
            "rank": 0,
            "local_rank": 0,
            "ddp_wrapped": True,
        },
    )
    append_gate_decision(
        outdir,
        run_id="run_distributed_missing_mode",
        iter_num=0,
        decision="pass",
        ok=True,
        warn=False,
        audit=gate_audit,
    )

    v = validate_outdir(outdir)
    assert not v.ok
    assert "ERROR:DISTRIBUTED_MODE_MISSING" in v.message
    assert any("ERROR:DISTRIBUTED_MODE_MISSING" in error for error in v.errors)


def test_validate_passes_when_world_size_gt_1_and_distributed_mode_valid(tmp_path: Path) -> None:
    outdir = tmp_path / "capsule"
    outdir.mkdir(parents=True, exist_ok=True)

    ws_obj = {
        "schema_version": 1,
        "transport": {"name": "test_transport"},
        "evidence": {"metrics": ["m"]},
        "gates": {"preset": "test"},
    }
    _write_json(outdir / "window_signature.json", ws_obj)
    ws_hash = canonical_json_sha256(ws_obj)

    gate_audit = {
        "evaluated": True,
        "reason": "ok",
        "policy": "test_policy",
        "per_metric_tv": {"m": 0.01},
        "evidence_total": 16,
        "min_evidence": 16,
    }
    common = {
        "schema_version": 1,
        "run_id": "run_distributed_valid_mode",
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
        "ended_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)),
    }
    _write_json(
        outdir / "results.json",
        {
            **common,
            "gates": [{"iter": 0, "decision": "pass", "ok": True, "warn": False, "audit": gate_audit}],
            "metrics": [],
        },
    )
    _write_json(
        outdir / "results_summary.json",
        {
            **common,
            "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
            "final_decision": "pass",
        },
    )

    append_run_started(
        outdir,
        run_id="run_distributed_valid_mode",
        outdir_path=outdir,
        argv=["pytest", "distributed_mode_contract"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.distributed_mode_contract"},
        distributed={
            "distributed_mode": "ddp_wrapped",
            "world_size_observed": 2,
            "backend": "nccl",
            "rank": 0,
            "local_rank": 0,
            "ddp_wrapped": True,
        },
    )
    append_gate_decision(
        outdir,
        run_id="run_distributed_valid_mode",
        iter_num=0,
        decision="pass",
        ok=True,
        warn=False,
        audit=gate_audit,
    )

    v = validate_outdir(outdir)
    assert v.ok


def test_validate_fails_when_distributed_mode_invalid(tmp_path: Path) -> None:
    outdir = tmp_path / "capsule"
    outdir.mkdir(parents=True, exist_ok=True)

    ws_obj = {
        "schema_version": 1,
        "transport": {"name": "test_transport"},
        "evidence": {"metrics": ["m"]},
        "gates": {"preset": "test"},
    }
    _write_json(outdir / "window_signature.json", ws_obj)
    ws_hash = canonical_json_sha256(ws_obj)

    gate_audit = {
        "evaluated": True,
        "reason": "ok",
        "policy": "test_policy",
        "per_metric_tv": {"m": 0.01},
        "evidence_total": 16,
        "min_evidence": 16,
    }
    common = {
        "schema_version": 1,
        "run_id": "run_distributed_invalid_mode",
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
        "ended_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)),
    }
    _write_json(
        outdir / "results.json",
        {
            **common,
            "gates": [{"iter": 0, "decision": "pass", "ok": True, "warn": False, "audit": gate_audit}],
            "metrics": [],
        },
    )
    _write_json(
        outdir / "results_summary.json",
        {
            **common,
            "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
            "final_decision": "pass",
        },
    )

    append_run_started(
        outdir,
        run_id="run_distributed_invalid_mode",
        outdir_path=outdir,
        argv=["pytest", "distributed_mode_contract"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.distributed_mode_contract"},
        distributed={
            "distributed_mode": "not_a_real_mode",
            "world_size_observed": 2,
            "backend": "nccl",
            "rank": 0,
            "local_rank": 0,
            "ddp_wrapped": True,
        },
    )
    append_gate_decision(
        outdir,
        run_id="run_distributed_invalid_mode",
        iter_num=0,
        decision="pass",
        ok=True,
        warn=False,
        audit=gate_audit,
    )

    v = validate_outdir(outdir)
    assert not v.ok
    assert "ERROR:DISTRIBUTED_MODE_INVALID" in v.message


def test_validate_passes_with_legacy_distributed_keys_when_mode_valid(tmp_path: Path) -> None:
    outdir = tmp_path / "capsule"
    outdir.mkdir(parents=True, exist_ok=True)

    ws_obj = {
        "schema_version": 1,
        "transport": {"name": "test_transport"},
        "evidence": {"metrics": ["m"]},
        "gates": {"preset": "test"},
    }
    _write_json(outdir / "window_signature.json", ws_obj)
    ws_hash = canonical_json_sha256(ws_obj)

    gate_audit = {
        "evaluated": True,
        "reason": "ok",
        "policy": "test_policy",
        "per_metric_tv": {"m": 0.01},
        "evidence_total": 16,
        "min_evidence": 16,
    }
    common = {
        "schema_version": 1,
        "run_id": "run_distributed_legacy_mode",
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
        "ended_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)),
    }
    _write_json(
        outdir / "results.json",
        {
            **common,
            "gates": [{"iter": 0, "decision": "pass", "ok": True, "warn": False, "audit": gate_audit}],
            "metrics": [],
        },
    )
    _write_json(
        outdir / "results_summary.json",
        {
            **common,
            "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
            "final_decision": "pass",
        },
    )

    append_run_started(
        outdir,
        run_id="run_distributed_legacy_mode",
        outdir_path=outdir,
        argv=["pytest", "distributed_mode_contract"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.distributed_mode_contract"},
        distributed={
            "distributed_mode": "ddp_wrapped",
            "world_size_observed": 2,
            "ddp_backend": "nccl",
            "rank_observed": 0,
            "local_rank_observed": 0,
            "ddp_active": True,
        },
    )
    append_gate_decision(
        outdir,
        run_id="run_distributed_legacy_mode",
        iter_num=0,
        decision="pass",
        ok=True,
        warn=False,
        audit=gate_audit,
    )

    v = validate_outdir(outdir)
    assert v.ok


def test_validate_fails_when_distributed_world_size_missing_with_hints(tmp_path: Path) -> None:
    outdir = tmp_path / "capsule"
    outdir.mkdir(parents=True, exist_ok=True)

    ws_obj = {
        "schema_version": 1,
        "transport": {"name": "test_transport"},
        "evidence": {"metrics": ["m"]},
        "gates": {"preset": "test"},
    }
    _write_json(outdir / "window_signature.json", ws_obj)
    ws_hash = canonical_json_sha256(ws_obj)

    gate_audit = {
        "evaluated": True,
        "reason": "ok",
        "policy": "test_policy",
        "per_metric_tv": {"m": 0.01},
        "evidence_total": 16,
        "min_evidence": 16,
    }
    common = {
        "schema_version": 1,
        "run_id": "run_distributed_missing_world_size",
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "profile": {"gate_preset": "test", "overrides": {}},
        "run_status": "success",
        "started_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)),
        "ended_ts_utc": _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)),
    }
    _write_json(
        outdir / "results.json",
        {
            **common,
            "gates": [{"iter": 0, "decision": "pass", "ok": True, "warn": False, "audit": gate_audit}],
            "metrics": [],
        },
    )
    _write_json(
        outdir / "results_summary.json",
        {
            **common,
            "counts": {"evaluated": 1, "skip": 0, "pass": 1, "warn": 0, "fail": 0},
            "final_decision": "pass",
        },
    )

    append_run_started(
        outdir,
        run_id="run_distributed_missing_world_size",
        outdir_path=outdir,
        argv=["pytest", "distributed_mode_contract"],
        code_identity={"package_version": "test"},
        window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
        entrypoint={"kind": "runner", "name": "tests.distributed_mode_contract"},
        distributed={
            "distributed_mode": "ddp_wrapped",
            "backend": "nccl",
            "rank": 0,
            "local_rank": 0,
            "ddp_wrapped": True,
        },
    )
    append_gate_decision(
        outdir,
        run_id="run_distributed_missing_world_size",
        iter_num=0,
        decision="pass",
        ok=True,
        warn=False,
        audit=gate_audit,
    )

    v = validate_outdir(outdir)
    assert not v.ok
    assert "ERROR:DISTRIBUTED_WORLD_SIZE_MISSING" in v.message
