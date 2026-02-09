from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from veriscope.core.artifacts import AuditV1, GateRecordV1, ProfileV1, ResultsV1, WindowSignatureRefV1
from veriscope.core.jsonutil import (
    atomic_write_json,
    atomic_write_pydantic_json,
    canonical_json_sha256,
    read_json_obj,
)
from veriscope.runners.hf.emit_artifacts import emit_hf_artifacts_v1


def test_atomic_write_pydantic_json_supports_kwargs(tmp_path: Path) -> None:
    audit = AuditV1(evaluated=False, per_metric_tv={})
    gate = GateRecordV1(iter=0, decision="skip", audit=audit)
    ws_ref = WindowSignatureRefV1(hash="a" * 64, path="window_signature.json")
    profile = ProfileV1(gate_preset="tuned_v0", overrides={})
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ended = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
    results = ResultsV1(
        schema_version=1,
        run_id="run-1",
        window_signature_ref=ws_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
        started_ts_utc=started,
        ended_ts_utc=ended,
        gates=[gate],
        metrics=[],
    )
    path = tmp_path / "results.json"
    atomic_write_pydantic_json(path, results, by_alias=True, exclude_none=True, fsync=True)
    payload = read_json_obj(path)
    assert payload["schema_version"] == 1
    assert isinstance(payload["gates"], list)


def test_window_signature_hash_matches_on_disk_canonical(tmp_path: Path) -> None:
    window_signature = {"schema_version": 1, "transport": {"name": "hf_hidden_state_v1", "cadence": "every_1_steps"}}
    path = tmp_path / "window_signature.json"
    atomic_write_json(path, window_signature)
    on_disk = read_json_obj(path)
    in_memory_hash = canonical_json_sha256(window_signature)
    on_disk_hash = canonical_json_sha256(on_disk)
    assert in_memory_hash == on_disk_hash


def test_emit_hf_artifacts_uses_on_disk_window_signature_hash(tmp_path: Path) -> None:
    outdir = tmp_path / "hf_emit"
    window_signature = {"schema_version": 1, "transport": {"name": "hf_hidden_state_v1", "cadence": "every_1_steps"}}
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ended = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)

    emitted = emit_hf_artifacts_v1(
        outdir=outdir,
        run_id="run-emit",
        started_ts_utc=started,
        ended_ts_utc=ended,
        gate_preset="tuned_v0",
        window_signature=window_signature,
        gate_records=[],
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
    )

    ws_on_disk = read_json_obj(emitted.window_signature_path)
    ws_hash = canonical_json_sha256(ws_on_disk)
    summary = read_json_obj(emitted.results_summary_path)
    assert summary["window_signature_ref"]["hash"] == ws_hash
    assert summary["window_signature_ref"]["path"] == "window_signature.json"
    results = read_json_obj(emitted.results_path)
    assert results["window_signature_ref"]["hash"] == ws_hash
    assert results["window_signature_ref"]["path"] == "window_signature.json"


def test_emit_hf_artifacts_writes_first_fail_marker(tmp_path: Path) -> None:
    outdir = tmp_path / "hf_emit_fail_marker"
    window_signature = {"schema_version": 1, "transport": {"name": "hf_hidden_state_v1", "cadence": "every_1_steps"}}
    started = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ended = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
    fail_audit = AuditV1(
        evaluated=True,
        reason="evaluated_fail",
        policy="test_policy",
        per_metric_tv={},
        evidence_total=1,
        min_evidence=1,
    )
    pass_audit = AuditV1(
        evaluated=True,
        reason="evaluated_pass",
        policy="test_policy",
        per_metric_tv={},
        evidence_total=1,
        min_evidence=1,
    )
    gate_records = [
        GateRecordV1(iter=5, decision="fail", audit=fail_audit, ok=False, warn=False),
        GateRecordV1(iter=7, decision="pass", audit=pass_audit, ok=True, warn=False),
    ]

    emitted = emit_hf_artifacts_v1(
        outdir=outdir,
        run_id="run-emit-fail",
        started_ts_utc=started,
        ended_ts_utc=ended,
        gate_preset="tuned_v0",
        window_signature=window_signature,
        gate_records=gate_records,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
    )

    summary = read_json_obj(emitted.results_summary_path)
    assert summary["counts"]["fail"] == 1
    assert summary["first_fail_iter"] == 5
    assert (outdir / "first_fail_iter.txt").read_text(encoding="utf-8") == "5\n"
