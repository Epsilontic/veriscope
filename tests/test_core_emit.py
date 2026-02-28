from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from veriscope.core.artifacts import AuditV1, GateRecordV1
from veriscope.core.emit import emit_capsule_v1
from veriscope.core.jsonutil import atomic_write_json, window_signature_sha256

pytestmark = pytest.mark.unit


def _write_window_signature(outdir: Path) -> str:
    window_signature = {
        "schema_version": 1,
        "code_identity": {"package_version": "test"},
        "gate_controls": {"gate_window": 5, "gate_epsilon": 0.12, "min_evidence": 5},
        "metric_interval": 1,
        "metric_pipeline": {"transport": "test"},
    }
    atomic_write_json(outdir / "window_signature.json", window_signature, fsync=True)
    return window_signature_sha256(window_signature)


def _skip_gate_record() -> GateRecordV1:
    return GateRecordV1(
        iter=0,
        decision="skip",
        audit=AuditV1(
            evaluated=False,
            reason="not_evaluated_no_steps",
            policy="persistence",
            per_metric_tv={},
            evidence_total=0,
            min_evidence=1,
        ),
        ok=True,
        warn=False,
    )


def test_emit_capsule_rejects_non_castable_runner_exit_code(tmp_path: Path) -> None:
    outdir = tmp_path / "emit_non_castable_exit"
    outdir.mkdir(parents=True, exist_ok=True)
    ws_hash = _write_window_signature(outdir)

    with pytest.raises(ValueError, match="runner_exit_code must be int-castable"):
        emit_capsule_v1(
            outdir=outdir,
            run_id="run_emit_non_castable",
            started_ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ended_ts_utc=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            gate_preset="test",
            ws_hash=ws_hash,
            gate_records=[_skip_gate_record()],
            runner_exit_code="bad-exit",
        )


def test_emit_capsule_rejects_out_of_range_runner_exit_code(tmp_path: Path) -> None:
    outdir = tmp_path / "emit_out_of_range_exit"
    outdir.mkdir(parents=True, exist_ok=True)
    ws_hash = _write_window_signature(outdir)

    with pytest.raises(ValueError, match=r"\[0, 255\]"):
        emit_capsule_v1(
            outdir=outdir,
            run_id="run_emit_out_of_range",
            started_ts_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ended_ts_utc=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            gate_preset="test",
            ws_hash=ws_hash,
            gate_records=[_skip_gate_record()],
            runner_exit_code=999,
        )
