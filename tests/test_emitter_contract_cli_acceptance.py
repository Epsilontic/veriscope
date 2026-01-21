# tests/test_emitter_contract_cli_acceptance.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from veriscope.cli.report import render_report_md
from veriscope.cli.validate import validate_outdir
from veriscope.runners.gpt.emit_artifacts import emit_gpt_artifacts_v1

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)


def _emitter_gate_event_ok(*, it: int = 1) -> dict[str, Any]:
    # Mirrors shapes already used in your existing emitter unit tests.
    return {
        "iter": int(it),
        "ok": True,
        "warn": False,
        "reason": "evaluated_ok",
        "audit": {
            "evaluated": True,
            "policy": "either",
            "reason": "evaluated_ok",
            "per_metric_tv": {},
            "evidence_total": 16,
            "min_evidence": 16,
        },
    }


@pytest.mark.integration
def test_end_to_end_emitter_contract(tmp_path: Path) -> None:
    outdir = tmp_path / "out_emitted"
    emitted = emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="test_run_emitted",
        started_ts_utc=T0,
        ended_ts_utc=T1,
        gate_preset="tuned_v0",
        overrides={"note": "contract"},
        resolved_gate_cfg={"gate_window": 16, "min_evidence": 16, "gate_epsilon": 0.08},
        metric_interval=16,
        metric_pipeline={"transport": "DeclTransport"},
        gate_history=[_emitter_gate_event_ok(it=1)],
        run_status="success",
    )

    v = validate_outdir(outdir)
    assert v.ok, f"Emitter produced invalid artifacts: {v.message}"
    assert v.window_signature_hash == emitted.window_signature_hash

    md = render_report_md(outdir, fmt="md")
    assert "# Veriscope Report" in md
    assert "test_run_emitted" in md
    assert "## Gate Summary" in md
