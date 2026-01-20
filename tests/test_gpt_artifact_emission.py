# tests/test_gpt_artifact_emission.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from veriscope.core.jsonutil import canonical_json_sha256
from veriscope.runners.gpt.emit_artifacts import emit_gpt_artifacts_v1

pytestmark = pytest.mark.unit


def test_emit_gpt_artifacts_writes_and_validates(tmp_path: Path) -> None:
    outdir = tmp_path / "out"

    gate_history = [
        {
            "iter": 16,
            "ok": True,
            "warn": False,
            "reason": "evaluated_ok",
            "audit": {
                "evaluated": True,
                # intentionally omit "reason" here to test fallback
                "policy": "either",
                "worst_DW": 0.01,
                "eps_eff": 0.08,
                "per_metric_tv": {"m1": 0.01},
                "evidence_total": 32,
                "min_evidence": 16,
            },
        },
        {
            "iter": 32,
            "audit": {
                "evaluated": False,
                "reason": "insufficient_evidence",
                "policy": "either",
                "worst_DW": None,
                "eps_eff": None,
                "per_metric_tv": {},
                "evidence_total": 8,
                "min_evidence": 16,
            },
        },
    ]

    emitted = emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_123",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=datetime(2026, 1, 1, 0, 1, 0),
        gate_preset="tuned_v0",
        overrides={"note": "unit"},
        resolved_gate_cfg={"gate_window": 16, "min_evidence": 16, "gate_epsilon": 0.08},
        metric_interval=16,
        metric_pipeline={"transport": "DeclTransport"},
        gate_history=gate_history,
    )

    assert (outdir / "window_signature.json").exists()
    assert (outdir / "results.json").exists()
    assert (outdir / "results_summary.json").exists()

    # Validate the window signature hash matches the emitted ref hash
    ws = (outdir / "window_signature.json").read_text(encoding="utf-8")
    import json

    ws_obj = json.loads(ws)
    assert canonical_json_sha256(ws_obj) == emitted.window_signature_hash

    # Pydantic validation (by construction) should succeed
    from veriscope.core.artifacts import ResultsSummaryV1, ResultsV1

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    summary = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text(encoding="utf-8"))

    # window signature ref consistency everywhere
    assert res.window_signature_ref.hash == emitted.window_signature_hash
    assert summary.window_signature_ref.hash == emitted.window_signature_hash

    # decision derivation
    assert res.gates[0].decision == "pass"
    assert res.gates[1].decision == "skip"

    # counts semantics: evaluated excludes skip
    assert summary.counts.evaluated == 1
    assert summary.counts.skip == 1
    assert summary.counts.pass_ == 1
    assert summary.counts.warn == 0
    assert summary.counts.fail == 0

    # timestamp serialization policy (naive treated as UTC)
    assert res.started_ts_utc == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert res.ended_ts_utc == datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

    # Regression guard: emitter must fall back to parent event "reason" when audit.reason is missing
    assert res.gates[0].audit.reason == "evaluated_ok"

    # git_sha omission behavior: absent when not provided
    assert "git_sha" not in ws_obj["code_identity"]


def test_emit_defaults_per_metric_tv_when_missing(tmp_path: Path) -> None:
    outdir = tmp_path / "out"

    emitted = emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_per_metric_tv_default",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=None,
        gate_preset="tuned_v0",
        overrides=None,
        resolved_gate_cfg={},
        gate_history=[
            {
                "iter": 1,
                "ok": True,
                "reason": "evaluated_ok",
                "audit": {
                    "evaluated": True,
                    "policy": "either",
                    # omit per_metric_tv to test defaulting
                },
            }
        ],
    )

    from veriscope.core.artifacts import ResultsV1

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    assert res.window_signature_ref.hash == emitted.window_signature_hash
    assert res.gates[0].audit.per_metric_tv == {}


def test_emit_rejects_non_string_policy(tmp_path: Path) -> None:
    outdir = tmp_path / "out"

    with pytest.raises(ValueError, match=r"audit\.policy must be a string"):
        emit_gpt_artifacts_v1(
            outdir=outdir,
            run_id="run_test_bad_policy",
            started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
            ended_ts_utc=None,
            gate_preset="tuned_v0",
            overrides=None,
            resolved_gate_cfg={},
            gate_history=[{"iter": 1, "audit": {"evaluated": True, "policy": {"not": "a string"}}}],
        )
