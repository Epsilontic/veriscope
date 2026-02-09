# tests/test_gpt_artifact_emission.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from veriscope.core.jsonutil import canonical_json_sha256
from veriscope.runners.gpt.emit_artifacts import (
    _first_fail_iter_from_events,
    _flatten_gate_events,
    emit_gpt_artifacts_v1,
)

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
        metrics_ref={"path": "run.json", "format": "legacy_v0", "count": 2},
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
    assert summary.first_fail_iter is None

    # timestamp serialization policy (naive treated as UTC)
    assert res.started_ts_utc == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert res.ended_ts_utc == datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

    # Regression guard: emitter must fall back to parent event "reason" when audit.reason is missing
    assert res.gates[0].audit.reason == "evaluated_ok"

    # git_sha omission behavior: absent when not provided
    assert "git_sha" not in ws_obj["code_identity"]

    # V1 stays gates-first, with an explicit pointer to the legacy metrics stream.
    raw_results = json.loads((outdir / "results.json").read_text(encoding="utf-8"))
    assert raw_results["metrics"] == []
    assert raw_results["metrics_ref"] == {"path": "run.json", "format": "legacy_v0", "count": 2}


def test_emit_skipped_gate_sanitizes_nonfinite_audit_values(tmp_path: Path) -> None:
    outdir = tmp_path / "out"

    emitted = emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_skip_nan",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=None,
        gate_preset="tuned_v0",
        overrides=None,
        resolved_gate_cfg={},
        gate_history=[
            {
                "iter": 1,
                "decision": "skip",
                "audit": {
                    "evaluated": False,
                    "reason": "not_evaluated",
                    "policy": "either",
                    "worst_DW": float("nan"),
                    "eps_eff": float("nan"),
                    "per_metric_tv": {},
                },
            }
        ],
    )

    from veriscope.core.artifacts import ResultsV1

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    assert res.window_signature_ref.hash == emitted.window_signature_hash
    assert res.gates[0].decision == "skip"
    assert res.gates[0].audit.evaluated is False
    assert res.gates[0].audit.worst_DW == 0.0
    assert res.gates[0].audit.eps_eff == 0.0


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


def test_emit_sanitizes_unevaluated_nonfinite(tmp_path: Path) -> None:
    outdir = tmp_path / "out"

    emitted = emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_unevaluated_nonfinite",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=None,
        gate_preset="tuned_v0",
        overrides=None,
        resolved_gate_cfg={},
        gate_history=[
            {
                "iter": 1,
                "audit": {
                    "evaluated": False,
                    "worst_DW": float("nan"),
                    "eps_eff": float("nan"),
                    "per_metric_tv": {"m1": float("nan")},
                },
            }
        ],
    )

    from veriscope.core.artifacts import ResultsV1

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    assert res.window_signature_ref.hash == emitted.window_signature_hash
    assert res.gates[0].audit.worst_DW == 0.0
    assert res.gates[0].audit.eps_eff == 0.0
    assert res.gates[0].audit.per_metric_tv["m1"] == 0.0
    raw = json.loads((outdir / "results.json").read_text(encoding="utf-8"))
    audit_raw = raw["gates"][0]["audit"]
    assert audit_raw["sanitized_nonfinite"] is True
    assert audit_raw["sanitized_nonfinite_count"] >= 1


def test_emit_legacy_bool_conflict_normalizes_to_fail(tmp_path: Path) -> None:
    outdir = tmp_path / "out"

    emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_fail_dominates_warn",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=None,
        gate_preset="tuned_v0",
        overrides=None,
        resolved_gate_cfg={},
        gate_history=[
            {
                "iter": 16,
                # Regression fixture: old precedence bug could emit warn with ok=False.
                "decision": "warn",
                "ok": False,
                "warn": True,
                "audit": {
                    "evaluated": True,
                    "reason": "change_persistence_fail",
                    "policy": "persistence",
                    "worst_DW": 0.22,
                    "eps_eff": 0.08,
                    "per_metric_tv": {"m1": 0.22},
                    "evidence_total": 32,
                    "min_evidence": 16,
                    "persistence_fail": True,
                    "regime_warn": True,
                },
            }
        ],
    )

    from veriscope.core.artifacts import ResultsV1

    results = ResultsV1.model_validate_json((outdir / "results.json").read_text(encoding="utf-8"))
    gate = results.gates[0]
    assert gate.decision == "fail"
    assert gate.ok is False
    assert gate.warn is False

    gov_lines = [
        json.loads(line) for line in (outdir / "governance_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gate_events = [line for line in gov_lines if (line.get("event") or line.get("event_type")) == "gate_decision_v1"]
    assert gate_events
    payload = gate_events[-1]["payload"]
    assert payload["decision"] == "fail"
    assert payload["ok"] is False
    assert payload["warn"] is False


def test_emit_rejects_non_monotonic_gate_iterations(tmp_path: Path) -> None:
    outdir = tmp_path / "out_non_monotonic"

    with pytest.raises(ValueError, match="strictly increasing"):
        emit_gpt_artifacts_v1(
            outdir=outdir,
            run_id="run_test_non_monotonic",
            started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
            ended_ts_utc=None,
            gate_preset="tuned_v0",
            overrides=None,
            resolved_gate_cfg={},
            gate_history=[
                {"iter": 16, "audit": {"evaluated": False, "reason": "not_evaluated", "policy": "either"}},
                {"iter": 8, "audit": {"evaluated": False, "reason": "not_evaluated", "policy": "either"}},
            ],
        )


def test_first_fail_iter_extractor_handles_heterogeneous_events() -> None:
    events: list[object] = [
        "noise",
        ["non-object"],
        None,
        {"decision": "warn", "iter": 900},
        {"decision": "fail", "audit": {"iter": "940"}},  # ignored (iter must be int)
        {"audit": {"final_decision": "fail", "step": "945"}},  # ignored (step must be int)
        {"audit": {"decision": "fail", "iter": 975}},
        {"decision": "fail", "iter": 950},
    ]
    assert _first_fail_iter_from_events(_flatten_gate_events(events)) == 950


def test_flatten_gate_events_unwraps_nested_gates() -> None:
    from veriscope.runners.gpt.emit_artifacts import _flatten_gate_events

    flat = [{"iter": 1, "decision": "pass"}, {"iter": 2, "decision": "fail"}]
    assert _flatten_gate_events(flat) == flat

    wrapped = [{"gates": [{"iter": 3, "decision": "pass"}]}, {"iter": 4, "decision": "warn"}]
    assert _flatten_gate_events(wrapped) == [{"iter": 3, "decision": "pass"}, {"iter": 4, "decision": "warn"}]

    top_level = {"gates": [{"iter": 5, "decision": "fail"}]}
    assert _flatten_gate_events(top_level) == [{"iter": 5, "decision": "fail"}]


@pytest.mark.parametrize(
    ("gate_history", "expected_first_fail_iter"),
    [
        (
            [
                {
                    "iter": 900,
                    "decision": "pass",
                    "audit": {
                        "evaluated": True,
                        "reason": "evaluated_ok",
                        "policy": "either",
                        "per_metric_tv": {},
                        "evidence_total": 16,
                        "min_evidence": 16,
                    },
                },
                {
                    "iter": 950,
                    "audit": {
                        "decision": "fail",
                        "evaluated": True,
                        "reason": "evaluated_fail",
                        "policy": "either",
                        "per_metric_tv": {"m1": 0.12},
                        "evidence_total": 16,
                        "min_evidence": 16,
                    },
                },
            ],
            950,
        ),
        (
            [
                {
                    "iter": 900,
                    "decision": "pass",
                    "audit": {
                        "evaluated": True,
                        "reason": "evaluated_ok",
                        "policy": "either",
                        "per_metric_tv": {},
                        "evidence_total": 16,
                        "min_evidence": 16,
                    },
                },
                {
                    "iter": 950,
                    "decision": "skip",
                    "audit": {
                        "evaluated": False,
                        "reason": "insufficient_evidence",
                        "policy": "either",
                        "per_metric_tv": {},
                    },
                },
            ],
            None,
        ),
    ],
)
def test_emit_first_fail_iter_summary_and_marker_file(
    tmp_path: Path,
    gate_history: list[dict[str, object]],
    expected_first_fail_iter: int | None,
) -> None:
    outdir = tmp_path / "out_first_fail"
    outdir.mkdir(parents=True, exist_ok=True)
    marker_path = outdir / "first_fail_iter.txt"
    marker_path.write_text("stale\n", encoding="utf-8")

    emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_first_fail_iter",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=datetime(2026, 1, 1, 0, 1, 0),
        gate_preset="tuned_v0",
        overrides=None,
        resolved_gate_cfg={"gate_window": 16, "min_evidence": 16, "gate_epsilon": 0.08},
        metric_interval=16,
        metric_pipeline={"transport": "DeclTransport"},
        gate_history=gate_history,
    )

    summary_obj = json.loads((outdir / "results_summary.json").read_text(encoding="utf-8"))
    assert "first_fail_iter" in summary_obj
    assert summary_obj["first_fail_iter"] == expected_first_fail_iter

    if expected_first_fail_iter is None:
        assert not marker_path.exists()
    else:
        assert marker_path.exists()
        assert marker_path.read_text(encoding="utf-8") == f"{expected_first_fail_iter}\n"


def test_emit_coerces_string_iter_for_first_fail_iter(tmp_path: Path) -> None:
    outdir = tmp_path / "out_coerced_first_fail_iter"
    emitted = emit_gpt_artifacts_v1(
        outdir=outdir,
        run_id="run_test_coerced_first_fail_iter",
        started_ts_utc=datetime(2026, 1, 1, 0, 0, 0),
        ended_ts_utc=None,
        gate_preset="tuned_v0",
        overrides=None,
        resolved_gate_cfg={"gate_window": 16, "min_evidence": 16, "gate_epsilon": 0.08},
        gate_history=[
            {
                "iter": "950",
                "decision": "fail",
                "audit": {
                    "evaluated": True,
                    "reason": "evaluated_fail",
                    "policy": "either",
                    "per_metric_tv": {"m1": 0.2},
                    "evidence_total": 16,
                    "min_evidence": 16,
                },
            }
        ],
    )
    assert emitted.results_summary_path.exists()
    summary_obj = json.loads((outdir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary_obj["first_fail_iter"] == 950
    assert (outdir / "first_fail_iter.txt").read_text(encoding="utf-8") == "950\n"
