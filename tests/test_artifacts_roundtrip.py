# tests/test_artifacts_roundtrip.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from veriscope.core.artifacts import (
    AuditV1,
    CountsV1,
    GateRecordV1,
    ProfileV1,
    ResultsSummaryV1,
    ResultsV1,
    WindowSignatureRefV1,
    derive_gate_decision,
)

pytestmark = pytest.mark.unit


def _dt_utc(
    y: int = 2026,
    m: int = 1,
    d: int = 18,
    hh: int = 0,
    mm: int = 0,
    ss: int = 0,
) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def test_results_v1_roundtrip_dump_validate_json_mode() -> None:
    """
    model_dump(mode="json") -> model_validate() is stable, and extra keys are accepted/preserved.
    """
    win_ref = WindowSignatureRefV1(hash="0" * 64, path="out/window_signature.json")
    profile = ProfileV1(gate_preset="default", overrides={"alpha": 1, "beta": True})

    audit = AuditV1(
        evaluated=True,
        reason="ok",
        policy="default",
        worst_DW=0.0,
        eps_eff=0.01,
        per_metric_tv={"m1": 0.1},
        evidence_total=10,
        min_evidence=5,
        extra_audit_note="hello",
    )
    gate = GateRecordV1(
        iter=0,
        decision="pass",
        audit=audit,
        ok=True,
        extra_gate_note="world",
    )
    r = ResultsV1(
        run_id="run_0001",
        window_signature_ref=win_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
        started_ts_utc=_dt_utc(),
        ended_ts_utc=None,
        gates=(gate,),
        metrics=(),
        extra_run_note="extra",
    )

    d = r.model_dump(mode="json", by_alias=True)
    r2 = ResultsV1.model_validate(d)

    assert r2.run_id == r.run_id
    assert r2.run_status == r.run_status
    assert r2.window_signature_ref.hash == r.window_signature_ref.hash
    assert len(r2.gates) == 1
    assert r2.gates[0].decision == "pass"

    # Extras preserved on round-trip (forward-compat contract posture).
    assert "extra_run_note" in r2.model_dump()
    assert "extra_gate_note" in r2.gates[0].model_dump()
    assert "extra_audit_note" in r2.gates[0].audit.model_dump()


def test_results_v1_roundtrip_json_and_datetime_z() -> None:
    """
    model_dump_json() -> model_validate_json() round-trips.
    Also asserts deterministic datetime serialization (UTC with trailing 'Z').
    """
    win_ref = WindowSignatureRefV1(hash="a" * 64, path="out/window_signature.json")
    profile = ProfileV1(gate_preset="default")

    audit = AuditV1(
        evaluated=False,
        per_metric_tv={},
        extra_audit_note="not_evaluated",
    )
    gate = GateRecordV1(iter=0, decision="skip", audit=audit)

    r = ResultsV1(
        run_id="run_0002",
        window_signature_ref=win_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        started_ts_utc=_dt_utc(2026, 1, 18, 12, 34, 56),
        ended_ts_utc=_dt_utc(2026, 1, 18, 12, 35, 0),
        gates=(gate,),
        metrics=(),
        extra_run_note="extra",
    )

    js = r.model_dump_json(by_alias=True)

    # Assert the wire format uses 'Z' (not '+00:00') and seconds precision.
    assert '"started_ts_utc":"2026-01-18T12:34:56Z"' in js
    assert '"ended_ts_utc":"2026-01-18T12:35:00Z"' in js

    r2 = ResultsV1.model_validate_json(js)
    assert r2.run_id == r.run_id
    assert r2.started_ts_utc == r.started_ts_utc
    assert r2.ended_ts_utc == r.ended_ts_utc
    assert "extra_run_note" in r2.model_dump()


def test_results_summary_alias_emission_and_ingestion() -> None:
    """
    ResultsSummaryV1:
    - emits counts['pass'] when dumping with by_alias=True
    - accepts counts['pass'] when validating (validation_alias)
    - accepts extra fields
    """
    win_ref = WindowSignatureRefV1(hash="b" * 64, path="out/window_signature.json")
    profile = ProfileV1(gate_preset="default")

    counts = CountsV1(
        evaluated=1,
        skip=0,
        pass_=1,
        warn=0,
        fail=0,
        extra_counts_note="x",
    )
    s = ResultsSummaryV1(
        run_id="run_0003",
        window_signature_ref=win_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        started_ts_utc=_dt_utc(),
        ended_ts_utc=None,
        counts=counts,
        final_decision="pass",
        extra_summary_note="y",
    )

    d = s.model_dump(mode="json", by_alias=True)
    assert "pass" in d["counts"]  # alias emission

    s2 = ResultsSummaryV1.model_validate(d)
    assert s2.final_decision == "pass"
    assert s2.counts.pass_ == 1
    assert "extra_summary_note" in s2.model_dump()
    assert "extra_counts_note" in s2.counts.model_dump()

    # Explicitly validate ingestion of the alias key "pass"
    d2 = dict(d)
    d2["counts"] = dict(d["counts"])
    d2["counts"]["pass"] = 1
    d2["counts"].pop("pass_", None)
    s3 = ResultsSummaryV1.model_validate(d2)
    assert s3.counts.pass_ == 1


def test_counts_invariant_violations_raise() -> None:
    with pytest.raises(ValueError, match=r"pass\+warn\+fail == evaluated"):
        CountsV1(evaluated=1, skip=0, pass_=0, warn=0, fail=0)


def test_temporal_order_violation_raises() -> None:
    win_ref = WindowSignatureRefV1(hash="c" * 64, path="out/window_signature.json")
    profile = ProfileV1(gate_preset="default")
    audit = AuditV1(
        evaluated=False,
        per_metric_tv={},
    )
    gate = GateRecordV1(iter=0, decision="skip", audit=audit)

    with pytest.raises(ValueError, match="ended_ts_utc cannot precede started_ts_utc"):
        ResultsV1(
            run_id="run_bad_time",
            window_signature_ref=win_ref,
            profile=profile,
            run_status="success",
            runner_exit_code=0,
            started_ts_utc=_dt_utc(2026, 1, 18, 12, 0, 1),
            ended_ts_utc=_dt_utc(2026, 1, 18, 12, 0, 0),
            gates=(gate,),
            metrics=(),
        )


def test_derive_gate_decision_fail_dominates_warn() -> None:
    truth_table = [
        (False, False, False, "skip"),
        (False, False, True, "skip"),
        (False, True, False, "skip"),
        (False, True, True, "skip"),
        (True, False, False, "fail"),
        (True, False, True, "fail"),
        (True, True, False, "pass"),
        (True, True, True, "warn"),
    ]
    for evaluated, ok, warn, expected in truth_table:
        assert derive_gate_decision(evaluated=evaluated, ok=ok, warn=warn) == expected


def test_gate_record_rejects_warn_when_ok_false() -> None:
    audit = AuditV1(
        evaluated=True,
        reason="evaluated_fail",
        policy="persistence",
        per_metric_tv={},
        evidence_total=1,
        min_evidence=1,
    )
    with pytest.raises(ValueError, match="gate.ok cannot be False when decision=='warn'"):
        GateRecordV1(iter=0, decision="warn", audit=audit, ok=False, warn=True)


@pytest.mark.parametrize(
    ("decision", "ok_value", "message"),
    [
        ("pass", False, "gate.ok cannot be False when decision=='pass'"),
        ("warn", False, "gate.ok cannot be False when decision=='warn'"),
        ("fail", True, "gate.ok cannot be True when decision=='fail'"),
    ],
)
def test_gate_record_enforces_decision_ok_implications(decision: str, ok_value: bool, message: str) -> None:
    audit = AuditV1(
        evaluated=True,
        reason="evaluated_state",
        policy="persistence",
        per_metric_tv={},
        evidence_total=1,
        min_evidence=1,
    )
    with pytest.raises(ValueError, match=message):
        GateRecordV1(iter=0, decision=decision, audit=audit, ok=ok_value, warn=(decision == "warn"))


def test_hash_normalization_and_validation() -> None:
    w = WindowSignatureRefV1(hash="A" * 64, path="x")
    assert w.hash == ("a" * 64)

    with pytest.raises(ValueError):
        WindowSignatureRefV1(hash="0" * 63, path="x")  # wrong length

    with pytest.raises(ValueError):
        WindowSignatureRefV1(hash="g" * 64, path="x")  # non-hex


def test_immutability_of_tuples_and_mappings() -> None:
    win_ref = WindowSignatureRefV1(hash="d" * 64, path="out/window_signature.json")
    profile = ProfileV1(gate_preset="default", overrides={"alpha": 1})

    audit = AuditV1(evaluated=False, per_metric_tv={})
    gate = GateRecordV1(iter=0, decision="skip", audit=audit)

    r = ResultsV1(
        run_id="run_immut",
        window_signature_ref=win_ref,
        profile=profile,
        run_status="success",
        runner_exit_code=0,
        started_ts_utc=_dt_utc(),
        ended_ts_utc=None,
        gates=(gate,),
        metrics=(),
    )

    # Tuple interior can't be mutated.
    with pytest.raises(TypeError):
        r.gates[0] = gate  # type: ignore[misc]

    # Frozen mapping (MappingProxyType) can't be mutated.
    with pytest.raises(TypeError):
        profile.overrides["new_key"] = 123  # type: ignore[index]
