# tests/test_governance_contract.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from veriscope.cli.governance import (
    GOVERNANCE_EVENT_TYPES,
    append_governance_log,
    read_governance_log,
    resolve_effective_status,
    validate_governance_log,
)
from veriscope.core.artifacts import ManualJudgementV1

pytestmark = pytest.mark.unit


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_manual_json(outdir: Path, *, run_id: str, status: str) -> None:
    judgement = ManualJudgementV1(
        run_id=run_id,
        status=status,
        reason="manual override",
        reviewer="tester",
        ts_utc=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    _write_json(outdir / "manual_judgement.json", judgement.model_dump(by_alias=True, mode="json"))


def _write_manual_jsonl(outdir: Path, *, run_id: str, status: str) -> None:
    judgement = ManualJudgementV1(
        run_id=run_id,
        status=status,
        reason="manual override log",
        reviewer=None,
        ts_utc=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    payload = judgement.model_dump(by_alias=True, mode="json")
    payload["rev"] = 1
    (outdir / "manual_judgement.jsonl").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_governance_log_hash_chain_detects_tamper(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    ts_0 = _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    ts_1 = _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "baseline"},
        ts_utc=ts_0,
        actor="tester",
    )
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "second"},
        ts_utc=ts_1,
        actor="tester",
    )

    log_path = outdir / "governance_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["note"] = "tampered"
    lines[0] = json.dumps(tampered, sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = validate_governance_log(log_path)
    assert not validation.ok
    assert any("GOVERNANCE_LOG_HASH_MISMATCH" in w for w in validation.errors)
    with pytest.raises(RuntimeError, match="Cannot append to invalid governance_log.jsonl"):
        append_governance_log(
            outdir,
            event_type="artifact_note",
            payload={"run_id": "run", "note": "blocked"},
            ts_utc=ts_1,
            actor="tester",
        )


def test_governance_log_rev_monotone_even_if_ts_decreases(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    ts_0 = _iso_z(datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc))
    ts_1 = _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "first"},
        ts_utc=ts_0,
        actor=None,
    )
    _, warnings = append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "second"},
        ts_utc=ts_1,
        actor=None,
    )
    assert any("GOVERNANCE_LOG_TS_DECREASE" in w for w in warnings)

    result = read_governance_log(outdir / "governance_log.jsonl")
    assert result.rev == 2


def test_governance_log_rev_nonmonotone_is_invalid(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    ts_0 = _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "first"},
        ts_utc=ts_0,
        actor="tester",
    )
    log_path = outdir / "governance_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["rev"] = 3
    lines[0] = json.dumps(entry, sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = validate_governance_log(log_path)
    assert not validation.ok
    assert any("GOVERNANCE_LOG_REV_NONMONOTONE" in w for w in validation.errors)


def test_governance_log_prev_hash_mismatch_is_invalid(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    ts_0 = _iso_z(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    ts_1 = _iso_z(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc))
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "first"},
        ts_utc=ts_0,
        actor="tester",
    )
    append_governance_log(
        outdir,
        event_type="artifact_note",
        payload={"run_id": "run", "note": "second"},
        ts_utc=ts_1,
        actor="tester",
    )
    log_path = outdir / "governance_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["prev_hash"] = "deadbeef"
    lines[1] = json.dumps(entry, sort_keys=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = validate_governance_log(log_path)
    assert not validation.ok
    assert any("GOVERNANCE_LOG_PREV_HASH_MISMATCH" in w for w in validation.errors)


def test_precedence_resolver_prefers_jsonl_and_fallbacks(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_manual_json(outdir, run_id="run", status="fail")
    _write_manual_jsonl(outdir, run_id="run", status="pass")

    resolved = resolve_effective_status(outdir, run_id="run", automated_decision="warn")
    assert resolved.status == "pass"
    assert resolved.source == "manual"

    (outdir / "manual_judgement.jsonl").write_text("not json\n", encoding="utf-8")
    resolved = resolve_effective_status(outdir, run_id="run", automated_decision="warn")
    assert resolved.status == "fail"
    assert resolved.source == "manual"

    (outdir / "manual_judgement.json").write_text("not json\n", encoding="utf-8")
    resolved = resolve_effective_status(outdir, run_id="run", automated_decision="warn")
    assert resolved.status == "warn"
    assert resolved.source == "automated"


def test_precedence_resolver_ignores_run_id_mismatch(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_manual_json(outdir, run_id="other", status="fail")

    resolved = resolve_effective_status(outdir, run_id="run", automated_decision="pass")
    assert resolved.status == "pass"
    assert resolved.source == "automated"
    assert any("MANUAL_JUDGEMENT_RUN_ID_MISMATCH" in w for w in resolved.warnings)


def test_precedence_resolver_uses_last_matching_jsonl_entry(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    outdir.mkdir(parents=True, exist_ok=True)
    entries = [
        ManualJudgementV1(
            run_id="run",
            status="pass",
            reason="first",
            reviewer="tester",
            ts_utc=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        ),
        ManualJudgementV1(
            run_id="other",
            status="fail",
            reason="other",
            reviewer="tester",
            ts_utc=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    lines = []
    for idx, judgement in enumerate(entries, start=1):
        payload = judgement.model_dump(by_alias=True, mode="json")
        payload["rev"] = idx
        lines.append(json.dumps(payload, sort_keys=True))
    (outdir / "manual_judgement.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    resolved = resolve_effective_status(outdir, run_id="run", automated_decision="warn")
    assert resolved.status == "pass"
    assert resolved.source == "manual"
    assert any("MANUAL_JUDGEMENT_RUN_ID_MISMATCH" in w for w in resolved.warnings)


def test_contract_doc_lists_event_types() -> None:
    doc = Path("docs/contract_v1.md").read_text(encoding="utf-8")
    for event_type in GOVERNANCE_EVENT_TYPES:
        assert event_type in doc
    for artifact in (
        "window_signature.json",
        "results.json",
        "results_summary.json",
        "manual_judgement.json",
        "manual_judgement.jsonl",
        "governance_log.jsonl",
    ):
        assert artifact in doc
