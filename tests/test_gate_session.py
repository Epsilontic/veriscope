from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from veriscope.cli.diff import diff_outdirs
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ResultsSummaryV1, ResultsV1
from veriscope.core.gate import GateEngine
from veriscope.core.gate_session import GateSession
from veriscope.core.governance import validate_governance_log
from veriscope.core.jsonutil import window_signature_sha256
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)


def _make_window_decl() -> WindowDecl:
    return WindowDecl(
        epsilon=0.12,
        metrics=["loss_delta_z"],
        weights={"loss_delta_z": 1.0},
        bins=16,
        interventions=(lambda x: x,),
        cal_ranges={"loss_delta_z": (-6.0, 6.0)},
    )


def _make_gate_engine(wd: WindowDecl) -> GateEngine:
    transport = DeclTransport(wd)
    wd.attach_transport(transport)
    fr_win = FRWindow(decl=wd, transport=transport, tests=())
    return GateEngine(
        frwin=fr_win,
        gain_thresh=0.0,
        eps_stat_alpha=0.05,
        eps_stat_max_frac=0.25,
        eps_sens=0.04,
        min_evidence=5,
        policy="persistence",
        persistence_k=2,
        min_metrics_exceeding=1,
    )


def _window_sig() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "code_identity": {"package_version": "test"},
        "gate_controls": {
            "gate_window": 5,
            "gate_epsilon": 0.12,
            "min_evidence": 5,
        },
        "metric_interval": 1,
        "metric_pipeline": {"transport": "test"},
    }


def _make_session(
    outdir: Path,
    *,
    observer_mode: bool = False,
    gate_window: int = 5,
    emit_governance: bool = True,
) -> GateSession:
    wd = _make_window_decl()
    ge = _make_gate_engine(wd)
    return GateSession(
        outdir=outdir,
        run_id="test_session",
        window_decl=wd,
        gate_engine=ge,
        gate_window=gate_window,
        gate_policy="persistence",
        gate_min_evidence=5,
        gate_preset="test",
        window_signature=_window_sig(),
        emit_governance=emit_governance,
        observer_mode=observer_mode,
        started_ts_utc=T0,
    )


def _load_governance_events(outdir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    gov_path = outdir / "governance_log.jsonl"
    for line in gov_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            continue
        events.append(obj)
    return events


def _events_by_name(outdir: Path, name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for obj in _load_governance_events(outdir):
        event_name = obj.get("event") or obj.get("event_type")
        if event_name == name:
            out.append(obj)
    return out


def test_session_observer_mode_skip_all(tmp_path: Path) -> None:
    outdir = tmp_path / "obs"
    session = _make_session(outdir, observer_mode=True, gate_window=5)
    for i in range(30):
        session.record_step(i, loss_delta_z=0.01 * i)
        if i % 5 == 0 and i > 0:
            record = session.evaluate(step=i)
            assert record.decision == "skip"
            assert record.audit.evaluated is False
            assert record.audit.reason == "not_evaluated_observer_mode"

    session.close(run_status="success", ended_ts_utc=T1)

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message

    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))
    assert summ.final_decision == "skip"
    assert summ.counts.evaluated == 0
    assert summ.counts.skip > 0
    assert summ.partial is not True

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text("utf-8"))
    assert dict(res.profile.overrides) == {}


def test_session_gated_evaluates_gates(tmp_path: Path) -> None:
    outdir = tmp_path / "gated"
    session = _make_session(outdir, observer_mode=False, gate_window=5)
    for i in range(25):
        session.record_step(i, loss_delta_z=0.01 * i)
        if i % 5 == 0 and i >= 10:
            record = session.evaluate(step=i)
            assert record.decision in {"pass", "warn", "fail", "skip"}

    session.close(run_status="success", ended_ts_utc=T1)

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message


def test_session_close_without_gates_emits_skip(tmp_path: Path) -> None:
    outdir = tmp_path / "empty"
    session = _make_session(outdir, gate_window=5)
    session.close(run_status="success", ended_ts_utc=T1)

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message

    gate_events = _events_by_name(outdir, "gate_decision_v1")
    assert len(gate_events) == 1


def test_session_observer_is_comparable_via_diff(tmp_path: Path) -> None:
    for name in ["a", "b"]:
        outdir = tmp_path / name
        session = _make_session(outdir, observer_mode=True, gate_window=5)
        for i in range(20):
            session.record_step(i, loss_delta_z=0.01 * i)
            if i % 5 == 0 and i > 0:
                session.evaluate(step=i)
        session.close(run_status="success", ended_ts_utc=T1)

    result = diff_outdirs(tmp_path / "a", tmp_path / "b")
    assert result.exit_code == 0, result.stderr


def test_session_governance_per_gate(tmp_path: Path) -> None:
    outdir = tmp_path / "gov"
    session = _make_session(outdir, observer_mode=True, gate_window=5)
    n_evals = 0
    for i in range(20):
        session.record_step(i, loss_delta_z=0.01 * i)
        if i % 5 == 0 and i > 0:
            session.evaluate(step=i)
            n_evals += 1
    session.close(run_status="success", ended_ts_utc=T1)

    gov_path = outdir / "governance_log.jsonl"
    assert gov_path.exists()
    chain_v = validate_governance_log(gov_path)
    assert chain_v.ok, chain_v.errors

    gate_events = _events_by_name(outdir, "gate_decision_v1")
    assert len(gate_events) == n_evals

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message


def test_session_close_raises_on_window_signature_tampered(tmp_path: Path) -> None:
    outdir = tmp_path / "tampered"
    session = _make_session(outdir)
    session.record_step(0, loss_delta_z=0.01)
    session.evaluate(step=0)

    ws_path = outdir / "window_signature.json"
    ws_obj = json.loads(ws_path.read_text("utf-8"))
    ws_obj["tampered_field"] = "malicious_value"
    ws_path.write_text(json.dumps(ws_obj, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="window_signature.*modified"):
        session.close(run_status="success", ended_ts_utc=T1)


def test_session_close_raises_on_window_signature_deleted(tmp_path: Path) -> None:
    outdir = tmp_path / "deleted"
    session = _make_session(outdir)
    (outdir / "window_signature.json").unlink()

    with pytest.raises(RuntimeError, match="window_signature.*deleted"):
        session.close(run_status="success", ended_ts_utc=T1)


def test_session_window_signature_hash_matches_governance(tmp_path: Path) -> None:
    outdir = tmp_path / "hash_match"
    session = _make_session(outdir)
    session.record_step(0, loss_delta_z=0.01)
    session.evaluate(step=0)
    session.close(run_status="success", ended_ts_utc=T1)

    started_events = _events_by_name(outdir, "run_started_v1")
    assert len(started_events) == 1
    started_obj = started_events[0]
    payload = started_obj.get("payload")
    assert isinstance(payload, dict)
    ws_ref = payload.get("window_signature_ref")
    assert isinstance(ws_ref, dict)
    gov_hash = ws_ref["hash"]

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text("utf-8"))
    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))
    assert gov_hash == res.window_signature_ref.hash == summ.window_signature_ref.hash

    ws_obj = json.loads((outdir / "window_signature.json").read_text("utf-8"))
    disk_hash = window_signature_sha256(ws_obj)
    assert disk_hash == gov_hash


def test_session_profile_overrides_always_empty(tmp_path: Path) -> None:
    outdir = tmp_path / "no_overrides"
    session = _make_session(outdir)
    session.record_step(0, loss_delta_z=0.01)
    session.evaluate(step=0)
    session.close(run_status="success", ended_ts_utc=T1)

    res = ResultsV1.model_validate_json((outdir / "results.json").read_text("utf-8"))
    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))
    assert dict(res.profile.overrides) == {}
    assert dict(summ.profile.overrides) == {}

    v = validate_outdir(
        outdir,
        strict_identity=True,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message
