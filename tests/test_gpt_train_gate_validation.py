from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from veriscope.runners.gpt.train_nanogpt import _validate_gate_row

pytestmark = pytest.mark.unit


def _gate_row(*, decision: str, ok: bool, warn: bool, reason: str, evaluated: bool = True) -> dict[str, object]:
    audit = {
        "reason": reason,
        "evaluated": evaluated,
        "policy": "persistence",
        "gain_bits": 0.0,
        "gain_thresh": 0.0,
        "gain_ok": bool(ok),
        "worst_DW": 0.25,
        "eps_eff": 0.08,
        "dw_exceeds_threshold": False,
        "metrics_exceeding": [],
        "min_metrics_exceeding": 1,
        "evidence_total": 64,
        "min_evidence": 16,
    }
    return {
        "decision": decision,
        "ok": ok,
        "warn": warn,
        "reason": reason,
        "audit": audit,
    }


def test_validate_gate_row_rejects_warn_fail_conflict() -> None:
    row = _gate_row(
        decision="warn",
        ok=False,
        warn=True,
        reason="change_warn_pending",
    )
    with pytest.raises(ValueError, match="decision='warn' requires ok=True"):
        _validate_gate_row(row)


def test_validate_gate_row_allows_fail_without_warn() -> None:
    row = _gate_row(
        decision="fail",
        ok=False,
        warn=False,
        reason="change_persistence_fail",
    )
    _validate_gate_row(row)
