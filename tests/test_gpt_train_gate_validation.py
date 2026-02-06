from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from veriscope.runners.gpt.train_nanogpt import TrainConfig, VeriscopeGatedTrainer, _validate_gate_row

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


def test_compute_gate_check_no_regime_strips_regime_audit_and_avoids_zero_tv_plateau(
    make_window_decl,
    make_fr_window,
    make_gate_engine,
    force_zero_tv,
) -> None:
    decl = make_window_decl(
        ["m1"],
        weights={"m1": 1.0},
        bins=16,
        epsilon=0.12,
        cal_ranges={"m1": (0.0, 1.0)},
    )
    fr = make_fr_window(decl)
    ge = make_gate_engine(fr, min_evidence=1, policy="persistence")

    cfg = TrainConfig(
        regime_enabled=False,
        gate_window=4,
        metric_interval=1,
        gate_min_evidence=1,
        gate_policy="persistence",
        gate_persistence_k=2,
    )

    trainer = object.__new__(VeriscopeGatedTrainer)
    trainer.config = cfg
    trainer.window_decl = decl
    trainer.gate_engine = ge
    trainer._regime_wrapper_enabled = False
    trainer.metric_history = [
        {"iter": 0, "m1": 0.10, "loss": 1.00, "ewma_loss": 1.10},
        {"iter": 1, "m1": 0.10, "loss": 0.99, "ewma_loss": 1.09},
        {"iter": 2, "m1": 0.10, "loss": 0.98, "ewma_loss": 1.08},
        {"iter": 3, "m1": 0.10, "loss": 0.97, "ewma_loss": 1.07},
        {"iter": 4, "m1": 0.20, "loss": 0.96, "ewma_loss": 1.06},
        {"iter": 5, "m1": 0.20, "loss": 0.95, "ewma_loss": 1.05},
        {"iter": 6, "m1": 0.20, "loss": 0.94, "ewma_loss": 1.04},
        {"iter": 7, "m1": 0.20, "loss": 0.93, "ewma_loss": 1.03},
    ]

    trainer.iter_num = 8
    row1 = trainer._compute_gate_check()

    trainer.metric_history.append({"iter": 8, "m1": 0.95, "loss": 0.92, "ewma_loss": 1.02})
    trainer.iter_num = 9
    row2 = trainer._compute_gate_check()

    all_zero_rows = 0
    for row in (row1, row2):
        audit = row["audit"]
        assert all(not str(key).startswith("regime_") for key in audit.keys())
        pm = audit.get("per_metric_tv", {})
        assert isinstance(pm, dict) and pm
        is_all_zero = all(np.isclose(float(v), 0.0, atol=1e-12) for v in pm.values())
        if is_all_zero:
            all_zero_rows += 1
            assert row["decision"] == "fail"
            assert audit.get("reason") in {"empty_window", "zero_tv_nonidentical_windows"}
    assert all_zero_rows >= 1
