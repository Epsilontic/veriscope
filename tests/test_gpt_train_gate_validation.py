from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
import numpy as np

torch = pytest.importorskip("torch")

from veriscope.runners.gpt.train_nanogpt import (
    TrainConfig,
    VeriscopeGatedTrainer,
    _DECL_TRANSPORT_KEY,
    _MAX_DEPTH,
    _to_jsonable,
    _validate_gate_row,
)

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


def test_validate_gate_row_reason_sync_warn_fail_only() -> None:
    row_warn = _gate_row(
        decision="warn",
        ok=True,
        warn=True,
        reason="none_ok",
    )
    row_warn["audit"]["reason"] = "gain_below_threshold"
    _validate_gate_row(row_warn)
    assert row_warn["reason"] == "gain_below_threshold"

    row_pass = _gate_row(
        decision="pass",
        ok=True,
        warn=False,
        reason="none_ok",
    )
    row_pass["audit"]["reason"] = "change_ok"
    _validate_gate_row(row_pass)
    assert row_pass["reason"] == "none_ok"


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


def test_compute_gate_check_regime_dw_alias_backcompat_and_strip_behavior(make_window_decl) -> None:
    decl = make_window_decl(
        ["m1"],
        weights={"m1": 1.0},
        bins=16,
        epsilon=0.12,
        cal_ranges={"m1": (0.0, 1.0)},
    )

    cfg = TrainConfig(
        regime_enabled=True,
        gate_window=4,
        metric_interval=1,
        gate_min_evidence=1,
        gate_policy="persistence",
        gate_persistence_k=2,
    )

    metric_history = [
        {"iter": 0, "m1": 0.10, "loss": 1.00, "ewma_loss": 1.10},
        {"iter": 1, "m1": 0.10, "loss": 0.99, "ewma_loss": 1.09},
        {"iter": 2, "m1": 0.10, "loss": 0.98, "ewma_loss": 1.08},
        {"iter": 3, "m1": 0.10, "loss": 0.97, "ewma_loss": 1.07},
        {"iter": 4, "m1": 0.95, "loss": 0.96, "ewma_loss": 1.06},
        {"iter": 5, "m1": 0.95, "loss": 0.95, "ewma_loss": 1.05},
        {"iter": 6, "m1": 0.95, "loss": 0.94, "ewma_loss": 1.04},
        {"iter": 7, "m1": 0.95, "loss": 0.93, "ewma_loss": 1.03},
    ]

    def _build_row(*, regime_wrapper_enabled: bool) -> dict[str, object]:
        trainer = object.__new__(VeriscopeGatedTrainer)
        trainer.config = cfg
        trainer.window_decl = decl
        trainer._regime_wrapper_enabled = regime_wrapper_enabled
        trainer.metric_history = list(metric_history)
        trainer.iter_num = 8

        class _StubGateEngine:
            def check(self, **kwargs):
                _ = kwargs
                return SimpleNamespace(
                    ok=False,
                    warn=False,
                    reason="regime_fail",
                    audit={
                        "evaluated": True,
                        "reason": "regime_fail",
                        "base_reason": "regime_fail",
                        "change_reason": "regime_fail",
                        "policy": "persistence",
                        "worst_DW": 0.95,
                        "eps_eff": 0.10,
                        "per_metric_tv": {"m1": 0.95},
                        "evidence_total": 8,
                        "min_evidence": 1,
                        "regime_check_ran": True,
                        "regime_ok": False,
                        "regime_warn": False,
                        "regime_enabled": True,
                        "regime_active": True,
                        "regime_worst_DW": 0.95,
                    },
                )

        trainer.gate_engine = _StubGateEngine()
        return trainer._compute_gate_check()

    row_regime_on = _build_row(regime_wrapper_enabled=True)
    row_regime_off = _build_row(regime_wrapper_enabled=False)
    produced = {"gates": [row_regime_on, row_regime_off]}

    regime_fail_row = produced["gates"][0]
    assert regime_fail_row["decision"] == "fail"
    audit = regime_fail_row["audit"]
    assert "regime_worst_DW" in audit
    assert np.isfinite(float(audit["regime_worst_DW"]))
    assert "regime_D_W" in audit
    assert audit["regime_D_W"] == audit["regime_worst_DW"]

    audit_disabled = produced["gates"][1]["audit"]
    assert all(not str(key).startswith("regime_") for key in audit_disabled.keys())


def test_to_jsonable_detects_recursive_cycles() -> None:
    recursive_dict: dict[str, object] = {}
    recursive_dict["self"] = recursive_dict
    assert _to_jsonable(recursive_dict) == {"self": "<recursive_ref>"}

    recursive_list: list[object] = []
    recursive_list.append(recursive_list)
    assert _to_jsonable(recursive_list) == ["<recursive_ref>"]


def test_to_jsonable_limits_depth() -> None:
    nested: object = "leaf"
    for _ in range(_MAX_DEPTH + 2):
        nested = [nested]

    converted = _to_jsonable(nested)
    cursor = converted
    for _ in range(_MAX_DEPTH + 1):
        assert isinstance(cursor, list)
        assert len(cursor) == 1
        cursor = cursor[0]
    assert cursor == "<max_depth>"


def test_to_jsonable_replaces_non_finite_floats() -> None:
    assert _to_jsonable(float("nan")) is None
    assert _to_jsonable(float("inf")) is None
    assert _to_jsonable(float("-inf")) is None


def test_to_jsonable_summarizes_large_ndarray() -> None:
    arr = np.zeros((101, 100), dtype=np.float32)  # size=10100 (>10_000)
    converted = _to_jsonable(arr)
    assert converted == {"__type__": "ndarray", "shape": [101, 100], "dtype": "float32"}


def test_to_jsonable_handles_torch_tensor() -> None:
    small = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    assert _to_jsonable(small) == [[1.0, 2.0], [3.0, 4.0]]

    large = torch.zeros((101, 100), dtype=torch.float32)  # numel=10100 (>10_000)
    converted = _to_jsonable(large)
    assert converted == {"__type__": "ndarray", "shape": [101, 100], "dtype": str(large.dtype)}


def test_to_jsonable_uses_pydantic_model_dump() -> None:
    pydantic = pytest.importorskip("pydantic")
    if not hasattr(pydantic.BaseModel, "model_dump"):
        pytest.skip("requires pydantic v2 model_dump")

    class Payload(pydantic.BaseModel):
        a: int
        b: float

    converted = _to_jsonable(Payload(a=1, b=2.5))
    assert converted == {"a": 1, "b": 2.5}


def test_to_jsonable_strips_decl_transport_from_dataclass() -> None:
    @dataclass
    class DeclLike:
        bins: int
        _DECL_TRANSPORT: object = object()

    converted = _to_jsonable(DeclLike(bins=16))
    assert converted["bins"] == 16
    assert _DECL_TRANSPORT_KEY not in converted
