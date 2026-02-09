from __future__ import annotations

import numpy as np
import pytest

from veriscope.core.regime import RegimeAnchoredGateEngine, RegimeConfig

pytestmark = pytest.mark.unit


def _run_pre_reference_persistence_pair(
    *,
    make_gate_engine,
    fr_window,
    pre_reference_change_policy: str | None,
    regime_enabled: bool = True,
):
    base_engine = make_gate_engine(
        fr_window,
        min_evidence=1,
        policy="persistence",
        persistence_k=2,
    )

    cfg_kwargs = {
        "enabled": regime_enabled,
        "reference_build_min_iter": 10_000,
        "reference_build_max_iter": 10_100,
        "min_evidence_per_metric": 1,
        "min_windows_for_reference": 1,
    }
    if pre_reference_change_policy is not None:
        cfg_kwargs["pre_reference_change_policy"] = pre_reference_change_policy

    gate = RegimeAnchoredGateEngine(
        base_engine=base_engine,
        fr_win=fr_window,
        config=RegimeConfig(**cfg_kwargs),
        gate_warmup=0,
        gate_window=10,
    )

    past = {"test_metric": np.zeros(64, dtype=float)}
    recent = {"test_metric": np.ones(64, dtype=float)}
    counts = {"test_metric": 64}

    first = gate.check(
        past=past,
        recent=recent,
        counts_by_metric=counts,
        gain_bits=0.1,
        kappa_sens=0.0,
        eps_stat_value=0.01,
        iter_num=100,
    )
    second = gate.check(
        past=past,
        recent=recent,
        counts_by_metric=counts,
        gain_bits=0.1,
        kappa_sens=0.0,
        eps_stat_value=0.01,
        iter_num=200,
    )
    return first, second


def test_pre_reference_warn_only_suppresses_change_persistence_fail(make_gate_engine, fr_window) -> None:
    _, second = _run_pre_reference_persistence_pair(
        make_gate_engine=make_gate_engine,
        fr_window=fr_window,
        pre_reference_change_policy=None,  # use RegimeConfig default
        regime_enabled=True,
    )

    assert second.ok is True
    assert second.warn is True
    assert second.audit.get("pre_reference") is True
    assert second.audit.get("pre_reference_change_policy") == "warn_only"
    assert second.audit.get("pre_reference_change_suppressed") is True
    assert second.audit.get("pre_reference_original_ok") is False
    assert "change_persistence_fail" in str(second.audit.get("pre_reference_original_reason", ""))
    assert second.audit.get("ref_ready") is False
    assert second.audit.get("regime_ref_ready") is False
    assert isinstance(second.audit.get("ref_windows_built"), int)
    assert isinstance(second.audit.get("regime_ref_windows_built"), int)


def test_pre_reference_enforce_preserves_change_persistence_fail(make_gate_engine, fr_window) -> None:
    _, second = _run_pre_reference_persistence_pair(
        make_gate_engine=make_gate_engine,
        fr_window=fr_window,
        pre_reference_change_policy="enforce",
        regime_enabled=True,
    )

    assert second.ok is False
    assert second.warn is False
    assert second.audit.get("pre_reference") is True
    assert second.audit.get("pre_reference_change_policy") == "enforce"
    assert second.audit.get("pre_reference_change_suppressed") is not True
    assert "change_persistence_fail" in str(second.audit.get("reason", ""))


def test_pre_reference_ignore_suppresses_change_and_persistence(make_gate_engine, fr_window) -> None:
    _, second = _run_pre_reference_persistence_pair(
        make_gate_engine=make_gate_engine,
        fr_window=fr_window,
        pre_reference_change_policy="ignore",
        regime_enabled=True,
    )

    assert second.ok is True
    assert second.warn is False
    assert second.audit.get("pre_reference") is True
    assert second.audit.get("pre_reference_change_policy") == "ignore"
    assert second.audit.get("pre_reference_change_suppressed") is True
    assert second.audit.get("change_ok") == "suppressed"
    assert int(second.audit.get("consecutive_exceedances", -1)) == 1
    assert int(second.audit.get("consecutive_exceedances_before", -1)) == 1
    assert int(second.audit.get("consecutive_exceedances_after", -1)) == 1


def test_pre_reference_warn_only_suppresses_change_persistence_fail_when_regime_disabled(
    make_gate_engine, fr_window
) -> None:
    _, second = _run_pre_reference_persistence_pair(
        make_gate_engine=make_gate_engine,
        fr_window=fr_window,
        pre_reference_change_policy="warn_only",
        regime_enabled=False,
    )

    assert second.ok is True
    assert second.warn is True
    assert second.audit.get("pre_reference") is True
    assert second.audit.get("pre_reference_change_policy") == "warn_only"
    assert second.audit.get("pre_reference_change_suppressed") is True
    assert str(second.audit.get("reason", "")).startswith("pre_reference_suppressed:")
    assert second.audit.get("ref_ready") is False
    assert second.audit.get("regime_ref_ready") is None
    assert isinstance(second.audit.get("ref_windows_built"), int)
    assert second.audit.get("regime_ref_windows_built") is None


def test_pre_reference_warn_only_suppresses_change_persistence_fail_with_gain_below_when_regime_disabled(
    make_gate_engine, fr_window
) -> None:
    base_engine = make_gate_engine(
        fr_window,
        min_evidence=1,
        gain_thresh=0.2,
        policy="persistence_stability",
        persistence_k=2,
    )

    gate = RegimeAnchoredGateEngine(
        base_engine=base_engine,
        fr_win=fr_window,
        config=RegimeConfig(
            enabled=False,
            reference_build_min_iter=10_000,
            reference_build_max_iter=10_100,
            min_evidence_per_metric=1,
            min_windows_for_reference=1,
            pre_reference_change_policy="warn_only",
        ),
        gate_warmup=0,
        gate_window=10,
    )

    past = {"test_metric": np.zeros(64, dtype=float)}
    recent = {"test_metric": np.ones(64, dtype=float)}
    counts = {"test_metric": 64}

    first = gate.check(
        past=past,
        recent=recent,
        counts_by_metric=counts,
        gain_bits=0.1,  # below gain_thresh, but persistence_stability keeps gain as WARN-only.
        kappa_sens=0.0,
        eps_stat_value=0.01,
        iter_num=100,
    )
    second = gate.check(
        past=past,
        recent=recent,
        counts_by_metric=counts,
        gain_bits=0.1,
        kappa_sens=0.0,
        eps_stat_value=0.01,
        iter_num=200,
    )

    assert first.ok is True
    assert first.warn is True
    assert second.ok is True
    assert second.warn is True
    assert second.audit.get("pre_reference") is True
    assert second.audit.get("pre_reference_change_policy") == "warn_only"
    assert second.audit.get("pre_reference_change_suppressed") is True
    assert second.audit.get("pre_reference_suppression_cause") == "reference_not_ready"
    assert second.audit.get("change_gain_failed") is True
    assert "persistence_fail" in str(second.audit.get("pre_reference_original_reason", ""))
    assert str(second.audit.get("reason", "")).startswith("pre_reference_suppressed:")
