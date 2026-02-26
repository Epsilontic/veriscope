from __future__ import annotations

from pathlib import Path

import pytest

from veriscope.cli.diff import diff_outdirs
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ResultsSummaryV1

pytestmark = pytest.mark.unit


def test_convenience_observer_validates(tmp_path: Path) -> None:
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "obs"
    logger = ConvenienceLogger(outdir, observer_mode=True, cadence=5, gate_window=10)
    for i in range(100):
        logger.step(i, loss=2.0 - 0.01 * i)
    logger.close()

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message

    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))
    assert summ.final_decision == "skip"
    assert summ.partial is not True
    assert dict(summ.profile.overrides) == {}


def test_convenience_gated_validates(tmp_path: Path) -> None:
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "gated"
    logger = ConvenienceLogger(outdir, cadence=2, gate_window=5)
    for i in range(60):
        logger.step(i, loss=2.0 - 0.005 * i)
    logger.close()

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message


def test_convenience_report_renders(tmp_path: Path) -> None:
    from veriscope.cli.report import render_report_md
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "report"
    logger = ConvenienceLogger(outdir, cadence=2, gate_window=5)
    for i in range(60):
        logger.step(i, loss=2.0 - 0.005 * i)
    logger.close()

    md = render_report_md(outdir, fmt="text")
    assert "Veriscope Report" in md
    assert "Gate Summary" in md


def test_convenience_two_runs_comparable(tmp_path: Path) -> None:
    from veriscope.core.convenience_logger import ConvenienceLogger

    for name in ["a", "b"]:
        outdir = tmp_path / name
        logger = ConvenienceLogger(outdir, cadence=2, gate_window=5)
        for i in range(60):
            logger.step(i, loss=2.0 - 0.005 * i)
        logger.close()

    result = diff_outdirs(tmp_path / "a", tmp_path / "b")
    assert result.exit_code == 0, result.stderr


def test_convenience_observer_comparable_with_gated(tmp_path: Path) -> None:
    from veriscope.core.convenience_logger import ConvenienceLogger

    for name, obs in [("observer", True), ("gated", False)]:
        outdir = tmp_path / name
        logger = ConvenienceLogger(outdir, cadence=2, gate_window=5, observer_mode=obs)
        for i in range(60):
            logger.step(i, loss=2.0 - 0.005 * i)
        logger.close()

    result = diff_outdirs(tmp_path / "observer", tmp_path / "gated")
    assert result.exit_code == 0, result.stderr


def test_convenience_spike_triggers_gate(tmp_path: Path) -> None:
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "spike"
    logger = ConvenienceLogger(
        outdir,
        cadence=1,
        gate_window=10,
        gate_epsilon=0.08,
        gate_min_evidence=5,
    )
    decisions = []
    for i in range(80):
        loss = 2.0 - 0.005 * i
        if 50 <= i < 60:
            loss += 5.0
        d = logger.step(i, loss=loss)
        if d is not None:
            decisions.append(d)
    logger.close()

    assert any(d in ("warn", "fail") for d in decisions), f"Expected gate trip; got {decisions}"

    v = validate_outdir(
        outdir,
        allow_missing_governance=False,
        allow_invalid_governance=False,
    )
    assert v.ok, v.message
