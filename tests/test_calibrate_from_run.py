from __future__ import annotations

import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_calibrate_from_gated_run(tmp_path: Path) -> None:
    from veriscope.cli.calibrate_from_run import calibrate_from_run
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "gated"
    logger = ConvenienceLogger(
        outdir,
        cadence=2,
        gate_window=5,
        gate_epsilon=0.12,
        gate_min_evidence=3,
    )
    for i in range(60):
        logger.step(i, loss=2.0 - 0.005 * i)
    logger.close()

    result = calibrate_from_run(outdir, quantile=0.95)
    assert math.isfinite(result["epsilon"])
    assert result["epsilon"] > 0
    assert result["n_samples"] > 0
    assert result["source"] == "gate_audit_per_metric_tv"


def test_calibrate_from_observer_run_raises(tmp_path: Path) -> None:
    from veriscope.cli.calibrate_from_run import calibrate_from_run
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "obs"
    logger = ConvenienceLogger(outdir, cadence=2, gate_window=5, observer_mode=True)
    for i in range(60):
        logger.step(i, loss=2.0 - 0.005 * i)
    logger.close()

    with pytest.raises(ValueError, match="No evaluated gates"):
        calibrate_from_run(outdir)


def test_calibrate_stable_run_tight_epsilon(tmp_path: Path) -> None:
    from veriscope.cli.calibrate_from_run import calibrate_from_run
    from veriscope.core.convenience_logger import ConvenienceLogger

    outdir = tmp_path / "stable"
    logger = ConvenienceLogger(
        outdir,
        cadence=1,
        gate_window=10,
        gate_epsilon=0.50,
        gate_min_evidence=3,
    )
    for i in range(100):
        logger.step(i, loss=2.0 - 0.001 * i)
    logger.close()

    result = calibrate_from_run(outdir, quantile=0.95)
    assert result["epsilon"] <= 0.50


def test_calibrate_rejects_invalid_capsule(tmp_path: Path) -> None:
    from veriscope.cli.calibrate_from_run import calibrate_from_run

    with pytest.raises(ValueError, match="Invalid capsule"):
        calibrate_from_run(tmp_path / "nonexistent")
