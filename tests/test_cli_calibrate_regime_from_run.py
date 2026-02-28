from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from veriscope.cli.main import _cmd_calibrate_regime_from_run

pytestmark = pytest.mark.unit


def test_cmd_calibrate_regime_from_run_write_failure_returns_exit_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import veriscope.cli.calibrate_regime_from_run as calibrate_regime_from_run_module

    def _fake_calibrate_regime_from_run(_capsule_dir: Path, *, quantile: float) -> dict[str, object]:
        _ = quantile
        return {
            "schema_version": 1,
            "capsule_dir": "unused",
            "source": "regime_D_W",
            "quantile": 0.95,
            "epsilon": 0.2,
            "n_samples": 5,
            "window_signature_hash": "0" * 64,
        }

    def _boom_write(self: Path, _payload: str, encoding: str = "utf-8") -> int:
        _ = self, encoding
        raise OSError("disk full")

    monkeypatch.setattr(calibrate_regime_from_run_module, "calibrate_regime_from_run", _fake_calibrate_regime_from_run)
    monkeypatch.setattr(Path, "write_text", _boom_write, raising=False)

    args = Namespace(capsule_dir=str(tmp_path / "capsule"), quantile=0.95, out=str(tmp_path / "out.json"))
    exit_code = _cmd_calibrate_regime_from_run(args)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "CALIBRATE_FAILED" in captured.err
