from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from veriscope.cli import main as main_mod
from veriscope.cli.main import _cmd_run_cifar
from veriscope.cli.validate import validate_outdir

pytestmark = pytest.mark.unit


def _write_fake_legacy_executable(path: Path, body: str) -> None:
    script = "\n".join(
        [
            "#!/usr/bin/env python3",
            "from __future__ import annotations",
            body,
            "",
        ]
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def test_run_cifar_wrapper_replaces_invalid_summary_with_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir = tmp_path / "cifar_run"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)

    legacy_exe = fake_bin / "veriscope-legacy"
    _write_fake_legacy_executable(
        legacy_exe,
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "outdir = Path(os.environ['SCAR_OUTDIR'])",
                "outdir.mkdir(parents=True, exist_ok=True)",
                "(outdir / 'results_summary.json').write_text('{bad json\\n', encoding='utf-8')",
                "raise SystemExit(0)",
            ]
        ),
    )

    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    args = argparse.Namespace(
        outdir=str(outdir),
        smoke=True,
        force=True,
        legacy_args=[],
    )
    exit_code = _cmd_run_cifar(args)
    assert exit_code == 2

    summary = json.loads((outdir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary.get("partial") is True
    assert summary.get("wrapper_emitted") is True
    assert "missing_or_invalid" in str(summary.get("note", ""))

    v = validate_outdir(outdir, allow_partial=True)
    assert v.ok, v.message


def test_run_cifar_wrapper_partial_summary_survives_recovery_signature_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "cifar_run_recovery"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)

    legacy_exe = fake_bin / "veriscope-legacy"
    _write_fake_legacy_executable(
        legacy_exe,
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "outdir = Path(os.environ['SCAR_OUTDIR'])",
                "outdir.mkdir(parents=True, exist_ok=True)",
                "(outdir / 'results_summary.json').write_text('{bad json\\n', encoding='utf-8')",
                "raise SystemExit(0)",
            ]
        ),
    )

    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    original_ensure = main_mod._ensure_window_signature

    def _ensure_with_recovery_failure(outdir_path: Path, *, reason: str, run_kind: str):
        if reason == "wrapper_emitted_partial_summary":
            raise RuntimeError("forced recovery signature failure")
        return original_ensure(outdir_path, reason=reason, run_kind=run_kind)

    monkeypatch.setattr(main_mod, "_ensure_window_signature", _ensure_with_recovery_failure)

    args = argparse.Namespace(
        outdir=str(outdir),
        smoke=True,
        force=True,
        legacy_args=[],
    )
    exit_code = _cmd_run_cifar(args)
    assert exit_code == 2

    summary_path = outdir / "results_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("partial") is True
    assert summary.get("wrapper_emitted") is True

    v = validate_outdir(outdir, allow_partial=True)
    assert v.ok, v.message
