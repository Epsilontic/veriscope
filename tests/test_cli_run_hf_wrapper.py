from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from veriscope.cli.report import render_report_md
from veriscope.cli.validate import validate_outdir


RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fake_hf_runner.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_wrapper(outdir: Path, runner_args: list[str], fake_deps_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-c",
        "from veriscope.cli.main import main; import sys; sys.exit(main(sys.argv[1:]))",
        "run",
        "hf",
        "--gate-preset",
        "tuned_v0",
        "--outdir",
        str(outdir),
        "--force",
        "--",
    ] + runner_args
    env = os.environ.copy()
    env["VERISCOPE_HF_RUNNER_CMD"] = f"{sys.executable} {RUNNER_SCRIPT}"
    pythonpath = env.get("PYTHONPATH", "")
    extra_paths = [str(REPO_ROOT), str(fake_deps_dir)]
    if pythonpath:
        extra_paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)


def test_run_hf_wrapper_with_fake_runner(tmp_path: Path) -> None:
    outdir = tmp_path / "hf_run"
    fake_deps_dir = tmp_path / "fake_deps"
    fake_deps_dir.mkdir()

    (fake_deps_dir / "datasets.py").write_text("__all__ = []\n", encoding="utf-8")
    (fake_deps_dir / "transformers.py").write_text("__all__ = []\n", encoding="utf-8")

    result = _run_wrapper(outdir, ["--max_steps", "1"], fake_deps_dir)

    assert result.returncode == 0, result.stderr
    assert (outdir / "run_config_resolved.json").exists()

    assert (outdir / "window_signature.json").exists()
    assert (outdir / "results.json").exists()
    assert (outdir / "results_summary.json").exists()

    v = validate_outdir(outdir)
    assert v.ok, v.message
    summary = json.loads((outdir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary.get("run_status") == "success"
    assert summary.get("profile", {}).get("gate_preset") == "tuned_v0"

    report = render_report_md(outdir, fmt="text")
    assert "Veriscope Report" in report
