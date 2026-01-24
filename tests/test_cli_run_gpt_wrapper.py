# tests/test_cli_run_gpt_wrapper.py
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from veriscope.cli.report import render_report_md
from veriscope.cli.validate import validate_outdir


RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fake_gpt_runner.py"


def _wait_for_file(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


def _wait_for_proc_running(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is None:
            return
        time.sleep(0.05)
    raise AssertionError("Process exited before signal could be sent")


def _run_wrapper(outdir: Path, runner_args: list[str]) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-c",
        "from veriscope.cli.main import main; import sys; sys.exit(main(sys.argv[1:]))",
        "run",
        "gpt",
        "--outdir",
        str(outdir),
        "--force",
        "--",
    ] + runner_args
    env = os.environ.copy()
    env["VERISCOPE_GPT_RUNNER_CMD"] = f"{sys.executable} {RUNNER_SCRIPT}"
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_run_gpt_sigterm_writes_fallback_summary(tmp_path: Path) -> None:
    outdir = tmp_path / "sigterm_run"
    proc = _run_wrapper(outdir, ["--sleep-seconds", "10"])
    _wait_for_file(outdir / "run_config_resolved.json")
    _wait_for_proc_running(proc)

    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=15)

    assert proc.returncode == 2
    _ = stdout + stderr

    summary_path = outdir / "results_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("partial") is True
    assert summary.get("wrapper_emitted") is True
    assert summary.get("runner_signal") is not None
    assert summary.get("note", "").startswith("wrapper-emitted")

    v = validate_outdir(outdir, allow_partial=True)
    assert v.ok, v.message
    assert render_report_md(outdir, fmt="text")


def test_run_gpt_interrupt_timeout_escalation(tmp_path: Path) -> None:
    outdir = tmp_path / "sigkill_run"
    proc = _run_wrapper(outdir, ["--sleep-seconds", "30", "--ignore-signals"])
    _wait_for_file(outdir / "run_config_resolved.json")
    _wait_for_proc_running(proc)

    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=15)

    assert proc.returncode == 2
    _ = stdout + stderr

    summary_path = outdir / "results_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary.get("partial") is True
    assert summary.get("wrapper_emitted") is True
    assert summary.get("runner_signal") is not None
    assert summary.get("note", "").startswith("wrapper-emitted")

    v = validate_outdir(outdir, allow_partial=True)
    assert v.ok, v.message
    assert render_report_md(outdir, fmt="text")
