from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_RUNNER = REPO_ROOT / "scripts" / "fake_gpt_runner.py"
GPT_SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_gpt_smoke.sh"
PILOT_RUN_SCRIPT = REPO_ROOT / "scripts" / "pilot" / "run.sh"


def _make_veriscope_shim(shim_dir: Path) -> Path:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "veriscope"
    shim.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'if [[ "${1:-}" == "report" && "${VERISCOPE_SHIM_FAIL_REPORT:-0}" == "1" ]]; then',
                '  echo "forced report failure (shim)" >&2',
                "  exit 2",
                "fi",
                f'exec "{sys.executable}" -m veriscope.cli.main "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _base_env(tmp_path: Path) -> dict[str, str]:
    shim_dir = tmp_path / "bin"
    _make_veriscope_shim(shim_dir)

    env = os.environ.copy()
    env["PATH"] = str(shim_dir) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["VERISCOPE_GPT_RUNNER_CMD"] = f"{sys.executable} {FAKE_RUNNER} --sleep-seconds 0 --emit-artifacts"
    return env


def _arg_value(args: list[str], key: str) -> str | None:
    for idx, token in enumerate(args):
        if token == key and idx + 1 < len(args):
            return args[idx + 1]
        if token.startswith(key + "="):
            return token.split("=", 1)[1]
    return None


def test_run_gpt_smoke_defaults_cpu_and_tuned_v0(tmp_path: Path) -> None:
    outdir = tmp_path / "smoke_out"
    nanogpt_root = tmp_path / "nanoGPT"
    ds_dir = nanogpt_root / "data" / "shakespeare_char"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "train.bin").write_bytes(b"")

    env = _base_env(tmp_path)
    env["NANOGPT_DIR"] = str(nanogpt_root)

    result = subprocess.run(
        ["bash", str(GPT_SMOKE_SCRIPT), str(outdir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "defaulting to --device cpu" in result.stdout

    run_cfg = json.loads((outdir / "run_config_resolved.json").read_text(encoding="utf-8"))
    normalized = list(run_cfg["argv"]["normalized_forwarded_args"])
    assert _arg_value(normalized, "--device") == "cpu"
    assert _arg_value(normalized, "--gate_preset") == "tuned_v0"

    window_signature = json.loads((outdir / "window_signature.json").read_text(encoding="utf-8"))
    assert window_signature["schema_version"] == 1
    assert window_signature.get("placeholder") is not True
    assert window_signature["gates"]["preset"] == "fake_runner"

    results_summary = json.loads((outdir / "results_summary.json").read_text(encoding="utf-8"))
    assert results_summary["profile"]["gate_preset"] == "fake_runner"


def test_pilot_run_strict_default_and_lenient_opt_out(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env["VERISCOPE_SHIM_FAIL_REPORT"] = "1"

    strict_outdir = tmp_path / "pilot_strict"
    strict = subprocess.run(
        ["bash", str(PILOT_RUN_SCRIPT), str(strict_outdir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode != 0
    assert "forced report failure" in (strict_outdir / "report_stderr.txt").read_text(encoding="utf-8")
    assert (strict_outdir / "capdir.txt").exists()

    lenient_outdir = tmp_path / "pilot_lenient"
    env_lenient = dict(env)
    env_lenient["VERISCOPE_PILOT_STRICT"] = "0"
    lenient = subprocess.run(
        ["bash", str(PILOT_RUN_SCRIPT), str(lenient_outdir)],
        cwd=str(REPO_ROOT),
        env=env_lenient,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lenient.returncode == 0, lenient.stderr
    assert "forced report failure" in (lenient_outdir / "report_stderr.txt").read_text(encoding="utf-8")
