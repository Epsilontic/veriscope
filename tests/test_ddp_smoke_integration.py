from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from veriscope.core.jsonutil import canonical_json_sha256, read_json_obj


def _build_env(env: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ)
    if env:
        base.update(env)
    repo_root = Path(__file__).resolve().parents[1]
    existing = base.get("PYTHONPATH", "")
    base["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    return base


def _run_torchrun(outdir: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "scripts/ddp_smoke_minimal.py",
        "--outdir",
        str(outdir),
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=_build_env(env))


@pytest.mark.integration
def test_ddp_smoke_emits_single_capsule(tmp_path: Path) -> None:
    outdir = tmp_path / "ddp_smoke"
    result = _run_torchrun(outdir)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    emitted = sorted(path.name for path in outdir.iterdir())
    assert emitted == ["results.json", "results_summary.json", "window_signature.json"]

    window_signature = read_json_obj(outdir / "window_signature.json")
    ws_hash = canonical_json_sha256(window_signature)
    results = read_json_obj(outdir / "results.json")
    summary = read_json_obj(outdir / "results_summary.json")

    assert results["run_id"] == "ddp-smoke"
    assert results["window_signature_ref"]["hash"] == ws_hash
    assert summary["window_signature_ref"]["hash"] == ws_hash


@pytest.mark.integration
def test_ddp_smoke_failure_propagates(tmp_path: Path) -> None:
    outdir = tmp_path / "ddp_smoke_fail"
    env = os.environ.copy()
    env["VS_FAIL_RANK"] = "1"

    result = _run_torchrun(outdir, env=env)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"

    emitted: list[str]
    if outdir.exists():
        emitted = sorted(path.name for path in outdir.iterdir())
    else:
        emitted = []
    assert emitted in ([], ["results.json", "results_summary.json", "window_signature.json"])

    if emitted:
        window_signature = read_json_obj(outdir / "window_signature.json")
        ws_hash = canonical_json_sha256(window_signature)
        results = read_json_obj(outdir / "results.json")
        summary = read_json_obj(outdir / "results_summary.json")

        assert results["window_signature_ref"]["hash"] == ws_hash
        assert summary["window_signature_ref"]["hash"] == ws_hash


@pytest.mark.integration
def test_ddp_smoke_rejects_invalid_fail_rank(tmp_path: Path) -> None:
    outdir = tmp_path / "ddp_smoke_bad_env"
    env = os.environ.copy()
    env["VS_FAIL_RANK"] = "bogus"

    result = _run_torchrun(outdir, env=env)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    combined = f"{result.stdout}\n{result.stderr}"
    assert "VS_FAIL_RANK must be int" in combined
