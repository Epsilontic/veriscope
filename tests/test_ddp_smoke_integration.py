from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from veriscope.core.jsonutil import canonical_json_sha256, read_json_obj

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _build_env(env: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ)
    if env:
        base.update(env)
    existing = base.get("PYTHONPATH", "")
    base["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)

    # Force loopback rendezvous so local hostname resolution cannot deadlock torchrun (common on macOS).
    base.setdefault("MASTER_ADDR", "127.0.0.1")
    base.setdefault("MASTER_PORT", str(_free_port()))

    # Prefer loopback interface for gloo if present.
    base.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")

    # macOS often probes IPv6/hostname paths that can stall local rendezvous; force IPv4.
    base.setdefault("GLOO_USE_IPV6", "0")
    # Avoid libuv store pathologies on some torch builds.
    base.setdefault("TORCH_DISTRIBUTED_USE_LIBUV", "0")

    return base


def _run_torchrun(outdir: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    script = REPO_ROOT / "scripts" / "ddp_smoke_minimal.py"
    env_built = _build_env(env)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--local_addr",
        env_built["MASTER_ADDR"],
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={env_built['MASTER_ADDR']}:{env_built['MASTER_PORT']}",
        str(script),
        "--outdir",
        str(outdir),
    ]
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env_built,
        cwd=str(REPO_ROOT),
        timeout=90,
    )


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


@pytest.mark.integration
def test_ddp_smoke_rejects_out_of_range_fail_rank(tmp_path: Path) -> None:
    outdir = tmp_path / "ddp_smoke_bad_range"
    env = os.environ.copy()
    env["VS_FAIL_RANK"] = "9"

    result = _run_torchrun(outdir, env=env)
    assert result.returncode != 0, f"{result.stdout}\n{result.stderr}"
    combined = f"{result.stdout}\n{result.stderr}"
    assert "VS_FAIL_RANK must be in [0, 1]" in combined
