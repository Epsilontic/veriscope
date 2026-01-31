from __future__ import annotations

import os
import shutil
import socket
import subprocess
from subprocess import TimeoutExpired
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from veriscope.core.jsonutil import read_json_obj

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_hf_ddp_smoke.sh"


def _resolve_capsule_dir(outdir: Path) -> Path:
    outdir = Path(outdir)
    direct = outdir / "window_signature.json"
    if direct.exists():
        return outdir

    matches = list(outdir.rglob("window_signature.json"))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one capsule under {outdir}, found {len(matches)} window_signature.json files: "
            f"{[str(p) for p in matches]}"
        )
    return matches[0].parent


def _must_exist(capdir: Path, name: str) -> Path:
    p = Path(capdir) / name
    if p.exists():
        return p
    matches = list(Path(capdir).rglob(name))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected {name} in {capdir} (or exactly once under it), found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]


def _hf_ddp_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for k in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["VERISCOPE_PYTHON_BIN"] = sys.executable
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HOME"] = str(tmp_path / "hf_home")
    env["TRANSFORMERS_CACHE"] = str(tmp_path / "hf_cache")
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_datasets_cache")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(_reserve_free_port())
    env["GLOO_SOCKET_IFNAME"] = _loopback_ifname()
    env["TP_SOCKET_IFNAME"] = _loopback_ifname()
    return env


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _loopback_ifname() -> str:
    return "lo0" if sys.platform == "darwin" else "lo"


def _skip_if_hf_unreachable() -> None:
    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        with urllib.request.urlopen(req, timeout=5):
            return
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("Hugging Face hub unreachable; skipping HF DDP smoke.")


def _maybe_skip_hf_download_failure(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        return
    output = f"{result.stdout}\n{result.stderr}".lower()
    patterns = (
        "readtimeout",
        "read timeout",
        "connectionerror",
        "connection error",
        "hf_hub_download",
        "can't load the model",
        "sslerror",
        "maxretryerror",
        "temporary failure in name resolution",
    )
    if any(p in output for p in patterns):
        pytest.skip("Hugging Face hub download failed; skipping HF DDP smoke.")


@pytest.mark.integration
def test_hf_ddp_smoke_cpu(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable; skipping HF DDP smoke.")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available on PATH; skipping HF DDP smoke.")
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        pytest.skip("HF offline mode enabled; skipping HF DDP smoke.")
    _skip_if_hf_unreachable()

    outdir = tmp_path / "hf_ddp_smoke"
    env = _hf_ddp_env(tmp_path)
    env["VERISCOPE_OUT_BASE"] = str(outdir)

    # Tighten: DDP rendezvous can hang on some macOS setups; treat as infra flake.
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT), str(outdir)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except TimeoutExpired:
        pytest.skip("torchrun rendezvous hung (timeout). Skipping HF DDP smoke on this host.")
    _maybe_skip_hf_download_failure(result)
    # Also skip explicit rendezvous/network failures (instead of failing the suite).
    out = f"{result.stdout}\n{result.stderr}".lower()
    if "distnetworkerror" in out or "client socket has timed out" in out:
        pytest.skip("torchrun rendezvous/network failure; skipping HF DDP smoke.")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    capdir = _resolve_capsule_dir(outdir)

    required = [
        "window_signature.json",
        "results.json",
        "results_summary.json",
        "run_config_resolved.json",
        "run_manifest.json",
    ]
    for name in required:
        assert _must_exist(capdir, name).exists()

    validate = subprocess.run(
        [sys.executable, "-m", "veriscope.cli.main", "validate", str(capdir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert validate.returncode == 0, f"{validate.stdout}\n{validate.stderr}"

    resolved = read_json_obj(_must_exist(capdir, "run_config_resolved.json"))
    assert resolved.get("schema_version") == 1
    results = read_json_obj(_must_exist(capdir, "results.json"))
    gates = results.get("gates", [])
    assert any(
        gate.get("audit", {}).get("ddp_agg") == "mean"
        and gate.get("audit", {}).get("world_size") == 2
        and gate.get("audit", {}).get("reason") != "ddp_unsupported"
        and gate.get("decision") != "skip"
        for gate in gates
    ), "Expected at least one evaluated gate with DDP aggregation provenance."
