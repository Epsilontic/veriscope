from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from veriscope.core.artifacts import CountsV1, derive_final_decision
from veriscope.core.jsonutil import read_json_obj, window_signature_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_hf_micro_smoke.sh"


def _resolve_capsule_dir(outdir: Path) -> Path:
    """Resolve the capsule directory.

    The HF run wrapper may emit artifacts directly into outdir OR into a single
    capsule subdirectory beneath outdir. Make the integration test robust to both.
    """

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
    """Return capdir/name, or (if missing) find exactly one match under capdir."""

    p = Path(capdir) / name
    if p.exists():
        return p
    matches = list(Path(capdir).rglob(name))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected {name} in {capdir} (or exactly once under it), found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]


def _hf_smoke_env(tmp_path: Path) -> dict[str, str]:
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
    env["VERISCOPE_FORCE_SINGLE_PROCESS"] = "1"
    pythonpath = env.get("PYTHONPATH", "")
    extra_paths = [str(REPO_ROOT)]
    if pythonpath:
        extra_paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    env["VERISCOPE_PYTHON_BIN"] = sys.executable
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HOME"] = str(tmp_path / "hf_home")
    env["TRANSFORMERS_CACHE"] = str(tmp_path / "hf_cache")
    env["HF_DATASETS_CACHE"] = str(tmp_path / "hf_datasets_cache")
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _skip_if_hf_unreachable() -> None:
    try:
        req = urllib.request.Request("https://huggingface.co", method="HEAD")
        with urllib.request.urlopen(req, timeout=5):
            return
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("Hugging Face hub unreachable; skipping HF micro smoke.")


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
        pytest.skip("Hugging Face hub download failed; skipping HF micro smoke.")


@pytest.mark.integration
def test_hf_micro_smoke_integration(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        pytest.skip("HF offline mode enabled; skipping HF micro smoke.")
    _skip_if_hf_unreachable()

    outdir = tmp_path / "hf_micro_smoke"
    env = _hf_smoke_env(tmp_path)
    env["VERISCOPE_OUT_BASE"] = str(outdir)
    env["VERISCOPE_HF_MICRO_SMOKE_TIMEOUT_SECS"] = "180"

    result = subprocess.run(
        ["bash", str(SCRIPT), str(outdir)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    _maybe_skip_hf_download_failure(result)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    capdir = _resolve_capsule_dir(outdir)

    window_signature_path = _must_exist(capdir, "window_signature.json")
    results_path = _must_exist(capdir, "results.json")
    summary_path = _must_exist(capdir, "results_summary.json")
    manifest_path = _must_exist(capdir, "run_manifest.json")
    resolved_path = _must_exist(capdir, "run_config_resolved.json")

    assert window_signature_path.exists()
    assert results_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()
    assert resolved_path.exists()

    ws_hash = window_signature_sha256(read_json_obj(window_signature_path))
    results = read_json_obj(results_path)
    summary = read_json_obj(summary_path)
    resolved = read_json_obj(resolved_path)

    assert resolved["schema_version"] == 1
    assert resolved["ts_utc"]
    assert resolved["run"]["kind"] == "hf"
    assert resolved["env_capture"]["redactions_applied"] in (True, False)

    assert results["window_signature_ref"]["hash"] == ws_hash
    assert summary["window_signature_ref"]["hash"] == ws_hash
    assert results["run_id"] == summary["run_id"]

    gate_decisions = [gate["decision"] for gate in results["gates"]]
    assert gate_decisions
    assert results["metrics"]

    counts = summary["counts"]
    expected = {"pass": 0, "warn": 0, "fail": 0, "skip": 0}
    for decision in gate_decisions:
        expected[decision] += 1
    expected["evaluated"] = expected["pass"] + expected["warn"] + expected["fail"]

    assert counts["pass"] == expected["pass"]
    assert counts["warn"] == expected["warn"]
    assert counts["fail"] == expected["fail"]
    assert counts["skip"] == expected["skip"]
    assert counts["evaluated"] == expected["evaluated"]

    counts_model = CountsV1.model_validate(counts)
    assert summary["final_decision"] == derive_final_decision(counts_model)


@pytest.mark.integration
def test_hf_micro_smoke_direct_runner_governance(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        pytest.skip("HF offline mode enabled; skipping HF micro smoke.")
    _skip_if_hf_unreachable()

    outdir = tmp_path / "hf_micro_direct"
    env = _hf_smoke_env(tmp_path)
    env["VERISCOPE_OUT_BASE"] = str(outdir)

    cmd = [
        sys.executable,
        "-m",
        "veriscope.runners.hf.train_hf",
        "--outdir",
        str(outdir),
        "--run_id",
        "hf-micro-direct",
        "--model",
        "sshleifer/tiny-gpt2",
        "--dataset",
        f"file:{REPO_ROOT}/tests/data/hf_micro_smoke.txt",
        "--dataset_split",
        "train",
        "--max_steps",
        "8",
        "--batch_size",
        "1",
        "--block_size",
        "32",
        "--cadence",
        "1",
        "--gate_window",
        "2",
        "--gate_min_evidence",
        "2",
        "--rp_dim",
        "8",
        "--seed",
        "1337",
        "--device",
        "cpu",
        "--force",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    _maybe_skip_hf_download_failure(result)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    capdir = _resolve_capsule_dir(outdir)
    gov_path = _must_exist(capdir, "governance_log.jsonl")
    lines = gov_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line).get("event") or json.loads(line).get("event_type") for line in lines]
    assert "run_started_v1" in events
    assert "gate_decision_v1" in events
