# tests/test_seed_intervention.py
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from veriscope.runners.hf.train_hf import HFRunConfig, _build_run_manifest, _seed_for_rank


# Helper: check if an environment variable is set "truthy"
def _env_truthy(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    s = str(val).strip().lower()
    return s not in {"", "0", "false", "no", "off"}


RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fake_hf_runner.py"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_nanogpt(tmp_path: Path, dataset: str = "tiny") -> Path:
    nanogpt_dir = tmp_path / "nanoGPT"
    data_dir = nanogpt_dir / "data" / dataset
    data_dir.mkdir(parents=True, exist_ok=True)

    tokens = (np.arange(0, 512, dtype=np.uint16) % 128).astype(np.uint16)
    (data_dir / "train.bin").write_bytes(tokens.tobytes())
    (data_dir / "val.bin").write_bytes(tokens.tobytes())
    with (data_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"vocab_size": 128}, f)

    model_py = textwrap.dedent(
        """
        import math
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class GPTConfig:
            def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias):
                self.block_size = int(block_size)
                self.vocab_size = int(vocab_size)
                self.n_layer = int(n_layer)
                self.n_head = int(n_head)
                self.n_embd = int(n_embd)
                self.dropout = float(dropout)
                self.bias = bool(bias)

        class CausalSelfAttention(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                if config.n_embd % config.n_head != 0:
                    raise ValueError("n_embd must be divisible by n_head")
                # MultiheadAttention gives us a real attention module + submodules for hooks/inspection.
                self.mha = nn.MultiheadAttention(
                    embed_dim=config.n_embd,
                    num_heads=config.n_head,
                    dropout=config.dropout,
                    bias=config.bias,
                    batch_first=True,
                )
                self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
                self.drop = nn.Dropout(config.dropout)

            def forward(self, x):
                # x: (B, T, C)
                B, T, C = x.shape
                # bool causal mask: True means "masked"
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                y, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
                y = self.proj(y)
                y = self.drop(y)
                return y

        class MLP(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                hidden = 4 * config.n_embd
                self.fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
                self.proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
                self.drop = nn.Dropout(config.dropout)

            def forward(self, x):
                x = self.fc(x)
                x = F.gelu(x)
                x = self.proj(x)
                x = self.drop(x)
                return x

        class Block(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                self.ln1 = nn.LayerNorm(config.n_embd)
                self.attn = CausalSelfAttention(config)
                self.ln2 = nn.LayerNorm(config.n_embd)
                self.mlp = MLP(config)

            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                x = x + self.mlp(self.ln2(x))
                return x

        class GPT(nn.Module):
            def __init__(self, config: GPTConfig):
                super().__init__()
                self.config = config

                # Match nanoGPT-ish structure: model.transformer.{wte,wpe,drop,h,ln_f}
                self.transformer = nn.ModuleDict(
                    dict(
                        wte=nn.Embedding(config.vocab_size, config.n_embd),
                        wpe=nn.Embedding(config.block_size, config.n_embd),
                        drop=nn.Dropout(config.dropout),
                        h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                        ln_f=nn.LayerNorm(config.n_embd),
                    )
                )
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            def forward(self, idx, targets=None):
                # idx: (B, T)
                B, T = idx.shape
                if T > self.config.block_size:
                    raise ValueError(f"Sequence length {T} > block_size {self.config.block_size}")

                pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)  # (1, T)
                tok = self.transformer["wte"](idx)                                           # (B, T, C)
                pe = self.transformer["wpe"](pos)                                           # (1, T, C)
                x = self.transformer["drop"](tok + pe)

                for block in self.transformer["h"]:
                    x = block(x)

                x = self.transformer["ln_f"](x)
                logits = self.lm_head(x)                                                    # (B, T, vocab)

                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                return logits, loss
        """
    )
    (nanogpt_dir / "model.py").write_text(model_py, encoding="utf-8")
    return nanogpt_dir


def _run_gpt(
    tmp_path: Path,
    nanogpt_dir: Path,
    dataset: str,
    seed: int,
    max_iters: int = 10,
    run_tag: Optional[str] = None,
) -> Path:
    dir_suffix = f"{seed}_{run_tag}" if run_tag else str(seed)
    out_dir = tmp_path / f"gpt_seed_{dir_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "veriscope.runners.gpt.train_nanogpt",
        "--dataset",
        dataset,
        "--nanogpt_dir",
        str(nanogpt_dir),
        "--device",
        "cpu",
        "--max_iters",
        str(max_iters),
        "--metric_interval",
        "1",
        "--out_dir",
        str(out_dir),
        "--out_json",
        "run.json",
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), pythonpath]) if pythonpath else str(REPO_ROOT)
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        raise RuntimeError(
            "nanoGPT runner failed.\n"
            f"returncode: {exc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        ) from exc
    return out_dir


def _quantize_losses(losses: List[float], ndigits: int = 8) -> List[float]:
    return [round(float(loss), ndigits) for loss in losses]


def _first_divergence(
    losses_a: List[float],
    losses_b: List[float],
    tol: float = 0.0,
) -> Optional[Tuple[int, float, float]]:
    for idx, (left, right) in enumerate(zip(losses_a, losses_b)):
        if abs(left - right) > tol:
            return idx, left, right
    return None


def _filtered_run_config(run_config_path: Path) -> dict[str, object]:
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    selected: dict[str, object] = {}
    keep_keys = {
        "resolved_seed",
        "gate_preset_effective",
        "resolved_gate_cfg",
        "metric_pipeline",
        "provenance",
    }
    for key in keep_keys:
        if key in run_config:
            selected[key] = run_config[key]

    window_signature_ref = run_config.get("window_signature_ref")
    if isinstance(window_signature_ref, dict):
        window_hash = window_signature_ref.get("hash")
        if window_hash is not None:
            selected["window_signature_hash"] = window_hash

    provenance = selected.get("provenance")
    if isinstance(provenance, dict):
        filtered_provenance = {}
        for key in ("policy_rev", "resolved_seed"):
            if key in provenance:
                filtered_provenance[key] = provenance[key]
        selected["provenance"] = filtered_provenance

    return selected


def _run_hf_wrapper(outdir: Path, runner_args: List[str], fake_deps_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-c",
        "from veriscope.cli.main import main; import sys; sys.exit(main(sys.argv[1:]))",
        "run",
        "hf",
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


def test_gpt_seed_resolved_and_recorded(tmp_path: Path) -> None:
    dataset = "tiny"
    nanogpt_dir = _write_minimal_nanogpt(tmp_path, dataset=dataset)
    out_dir = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=123, max_iters=10)

    run_cfg = json.loads((out_dir / "run_config_resolved.json").read_text(encoding="utf-8"))
    assert run_cfg.get("resolved_seed") == 123
    assert run_cfg.get("provenance", {}).get("resolved_seed") == 123

    run_json = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json.get("resolved_seed") == 123

    window_signature = json.loads((out_dir / "window_signature.json").read_text(encoding="utf-8"))
    metrics_sig = window_signature.get("metrics")
    assert isinstance(metrics_sig, dict)
    assert isinstance(metrics_sig.get("names"), list) and metrics_sig["names"]
    assert metrics_sig["names"] == sorted(metrics_sig["names"])
    assert isinstance(metrics_sig.get("weights"), dict)
    assert sorted(metrics_sig["weights"].keys()) == metrics_sig["names"]

    window_decl = run_json.get("window_decl", {})
    decl_metrics = sorted(str(m) for m in (window_decl.get("metrics") or []))
    decl_weights = window_decl.get("weights", {})
    assert metrics_sig["names"] == decl_metrics
    for name in metrics_sig["names"]:
        expected = float(decl_weights.get(name, 1.0))
        assert float(metrics_sig["weights"][name]) == pytest.approx(expected, abs=1e-12)

    dw_aggregator = window_signature.get("dw_aggregator")
    assert isinstance(dw_aggregator, dict)
    assert dw_aggregator.get("name") == "weighted_sum_product_tv"


def test_gpt_seed_changes_metrics(tmp_path: Path) -> None:
    if not _env_truthy("VERISCOPE_SLOW"):
        pytest.skip("slow: set VERISCOPE_SLOW=1 to run")
    dataset = "tiny"
    nanogpt_dir = _write_minimal_nanogpt(tmp_path, dataset=dataset)

    out_dir_a = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=111, max_iters=50, run_tag="a")
    out_dir_b = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=222, max_iters=50, run_tag="b")
    out_dir_c = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=111, max_iters=50, run_tag="c")

    metrics_a = json.loads((out_dir_a / "run.json").read_text(encoding="utf-8"))["metrics"]
    metrics_b = json.loads((out_dir_b / "run.json").read_text(encoding="utf-8"))["metrics"]
    metrics_c = json.loads((out_dir_c / "run.json").read_text(encoding="utf-8"))["metrics"]

    losses_a = [m["loss"] for m in metrics_a]
    losses_b = [m["loss"] for m in metrics_b]
    losses_c = [m["loss"] for m in metrics_c]

    assert len(losses_a) == len(losses_b) == len(losses_c)
    quantized_a = _quantize_losses(losses_a)
    quantized_c = _quantize_losses(losses_c)

    divergence_ac = _first_divergence(losses_a, losses_c, tol=1e-8)
    if divergence_ac is not None:
        idx, left, right = divergence_ac
        start = max(idx - 2, 0)
        end = idx + 3
        pytest.fail(
            "Expected identical loss trajectories; first divergence at index "
            f"{idx}: {left} vs {right}. "
            f"Context A[{start}:{end}]={quantized_a[start:end]} "
            f"C[{start}:{end}]={quantized_c[start:end]}"
        )

    divergence_ab = _first_divergence(losses_a, losses_b, tol=1e-8)
    if divergence_ab is None:
        pytest.fail("Expected different loss trajectories but found none.")

    filtered_a = _filtered_run_config(out_dir_a / "run_config_resolved.json")
    filtered_c = _filtered_run_config(out_dir_c / "run_config_resolved.json")
    if filtered_a != filtered_c:
        keys = sorted(set(filtered_a) | set(filtered_c))
        diffs = [
            f"{key}: {filtered_a.get(key)!r} != {filtered_c.get(key)!r}"
            for key in keys
            if filtered_a.get(key) != filtered_c.get(key)
        ]
        pytest.fail("Expected semantic run config matches. Differences:\n" + "\n".join(diffs))


def test_hf_seed_resolved_in_wrapper(tmp_path: Path) -> None:
    outdir = tmp_path / "hf_seed_wrapper"
    fake_deps_dir = tmp_path / "fake_deps_resolved"
    fake_deps_dir.mkdir()
    (fake_deps_dir / "datasets.py").write_text("__all__ = []\n", encoding="utf-8")
    (fake_deps_dir / "transformers.py").write_text("__all__ = []\n", encoding="utf-8")
    if not RUNNER_SCRIPT.exists():
        pytest.skip(f"fake_hf_runner.py missing at {RUNNER_SCRIPT}")

    result = _run_hf_wrapper(outdir, ["--max_steps", "1", "--seed", "987"], fake_deps_dir)
    assert result.returncode == 0, result.stderr

    run_cfg = json.loads((outdir / "run_config_resolved.json").read_text(encoding="utf-8"))
    assert run_cfg.get("resolved_seed") == 987


def test_hf_seed_withheld_when_overridden_without_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outdir = tmp_path / "hf_seed_wrapper_default"
    fake_deps_dir = tmp_path / "fake_deps_default"
    fake_deps_dir.mkdir()
    (fake_deps_dir / "datasets.py").write_text("__all__ = []\n", encoding="utf-8")
    (fake_deps_dir / "transformers.py").write_text("__all__ = []\n", encoding="utf-8")
    if not RUNNER_SCRIPT.exists():
        pytest.skip(f"fake_hf_runner.py missing at {RUNNER_SCRIPT}")

    monkeypatch.delenv("VERISCOPE_SEED", raising=False)

    result = _run_hf_wrapper(outdir, ["--max_steps", "1"], fake_deps_dir)
    assert result.returncode == 0, result.stderr

    run_cfg = json.loads((outdir / "run_config_resolved.json").read_text(encoding="utf-8"))
    assert "resolved_seed" not in run_cfg


def test_hf_seed_manifest_includes_rank_seed(tmp_path: Path) -> None:
    cfg = HFRunConfig(
        model="gpt2",
        dataset_name="wikitext",
        dataset_config=None,
        dataset_path=None,
        dataset_split="train",
        dataset_text_column="text",
        outdir=tmp_path,
        run_id="run123",
        force=False,
        max_steps=1,
        batch_size=1,
        lr=1e-3,
        seed=101,
        cadence=1,
        block_size=8,
        device="cpu",
        grad_clip=1.0,
        gate_preset="tuned_v0",
        gate_window=4,
        gate_epsilon=0.1,
        gate_min_evidence=2,
        gate_gain_thresh=0.0,
        gate_policy="persistence",
        gate_persistence_k=2,
        rp_dim=8,
        lr_spike_at=-1,
        lr_spike_len=0,
        lr_spike_mult=1.0,
        lr_spike_verify=False,
        data_corrupt_at=-1,
        data_corrupt_len=0,
        data_corrupt_frac=0.0,
        data_corrupt_mode="permute",
        data_corrupt_target="clean",
        data_corrupt_mask_id=None,
    )
    seed_rank = _seed_for_rank(cfg.seed, rank=2)
    manifest = _build_run_manifest(
        cfg,
        argv=["train_hf.py"],
        started_ts_utc=datetime.now(timezone.utc),
        ended_ts_utc=None,
        run_status="success",
        runner_exit_code=0,
        runner_signal=None,
        failure_reason=None,
        failure_traceback=None,
        seed_rank=seed_rank,
        seed_rank_scheme="base_seed+1000*rank",
        rank=2,
        world_size=4,
        rank_used_for_corrupt_seed=2,
        world_size_used_for_corrupt_seed=4,
    )
    seeds = manifest.get("seeds", {})
    assert seeds.get("base_seed") == 101
    assert seeds.get("seed_rank") == seed_rank
    assert seeds.get("rank_seed_scheme") == "base_seed+1000*rank"
