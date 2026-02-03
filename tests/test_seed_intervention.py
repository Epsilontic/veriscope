from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from veriscope.runners.hf.train_hf import HFRunConfig, _build_run_manifest, _seed_for_rank


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


def _run_gpt(tmp_path: Path, nanogpt_dir: Path, dataset: str, seed: int) -> Path:
    out_dir = tmp_path / f"gpt_seed_{seed}"
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
        "10",
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
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    return out_dir


def _run_hf_wrapper(outdir: Path, runner_args: list[str], fake_deps_dir: Path) -> subprocess.CompletedProcess:
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
    out_dir = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=123)

    run_cfg = json.loads((out_dir / "run_config_resolved.json").read_text(encoding="utf-8"))
    assert run_cfg.get("resolved_seed") == 123
    assert run_cfg.get("provenance", {}).get("resolved_seed") == 123

    run_json = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json.get("resolved_seed") == 123


def test_gpt_seed_changes_metrics(tmp_path: Path) -> None:
    dataset = "tiny"
    nanogpt_dir = _write_minimal_nanogpt(tmp_path, dataset=dataset)

    out_dir_a = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=111)
    out_dir_b = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=222)
    out_dir_c = _run_gpt(tmp_path, nanogpt_dir, dataset, seed=111)

    metrics_a = json.loads((out_dir_a / "run.json").read_text(encoding="utf-8"))["metrics"]
    metrics_b = json.loads((out_dir_b / "run.json").read_text(encoding="utf-8"))["metrics"]
    metrics_c = json.loads((out_dir_c / "run.json").read_text(encoding="utf-8"))["metrics"]

    losses_a = [m["loss"] for m in metrics_a]
    losses_b = [m["loss"] for m in metrics_b]
    losses_c = [m["loss"] for m in metrics_c]

    assert len(losses_a) == len(losses_b) == len(losses_c)
    assert any(abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)) for a, b in zip(losses_a, losses_b))
    assert losses_a == pytest.approx(losses_c, rel=1e-6, abs=1e-6)


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
