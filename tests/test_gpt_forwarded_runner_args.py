from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import textwrap
from array import array
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from veriscope.runners.gpt import train_nanogpt


REPO_ROOT = Path(__file__).resolve().parents[1]


def _wrapper_cmd(outdir: Path, runner_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-c",
        "from veriscope.cli.main import main; import sys; sys.exit(main(sys.argv[1:]))",
        "run",
        "gpt",
        "--outdir",
        str(outdir),
        "--force",
        "--",
        *runner_args,
    ]


def _arg_value(args: list[str], key: str) -> str | None:
    for idx, token in enumerate(args):
        if token == key and idx + 1 < len(args):
            return args[idx + 1]
        if token.startswith(key + "="):
            return token.split("=", 1)[1]
    return None


def _write_minimal_nanogpt(tmp_path: Path, dataset: str = "tiny") -> Path:
    nanogpt_dir = tmp_path / "nanoGPT"
    data_dir = nanogpt_dir / "data" / dataset
    data_dir.mkdir(parents=True, exist_ok=True)

    tokens = array("H", (i % 128 for i in range(512)))
    payload = tokens.tobytes()
    (data_dir / "train.bin").write_bytes(payload)
    (data_dir / "val.bin").write_bytes(payload)

    model_py = textwrap.dedent(
        """
        import pickle
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
                _, t, _ = x.shape
                mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
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
                _, t = idx.shape
                if t > self.config.block_size:
                    raise ValueError(f"Sequence length {t} > block_size {self.config.block_size}")

                pos = torch.arange(0, t, device=idx.device, dtype=torch.long).unsqueeze(0)
                tok = self.transformer["wte"](idx)
                pe = self.transformer["wpe"](pos)
                x = self.transformer["drop"](tok + pe)

                for block in self.transformer["h"]:
                    x = block(x)

                x = self.transformer["ln_f"](x)
                logits = self.lm_head(x)

                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                return logits, loss
        """
    )
    (nanogpt_dir / "model.py").write_text(model_py, encoding="utf-8")

    meta_path = data_dir / "meta.pkl"
    meta_path.write_bytes(b"")
    with meta_path.open("wb") as f:
        import pickle

        pickle.dump({"vocab_size": 128}, f)

    return nanogpt_dir


def _write_fast_launcher(tmp_path: Path) -> Path:
    launcher = tmp_path / "fast_train_nanogpt.py"
    launcher.write_text(
        textwrap.dedent(
            f"""
            from __future__ import annotations

            import sys
            from pathlib import Path

            repo_root = Path({str(REPO_ROOT)!r})
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            from veriscope.runners.gpt import train_nanogpt as runner

            def _patched_train(self, get_batch_fn):
                return [], []

            runner.VeriscopeGatedTrainer.train = _patched_train

            if __name__ == "__main__":
                raise SystemExit(runner.main(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    return launcher


def test_normalize_runner_cli_argv_preserves_authoritative_forwarded_order() -> None:
    raw = ["--max_iters", "5000", "--", "--max_iters", "10000", "--no_regime"]
    assert train_nanogpt._normalize_runner_cli_argv(raw) == [
        "--max_iters",
        "5000",
        "--max_iters",
        "10000",
        "--no_regime",
    ]


def test_wrapper_forwarded_args_after_separator_override_effective_runtime_config(tmp_path: Path) -> None:
    dataset = "tiny"
    nanogpt_dir = _write_minimal_nanogpt(tmp_path, dataset=dataset)
    launcher = _write_fast_launcher(tmp_path)
    outdir = tmp_path / "gpt_forwarded_after_separator"

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), pythonpath]) if pythonpath else str(REPO_ROOT)
    env["VERISCOPE_GPT_RUNNER_CMD"] = shlex.join(
        [
            sys.executable,
            str(launcher),
            "--dataset",
            dataset,
            "--nanogpt_dir",
            str(nanogpt_dir),
            "--device",
            "cpu",
            "--metric_interval",
            "1",
            "--",
        ]
    )

    result = subprocess.run(
        _wrapper_cmd(outdir, ["--max_iters", "10000", "--no_regime"]),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    run_cfg = json.loads((outdir / "run_config_resolved.json").read_text(encoding="utf-8"))
    assert run_cfg.get("max_iters") == 10000
    assert run_cfg.get("resolved_gate_cfg", {}).get("regime_enabled") is False

    argv = run_cfg.get("argv")
    assert isinstance(argv, dict)

    normalized = argv.get("normalized_forwarded_args")
    assert isinstance(normalized, list)
    assert _arg_value(normalized, "--max_iters") == "10000"
    assert "--no_regime" in normalized

    runner_raw = argv.get("runner_raw_argv")
    assert isinstance(runner_raw, list)
    assert "--" in runner_raw

    runner_effective = argv.get("runner_effective_argv")
    assert isinstance(runner_effective, list)
    assert _arg_value(runner_effective, "--max_iters") == "10000"
    assert "--no_regime" in runner_effective
    assert "--" not in runner_effective

    run_json_path = Path(str(run_cfg["out_json"]))
    run_json = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert run_json["config"]["regime"]["regime_enabled"] is False
    assert list(run_json["config"]["argv"]) == list(runner_effective)

    governance_path = outdir / "governance_log.jsonl"
    governance_entry = json.loads(governance_path.read_text(encoding="utf-8").splitlines()[0])
    payload_argv = governance_entry["payload"]["argv"]
    assert list(payload_argv) == list(runner_effective)
    assert _arg_value(payload_argv, "--max_iters") == "10000"
    assert "--no_regime" in payload_argv
