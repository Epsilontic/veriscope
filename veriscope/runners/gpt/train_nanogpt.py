# veriscope/runners/gpt/train_nanogpt.py
"""  
nanoGPT training with veriscope FR gating.  
  
Minimal modifications to the standard nanoGPT train.py.  
"""
from __future__ import annotations
  
import os
import sys
from pathlib import Path
import time
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
  
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
  

# Helper to ensure nanoGPT is importable
def _ensure_nanogpt_on_path(nanogpt_dir: str) -> None:
    """Ensure the nanoGPT checkout (containing model.py) is importable."""
    p = str(Path(nanogpt_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
  
# Your veriscope imports
from veriscope.runners.gpt.adapter import (
    GPTMetricConfig,
    GPTFeatureExtractor,
    GPTMetricComputer,
    create_gpt_window_decl,
    create_gpt_gate_engine,
)
from veriscope.core.calibration import aggregate_epsilon_stat
  
  
@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    dataset: str = "openwebtext"
    batch_size: int = 12
    block_size: int = 1024
      
    # Model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
      
    # Training
    max_iters: int = 600000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
      
    # LR schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # Pathology injection (optional; for gate validation)
    lr_spike_at: int = -1          # iteration to start spike; <0 disables
    lr_spike_len: int = 0          # number of iterations to spike
    lr_spike_mult: float = 1.0     # multiplier during spike
      
    # Logging
    eval_interval: int = 1000
    log_interval: int = 10
    eval_iters: int = 200
      
    # Veriscope gating
    gate_enabled: bool = True
    gate_window: int = 50  # iterations, not epochs
    gate_warmup: int = 1000  # don't gate until model is warmed up
    gate_epsilon: float = 0.12
    gate_gain_thresh: float = 0.0  # stability-only by default; tune upward if you want "learning+stability"
    gate_min_evidence: int = 16
      
    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
  
  
class VeriscopeGatedTrainer:
    """  
    GPT trainer with veriscope finite-window gating.  
    """
      
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.ctx = self._get_autocast_context()

        # Initialize model UNCOMPILED first (hooks + torch.compile can be flaky otherwise)
        self.model = self._init_model()

        # Veriscope components (set up hooks BEFORE optional compilation)
        self.metric_config = GPTMetricConfig(
            probe_layers="last",
            max_tokens_per_batch=1024,
            geom_rp_dim=64,
        )
        self.extractor = GPTFeatureExtractor(self.model, self.metric_config, self.device)

        # Optionally compile AFTER hooks are registered.
        if config.compile:
            print("Compiling model...")
            self.model = torch.compile(self.model)
            # Ensure extractor forwards through the compiled wrapper
            self.extractor.model = self.model

        self.metric_computer = GPTMetricComputer(
            self.extractor, self.metric_config, self.device
        )

        # Optimizer/scaler AFTER potential compilation
        self.optimizer = self._init_optimizer()

        # Prefer torch.amp.GradScaler API; enable only for float16 on CUDA
        scaler_enabled = (config.dtype == "float16") and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        # Create window declaration and gate engine
        self.window_decl = create_gpt_window_decl(
            epsilon=config.gate_epsilon,
            bins=16,
        )
        self.fr_win, self.gate_engine = create_gpt_gate_engine(
            self.window_decl,
            {
                "gate_gain_thresh": config.gate_gain_thresh,
                "gate_min_evidence": config.gate_min_evidence,
                "gate_eps_stat_max_frac": 0.25,
                "gate_epsilon_sens": 0.04,
            },
        )

        # Metric history for gating
        self.metric_history: List[Dict[str, Any]] = []
        self.loss_history: List[float] = []
        self.gate_history: List[Dict[str, Any]] = []

        # EWMA baseline for prequential gain
        self.ewma_loss: Optional[float] = None
        self.ewma_alpha = 0.1

        # Previous JL-projected features for SW2 (avoid re-projecting each metric step)
        self._prev_H_jl: Optional[torch.Tensor] = None

        # Training state
        self.iter_num = 0
        self.best_val_loss = float("inf")
      
    def _get_autocast_context(self):
        """Get appropriate autocast context."""
        cfg = self.config
        if cfg.dtype == "bfloat16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif cfg.dtype == "float16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        else:
            return nullcontext()
      
    def _init_model(self, vocab_size: Optional[int] = None) -> nn.Module:
        """Initialize GPT model (uncompiled; compilation happens in __init__ after hooks)."""
        cfg = self.config

        # Ensure nanoGPT (model.py) is importable regardless of cwd.
        # The CLI sets NANOGPT_DIR from --nanogpt_dir.
        nanogpt_dir = os.environ.get("NANOGPT_DIR") or "nanoGPT"
        _ensure_nanogpt_on_path(nanogpt_dir)

        # Import nanoGPT model components after sys.path is prepared.
        from model import GPTConfig, GPT

        # Vocab size should come from data preparation, not hardcoded.
        # nanoGPT convention is: <nanogpt_root>/data/<dataset>/meta.pkl
        if vocab_size is None:
            import pickle

            meta_path: Optional[Path] = None

            # 1) If user provides NANOGPT_DIR, prefer it.
            nanogpt_dir_env = os.environ.get("NANOGPT_DIR")
            if nanogpt_dir_env:
                cand = Path(nanogpt_dir_env) / "data" / cfg.dataset / "meta.pkl"
                if cand.exists():
                    meta_path = cand

            # 2) If running from nanoGPT root, this works.
            if meta_path is None:
                cand = Path("data") / cfg.dataset / "meta.pkl"
                if cand.exists():
                    meta_path = cand

            # 3) If running from the veriscope repo root with a ./nanoGPT checkout.
            if meta_path is None:
                cand = Path("nanoGPT") / "data" / cfg.dataset / "meta.pkl"
                if cand.exists():
                    meta_path = cand

            # 4) Search upwards from this file for a nanoGPT checkout.
            if meta_path is None:
                here = Path(__file__).resolve()
                for parent in here.parents:
                    cand = parent / "nanoGPT" / "data" / cfg.dataset / "meta.pkl"
                    if cand.exists():
                        meta_path = cand
                        break

            if meta_path is not None:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                vocab_size = int(meta["vocab_size"])
                print(f"Found vocab_size={vocab_size} from {meta_path}")
            else:
                # Fallback: GPT-2 default (but warn). This is not correct for shakespeare_char.
                vocab_size = 50304
                print(
                    f"WARNING: meta.pkl not found for dataset='{cfg.dataset}'. "
                    f"Defaulting to vocab_size={vocab_size}. "
                    f"Set NANOGPT_DIR or run from nanoGPT root to pick up data/<dataset>/meta.pkl."
                )

        model_config = GPTConfig(
            block_size=cfg.block_size,
            vocab_size=int(vocab_size),
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
            bias=False,
        )
        model = GPT(model_config)
        model = model.to(self.device)
        # NOTE: Do NOT compile here - hooks must be registered first.
        # Compilation (if enabled) happens in __init__ after GPTFeatureExtractor setup.
        return model
      
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize AdamW optimizer with weight decay."""
        cfg = self.config
          
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
          
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
          
        return torch.optim.AdamW(
            optim_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            fused=True,  # H100 optimization
        )
      
    def _get_lr(self, it: int) -> float:
        """Learning rate schedule with warmup and cosine decay."""
        cfg = self.config

        # Guard against zero warmup; avoid lr=0 at it=0 by using (it+1)
        if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
            return cfg.learning_rate * float(it + 1) / float(cfg.warmup_iters)

        # Clamp at/after decay end
        if it >= cfg.lr_decay_iters:
            return cfg.min_lr

        # Guard against bad ranges
        decay_range = int(cfg.lr_decay_iters) - int(cfg.warmup_iters)
        if decay_range <= 0:
            return cfg.min_lr

        decay_ratio = (float(it) - float(cfg.warmup_iters)) / float(decay_range)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)
    def _save_checkpoint(self, path: str):
        """Save model checkpoint."""
        # Handle compiled model
        model_to_save = self.model
        if hasattr(self.model, "_orig_mod"):
            model_to_save = self.model._orig_mod

        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iter_num": self.iter_num,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")
      
    def _compute_gate_check(self) -> Dict[str, Any]:
        """  
        Perform finite-window gate check using recent metric history.  
          
        Returns audit dict with gate decision and diagnostics.  
        """
        W = self.config.gate_window
          
        if len(self.metric_history) < 2 * W:
            return {"ok": True, "reason": "insufficient_history"}
          
        # Get past and recent windows
        recent = self.metric_history[-(2 * W):]
        past_slice = recent[:W]
        recent_slice = recent[W:]
          
        # Build metric arrays
        metrics = list(self.window_decl.weights.keys())
          
        def _extract(slice_data: List[Dict], key: str) -> np.ndarray:
            vals = [float(d.get(key, np.nan)) for d in slice_data]
            arr = np.array(vals, dtype=float)
            return arr[np.isfinite(arr)]
          
        past_dict = {m: _extract(past_slice, m) for m in metrics}
        recent_dict = {m: _extract(recent_slice, m) for m in metrics}
          
        # Count evidence
        counts = {}
        for m in metrics:
            counts[m] = min(len(past_dict[m]), len(recent_dict[m]))
          
        # Compute prequential gain (bits/sample)
        recent_losses = [d.get("loss", np.nan) for d in recent_slice]
        recent_baselines = [d.get("ewma_loss", np.nan) for d in recent_slice]
          
        gain_vals = []
        for l, b in zip(recent_losses, recent_baselines):
            if np.isfinite(l) and np.isfinite(b):
                gain_vals.append((b - l) / math.log(2))  # bits
          
        gain_bits = float(np.mean(gain_vals)) if gain_vals else float("nan")
          
        # Use gate engine
        result = self.gate_engine.check(
            past=past_dict,
            recent=recent_dict,
            counts_by_metric=counts,
            gain_bits=gain_bits,
            kappa_sens=0.0,  # Skip κ_sens for now
            eps_stat_value=aggregate_epsilon_stat(
                self.window_decl, counts, alpha=0.05
            ),
        )
          
        return {
            "ok": result.ok,
            "audit": result.audit,
            "gain_bits": gain_bits,
            "iter": self.iter_num,
        }
      
    def _log_metrics(self, loss: float, input_ids: torch.Tensor):
        """Compute and log metrics for this iteration."""
        # Update EWMA baseline
        if self.ewma_loss is None:
            self.ewma_loss = loss
        else:
            self.ewma_loss = self.ewma_alpha * loss + (1 - self.ewma_alpha) * self.ewma_loss

        # Compute veriscope metrics (every N iterations to save compute)
        if self.iter_num % 5 == 0:
            metrics = self.metric_computer.compute_all(
                input_ids,
                run_key=42,
                epoch=self.iter_num,
                prev_H_jl=self._prev_H_jl,
            )

            # Store projected features for next SW2
            self._prev_H_jl = metrics.pop("_H_jl", None)
            # Drop raw features unless you explicitly want to persist them
            metrics.pop("_H_norm", None)

            metrics["loss"] = loss
            metrics["ewma_loss"] = self.ewma_loss
            metrics["iter"] = self.iter_num
            metrics["lr"] = self._get_lr(self.iter_num)

            self.metric_history.append(metrics)
            self.loss_history.append(loss)
      
    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Execute one training step."""
        cfg = self.config
        
        # Update learning rate
        lr = self._get_lr(self.iter_num)

        # Optional LR spike (gate validation / pathology injection)
        if cfg.lr_spike_at >= 0 and cfg.lr_spike_len > 0 and cfg.lr_spike_mult != 1.0:
            if cfg.lr_spike_at <= self.iter_num < (cfg.lr_spike_at + cfg.lr_spike_len):
                lr = lr * float(cfg.lr_spike_mult)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
          
        # Forward pass
        with self.ctx:
            logits, loss = self.model(X, Y)
          
        # Backward pass
        self.scaler.scale(loss).backward()
          
        # Gradient clipping
        if cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
          
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
          
        loss_val = loss.item()
          
        # Log metrics
        self._log_metrics(loss_val, X)
          
        # Gate check (after warmup)
        if cfg.gate_enabled and self.iter_num >= cfg.gate_warmup:
            if self.iter_num % cfg.gate_window == 0:
                gate_result = self._compute_gate_check()
                self.gate_history.append(gate_result)
                  
                if not gate_result["ok"]:
                    print(f"\n[GATE] iter={self.iter_num} FAIL: {gate_result['audit']}")
                elif self.iter_num % (cfg.gate_window * 10) == 0:
                    print(f"[GATE] iter={self.iter_num} OK, gain={gate_result['gain_bits']:.4f} bits")
          
        self.iter_num += 1
        return loss_val
      
    def train(self, get_batch_fn):
        """Main training loop."""
        cfg = self.config
          
        t0 = time.time()
        running_loss = 0.0
          
        while self.iter_num < cfg.max_iters:
            # Get batch
            X, Y = get_batch_fn("train")
            X, Y = X.to(self.device), Y.to(self.device)
              
            # Train step
            loss = self.train_step(X, Y)
            running_loss += loss
              
            # Logging
            if self.iter_num % cfg.log_interval == 0:
                dt = time.time() - t0
                avg_loss = running_loss / cfg.log_interval
                print(f"iter {self.iter_num}: loss {avg_loss:.4f}, time {dt*1000:.2f}ms")
                running_loss = 0.0
                t0 = time.time()
              
            # Evaluation
            if self.iter_num % cfg.eval_interval == 0:
                self._evaluate(get_batch_fn)
          
        print("Training complete!")
        return self.metric_history, self.gate_history
      
    @torch.no_grad()
    def _evaluate(self, get_batch_fn):
        """Evaluate on validation set."""
        self.model.eval()
        losses = []

        for _ in range(self.config.eval_iters):
            X, Y = get_batch_fn("val")
            X, Y = X.to(self.device), Y.to(self.device)

            with self.ctx:
                _, loss = self.model(X, Y)
            losses.append(loss.item())

        val_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"val loss: {val_loss:.4f}")

        if np.isfinite(val_loss) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint("best_ckpt.pt")

        self.model.train()
  
  

# -----------------
# Data loader helper
# -----------------

def get_batch_factory(data_dir: str, block_size: int, batch_size: int, device: str):
    """Create get_batch function compatible with nanoGPT .bin data format."""
    import numpy as _np
    from pathlib import Path

    dd = Path(data_dir)
    train_data = _np.memmap(dd / "train.bin", dtype=_np.uint16, mode="r")
    val_data = _np.memmap(dd / "val.bin", dtype=_np.uint16, mode="r")

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        # need room for x of length block_size and y shifted by 1
        max_start = int(len(data)) - int(block_size) - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset too small for block_size={block_size}: len(data)={len(data)}"
            )
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(_np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + block_size]).astype(_np.int64)
                )
                for i in ix
            ]
        )

        if "cuda" in str(device):
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

    return get_batch


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--nanogpt_dir", default="./nanoGPT")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lr_spike_at",
        type=int,
        default=-1,
        help="Iteration to start an LR spike (>=0 enables).",
    )
    parser.add_argument(
        "--lr_spike_len",
        type=int,
        default=0,
        help="Number of iterations to apply the LR spike.",
    )
    parser.add_argument(
        "--lr_spike_mult",
        type=float,
        default=1.0,
        help="LR multiplier during the spike window.",
    )
    args = parser.parse_args()

    # Set env for meta.pkl discovery
    os.environ["NANOGPT_DIR"] = args.nanogpt_dir

    config = TrainConfig(
        dataset=args.dataset,
        # Small model for testing
        n_layer=6,
        n_head=6,
        n_embd=384,
        batch_size=64,
        block_size=256,
        max_iters=5000,
        eval_interval=500,
        log_interval=10,
        device=args.device,
        lr_spike_at=args.lr_spike_at,
        lr_spike_len=args.lr_spike_len,
        lr_spike_mult=args.lr_spike_mult,
        # Gate config
        gate_enabled=True,
        gate_window=50,
        gate_warmup=500,
        gate_gain_thresh=0.0,
        # Safer default for initial hook validation
        compile=False,
    )

    data_dir = os.path.join(args.nanogpt_dir, "data", args.dataset)
    get_batch = get_batch_factory(
        data_dir=data_dir,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=config.device,
    )

    trainer = VeriscopeGatedTrainer(config)
    metrics, gates = trainer.train(get_batch)

    with open("veriscope_gpt_run.json", "w") as f:
        json.dump(
            {
                "metrics": metrics[-100:],  # last 100 for brevity
                "gates": gates,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"Saved {len(metrics)} metric snapshots, {len(gates)} gate checks")
