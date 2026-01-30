# veriscope/runners/hf/train_hf.py
from __future__ import annotations

import argparse
import math
import contextlib
import logging
import os
import random
import signal
import subprocess
import sys
import traceback
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import veriscope
from veriscope.core.artifacts import AuditV1, GateRecordV1, MetricRecordV1
from veriscope.core.calibration import aggregate_epsilon_stat
from veriscope.core.ddp import (
    ddp_barrier,
    ddp_can_communicate,
    ddp_destroy_process_group,
    ddp_is_active,
    ddp_is_chief,
    ddp_reduce_mean_scalars_masked,
    env_truthy,
)
from veriscope.core.gate import GateEngine
from veriscope.core.transport import DeclTransport
from veriscope.core.window import FRWindow, WindowDecl
from veriscope.core.governance import append_gate_decision, append_run_started, build_code_identity
from veriscope.core.jsonutil import atomic_write_json, read_json_obj, window_signature_sha256
from veriscope.runners.hf.adapter import HFMetricComputer, HFMetricConfig
from veriscope.runners.hf.emit_artifacts import emit_hf_artifacts_v1

logger = logging.getLogger(__name__)
_DDP_BARRIER_WARNED = False
DDP_AGG_METRICS = {
    "var_out_k": "mean",
    "eff_dim": "mean",
    "loss_delta_z": "mean",
}


@dataclass(frozen=True)
class HFRunConfig:
    model: str
    dataset_name: str
    dataset_config: Optional[str]
    dataset_path: Optional[Path]
    dataset_split: str
    dataset_text_column: str
    outdir: Path
    run_id: str
    force: bool
    max_steps: int
    batch_size: int
    lr: float
    seed: int
    cadence: int
    block_size: int
    device: str
    grad_clip: float
    gate_preset: str
    gate_window: int
    gate_epsilon: float
    gate_min_evidence: int
    gate_gain_thresh: float
    gate_policy: str
    gate_persistence_k: int
    rp_dim: int
    lr_spike_at: int
    lr_spike_len: int
    lr_spike_mult: float
    lr_spike_verify: bool
    data_corrupt_at: int
    data_corrupt_len: int
    data_corrupt_frac: float
    data_corrupt_mode: str
    data_corrupt_target: str
    data_corrupt_mask_id: Optional[int]


def _jsonable_float(x: float) -> Optional[float]:
    """Convert float to JSON-safe value: finite -> float, non-finite -> None."""
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_outdir() -> Path:
    base = Path(os.environ.get("VERISCOPE_OUT_BASE", "./out")).expanduser()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return base / f"veriscope_hf_{ts}_{os.getpid()}"


def _get_rank_and_world() -> tuple[int, int]:
    """Determine (rank, world_size) without being confused by unrelated env vars.

    Policy:
      1) If a torch.distributed process group is initialized, trust it.
      2) Else, only trust env vars if we appear to be under torchrun/elastic
         (LOCAL_RANK or TORCHELASTIC_RUN_ID present).
      3) Otherwise treat as single-process (0, 1).
    """
    # For smokes/CI, allow an explicit override to force single-process emission.
    force = (os.environ.get("VERISCOPE_FORCE_SINGLE_PROCESS") or "").strip().lower()
    if force in {"1", "true", "yes", "y", "on"}:
        return 0, 1
    # 1) Trust an initialized process group (most reliable).
    try:
        import torch.distributed as dist  # local import

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        pass

    # 2) Only trust env rank/world when torchrun/elastic hints are present.
    if os.environ.get("LOCAL_RANK") is None and os.environ.get("TORCHELASTIC_RUN_ID") is None:
        return 0, 1

    def _as_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    rank = _as_int("RANK", 0)
    world_size = _as_int("WORLD_SIZE", 1)
    if world_size < 1:
        world_size = 1
    if rank < 0:
        rank = 0
    if rank >= world_size:
        # Clamp rather than silently acting non-chief and emitting nothing.
        rank = 0
        world_size = 1
    return rank, world_size


def _is_chief() -> bool:
    return ddp_is_chief()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _maybe_init_ddp() -> bool:
    """Initialize torch.distributed process group when appropriate.

    Returns True iff we initialized the process group in this function.

    Policy:
      - Treat rank/world env vars as *inert* unless we are actually under torchrun/elastic
        (LOCAL_RANK or TORCHELASTIC_RUN_ID present) OR an explicit escape hatch is enabled.
      - In strict torchrun/elastic multi-rank context, failures are fatal (raise) to avoid rank skew.
    """

    # Force single-process semantics even if launcher env vars are present.
    # This is the strongest override and must dominate all other DDP detection.
    if env_truthy("VERISCOPE_FORCE_SINGLE_PROCESS"):
        logger.debug(
            "FORCE_SINGLE_PROCESS=1; skipping DDP init (WORLD_SIZE=%r RANK=%r LOCAL_RANK=%r TORCHELASTIC_RUN_ID=%r).",
            os.environ.get("WORLD_SIZE"),
            os.environ.get("RANK"),
            os.environ.get("LOCAL_RANK"),
            os.environ.get("TORCHELASTIC_RUN_ID"),
        )
        return False

    # ---- NEW: treat rank/world env as inert unless we're really under torchrun/elastic ----
    def _as_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except Exception:
            return default

    world_size_env = _as_int("WORLD_SIZE", 1)
    strict = (world_size_env > 1) and (
        os.environ.get("LOCAL_RANK") is not None or os.environ.get("TORCHELASTIC_RUN_ID") is not None
    )

    # Optional escape hatch for advanced/manual rendezvous setups (default off).
    allow_env_rendezvous = env_truthy("VERISCOPE_DDP_ALLOW_ENV_RANKWORLD")

    if not strict and not allow_env_rendezvous:
        # If env is polluted (WORLD_SIZE>1 etc) but not torchrun, do NOT attempt PG init.
        if world_size_env > 1:
            logger.debug(
                "DDP env vars present but not in torchrun/elastic context; treating as inert. "
                "Set VERISCOPE_DDP_ALLOW_ENV_RANKWORLD=1 to force env:// init."
            )
        return False
    # ---- end NEW gate ----

    if not ddp_is_active():
        logger.debug("Skipping DDP init: ddp_is_active() is false.")
        return False

    try:
        import torch.distributed as dist  # local import
    except Exception as exc:
        msg = f"DDP env active but torch.distributed import failed: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning("%s; skipping init.", msg)
        return False

    if not getattr(dist, "is_available", lambda: False)():
        msg = "DDP env active but torch.distributed unavailable."
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s; skipping init.", msg)
        return False

    try:
        if dist.is_initialized():
            logger.debug("Skipping DDP init: process group already initialized.")
            return False
    except Exception as exc:
        msg = f"DDP env active but could not check init status: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning("%s; skipping init.", msg)
        return False

    # Required env for env://
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    rank_raw = os.environ.get("RANK")
    world_size_raw = os.environ.get("WORLD_SIZE")

    if not master_addr or not master_port or not rank_raw or not world_size_raw:
        msg = (
            "Skipping DDP init: required env vars missing "
            f"(MASTER_ADDR={master_addr!r}, MASTER_PORT={master_port!r}, "
            f"RANK={rank_raw!r}, WORLD_SIZE={world_size_raw!r})."
        )
        if strict:
            raise RuntimeError(msg)
        logger.debug(msg)
        return False

    try:
        rank = int(rank_raw)
        world_size = int(world_size_raw)
        _ = int(master_port)
    except (TypeError, ValueError) as exc:
        msg = (
            "Skipping DDP init: required env vars not parseable "
            f"(RANK={rank_raw!r}, WORLD_SIZE={world_size_raw!r}, MASTER_PORT={master_port!r})."
        )
        if strict:
            raise RuntimeError(msg) from exc
        logger.debug(msg)
        return False

    # ---- FAIL FAST GUARDS (must run before init_process_group) ----
    if world_size < 1:
        msg = f"Invalid WORLD_SIZE={world_size} (must be >= 1)."
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s Skipping DDP init.", msg)
        return False

    if not (0 <= rank < world_size):
        msg = f"Invalid rank/world size: RANK={rank} must be in [0, {world_size - 1}] for WORLD_SIZE={world_size}."
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s Skipping DDP init.", msg)
        return False
    # ---- end fail fast guards ----

    if world_size <= 1 or rank < 0:
        msg = f"Skipping DDP init: WORLD_SIZE={world_size} <= 1 or rank invalid (rank={rank})."
        if strict:
            raise RuntimeError(msg)
        logger.debug(msg)
        return False

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    local_rank = None
    local_rank_raw = os.environ.get("LOCAL_RANK")
    local_rank_missing = local_rank_raw in (None, "")
    if not local_rank_missing:
        try:
            local_rank = int(local_rank_raw)
        except (TypeError, ValueError):
            local_rank = None

    # Torchrun sets LOCAL_WORLD_SIZE; fall back to WORLD_SIZE if missing.
    try:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "") or "0")
    except (TypeError, ValueError):
        local_world_size = 0
    if local_world_size <= 0:
        local_world_size = world_size

    if torch.cuda.is_available():
        if local_rank_missing:
            logger.debug("LOCAL_RANK missing; using gloo backend.")
        elif local_rank is None:
            logger.debug("LOCAL_RANK unparseable; using gloo backend.")

        # IMPORTANT: backend choice must be consistent across ranks.
        # Only use NCCL if *every* local rank can map to a CUDA device.
        if available_gpus > 0 and available_gpus < local_world_size:
            if _is_chief():
                logger.warning(
                    "DDP requested with LOCAL_WORLD_SIZE=%s but only %s CUDA device(s) available; "
                    "forcing gloo backend for all ranks.",
                    local_world_size,
                    available_gpus,
                )

    use_nccl = (
        torch.cuda.is_available()
        and (local_rank is not None)
        and (0 <= local_rank < available_gpus)
        and (available_gpus >= local_world_size)
    )

    backend = "nccl" if use_nccl else "gloo"

    if backend == "nccl":
        try:
            torch.cuda.set_device(local_rank)
        except Exception as exc:
            # In strict context, don't silently flip backend; fail loudly.
            msg = f"DDP init failed to set CUDA device for LOCAL_RANK={local_rank_raw!r}: {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            logger.warning("%s; falling back to gloo.", msg)
            backend = "gloo"

    try:
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=30))
    except TypeError as exc:
        # Older torch may not accept timeout=
        if not env_truthy("VERISCOPE_DDP_ALLOW_NO_TIMEOUT_INIT"):
            msg = "DDP init_process_group timeout unsupported."
            if strict:
                raise RuntimeError(msg) from exc
            logger.warning("%s; skipping init.", msg)
            return False

        logger.debug("DDP init_process_group timeout unsupported; retrying without timeout.")
        try:
            dist.init_process_group(backend=backend, init_method="env://")
            logger.debug("DDP init_process_group succeeded without timeout.")
        except Exception as exc2:
            msg = f"DDP env active but init_process_group failed (no-timeout retry): {exc2}"
            if strict:
                raise RuntimeError(msg) from exc2
            logger.warning("%s; continuing without DDP.", msg)
            return False
    except Exception as exc:
        msg = f"DDP env active but init_process_group failed: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning("%s; continuing without DDP.", msg)
        return False

    return True


def _should_cleanup_ddp(initialized_here: bool) -> bool:
    return bool(initialized_here) or env_truthy("VERISCOPE_DDP_CLEANUP")


def _build_window_decl(cfg: HFRunConfig) -> WindowDecl:
    # NOTE: Use a broad loss z-score calibration range to avoid spurious gate trips
    # on tiny HF models and short runs.
    return WindowDecl(
        epsilon=float(cfg.gate_epsilon),
        # Evidence: DDP-correct loss z-score computed from cross-rank mean loss history.
        metrics=["loss_delta_z"],
        weights={"loss_delta_z": 1.0},
        bins=16,
        interventions=(lambda x: x,),
        cal_ranges={"loss_delta_z": (-6.0, 6.0)},
    )


def _fallback_metrics_from_hidden(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    rp_dim: int,
) -> Dict[str, float]:
    """Numeric-stable fallback for CI/smokes if the adapter metric path is degenerate.

    Produces:
      - var_out_k in [0, 1] (top-k variance fraction via SVD)
      - eff_dim in [0, rp_dim] (participation ratio proxy)
    """
    try:
        hs = hidden_states.detach()
        if hs.dim() != 3:
            return {"var_out_k": float("nan"), "eff_dim": float("nan")}
        x = hs.reshape(-1, hs.shape[-1]).float()
        if attention_mask is not None:
            m = attention_mask.reshape(-1).to(dtype=torch.bool)
            x = x[m]
        if x.shape[0] < 2 or x.shape[1] < 1:
            return {"var_out_k": float("nan"), "eff_dim": float("nan")}
        # cap tokens for safety
        if x.shape[0] > 2048:
            idx = torch.randperm(x.shape[0], device=x.device)[:2048]
            x = x[idx]
        x = x - x.mean(dim=0, keepdim=True)
        with contextlib.suppress(Exception):
            s = torch.linalg.svdvals(x)
            var = s.square()
            tot = var.sum()
            if torch.isfinite(tot) and float(tot.item()) > 0.0:
                k = min(int(rp_dim), int(var.numel()))
                frac = (var[:k].sum() / (tot + 1e-12)).clamp(0.0, 1.0)
                lam = var / max(1, x.shape[0] - 1)
                tr = lam.sum()
                tr2 = lam.square().sum()
                eff = (tr.square() / (tr2 + 1e-12)).clamp(0.0, float(rp_dim))
                return {"var_out_k": float(frac.item()), "eff_dim": float(eff.item())}
    except Exception:
        pass
    return {"var_out_k": float("nan"), "eff_dim": float("nan")}


def _lr_spike_active(cfg: HFRunConfig, step: int) -> bool:
    if cfg.lr_spike_at < 0 or cfg.lr_spike_len <= 0:
        return False
    return int(cfg.lr_spike_at) <= int(step) < (int(cfg.lr_spike_at) + int(cfg.lr_spike_len))


def _effective_lr(cfg: HFRunConfig, step: int) -> float:
    lr = float(cfg.lr)
    if _lr_spike_active(cfg, step):
        lr = lr * float(cfg.lr_spike_mult)
    return lr


def _data_corrupt_active(cfg: HFRunConfig, step: int) -> bool:
    if cfg.data_corrupt_at < 0 or cfg.data_corrupt_len <= 0 or cfg.data_corrupt_frac <= 0.0:
        return False
    end = int(cfg.data_corrupt_at) + int(cfg.data_corrupt_len)
    return int(cfg.data_corrupt_at) <= int(step) < end


def _corrupt_seed(cfg: HFRunConfig, step: int, rank: int, batch_idx: int) -> int:
    base = int(cfg.seed) & 0xFFFFFFFF
    rid = zlib.crc32(cfg.run_id.encode("utf-8")) & 0xFFFFFFFF
    return (
        base ^ (rid << 1) ^ (int(step) * 0x9E3779B1) ^ (int(rank) * 0x85EBCA77) ^ (int(batch_idx) * 0xC2B2AE3D)
    ) & 0xFFFFFFFF


def _maybe_corrupt_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    cfg: HFRunConfig,
    step: int,
    vocab_size: int,
    mask_token_id: int,
    attention_mask: Optional[torch.Tensor],
    rank: int,
    batch_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Apply token corruption for pathology injection (HF runner).

    Modes:
      - permute: permute a fraction of positions within each sequence
      - random: replace a fraction of positions with random token IDs
      - mask: replace a fraction of positions with mask_token_id
    """
    if not _data_corrupt_active(cfg, step):
        return input_ids, labels, {"corrupt_frac_effective": 0.0}
    if input_ids.ndim != 2:
        return input_ids, labels, {"corrupt_frac_effective": 0.0}

    bsz, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    n_corrupt = int(round(float(seq_len) * float(cfg.data_corrupt_frac)))
    n_corrupt = max(0, min(seq_len, n_corrupt))
    if n_corrupt == 0:
        return input_ids, labels, {"corrupt_frac_effective": 0.0}

    x_corrupt = input_ids.clone()
    gen = torch.Generator(device=input_ids.device)
    gen.manual_seed(_corrupt_seed(cfg, step, rank, batch_idx))

    mode = str(cfg.data_corrupt_mode).lower().strip()
    if mode not in ("permute", "random", "mask"):
        raise ValueError(f"Unknown data_corrupt_mode={cfg.data_corrupt_mode!r}")

    vocab_size = max(1, int(vocab_size))
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    active_mask = attention_mask.to(dtype=torch.bool)
    active_mask = active_mask & (labels != -100)

    total_picked = 0
    total_active = 0
    for b in range(bsz):
        valid_pos = torch.nonzero(active_mask[b], as_tuple=False).flatten()
        if valid_pos.numel() == 0:
            continue
        total_active += int(valid_pos.numel())
        n_pick = min(int(n_corrupt), int(valid_pos.numel()))
        if n_pick <= 0:
            continue
        total_picked += int(n_pick)
        pos = valid_pos[torch.randperm(valid_pos.numel(), generator=gen, device=input_ids.device)[:n_pick]]
        if mode == "permute":
            shuf = torch.randperm(n_pick, generator=gen, device=input_ids.device)
            x_corrupt[b, pos] = input_ids[b, pos[shuf]]
        elif mode == "random":
            rnd = torch.randint(0, vocab_size, (n_pick,), generator=gen, device=input_ids.device)
            x_corrupt[b, pos] = rnd
        else:
            x_corrupt[b, pos] = int(mask_token_id)

    target_mode = str(cfg.data_corrupt_target).lower().strip()
    if target_mode == "same":
        new_labels = labels.clone()
        keep = labels != -100
        new_labels[keep] = x_corrupt[keep]
        frac_effective = float(total_picked) / float(total_active) if total_active > 0 else 0.0
        return x_corrupt, new_labels.detach(), {"corrupt_frac_effective": frac_effective}
    frac_effective = float(total_picked) / float(total_active) if total_active > 0 else 0.0
    return x_corrupt, labels, {"corrupt_frac_effective": frac_effective}


def _build_gate_engine(cfg: HFRunConfig, window_decl: WindowDecl) -> GateEngine:
    transport = DeclTransport(window_decl)
    window_decl.attach_transport(transport)
    fr_win = FRWindow(decl=window_decl, transport=transport, tests=())
    return GateEngine(
        frwin=fr_win,
        gain_thresh=float(cfg.gate_gain_thresh),
        eps_stat_alpha=0.05,
        eps_stat_max_frac=0.25,
        eps_sens=0.04,
        min_evidence=int(cfg.gate_min_evidence),
        policy=str(cfg.gate_policy),
        persistence_k=int(cfg.gate_persistence_k),
        min_metrics_exceeding=1,
    )


def _build_window_signature(cfg: HFRunConfig, *, created_ts_utc: datetime) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_ts_utc": _iso_utc(created_ts_utc),
        "description": "HF transformer runner (custom loop)",
        "code_identity": {"package_version": veriscope.__version__},
        "transport": {"name": "hf_hidden_state_v1", "cadence": f"every_{cfg.cadence}_steps"},
        "evidence": {
            "metrics": ["loss_delta_z"],
            "window": {"kind": "fixed", "size": cfg.gate_window, "stride": cfg.cadence},
        },
        "gates": {
            "preset": cfg.gate_preset,
            "params": {
                "epsilon": cfg.gate_epsilon,
                "min_evidence": cfg.gate_min_evidence,
                "gain_thresh": cfg.gate_gain_thresh,
                "policy": cfg.gate_policy,
                "persistence_k": cfg.gate_persistence_k,
            },
        },
        "model": {"name": cfg.model},
        "dataset": {
            "name": cfg.dataset_name,
            "config": cfg.dataset_config,
            "path": str(cfg.dataset_path) if cfg.dataset_path else None,
            "split": cfg.dataset_split,
            "text_column": cfg.dataset_text_column,
        },
        "sketch": {"kind": "jl", "dim": cfg.rp_dim, "seed": cfg.seed},
    }


def _best_effort_git_sha() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return (result.stdout or "").strip() or None
    except Exception:
        return None


def _best_effort_transformers_version() -> Optional[str]:
    try:
        import transformers  # local import to avoid module-level dependency

        return getattr(transformers, "__version__", None)
    except Exception:
        return None


def _build_run_manifest(
    cfg: HFRunConfig,
    *,
    argv: List[str],
    started_ts_utc: datetime,
    ended_ts_utc: Optional[datetime],
    run_status: str,
    runner_exit_code: Optional[int],
    runner_signal: Optional[str],
    failure_reason: Optional[str],
    failure_traceback: Optional[str],
    rank_used_for_corrupt_seed: Optional[int],
    world_size_used_for_corrupt_seed: Optional[int],
) -> Dict[str, Any]:
    env = os.environ

    def _env_int(name: str) -> Optional[int]:
        raw = env.get(name)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except Exception:
            return None

    if rank_used_for_corrupt_seed is None:
        rank_used_for_corrupt_seed = _env_int("RANK")
    if world_size_used_for_corrupt_seed is None:
        world_size_used_for_corrupt_seed = _env_int("WORLD_SIZE")

    pathology_payload = {
        "lr_spike_at": cfg.lr_spike_at,
        "lr_spike_len": cfg.lr_spike_len,
        "lr_spike_mult": cfg.lr_spike_mult,
        "lr_spike_verify": cfg.lr_spike_verify,
        "data_corrupt_at": cfg.data_corrupt_at,
        "data_corrupt_len": cfg.data_corrupt_len,
        "data_corrupt_frac": cfg.data_corrupt_frac,
        "data_corrupt_mode": cfg.data_corrupt_mode,
        "data_corrupt_target": cfg.data_corrupt_target,
        "data_corrupt_mask_id": cfg.data_corrupt_mask_id,
        "corrupt_seed_scheme": "seed^runid^step^rank^batch_idx",
        "rank_used_for_corrupt_seed": rank_used_for_corrupt_seed,
        "world_size_observed_for_corrupt_seed": world_size_used_for_corrupt_seed,
        "corrupt_seed_note": "seed uses rank; world_size recorded for provenance",
    }
    if cfg.data_corrupt_at >= 0 and cfg.data_corrupt_len > 0 and cfg.data_corrupt_frac > 0.0:
        s0 = int(cfg.data_corrupt_at)
        seed_rank = int(rank_used_for_corrupt_seed or 0)
        pathology_payload[f"corrupt_seed_at_start_rank{seed_rank}_batch0"] = _corrupt_seed(cfg, s0, seed_rank, 0)
        required_snaps = max(
            int(cfg.gate_window) + max(0, int(cfg.gate_persistence_k) - 1),
            int(cfg.gate_min_evidence),
        )
        required_len_steps_min = (required_snaps - 1) * int(cfg.cadence) + 1 if required_snaps > 0 else 0
        injected_snaps_est = (int(cfg.data_corrupt_len) + int(cfg.cadence) - 1) // int(cfg.cadence)
        pathology_payload["data_corrupt_required_snaps"] = required_snaps
        pathology_payload["data_corrupt_required_len_steps_min"] = required_len_steps_min
        pathology_payload["data_corrupt_injected_snaps_est"] = injected_snaps_est
    if cfg.lr_spike_at >= 0 and cfg.lr_spike_len > 0:
        required_snaps = max(
            int(cfg.gate_window) + max(0, int(cfg.gate_persistence_k) - 1),
            int(cfg.gate_min_evidence),
        )
        required_len_steps_min = (required_snaps - 1) * int(cfg.cadence) + 1 if required_snaps > 0 else 0
        injected_snaps_est = (int(cfg.lr_spike_len) + int(cfg.cadence) - 1) // int(cfg.cadence)
        pathology_payload["lr_spike_required_snaps"] = required_snaps
        pathology_payload["lr_spike_required_len_steps_min"] = required_len_steps_min
        pathology_payload["lr_spike_injected_snaps_est"] = injected_snaps_est
    return {
        "schema_version": 1,
        "argv": list(argv),
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "WORLD_SIZE": env.get("WORLD_SIZE"),
            "RANK": env.get("RANK"),
            "MASTER_ADDR": env.get("MASTER_ADDR"),
            "MASTER_PORT": env.get("MASTER_PORT"),
            "HF_HOME": env.get("HF_HOME"),
            "TRANSFORMERS_CACHE": env.get("TRANSFORMERS_CACHE"),
            "PYTHONPATH": env.get("PYTHONPATH"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
        },
        "seeds": {
            "seed": cfg.seed,
            "torch_manual_seed": cfg.seed,
            "numpy_seed": cfg.seed,
            "python_random_seed": cfg.seed,
        },
        "determinism": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "torch_deterministic_algorithms": bool(
                getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)()
            ),
        },
        "pathology": pathology_payload,
        "timestamps": {
            "started_ts_utc": _iso_utc(started_ts_utc),
            "ended_ts_utc": _iso_utc(ended_ts_utc) if ended_ts_utc else None,
        },
        "run_status": run_status,
        "runner_exit_code": runner_exit_code,
        "runner_signal": runner_signal,
        "failure_reason": failure_reason,
        "failure_traceback": failure_traceback,
        "git": {"commit_sha": _best_effort_git_sha()},
        "versions": {
            "veriscope": veriscope.__version__,
            "torch": getattr(torch, "__version__", None),
            "transformers": _best_effort_transformers_version(),
        },
    }


def _write_run_manifest(outdir: Path, manifest: Dict[str, Any]) -> None:
    atomic_write_json(outdir / "run_manifest.json", manifest, fsync=True)


def _tokenize_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_path: Optional[Path],
    dataset_split: str,
    dataset_text_column: str,
    tokenizer: Any,
    *,
    block_size: int,
    batch_size: int,
    seed: int,
) -> Iterable[Dict[str, torch.Tensor]]:
    from datasets import load_dataset

    if dataset_path is not None:
        dataset = load_dataset("text", data_files={dataset_split: str(dataset_path)}, split=dataset_split)
    elif dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    if dataset_text_column not in dataset.column_names:
        raise ValueError(f"Dataset column '{dataset_text_column}' not found. Available columns: {dataset.column_names}")

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # datasets.Dataset.map requires a dict (or pyarrow table) when transforming data.
        # For batched tokenization, transformers returns a list-of-lists for input_ids.
        return {"input_ids": tokenizer(batch[dataset_text_column], add_special_tokens=False)["input_ids"]}

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    def group_texts(batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated: List[int] = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
        total_len = (len(concatenated) // block_size) * block_size
        result = {"input_ids": [concatenated[i : i + block_size] for i in range(0, total_len, block_size)]}
        return result

    grouped = tokenized.map(group_texts, batched=True)

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids}

    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(grouped, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator)


def _metric_snapshot(
    metric_history: List[Dict[str, Any]], gate_window: int
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if gate_window <= 0:
        return [], []
    if len(metric_history) < gate_window:
        return [], list(metric_history)
    past = metric_history[-2 * gate_window : -gate_window]
    recent = metric_history[-gate_window:]
    return past, recent


def _ddp_aggregate_slice(
    slice_data: List[Dict[str, Any]], agg_metrics: Dict[str, Optional[str]], metrics: List[str]
) -> Optional[List[Dict[str, Any]]]:
    if not ddp_can_communicate():
        return None
    aggregated: List[Dict[str, Any]] = []
    for step in slice_data:
        step_out = dict(step)
        values: List[float] = []
        for metric in metrics:
            agg = agg_metrics.get(metric)
            if agg != "mean":
                logger.debug("DDP aggregation unsupported for metric=%s agg=%s", metric, agg)
                return None
            raw = step.get(metric, float("nan"))
            try:
                val = float(raw)
            except Exception:
                val = float("nan")
            values.append(val)
        reduced = ddp_reduce_mean_scalars_masked(values)
        if reduced is None:
            return None
        for metric, value in zip(metrics, reduced):
            step_out[metric] = float(value)
        aggregated.append(step_out)
    return aggregated


def _gate_from_history(
    gate_engine: GateEngine,
    window_decl: WindowDecl,
    metric_history: List[Dict[str, Any]],
    gate_window: int,
    iter_num: int,
    gate_policy: str,
    gate_min_evidence: int,
) -> GateRecordV1:
    if gate_window <= 0:
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_insufficient_evidence",
            policy=gate_policy,
            per_metric_tv={},
            evidence_total=0,
            min_evidence=gate_min_evidence,
        )
        return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)

    metrics = list(window_decl.weights.keys())

    def _count_finite(slice_data: List[Dict[str, Any]], key: str) -> int:
        vals = [float(d.get(key, np.nan)) for d in slice_data]
        arr = np.array(vals, dtype=float)
        return int(np.isfinite(arr).sum())

    def _extract(slice_data: List[Dict[str, Any]], key: str) -> np.ndarray:
        vals = [float(d.get(key, np.nan)) for d in slice_data]
        arr = np.array(vals, dtype=float)
        return arr[np.isfinite(arr)]

    ddp_agg_used = False
    ddp_barrier_status: Optional[str] = None
    agg_metrics: Dict[str, Optional[str]] = {m: DDP_AGG_METRICS.get(m) for m in metrics}
    ddp_env = ddp_is_active()
    ddp_live = ddp_can_communicate()
    if ddp_env and not ddp_live:
        logger.debug(
            "DDP env detected (WORLD_SIZE=%s RANK=%s MASTER_ADDR=%s MASTER_PORT=%s) but process group inactive; "
            "skipping gates: ddp_unsupported.",
            os.environ.get("WORLD_SIZE"),
            os.environ.get("RANK"),
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
        )
        recent_slice = list(metric_history[-gate_window:])
        if len(recent_slice) < gate_window:
            recent_slice = [{}] * (gate_window - len(recent_slice)) + recent_slice
        past_end = max(0, len(metric_history) - gate_window)
        past_start = max(0, past_end - gate_window)
        past_slice = list(metric_history[past_start:past_end])
        if len(past_slice) < gate_window:
            past_slice = [{}] * (gate_window - len(past_slice)) + past_slice
        evidence_total = 0
        if past_slice and recent_slice:
            counts = {m: min(_count_finite(past_slice, m), _count_finite(recent_slice, m)) for m in metrics}
            evidence_total = int(sum(counts.values()))
        audit = AuditV1(
            evaluated=False,
            reason="ddp_unsupported",
            policy=gate_policy,
            per_metric_tv={},
            evidence_total=evidence_total,
            min_evidence=gate_min_evidence,
            ddp_barrier_status="skipped_inactive",
        )
        return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)
    if ddp_env and ddp_live:
        aggregated_past = None
        aggregated_recent = None
        strict_sync = env_truthy("VERISCOPE_DDP_STRICT_GATE_SYNC")
        # Permissive mode assumes all ranks call aggregation collectives in lockstep; strict sync
        # runs a best-effort timed barrier first to reduce deadlock risk.
        if strict_sync:
            ddp_barrier_status = ddp_barrier(timeout_s=10.0)
        else:
            ddp_barrier_status = "not_requested"

        recent_slice = list(metric_history[-gate_window:])
        if len(recent_slice) < gate_window:
            recent_slice = [{}] * (gate_window - len(recent_slice)) + recent_slice
        past_end = max(0, len(metric_history) - gate_window)
        past_start = max(0, past_end - gate_window)
        past_slice = list(metric_history[past_start:past_end])
        if len(past_slice) < gate_window:
            past_slice = [{}] * (gate_window - len(past_slice)) + past_slice

        if (not strict_sync or ddp_barrier_status == "performed") and not any(agg_metrics[m] is None for m in metrics):
            aggregated_past = _ddp_aggregate_slice(past_slice, agg_metrics, metrics)
            aggregated_recent = _ddp_aggregate_slice(recent_slice, agg_metrics, metrics)
        if aggregated_past is None or aggregated_recent is None:
            logger.info(
                "DDP env detected (WORLD_SIZE=%s RANK=%s MASTER_ADDR=%s MASTER_PORT=%s). "
                "Skipping gates: ddp_unsupported (no aggregation).",
                os.environ.get("WORLD_SIZE"),
                os.environ.get("RANK"),
                os.environ.get("MASTER_ADDR"),
                os.environ.get("MASTER_PORT"),
            )
            evidence_total = 0
            if past_slice and recent_slice:
                counts = {m: min(_count_finite(past_slice, m), _count_finite(recent_slice, m)) for m in metrics}
                evidence_total = int(sum(counts.values()))
            audit = AuditV1(
                evaluated=False,
                reason="ddp_unsupported",
                policy=gate_policy,
                per_metric_tv={},
                evidence_total=evidence_total,
                min_evidence=gate_min_evidence,
                ddp_barrier_status=ddp_barrier_status,
            )
            return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)
        ddp_agg_used = True
        past_slice = aggregated_past
        recent_slice = aggregated_recent
    else:
        past_slice, recent_slice = _metric_snapshot(metric_history, gate_window)
    if not past_slice or not recent_slice:
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_insufficient_evidence",
            policy=gate_policy,
            per_metric_tv={},
            evidence_total=0,
            min_evidence=gate_min_evidence,
        )
        return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)

    past_dict = {m: _extract(past_slice, m) for m in metrics}
    recent_dict = {m: _extract(recent_slice, m) for m in metrics}
    counts = {m: min(len(past_dict[m]), len(recent_dict[m])) for m in metrics}
    evidence_total = int(sum(counts.values()))

    eps_stat_value = float("nan")
    if evidence_total > 0:
        eps_stat_value = aggregate_epsilon_stat(window_decl, counts, alpha=0.05)

    # If we don't have enough finite evidence, don't “evaluate” (prevents spurious fail).
    if evidence_total < int(gate_min_evidence):
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_insufficient_finite_evidence",
            policy=gate_policy,
            per_metric_tv={},
            evidence_total=evidence_total,
            min_evidence=gate_min_evidence,
        )
        return GateRecordV1(iter=iter_num, decision="skip", audit=audit, ok=True, warn=False)

    result = gate_engine.check(
        past=past_dict,
        recent=recent_dict,
        counts_by_metric=counts,
        gain_bits=0.0,
        kappa_sens=0.0,
        eps_stat_value=eps_stat_value,
        iter_num=iter_num,
    )

    audit_payload = dict(result.audit or {})
    audit_payload.setdefault("per_metric_tv", {})
    # AuditV1 requires policy always, and requires reason when evaluated=True.
    audit_payload.setdefault("policy", gate_policy)
    audit_payload.setdefault("evidence_total", evidence_total)
    audit_payload.setdefault("min_evidence", int(gate_min_evidence))
    if ddp_agg_used:
        world_size: Optional[int] = None
        try:
            import torch.distributed as dist  # local import

            if dist.is_available() and dist.is_initialized():
                world_size = int(dist.get_world_size())
        except Exception:
            world_size = None
        if world_size is None:
            raw_world_size = os.environ.get("WORLD_SIZE")
            if raw_world_size:
                try:
                    world_size = int(raw_world_size)
                except Exception:
                    world_size = None
        audit_payload.setdefault("ddp_agg", "mean")
        audit_payload.setdefault("aggregation_method", "mean_allreduce")
        audit_payload.setdefault("world_size", world_size)
    if ddp_barrier_status is not None:
        audit_payload.setdefault("ddp_barrier_status", ddp_barrier_status)

    evaluated = bool(audit_payload.get("evaluated", True))
    audit_payload["evaluated"] = evaluated
    if evaluated:
        audit_payload.setdefault("reason", "evaluated")
    else:
        audit_payload.setdefault("reason", "not_evaluated")
    audit = AuditV1(**audit_payload)

    if not audit.evaluated:
        decision = "skip"
    elif result.warn:
        decision = "warn"
    elif result.ok:
        decision = "pass"
    else:
        decision = "fail"

    return GateRecordV1(iter=iter_num, decision=decision, audit=audit, ok=result.ok, warn=result.warn)


def _exit_code_for_signal(signal_name: Optional[str]) -> Optional[int]:
    if not signal_name:
        return None
    mapping = {"SIGINT": 130, "SIGTERM": 143}
    return mapping.get(signal_name, 128)


def _force_cleanup_outdir(outdir: Path) -> None:
    """Remove only the capsule marker files that can deterministically collide.

    We avoid deleting the entire outdir because callers (pytest) may own the temp root.
    """
    targets = [
        "window_signature.json",
        "results.json",
        "results_summary.json",
        "run_manifest.json",
    ]
    for name in targets:
        try:
            p = outdir / name
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _run_body(cfg: HFRunConfig, *, argv: List[str]) -> int:
    if _is_chief():
        # If any outer wrapper pre-created capsule markers (esp window_signature.json),
        # force mode must ensure the runner is authoritative.
        if bool(cfg.force):
            _force_cleanup_outdir(cfg.outdir)
        # Chief-only emission: only rank 0 writes artifacts and manifest.
        cfg.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model)
    # Silence HF warning about embedding tying for models whose checkpoints include both tensors.
    # This is cosmetic/log-hygiene; it does not change training semantics for our runner.
    with contextlib.suppress(Exception):
        if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
    model.config.use_cache = False
    model.to(device)
    model.train()
    vocab_size = int(
        getattr(model.config, "vocab_size", None) or getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    )
    mask_token_id = cfg.data_corrupt_mask_id
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        if str(cfg.data_corrupt_mode).lower().strip() == "mask":
            raise ValueError("--data_corrupt_mask_id required when tokenizer has no mask_token_id")
        mask_token_id = tokenizer.eos_token_id
    mask_token_id = int(mask_token_id or 0)

    data_loader = _tokenize_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.dataset_path,
        cfg.dataset_split,
        cfg.dataset_text_column,
        tokenizer,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    current_lr = float(cfg.lr)

    metric_config = HFMetricConfig(
        max_tokens_per_batch=cfg.batch_size * cfg.block_size,
        rp_dim=cfg.rp_dim,
    )
    metric_computer = HFMetricComputer(config=metric_config, seed=cfg.seed)

    window_decl = _build_window_decl(cfg)
    gate_engine = _build_gate_engine(cfg, window_decl)

    run_id = cfg.run_id
    started_ts = datetime.now(timezone.utc)
    window_signature = _build_window_signature(cfg, created_ts_utc=started_ts)
    window_signature_ref: Dict[str, Any] = {"path": "window_signature.json"}
    if _is_chief():
        try:
            window_signature_path = cfg.outdir / "window_signature.json"
            atomic_write_json(window_signature_path, window_signature)
            window_signature_ref["hash"] = window_signature_sha256(read_json_obj(window_signature_path))
        except Exception as exc:
            logger.warning("Failed to write window_signature.json before governance: %s", exc)

        try:
            code_identity = build_code_identity(git_sha=_best_effort_git_sha())
            append_run_started(
                cfg.outdir,
                run_id=run_id,
                outdir_path=cfg.outdir,
                argv=argv,
                code_identity=code_identity,
                window_signature_ref=window_signature_ref,
                entrypoint={"kind": "runner", "name": "veriscope.runners.hf.train_hf"},
            )
        except Exception as exc:
            logger.warning("Failed to append governance run_started entry: %s", exc)

    if _is_chief():
        if cfg.data_corrupt_len > 0 and cfg.data_corrupt_frac > 0.0:
            if int(cfg.data_corrupt_at) < 0:
                print("[veriscope:hf] WARNING: data_corrupt_len set but data_corrupt_at < 0; corruption is disabled.")
        if cfg.cadence > 0 and cfg.data_corrupt_len > 0 and cfg.data_corrupt_frac > 0.0 and cfg.data_corrupt_at >= 0:
            injected_snaps = (int(cfg.data_corrupt_len) + int(cfg.cadence) - 1) // int(cfg.cadence)
            required_snaps = max(
                int(cfg.gate_window) + max(0, int(cfg.gate_persistence_k) - 1),
                int(cfg.gate_min_evidence),
            )
            required_len_steps_min = (required_snaps - 1) * int(cfg.cadence) + 1 if required_snaps > 0 else 0
            if injected_snaps < required_snaps:
                print(
                    "[veriscope:hf] WARNING: data corruption provides only "
                    f"{injected_snaps} cadenced samples (< required={required_snaps}); "
                    f"min required steps ~{required_len_steps_min}; "
                    "gate trip may be delayed or never occur."
                )
            if int(cfg.data_corrupt_at) >= 0 and int(cfg.data_corrupt_at) % int(cfg.cadence) != 0:
                print(
                    "[veriscope:hf] WARNING: data_corrupt_at is not aligned to cadence; "
                    "first injected snapshot may be delayed."
                )
        if cfg.lr_spike_len > 0:
            if int(cfg.lr_spike_at) < 0:
                print("[veriscope:hf] WARNING: lr_spike_len set but lr_spike_at < 0; lr spike is disabled.")
        if cfg.cadence > 0 and cfg.lr_spike_len > 0 and cfg.lr_spike_at >= 0:
            injected_snaps = (int(cfg.lr_spike_len) + int(cfg.cadence) - 1) // int(cfg.cadence)
            required_snaps = max(
                int(cfg.gate_window) + max(0, int(cfg.gate_persistence_k) - 1),
                int(cfg.gate_min_evidence),
            )
            required_len_steps_min = (required_snaps - 1) * int(cfg.cadence) + 1 if required_snaps > 0 else 0
            if injected_snaps < required_snaps:
                print(
                    "[veriscope:hf] WARNING: lr spike provides only "
                    f"{injected_snaps} cadenced samples (< required={required_snaps}); "
                    f"min required steps ~{required_len_steps_min}; "
                    "gate trip may be delayed or never occur."
                )
            if int(cfg.lr_spike_at) >= 0 and int(cfg.lr_spike_at) % int(cfg.cadence) != 0:
                print(
                    "[veriscope:hf] WARNING: lr_spike_at is not aligned to cadence; "
                    "first injected snapshot may be delayed."
                )

    interrupt_signal: Optional[str] = None
    stop_requested = False
    previous_handlers: Dict[int, Any] = {}
    failure_reason: Optional[str] = None
    failure_traceback: Optional[str] = None

    def _signal_handler(signum: int, _frame: Any) -> None:
        nonlocal interrupt_signal, stop_requested
        stop_requested = True
        try:
            interrupt_signal = signal.Signals(signum).name
        except Exception:
            interrupt_signal = f"SIG{signum}"

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            sig_num = sig.value if hasattr(sig, "value") else int(sig)
            previous_handlers[sig_num] = signal.getsignal(sig_num)
            signal.signal(sig_num, _signal_handler)
        except Exception:
            continue

    metric_history: List[Dict[str, Any]] = []
    gate_records: List[GateRecordV1] = []
    metric_records: List[MetricRecordV1] = []

    step = 0
    batch_idx = 0
    rank, world_size = _get_rank_and_world()
    data_iter = iter(data_loader)
    try:
        while step < cfg.max_steps:
            if stop_requested:
                break
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Always provide a real attention_mask (important for adapter metrics on tiny models).
            attention_mask: Optional[torch.Tensor] = None
            if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            input_ids, labels, corrupt_stats = _maybe_corrupt_batch(
                input_ids,
                labels,
                cfg=cfg,
                step=step,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                attention_mask=attention_mask,
                rank=rank,
                batch_idx=batch_idx,
            )

            effective_lr = _effective_lr(cfg, step)
            if effective_lr != current_lr:
                for group in optimizer.param_groups:
                    group["lr"] = effective_lr
                current_lr = effective_lr
            if cfg.lr_spike_verify and _lr_spike_active(cfg, step) and step == cfg.lr_spike_at and _is_chief():
                ratio = effective_lr / float(cfg.lr) if float(cfg.lr) != 0.0 else float("inf")
                print(f"[veriscope:hf] lr_spike_verify ratio={ratio:.4f} step={step}")
            need_hidden = step % cfg.cadence == 0
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=need_hidden,
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if step % cfg.cadence == 0:
                with torch.no_grad():
                    hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
                    if hidden_states is None:
                        m = {"var_out_k": float("nan"), "eff_dim": float("nan")}
                    else:
                        try:
                            m = metric_computer.compute_metrics(
                                hidden_states=hidden_states.detach(),
                                attention_mask=attention_mask,
                                step=step,
                            )
                        except Exception:
                            m = _fallback_metrics_from_hidden(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                rp_dim=cfg.rp_dim,
                            )
                        # If adapter returns non-finite values, use fallback (keeps gates evaluable).
                        v_ok = math.isfinite(float(m.get("var_out_k", float("nan"))))
                        e_ok = math.isfinite(float(m.get("eff_dim", float("nan"))))
                        if not (v_ok and e_ok):
                            m = _fallback_metrics_from_hidden(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                rp_dim=cfg.rp_dim,
                            )

                # Telemetry: local loss and cross-rank mean loss (for DDP-correct evidence computation).
                loss_local = float(loss.detach().cpu().item())
                loss_mean = loss_local
                if ddp_can_communicate():
                    reduced = ddp_reduce_mean_scalars_masked([loss_local])
                    if reduced is not None and len(reduced) == 1:
                        reduced_value = float(reduced[0])
                        if math.isfinite(reduced_value):
                            loss_mean = reduced_value

                loss_delta = float("nan")
                loss_delta_z = float("nan")
                # Evidence: compute from aggregated (cross-rank mean) loss stream ONLY.
                # Past-only, variable-length (<=gate_window) reference; require >=2 finite.
                gw = int(cfg.gate_window)
                if gw > 0 and metric_history:
                    ref_window = metric_history[-min(len(metric_history), gw) :]
                    ref_losses = [float(d.get("loss_mean", d.get("loss", float("nan")))) for d in ref_window]
                    ref_arr = np.array([v for v in ref_losses if math.isfinite(v)], dtype=float)
                    if ref_arr.size >= 2:
                        ref_mean = float(ref_arr.mean())
                        ref_std = float(ref_arr.std(ddof=1))
                        loss_delta = loss_mean - ref_mean
                        loss_delta_z = loss_delta / max(ref_std, 1e-4)
                        loss_delta_z = float(np.clip(loss_delta_z, -6.0, 6.0))

                var_out_k_raw = float(m.get("var_out_k", float("nan")))
                eff_dim_raw = float(m.get("eff_dim", float("nan")))

                metric_history.append(
                    {
                        "iter": step,
                        "loss": loss_local,
                        "loss_mean": loss_mean,
                        "loss_delta": loss_delta,
                        "loss_delta_z": loss_delta_z,
                        "var_out_k": var_out_k_raw,
                        "eff_dim": eff_dim_raw,
                        "lr": effective_lr,
                        "data_corrupt_active": int(_data_corrupt_active(cfg, step)),
                        "lr_spike_active": int(_lr_spike_active(cfg, step)),
                        "data_corrupt_frac_effective": float(corrupt_stats.get("corrupt_frac_effective", 0.0)),
                    }
                )

                # Emit JSON-safe MetricRecordV1 for declared evidence metrics (and loss).
                metric_records.append(MetricRecordV1(name="loss", iter=step, value=_jsonable_float(loss_local)))
                metric_records.append(MetricRecordV1(name="loss_mean", iter=step, value=_jsonable_float(loss_mean)))
                metric_records.append(MetricRecordV1(name="loss_delta", iter=step, value=_jsonable_float(loss_delta)))
                metric_records.append(
                    MetricRecordV1(name="loss_delta_z", iter=step, value=_jsonable_float(loss_delta_z))
                )
                metric_records.append(MetricRecordV1(name="var_out_k", iter=step, value=_jsonable_float(var_out_k_raw)))
                metric_records.append(MetricRecordV1(name="eff_dim", iter=step, value=_jsonable_float(eff_dim_raw)))
                metric_records.append(MetricRecordV1(name="lr", iter=step, value=_jsonable_float(effective_lr)))
                metric_records.append(
                    MetricRecordV1(
                        name="data_corrupt_frac_effective",
                        iter=step,
                        value=_jsonable_float(float(corrupt_stats.get("corrupt_frac_effective", 0.0))),
                    )
                )
                metric_records.append(
                    MetricRecordV1(
                        name="data_corrupt_active",
                        iter=step,
                        value=_jsonable_float(float(_data_corrupt_active(cfg, step))),
                    )
                )
                metric_records.append(
                    MetricRecordV1(
                        name="lr_spike_active",
                        iter=step,
                        value=_jsonable_float(float(_lr_spike_active(cfg, step))),
                    )
                )

                gate_record = _gate_from_history(
                    gate_engine,
                    window_decl,
                    metric_history,
                    cfg.gate_window,
                    step,
                    cfg.gate_policy,
                    cfg.gate_min_evidence,
                )
                gate_records.append(gate_record)
                if _is_chief():
                    try:
                        audit_payload = gate_record.audit.model_dump(mode="json", by_alias=True, exclude_none=True)
                        append_gate_decision(
                            cfg.outdir,
                            run_id=run_id,
                            iter_num=int(gate_record.iter),
                            decision=str(gate_record.decision),
                            ok=gate_record.ok,
                            warn=gate_record.warn,
                            audit=audit_payload,
                        )
                    except Exception as exc:
                        logger.warning("Failed to append governance gate decision: %s", exc)

            step += 1
            batch_idx += 1
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
        # Make failures visible in CI logs (otherwise you just see rc!=0 with no clue).
        if _is_chief():
            print(f"[veriscope:hf] failure_reason={failure_reason}\n{failure_traceback}", file=sys.stderr)

    for sig_num, handler in previous_handlers.items():
        try:
            signal.signal(sig_num, handler)
        except Exception:
            continue

    ended_ts = datetime.now(timezone.utc)
    if not gate_records:
        audit = AuditV1(
            evaluated=False,
            reason="not_evaluated_no_steps",
            policy=cfg.gate_policy,
            per_metric_tv={},
            evidence_total=0,
            min_evidence=cfg.gate_min_evidence,
        )
        gate_records.append(GateRecordV1(iter=0, decision="skip", audit=audit, ok=True, warn=False))

    metrics = list(metric_records)
    if not metrics:
        # Ensure non-degenerate emission even when no steps completed.
        metrics = [
            MetricRecordV1(name="loss", iter=0, value=None),
            MetricRecordV1(name="loss_mean", iter=0, value=None),
            MetricRecordV1(name="loss_delta", iter=0, value=None),
            MetricRecordV1(name="loss_delta_z", iter=0, value=None),
            MetricRecordV1(name="var_out_k", iter=0, value=None),
            MetricRecordV1(name="eff_dim", iter=0, value=None),
        ]

    run_status = "success"
    runner_signal = interrupt_signal
    runner_exit_code = 0
    if failure_reason:
        run_status = "user_code_failure"
        runner_exit_code = 1
    elif stop_requested:
        run_status = "user_code_failure"
        runner_exit_code = _exit_code_for_signal(interrupt_signal) or 1

    # DDP barrier default skips if backend lacks timeout support (avoid hangs).
    if ddp_can_communicate():
        barrier_status = ddp_barrier()
    else:
        barrier_status = "skipped_inactive"
    try:
        if barrier_status == "skipped_no_timeout":
            global _DDP_BARRIER_WARNED
            if not _DDP_BARRIER_WARNED:
                logger.warning(
                    "DDP barrier timeout unsupported; barrier skipped (default non-hanging policy). "
                    "Partial capsule possible under rank skew. Set VERISCOPE_DDP_STRICT_BARRIER=1 "
                    "to force unbounded barrier (may hang)."
                )
                _DDP_BARRIER_WARNED = True
        if _is_chief():
            # Chief-only emission: only rank 0 writes artifacts and manifest.
            emit_hf_artifacts_v1(
                outdir=cfg.outdir,
                run_id=run_id,
                started_ts_utc=started_ts,
                ended_ts_utc=ended_ts,
                gate_preset=cfg.gate_preset,
                window_signature=window_signature,
                gate_records=gate_records,
                metrics=metrics,
                run_status=run_status,
                runner_exit_code=runner_exit_code,
                runner_signal=runner_signal,
            )
            manifest = _build_run_manifest(
                cfg,
                argv=argv,
                started_ts_utc=started_ts,
                ended_ts_utc=ended_ts,
                run_status=run_status,
                runner_exit_code=runner_exit_code,
                runner_signal=runner_signal,
                failure_reason=failure_reason,
                failure_traceback=failure_traceback,
                rank_used_for_corrupt_seed=rank,
                world_size_used_for_corrupt_seed=world_size,
            )
            _write_run_manifest(cfg.outdir, manifest)
    finally:
        # Best-effort teardown sync (same non-hanging policy as pre-finalize barrier).
        if ddp_can_communicate():
            ddp_barrier()
    return int(runner_exit_code if runner_exit_code is not None else 1)


def _run(cfg: HFRunConfig, *, argv: List[str]) -> int:
    initialized_here = _maybe_init_ddp()
    _set_seed(cfg.seed)
    try:
        return _run_body(cfg, argv=argv)
    finally:
        if _should_cleanup_ddp(initialized_here):
            ddp_destroy_process_group()


def _parse_args() -> HFRunConfig:
    parser = argparse.ArgumentParser(description="Veriscope HF transformer runner (custom loop).")
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name (default: gpt2)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext:wikitext-2-raw-v1",
        help="HF dataset spec (name[:config]) (default: wikitext:wikitext-2-raw-v1)",
    )
    parser.add_argument("--dataset_name", type=str, default="", help="HF dataset name override")
    parser.add_argument("--dataset_config", type=str, default="", help="HF dataset config override")
    parser.add_argument("--dataset_split", type=str, default="train", help="HF dataset split")
    parser.add_argument("--dataset_text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--outdir", type=str, default="", help="Output directory for artifacts")
    parser.add_argument("--run_id", type=str, default="", help="Run identifier (wrapper overrides)")
    parser.add_argument("--force", action="store_true", help="Overwrite/repair existing artifacts in outdir")
    parser.add_argument("--max_steps", type=int, default=200, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_spike_at", type=int, default=-1, help="Step to start LR spike (<0 disables)")
    parser.add_argument("--lr_spike_len", type=int, default=0, help="Number of steps to spike LR")
    parser.add_argument("--lr_spike_mult", type=float, default=1.0, help="LR multiplier during spike")
    parser.add_argument(
        "--lr_spike_verify",
        action="store_true",
        help="Log LR spike verification ratio when spike begins",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--cadence", type=int, default=10, help="Instrumentation cadence (steps)")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length for training")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (0 to disable)")
    parser.add_argument("--gate_preset", type=str, default="tuned_v0", help="Gate preset name")
    parser.add_argument("--gate_window", type=int, default=20, help="Gate window (metric snapshots)")
    parser.add_argument("--gate_epsilon", type=float, default=0.12, help="Gate epsilon")
    parser.add_argument("--gate_min_evidence", type=int, default=8, help="Gate minimum evidence")
    parser.add_argument("--gate_gain_thresh", type=float, default=0.0, help="Gate gain threshold")
    parser.add_argument("--gate_policy", type=str, default="persistence", help="Gate policy")
    parser.add_argument("--gate_persistence_k", type=int, default=2, help="Gate persistence K")
    parser.add_argument("--rp_dim", type=int, default=64, help="JL projection dimension")
    parser.add_argument("--data_corrupt_at", type=int, default=-1, help="Step to start data corruption (<0 disables)")
    parser.add_argument("--data_corrupt_len", type=int, default=0, help="Number of steps to corrupt data")
    parser.add_argument(
        "--data_corrupt_frac",
        type=float,
        default=0.0,
        help="Fraction of tokens to corrupt per sequence",
    )
    parser.add_argument(
        "--data_corrupt_mode",
        type=str,
        default="permute",
        help="Token corruption mode: permute, random, or mask",
    )
    parser.add_argument(
        "--data_corrupt_target",
        type=str,
        default="clean",
        help="Corruption target: clean (default) or same",
    )
    parser.add_argument(
        "--data_corrupt_mask_id",
        type=int,
        default=None,
        help="Override mask token id for data corruption (used when mode=mask)",
    )

    args = parser.parse_args()
    if int(args.cadence) <= 0:
        raise ValueError("--cadence must be >= 1")
    outdir = Path(args.outdir).expanduser() if args.outdir else _default_outdir()
    run_id = args.run_id.strip() or uuid.uuid4().hex[:12]
    dataset_name = args.dataset_name.strip()
    dataset_config = args.dataset_config.strip()
    dataset_path: Optional[Path] = None
    if not dataset_name:
        dataset_spec = args.dataset.strip()
        if dataset_spec.startswith("file:"):
            dataset_path_raw = dataset_spec[len("file:") :].strip()
            if not dataset_path_raw:
                raise ValueError("--dataset file: requires a path")
            candidate = Path(dataset_path_raw).expanduser()
            if not candidate.is_absolute():
                repo_root = Path(__file__).resolve().parents[3]
                candidate = (repo_root / candidate).resolve()
            dataset_path = candidate
            dataset_name = "file"
            dataset_config = ""
        elif ":" in dataset_spec:
            dataset_name, dataset_config = dataset_spec.split(":", 1)
        else:
            dataset_name = dataset_spec
    dataset_name = dataset_name or "wikitext"
    dataset_config = dataset_config or None
    if not math.isfinite(float(args.lr)):
        raise ValueError("--lr must be finite")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be > 0")
    if not math.isfinite(float(args.lr_spike_mult)):
        raise ValueError("--lr_spike_mult must be finite")
    if not math.isfinite(float(args.data_corrupt_frac)):
        raise ValueError("--data_corrupt_frac must be finite")
    if float(args.data_corrupt_frac) < 0.0 or float(args.data_corrupt_frac) > 1.0:
        raise ValueError("--data_corrupt_frac must be within [0, 1]")
    if int(args.data_corrupt_len) < 0:
        raise ValueError("--data_corrupt_len must be >= 0")
    if int(args.data_corrupt_at) < -1:
        raise ValueError("--data_corrupt_at must be >= -1")
    if float(args.lr_spike_mult) <= 0.0:
        raise ValueError("--lr_spike_mult must be > 0")
    if int(args.lr_spike_len) < 0:
        raise ValueError("--lr_spike_len must be >= 0")
    if int(args.lr_spike_at) < -1:
        raise ValueError("--lr_spike_at must be >= -1")
    if args.data_corrupt_mask_id is not None and int(args.data_corrupt_mask_id) < 0:
        raise ValueError("--data_corrupt_mask_id must be >= 0")
    mode = str(args.data_corrupt_mode).lower().strip()
    if mode not in {"permute", "random", "mask"}:
        raise ValueError("--data_corrupt_mode must be one of: permute, random, mask")
    target_mode = str(args.data_corrupt_target).lower().strip()
    if target_mode not in {"clean", "same"}:
        raise ValueError("--data_corrupt_target must be one of: clean, same")
    # Honor either runner --force or wrapper-side VERISCOPE_FORCE=1
    force = bool(args.force) or env_truthy("VERISCOPE_FORCE")
    return HFRunConfig(
        model=args.model,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_path=dataset_path,
        dataset_split=args.dataset_split,
        dataset_text_column=args.dataset_text_column,
        outdir=outdir,
        run_id=run_id,
        force=force,
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        cadence=int(args.cadence),
        block_size=int(args.block_size),
        device=str(args.device),
        grad_clip=float(args.grad_clip),
        gate_preset=str(args.gate_preset),
        gate_window=int(args.gate_window),
        gate_epsilon=float(args.gate_epsilon),
        gate_min_evidence=int(args.gate_min_evidence),
        gate_gain_thresh=float(args.gate_gain_thresh),
        gate_policy=str(args.gate_policy),
        gate_persistence_k=int(args.gate_persistence_k),
        rp_dim=int(args.rp_dim),
        lr_spike_at=int(args.lr_spike_at),
        lr_spike_len=int(args.lr_spike_len),
        lr_spike_mult=float(args.lr_spike_mult),
        lr_spike_verify=bool(args.lr_spike_verify),
        data_corrupt_at=int(args.data_corrupt_at),
        data_corrupt_len=int(args.data_corrupt_len),
        data_corrupt_frac=float(args.data_corrupt_frac),
        data_corrupt_mode=mode,
        data_corrupt_target=target_mode,
        data_corrupt_mask_id=args.data_corrupt_mask_id,
    )


def main() -> int:
    cfg = _parse_args()
    if _is_chief():
        cfg.outdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            cfg.outdir / "runner_config.json",
            {
                "schema_version": 1,
                "ts_utc": _iso_utc(datetime.now(timezone.utc)),
                "runner": "hf",
                "config": {
                    "run_id": cfg.run_id,
                    "model": cfg.model,
                    "dataset_name": cfg.dataset_name,
                    "dataset_config": cfg.dataset_config,
                    "dataset_path": str(cfg.dataset_path) if cfg.dataset_path else None,
                    "dataset_split": cfg.dataset_split,
                    "dataset_text_column": cfg.dataset_text_column,
                    "max_steps": cfg.max_steps,
                    "batch_size": cfg.batch_size,
                    "lr": cfg.lr,
                    "lr_spike_at": cfg.lr_spike_at,
                    "lr_spike_len": cfg.lr_spike_len,
                    "lr_spike_mult": cfg.lr_spike_mult,
                    "lr_spike_verify": cfg.lr_spike_verify,
                    "seed": cfg.seed,
                    "cadence": cfg.cadence,
                    "block_size": cfg.block_size,
                    "device": cfg.device,
                    "grad_clip": cfg.grad_clip,
                    "gate_preset": cfg.gate_preset,
                    "gate_window": cfg.gate_window,
                    "gate_epsilon": cfg.gate_epsilon,
                    "gate_min_evidence": cfg.gate_min_evidence,
                    "gate_gain_thresh": cfg.gate_gain_thresh,
                    "gate_policy": cfg.gate_policy,
                    "gate_persistence_k": cfg.gate_persistence_k,
                    "rp_dim": cfg.rp_dim,
                    "data_corrupt_at": cfg.data_corrupt_at,
                    "data_corrupt_len": cfg.data_corrupt_len,
                    "data_corrupt_frac": cfg.data_corrupt_frac,
                    "data_corrupt_mode": cfg.data_corrupt_mode,
                    "data_corrupt_target": cfg.data_corrupt_target,
                    "data_corrupt_mask_id": cfg.data_corrupt_mask_id,
                },
            },
        )
    return _run(cfg, argv=sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
