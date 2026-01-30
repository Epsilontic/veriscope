# veriscope/core/ddp.py
from __future__ import annotations

import logging
import math
import os
from datetime import timedelta
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = logging.getLogger(__name__)


def env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _dist_module() -> Optional[Any]:
    try:
        import torch.distributed as dist  # local import
    except Exception:
        return None
    return dist if getattr(dist, "is_available", lambda: False)() else None


def _dist_rank_and_world(dist: Any) -> Optional[tuple[int, int]]:
    try:
        if dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        return None
    return None


def _env_ddp_signature() -> bool:
    def _has(name: str) -> bool:
        val = os.environ.get(name)
        return val is not None and val != ""

    return _env_int("WORLD_SIZE", 1) > 1 and _has("RANK") and _has("MASTER_ADDR") and _has("MASTER_PORT")


def _env_allows_ddp_rank_world() -> bool:
    if env_truthy("VERISCOPE_FORCE_SINGLE_PROCESS"):
        return False
    if env_truthy("VERISCOPE_DDP_ALLOW_ENV_RANKWORLD"):
        return True
    return os.environ.get("LOCAL_RANK") not in (None, "") or os.environ.get("TORCHELASTIC_RUN_ID") not in (None, "")


def ddp_is_active() -> bool:
    if env_truthy("VERISCOPE_FORCE_SINGLE_PROCESS"):
        return False
    dist = _dist_module()
    if dist is not None:
        rw = _dist_rank_and_world(dist)
        if rw:
            return rw[1] > 1
    if not _env_allows_ddp_rank_world():
        return False
    return _env_ddp_signature()


def ddp_can_communicate() -> bool:
    """True when torch.distributed is initialized with world_size > 1."""
    dist = _dist_module()
    if dist is None:
        return False
    rw = _dist_rank_and_world(dist)
    if not rw:
        return False
    return rw[1] > 1


def ddp_rank() -> int:
    dist = _dist_module()
    if dist is not None:
        rw = _dist_rank_and_world(dist)
        if rw:
            return rw[0]
    if not _env_allows_ddp_rank_world():
        return 0
    return _env_int("RANK", 0)


def ddp_local_rank() -> int:
    env = os.environ.get("LOCAL_RANK")
    if env is not None and env != "" and _env_allows_ddp_rank_world():
        return _env_int("LOCAL_RANK", 0)
    return 0


def ddp_world_size() -> int:
    dist = _dist_module()
    if dist is not None:
        rw = _dist_rank_and_world(dist)
        if rw:
            return max(1, rw[1])
    if not _env_allows_ddp_rank_world():
        return 1
    size = _env_int("WORLD_SIZE", 1)
    return size if size > 0 else 1


def ddp_is_chief() -> bool:
    return ddp_rank() == 0


def ddp_barrier(timeout_s: float = 30.0) -> str:
    """Best-effort barrier with a non-hanging default policy.

    Returns:
        "performed" when a barrier was executed,
        "skipped_no_timeout" when timeout is unsupported and strict mode is off,
        "skipped_inactive" when no multi-rank process group is active,
        "skipped_error" on failures.
    """
    dist = _dist_module()
    if dist is None:
        return "skipped_inactive"
    try:
        rw = _dist_rank_and_world(dist)
        if not rw:
            return "skipped_inactive"
        world_size = rw[1]
        if world_size <= 1:
            return "skipped_inactive"
        timeout = timedelta(seconds=float(timeout_s))
        if hasattr(dist, "monitored_barrier"):
            dist.monitored_barrier(timeout=timeout)
            return "performed"
        try:
            dist.barrier(timeout=timeout)
        except TypeError:
            if env_truthy("VERISCOPE_DDP_STRICT_BARRIER"):
                dist.barrier()
                return "performed"
            logger.debug("DDP barrier timeout unsupported; skipping barrier.")
            return "skipped_no_timeout"
        return "performed"
    except Exception:
        logger.debug("DDP barrier failed; skipping barrier.", exc_info=True)
        return "skipped_error"


def _ddp_reduce_device() -> Optional["torch.device"]:
    try:
        import torch  # local import
        import torch.distributed as dist  # local import

        backend = None
        try:
            backend = dist.get_backend()
        except Exception:
            backend = None
        if backend == "nccl":
            if not torch.cuda.is_available():
                logger.debug("DDP backend nccl but CUDA unavailable; skipping reduction.")
                return None
            if os.environ.get("LOCAL_RANK") in (None, ""):
                logger.debug("DDP backend nccl but LOCAL_RANK unset; skipping reduction.")
                return None
            return torch.device("cuda", ddp_local_rank())
        return torch.device("cpu")
    except Exception:
        return None


def ddp_reduce_mean_scalar(x: float) -> Optional[float]:
    try:
        if not ddp_can_communicate():
            return None
        import torch  # local import
        import torch.distributed as dist  # local import

        device = _ddp_reduce_device()
        if device is None:
            return None
        tensor = torch.tensor(float(x), dtype=torch.float32, device=device)
        op = getattr(dist, "ReduceOp", None)
        op_sum = op.SUM if op is not None else None
        if op_sum is not None:
            dist.all_reduce(tensor, op=op_sum)
        else:
            dist.all_reduce(tensor)
        world_size = int(dist.get_world_size())
        if world_size <= 0:
            return None
        return float(tensor.item()) / float(world_size)
    except Exception:
        logger.debug("DDP mean reduction failed.", exc_info=True)
        return None


def ddp_reduce_mean_scalar_masked(x: float) -> Optional[float]:
    try:
        if not ddp_can_communicate():
            return None
        import torch  # local import
        import torch.distributed as dist  # local import

        device = _ddp_reduce_device()
        if device is None:
            return None
        val = float(x)
        is_finite = math.isfinite(val)
        sum_val = val if is_finite else 0.0
        count_val = 1.0 if is_finite else 0.0
        tensor = torch.tensor([sum_val, count_val], dtype=torch.float32, device=device)
        op = getattr(dist, "ReduceOp", None)
        op_sum = op.SUM if op is not None else None
        if op_sum is not None:
            dist.all_reduce(tensor, op=op_sum)
        else:
            dist.all_reduce(tensor)
        total_sum = float(tensor[0].item())
        total_count = float(tensor[1].item())
        if total_count <= 0.0:
            return float("nan")
        return total_sum / total_count
    except Exception:
        logger.debug("DDP masked mean reduction failed.", exc_info=True)
        return None


def ddp_reduce_mean_scalars_masked(values: list[float]) -> Optional[list[float]]:
    try:
        if not ddp_can_communicate():
            return None
        if not values:
            return []
        import torch  # local import
        import torch.distributed as dist  # local import

        device = _ddp_reduce_device()
        if device is None:
            return None
        packed: list[float] = []
        for val in values:
            is_finite = math.isfinite(float(val))
            packed.append(float(val) if is_finite else 0.0)
            packed.append(1.0 if is_finite else 0.0)
        tensor = torch.tensor(packed, dtype=torch.float32, device=device)
        op = getattr(dist, "ReduceOp", None)
        op_sum = op.SUM if op is not None else None
        if op_sum is not None:
            dist.all_reduce(tensor, op=op_sum)
        else:
            dist.all_reduce(tensor)
        results: list[float] = []
        for i in range(len(values)):
            total_sum = float(tensor[2 * i].item())
            total_count = float(tensor[2 * i + 1].item())
            if total_count <= 0.0:
                results.append(float("nan"))
            else:
                results.append(total_sum / total_count)
        return results
    except Exception:
        logger.debug("DDP masked mean reduction failed.", exc_info=True)
        return None


def ddp_destroy_process_group() -> None:
    dist = _dist_module()
    if dist is None:
        return
    try:
        rw = _dist_rank_and_world(dist)
        if rw and rw[1] > 1:
            dist.destroy_process_group()
    except Exception:
        logger.debug("DDP destroy_process_group failed; continuing.", exc_info=True)
        return
