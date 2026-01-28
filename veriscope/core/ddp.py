# veriscope/core/ddp.py
from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


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


def ddp_is_active() -> bool:
    dist = _dist_module()
    if dist is not None:
        rw = _dist_rank_and_world(dist)
        if rw:
            return rw[1] > 1
    return _env_ddp_signature()


def ddp_rank() -> int:
    dist = _dist_module()
    if dist is not None:
        rw = _dist_rank_and_world(dist)
        if rw:
            return rw[0]
    return _env_int("RANK", 0)


def ddp_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def ddp_world_size() -> int:
    dist = _dist_module()
    if dist is not None:
        rw = _dist_rank_and_world(dist)
        if rw:
            return max(1, rw[1])
    size = _env_int("WORLD_SIZE", 1)
    return size if size > 0 else 1


def ddp_is_chief() -> bool:
    return ddp_rank() == 0


def env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


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
