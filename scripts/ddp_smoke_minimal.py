from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _build_window_signature() -> dict:
    return {
        "schema_version": 1,
        "transport": {"name": "hf_hidden_state_v1", "cadence": "every_1_steps"},
        "evidence": {"metrics": ["loss"]},
        "gates": {"preset": "tuned_v0"},
        "description": "ddp_smoke_minimal",
    }


def _early_validate_fail_rank() -> tuple[int | None, int]:
    fail_rank_env = os.environ.get("VS_FAIL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "1"))
    try:
        world_size = int(world_size_env)
    except (TypeError, ValueError):
        world_size = 1
    if fail_rank_env is None:
        return None, world_size
    try:
        fail_rank = int(fail_rank_env)
    except (TypeError, ValueError):
        sys.stderr.write(f"VS_FAIL_RANK must be int, got {fail_rank_env!r}\n")
        raise SystemExit(2)
    if fail_rank < 0 or fail_rank >= world_size:
        sys.stderr.write(f"ERROR: VS_FAIL_RANK out of range: {fail_rank} (world_size={world_size})\n")
        sys.stderr.write(f"VS_FAIL_RANK must be in [0, {world_size - 1}], got {fail_rank}\n")
        raise SystemExit(2)
    return fail_rank, world_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal DDP smoke that emits veriscope artifacts.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-id", default="ddp-smoke")
    args = parser.parse_args()
    fail_rank, _ = _early_validate_fail_rank()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.distributed as dist

    from veriscope.core.artifacts import AuditV1, GateRecordV1
    from veriscope.runners.hf.emit_artifacts import emit_hf_artifacts_v1

    random.seed(0)
    torch.manual_seed(0)

    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=30))

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(f"Expected world_size=2, got {world_size}")
        should_fail = fail_rank is not None and fail_rank == rank

        if rank == 0:
            # Best-effort policy: rank 0 may emit artifacts even if another rank fails.
            gate_records = [GateRecordV1(iter=0, decision="skip", audit=AuditV1(evaluated=False, per_metric_tv={}))]
            now = datetime.now(timezone.utc)
            emit_hf_artifacts_v1(
                outdir=args.outdir,
                run_id=args.run_id,
                started_ts_utc=now,
                ended_ts_utc=now,
                gate_preset="tuned_v0",
                window_signature=_build_window_signature(),
                gate_records=gate_records,
                run_status="success",
                runner_exit_code=0,
                runner_signal=None,
            )

        if should_fail:
            raise RuntimeError(f"Intentional failure on rank {rank}")

        if fail_rank is None:
            dist.barrier()
    finally:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
