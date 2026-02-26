"""Core capsule emitter. No runner imports. No DDP logic.

Contract:
  - window_signature.json MUST exist on disk before calling emit_capsule_v1.
  - Caller provides ws_hash (the frozen hash from GateSession or equivalent).
  - Emitter verifies disk hash matches ws_hash; raises RuntimeError on mismatch.
  - ProfileV1.overrides is always empty; overrides are not supported in v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from veriscope.core.artifacts import (
    CountsV1,
    GateRecordV1,
    MetricRecordV1,
    ProfileV1,
    ResultsSummaryV1,
    ResultsV1,
    WindowSignatureRefV1,
    derive_final_decision,
)
from veriscope.core.jsonutil import atomic_write_json, atomic_write_text, window_signature_sha256


@dataclass(frozen=True)
class EmitResult:
    outdir: Path
    window_signature_hash: str
    emitted: bool


def _derive_counts(
    gates: List[GateRecordV1],
) -> tuple[CountsV1, Optional[int]]:
    pass_n = 0
    warn_n = 0
    fail_n = 0
    skip_n = 0
    first_fail_iter: Optional[int] = None

    for gate in gates:
        decision = str(gate.decision)
        if decision == "pass":
            pass_n += 1
        elif decision == "warn":
            warn_n += 1
        elif decision == "fail":
            fail_n += 1
            gate_iter = int(gate.iter)
            if first_fail_iter is None or gate_iter < first_fail_iter:
                first_fail_iter = gate_iter
        elif decision == "skip":
            skip_n += 1

    evaluated = pass_n + warn_n + fail_n
    counts = CountsV1(
        evaluated=evaluated,
        skip=skip_n,
        pass_=pass_n,
        warn=warn_n,
        fail=fail_n,
    )
    return counts, first_fail_iter


def emit_capsule_v1(
    *,
    outdir: Path,
    run_id: str,
    started_ts_utc: datetime,
    ended_ts_utc: datetime,
    gate_preset: str,
    ws_hash: str,
    gate_records: List[GateRecordV1],
    run_status: str = "success",
    runner_exit_code: Optional[int] = None,
    runner_signal: Optional[str] = None,
    metrics: Optional[List[MetricRecordV1]] = None,
) -> EmitResult:
    """Write canonical v1 capsule artifacts using only core types."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ws_path = outdir / "window_signature.json"
    if not ws_path.exists():
        raise RuntimeError(
            "window_signature.json must exist before emit_capsule_v1. "
            "Caller (GateSession or equivalent) owns the write."
        )
    ws_on_disk = json.loads(ws_path.read_text(encoding="utf-8"))
    disk_hash = window_signature_sha256(ws_on_disk)
    if disk_hash != str(ws_hash):
        raise RuntimeError(
            "window_signature.json hash mismatch: "
            f"expected={ws_hash} disk={disk_hash}. "
            "File was modified after governance binding."
        )

    ws_ref = WindowSignatureRefV1(hash=str(ws_hash), path="window_signature.json")
    gate_records_list = list(gate_records)
    counts, first_fail_iter = _derive_counts(gate_records_list)
    final_decision = derive_final_decision(counts)
    profile = ProfileV1(gate_preset=str(gate_preset), overrides={})

    safe_exit: Optional[int]
    if runner_exit_code is None:
        safe_exit = None
    else:
        try:
            exit_i = int(runner_exit_code)
        except Exception:
            exit_i = -1
        safe_exit = exit_i if 0 <= exit_i <= 255 else None

    metric_records_list = list(metrics or [])

    results = ResultsV1(
        run_id=str(run_id),
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=str(run_status),
        runner_exit_code=safe_exit,
        runner_signal=runner_signal,
        started_ts_utc=started_ts_utc,
        ended_ts_utc=ended_ts_utc,
        gates=tuple(gate_records_list),
        metrics=tuple(metric_records_list),
    )
    atomic_write_json(
        outdir / "results.json",
        results.model_dump(mode="json", by_alias=True, exclude_none=True),
        fsync=True,
    )

    summary = ResultsSummaryV1(
        run_id=str(run_id),
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=str(run_status),
        runner_exit_code=safe_exit,
        runner_signal=runner_signal,
        started_ts_utc=started_ts_utc,
        ended_ts_utc=ended_ts_utc,
        counts=counts,
        final_decision=final_decision,
        first_fail_iter=first_fail_iter,
    )
    atomic_write_json(
        outdir / "results_summary.json",
        summary.model_dump(mode="json", by_alias=True, exclude_none=True),
        fsync=True,
    )

    marker = outdir / "first_fail_iter.txt"
    if counts.fail > 0 and first_fail_iter is not None:
        atomic_write_text(marker, f"{int(first_fail_iter)}\n", fsync=True)
    elif marker.exists():
        marker.unlink()

    return EmitResult(
        outdir=outdir,
        window_signature_hash=str(ws_hash),
        emitted=True,
    )
