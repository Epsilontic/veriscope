# veriscope/runners/hf/emit_artifacts.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
from veriscope.core.ddp import ddp_is_chief
from veriscope.core.jsonutil import (
    atomic_write_json,
    atomic_write_pydantic_json,
    canonical_dumps,
    canonical_json_sha256,
    read_json_obj,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmittedArtifactsV1:
    window_signature_path: Path
    results_path: Path
    results_summary_path: Path
    window_signature_hash: str
    emitted: bool = True


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _counts_from_gates(gates: Iterable[GateRecordV1]) -> CountsV1:
    skip = warn = fail = pass_n = 0
    for record in gates:
        if record.decision == "skip":
            skip += 1
        elif record.decision == "warn":
            warn += 1
        elif record.decision == "fail":
            fail += 1
        elif record.decision == "pass":
            pass_n += 1
        else:
            raise ValueError(f"Unexpected gate decision={record.decision!r} at iter={record.iter!r}")
    evaluated = pass_n + warn + fail
    return CountsV1(evaluated=evaluated, skip=skip, pass_=pass_n, warn=warn, fail=fail)


def emit_hf_artifacts_v1(
    *,
    outdir: Path,
    run_id: str,
    started_ts_utc: datetime,
    ended_ts_utc: Optional[datetime],
    gate_preset: str,
    window_signature: Dict[str, Any],
    gate_records: Iterable[GateRecordV1],
    metrics: Optional[Iterable[MetricRecordV1]] = None,
    run_status: str = "success",
    runner_exit_code: Optional[int] = None,
    runner_signal: Optional[str] = None,
) -> EmittedArtifactsV1:
    outdir = Path(outdir)
    if not ddp_is_chief():
        logger.info("Skipping HF artifact emission on non-chief rank for %s", outdir)
        return EmittedArtifactsV1(
            window_signature_path=outdir / "window_signature.json",
            results_path=outdir / "results.json",
            results_summary_path=outdir / "results_summary.json",
            window_signature_hash=canonical_json_sha256(window_signature),
            emitted=False,
        )
    outdir.mkdir(parents=True, exist_ok=True)

    window_signature_path = outdir / "window_signature.json"
    if window_signature_path.exists():
        ws_on_disk = read_json_obj(window_signature_path)
        ws_hash = canonical_json_sha256(ws_on_disk)
        ws_on_disk_canonical = canonical_dumps(ws_on_disk)
        ws_in_memory_canonical = canonical_dumps(window_signature)
        if ws_on_disk_canonical != ws_in_memory_canonical:
            ws_in_memory_hash = canonical_json_sha256(window_signature)
            raise ValueError(
                "window_signature.json already exists and differs from the provided window_signature; "
                f"on-disk hash={ws_hash} in-memory hash={ws_in_memory_hash}. "
                "Use a new outdir, remove the existing capsule, or rerun with --force if the wrapper handles cleanup."
            )
    else:
        atomic_write_json(window_signature_path, window_signature)
        ws_on_disk = read_json_obj(window_signature_path)
        ws_hash = canonical_json_sha256(ws_on_disk)
    ws_ref = WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")

    gate_records_list = list(gate_records)
    metrics_list = list(metrics or [])
    profile = ProfileV1(gate_preset=gate_preset, overrides={})

    results = ResultsV1(
        schema_version=1,
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=runner_signal,
        started_ts_utc=_iso_z(started_ts_utc),
        ended_ts_utc=_iso_z(ended_ts_utc) if ended_ts_utc else None,
        gates=gate_records_list,
        metrics=metrics_list,
    )

    results_path = outdir / "results.json"
    atomic_write_pydantic_json(results_path, results, by_alias=True, exclude_none=True, fsync=True)

    counts = _counts_from_gates(gate_records_list)
    summary = ResultsSummaryV1(
        schema_version=1,
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=runner_signal,
        started_ts_utc=_iso_z(started_ts_utc),
        ended_ts_utc=_iso_z(ended_ts_utc) if ended_ts_utc else None,
        counts=counts,
        final_decision=derive_final_decision(counts),
    )

    results_summary_path = outdir / "results_summary.json"
    atomic_write_pydantic_json(results_summary_path, summary, by_alias=True, exclude_none=True, fsync=True)

    logger.info("Wrote HF artifacts to %s (run_id=%s)", outdir, run_id)
    return EmittedArtifactsV1(
        window_signature_path=window_signature_path,
        results_path=results_path,
        results_summary_path=results_summary_path,
        window_signature_hash=ws_hash,
    )
