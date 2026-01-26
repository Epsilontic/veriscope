from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from veriscope.core.artifacts import (
    CountsV1,
    GateRecordV1,
    ProfileV1,
    ResultsSummaryV1,
    ResultsV1,
    WindowSignatureRefV1,
    derive_final_decision,
)
from veriscope.core.jsonutil import atomic_write_json, atomic_write_pydantic_json, canonical_json_sha256

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmittedArtifactsV1:
    window_signature_path: Path
    results_path: Path
    results_summary_path: Path
    window_signature_hash: str


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


def _read_json_obj(path: Path) -> Dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return obj


def emit_hf_artifacts_v1(
    *,
    outdir: Path,
    run_id: str,
    started_ts_utc: datetime,
    ended_ts_utc: Optional[datetime],
    gate_preset: str,
    window_signature: Dict[str, Any],
    gate_records: Iterable[GateRecordV1],
    run_status: str = "success",
    runner_exit_code: Optional[int] = None,
    runner_signal: Optional[str] = None,
) -> EmittedArtifactsV1:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    window_signature_path = outdir / "window_signature.json"
    atomic_write_json(window_signature_path, window_signature)

    ws_on_disk = _read_json_obj(window_signature_path)
    ws_hash = canonical_json_sha256(ws_on_disk)
    ws_ref = WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")

    gate_records_list = list(gate_records)
    profile = ProfileV1(gate_preset=gate_preset, overrides={})

    results = ResultsV1(
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=runner_signal,
        started_ts_utc=_iso_z(started_ts_utc),
        ended_ts_utc=_iso_z(ended_ts_utc) if ended_ts_utc else None,
        gates=tuple(gate_records_list),
        metrics=tuple(),
    )

    results_path = outdir / "results.json"
    atomic_write_pydantic_json(results_path, results)

    counts = _counts_from_gates(gate_records_list)
    summary = ResultsSummaryV1(
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
    atomic_write_pydantic_json(results_summary_path, summary)

    logger.info("Wrote HF artifacts to %s (run_id=%s)", outdir, run_id)
    return EmittedArtifactsV1(
        window_signature_path=window_signature_path,
        results_path=results_path,
        results_summary_path=results_summary_path,
        window_signature_hash=ws_hash,
    )
