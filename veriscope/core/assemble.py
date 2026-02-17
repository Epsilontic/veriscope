from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from veriscope.core.artifacts import (
    AuditV1,
    CountsV1,
    GateRecordV1,
    MetricsRefV1,
    ProfileV1,
    ResultsSummaryV1,
    ResultsV1,
    RunStatus,
    WindowSignatureRefV1,
    derive_final_decision,
)
from veriscope.core._coerce import (
    coerce_nonneg_int,
    coerce_optional_float,
    coerce_optional_nonneg_int,
    iso_z,
)
from veriscope.core.governance import (
    append_gate_decision,
    append_run_started,
    build_code_identity,
    build_distributed_context,
)
from veriscope.core.jsonutil import (
    atomic_write_json,
    atomic_write_pydantic_json,
    atomic_write_text,
    read_json_obj,
    window_signature_sha256,
)

_ALLOWED_RUN_STATUSES = {"success", "user_code_failure", "veriscope_failure"}
_UNIVERSAL_SCALAR_KEYS = ("loss", "lr", "step_time", "grad_norm", "update_norm")
_UNIVERSAL_DEFAULT_MIN_EVIDENCE = 1
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _read_optional_manifest(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"manifest file not found: {path}")
    return read_json_obj(path)


def _sha256_file_chunks(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class UniversalStepRecord:
    step: int
    ts_utc: Optional[datetime]
    loss: Optional[float]
    lr: Optional[float]
    step_time: Optional[float]
    grad_norm: Optional[float]
    update_norm: Optional[float]
    overflow: Optional[int]
    nan_count: Optional[int]
    metrics: dict[str, Any]


def _extract_metric(raw: Mapping[str, Any], key: str) -> Any:
    if key in raw:
        return raw.get(key)
    metrics = raw.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics.get(key)
    return None


def _normalize_overflow(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        try:
            parsed = int(value)
        except Exception:
            parsed = None
        if parsed is not None:
            return 1 if parsed > 0 else 0
    return 1 if bool(value) else 0


def _normalize_step_record(raw: Mapping[str, Any], *, line_no: int) -> UniversalStepRecord:
    step_val = raw.get("step", raw.get("iter"))
    if step_val is None:
        raise ValueError(f"line {line_no}: missing required field step")
    step = coerce_nonneg_int(step_val, field="step")

    overflow_raw = _extract_metric(raw, "overflow")
    overflow = _normalize_overflow(overflow_raw)

    metrics_raw = raw.get("metrics")
    metrics: dict[str, Any] = dict(metrics_raw) if isinstance(metrics_raw, Mapping) else {}
    return UniversalStepRecord(
        step=step,
        ts_utc=_parse_iso_utc(raw.get("ts_utc")),
        loss=coerce_optional_float(_extract_metric(raw, "loss")),
        lr=coerce_optional_float(_extract_metric(raw, "lr")),
        step_time=coerce_optional_float(_extract_metric(raw, "step_time")),
        grad_norm=coerce_optional_float(_extract_metric(raw, "grad_norm")),
        update_norm=coerce_optional_float(_extract_metric(raw, "update_norm")),
        overflow=overflow,
        nan_count=coerce_optional_nonneg_int(_extract_metric(raw, "nan_count")),
        metrics=metrics,
    )


def _read_step_records(path: Path) -> list[UniversalStepRecord]:
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    rows: list[UniversalStepRecord] = []
    prev_step: Optional[int] = None
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON ({exc.msg})") from exc
            if not isinstance(obj, Mapping):
                raise ValueError(f"line {line_no}: expected a JSON object")
            row = _normalize_step_record(obj, line_no=line_no)
            if prev_step is not None and row.step <= prev_step:
                raise ValueError(
                    f"line {line_no}: step must be strictly increasing (got {row.step} after {prev_step})"
                )
            rows.append(row)
            prev_step = row.step
    if not rows:
        raise ValueError("no valid step records were found in the JSONL log")
    return rows


def _sanitize_run_status(value: Any) -> RunStatus:
    text = str(value).strip()
    if text in _ALLOWED_RUN_STATUSES:
        return text  # type: ignore[return-value]
    return "success"


def _sanitize_runner_exit(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0 or parsed > 255:
        return None
    return parsed


def _first_fail_iter(records: Iterable[GateRecordV1]) -> Optional[int]:
    first: Optional[int] = None
    for rec in records:
        if rec.decision != "fail":
            continue
        iter_num = int(rec.iter)
        if first is None or iter_num < first:
            first = iter_num
    return first


def _counts_from_gates(gates: Iterable[GateRecordV1]) -> CountsV1:
    skip = warn = fail = pass_n = 0
    for gate in gates:
        if gate.decision == "skip":
            skip += 1
        elif gate.decision == "warn":
            warn += 1
        elif gate.decision == "fail":
            fail += 1
        elif gate.decision == "pass":
            pass_n += 1
        else:  # pragma: no cover - enum constrained upstream
            raise ValueError(f"unsupported decision={gate.decision!r}")
    return CountsV1(evaluated=pass_n + warn + fail, skip=skip, pass_=pass_n, warn=warn, fail=fail)


def _gate_from_universal_step(record: UniversalStepRecord, *, min_evidence: int) -> GateRecordV1:
    scalar_values = (
        record.loss,
        record.lr,
        record.step_time,
        record.grad_norm,
        record.update_norm,
    )
    evidence_total = sum(1 for value in scalar_values if value is not None)
    if record.overflow is not None:
        evidence_total += 1
    if record.nan_count is not None:
        evidence_total += 1

    evaluated = int(evidence_total) >= int(min_evidence)
    if not evaluated:
        decision = "skip"
        reason = "insufficient_universal_evidence"
        ok = True
        warn = False
    else:
        nan_count = int(record.nan_count or 0)
        overflow = int(record.overflow or 0)
        missing_core = any(getattr(record, key) is None for key in _UNIVERSAL_SCALAR_KEYS[:3])
        if nan_count > 0:
            decision = "fail"
            reason = "nan_count_detected"
            ok = False
            warn = False
        elif overflow > 0:
            decision = "fail"
            reason = "overflow_detected"
            ok = False
            warn = False
        elif missing_core:
            decision = "warn"
            reason = "missing_core_universal_metrics"
            ok = True
            warn = True
        else:
            decision = "pass"
            reason = "universal_metrics_stable"
            ok = True
            warn = False

    audit = AuditV1(
        evaluated=evaluated,
        reason=reason,
        policy="universal_v0",
        per_metric_tv={},
        evidence_total=int(evidence_total),
        min_evidence=int(min_evidence),
    )
    return GateRecordV1(
        iter=int(record.step),
        decision=decision,
        audit=audit,
        ok=ok,
        warn=warn,
    )


def _resolved_run_id(cli_run_id: Optional[str], manifest: Mapping[str, Any]) -> str:
    if cli_run_id and cli_run_id.strip():
        return cli_run_id.strip()
    manifest_run_id = manifest.get("run_id")
    if isinstance(manifest_run_id, str) and manifest_run_id.strip():
        return manifest_run_id.strip()
    return f"assemble_{uuid.uuid4().hex[:12]}"


def _window_signature(
    *,
    gate_preset: str,
    min_evidence: int,
    calibration_hooks: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    # Comparability stability policy:
    # - no per-run timestamps
    # - no code identity
    # This keeps window-signature hashes identical across repeated assemble runs
    # when gate settings and hooks are unchanged.
    payload: dict[str, Any] = {
        "schema_version": 1,
        "metric_pipeline": {"transport": "universal_jsonl_v1"},
        "metrics": [
            "loss",
            "lr",
            "step_time",
            "grad_norm",
            "update_norm",
            "overflow",
            "nan_count",
        ],
        "gate_controls": {
            "policy": "universal_v0",
            "gate_preset": gate_preset,
            "min_evidence": int(min_evidence),
            "fail_on_overflow": True,
            "fail_on_nan_count_gt": 0,
            "warn_on_missing_core_metrics": True,
        },
    }
    if calibration_hooks:
        payload["calibration_hooks"] = dict(calibration_hooks)
    return payload


def _unique_outdir_filename(outdir: Path, name: str) -> Path:
    base_name = Path(name).name or "universal_steps.jsonl"
    candidate = outdir / base_name
    if not candidate.exists():
        return candidate

    suffixes = "".join(Path(base_name).suffixes)
    if suffixes:
        stem = base_name[: -len(suffixes)]
    else:
        stem = base_name
    if not stem:
        stem = "universal_steps"
    for idx in range(1, 10000):
        next_candidate = outdir / f"{stem}_{idx}{suffixes}"
        if not next_candidate.exists():
            return next_candidate
    raise RuntimeError("could not allocate destination filename for metrics JSONL")


def _prepare_metrics_payload_path(*, outdir: Path, logs_path: Path) -> tuple[Path, str]:
    outdir_resolved = outdir.resolve()
    logs_resolved = logs_path.resolve()
    try:
        rel = logs_resolved.relative_to(outdir_resolved)
        payload_path = logs_resolved
    except Exception:
        payload_path = _unique_outdir_filename(outdir_resolved, logs_resolved.name)
        shutil.copyfile(logs_resolved, payload_path)
        rel = payload_path.relative_to(outdir_resolved)

    rel_path = rel.as_posix()
    if not rel_path:
        raise ValueError("metrics_ref.path cannot be empty")
    rel_parts = Path(rel_path).parts
    if Path(rel_path).is_absolute() or rel_path.startswith("/") or any(part == ".." for part in rel_parts):
        raise ValueError(f"metrics_ref.path must be a relative in-outdir path (got {rel_path!r})")
    if len(rel_path) >= 2 and rel_path[1] == ":" and rel_path[0].isalpha():
        raise ValueError(f"metrics_ref.path must not include a drive prefix (got {rel_path!r})")
    return payload_path, rel_path


@dataclass(frozen=True)
class AssembledCapsule:
    outdir: Path
    run_id: str
    window_signature_hash: str
    records: int
    run_status: RunStatus


def assemble_capsule_from_jsonl(
    *,
    outdir: Path | str,
    logs_path: Path | str,
    manifest_path: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    gate_preset: str = "universal_v0",
    min_evidence: int = _UNIVERSAL_DEFAULT_MIN_EVIDENCE,
    write_governance: bool = True,
) -> AssembledCapsule:
    outdir_path = Path(outdir).expanduser()
    outdir_path.mkdir(parents=True, exist_ok=True)
    logs_path_resolved = Path(logs_path).expanduser().resolve()
    payload_logs_path, metrics_ref_path = _prepare_metrics_payload_path(outdir=outdir_path, logs_path=logs_path_resolved)

    manifest_candidate: Optional[Path] = None
    if manifest_path is not None:
        manifest_candidate = Path(manifest_path).expanduser()
    else:
        default_manifest = logs_path_resolved.with_name("universal_manifest.json")
        if default_manifest.exists():
            manifest_candidate = default_manifest
    manifest = _read_optional_manifest(manifest_candidate)

    min_evidence_i = coerce_nonneg_int(min_evidence, field="min_evidence")
    if min_evidence_i < 1:
        raise ValueError("min_evidence must be >= 1")

    rows = _read_step_records(payload_logs_path)
    resolved_run_id = _resolved_run_id(run_id, manifest)
    resolved_gate_preset = str(gate_preset or manifest.get("gate_preset") or "universal_v0").strip() or "universal_v0"

    created_ts = _parse_iso_utc(manifest.get("created_ts_utc")) or rows[0].ts_utc or datetime.now(timezone.utc)
    ended_ts = _parse_iso_utc(manifest.get("ended_ts_utc")) or rows[-1].ts_utc or datetime.now(timezone.utc)

    run_status = _sanitize_run_status(manifest.get("run_status"))
    runner_exit_code = _sanitize_runner_exit(manifest.get("runner_exit_code"))
    if run_status == "success" and runner_exit_code is None:
        runner_exit_code = 0

    calibration_hooks = manifest.get("calibration_hooks")
    calibration_hooks_map = calibration_hooks if isinstance(calibration_hooks, Mapping) else None

    window_signature = _window_signature(
        gate_preset=resolved_gate_preset,
        min_evidence=min_evidence_i,
        calibration_hooks=calibration_hooks_map,
    )
    ws_path = outdir_path / "window_signature.json"
    atomic_write_json(ws_path, window_signature)
    ws_hash = window_signature_sha256(read_json_obj(ws_path))
    ws_ref = WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")

    profile = ProfileV1(gate_preset=resolved_gate_preset, overrides={})
    gate_records = [_gate_from_universal_step(row, min_evidence=min_evidence_i) for row in rows]

    if write_governance:
        append_run_started(
            outdir_path,
            run_id=resolved_run_id,
            outdir_path=outdir_path,
            argv=["veriscope", "assemble", str(outdir_path), "--from", str(payload_logs_path)],
            code_identity=build_code_identity(),
            window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
            entrypoint={"kind": "cli", "name": "veriscope.cli.main:assemble"},
            distributed=build_distributed_context(),
        )
        for gate in gate_records:
            audit_payload = gate.audit.model_dump(mode="json", by_alias=True, exclude_none=True)
            append_gate_decision(
                outdir_path,
                run_id=resolved_run_id,
                iter_num=int(gate.iter),
                decision=str(gate.decision),
                ok=gate.ok,
                warn=gate.warn,
                audit=audit_payload,
            )

    metrics_ref = MetricsRefV1(
        path=metrics_ref_path,
        format="universal_jsonl_v1",
        count=len(rows),
        sha256=_sha256_file_chunks(payload_logs_path),
    )
    results = ResultsV1(
        schema_version=1,
        run_id=resolved_run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=None,
        started_ts_utc=created_ts,
        ended_ts_utc=ended_ts,
        gates=gate_records,
        metrics=[],
        metrics_ref=metrics_ref,
    )
    atomic_write_pydantic_json(outdir_path / "results.json", results, by_alias=True, exclude_none=True, fsync=True)

    counts = _counts_from_gates(gate_records)
    first_fail_iter = _first_fail_iter(gate_records)
    summary = ResultsSummaryV1(
        schema_version=1,
        run_id=resolved_run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=None,
        started_ts_utc=created_ts,
        ended_ts_utc=ended_ts,
        counts=counts,
        final_decision=derive_final_decision(counts),
        first_fail_iter=first_fail_iter,
    )
    atomic_write_pydantic_json(
        outdir_path / "results_summary.json",
        summary,
        by_alias=True,
        exclude_none=True,
        fsync=True,
    )

    first_fail_marker = outdir_path / "first_fail_iter.txt"
    if first_fail_iter is None:
        if first_fail_marker.exists():
            first_fail_marker.unlink()
    else:
        atomic_write_text(first_fail_marker, f"{int(first_fail_iter)}\n", fsync=True)

    run_config_resolved = {
        "schema_version": 1,
        "ts_utc": iso_z(),
        "run": {"kind": "assemble", "run_id": resolved_run_id, "outdir": str(outdir_path)},
        "code_identity": build_code_identity(),
        "assembly": {
            "source_jsonl": str(payload_logs_path),
            "manifest_path": str(manifest_candidate) if manifest_candidate is not None else None,
            "records": len(rows),
            "gate_preset": resolved_gate_preset,
            "min_evidence": min_evidence_i,
            "write_governance": bool(write_governance),
            "metrics_ref": {
                "path": metrics_ref.path,
                "format": metrics_ref.format,
                "count": metrics_ref.count,
                "sha256": metrics_ref.sha256,
            },
        },
        "window_signature_ref": {"hash": ws_hash, "path": "window_signature.json"},
        "run_status": run_status,
        "wrapper_exit_code": 0,
        "runner_exit_code": runner_exit_code,
        "runner_signal": None,
        "started_ts_utc": iso_z(created_ts),
        "ended_ts_utc": iso_z(ended_ts),
    }
    atomic_write_json(outdir_path / "run_config_resolved.json", run_config_resolved)

    return AssembledCapsule(
        outdir=outdir_path,
        run_id=resolved_run_id,
        window_signature_hash=ws_hash,
        records=len(rows),
        run_status=run_status,
    )
