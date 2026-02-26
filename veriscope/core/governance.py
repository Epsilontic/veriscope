# veriscope/core/governance.py
from __future__ import annotations

import os
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional

from veriscope.core.artifacts import AuditV1, DistributedRunConfigV1, derive_gate_decision
from veriscope.core.ddp import ddp_is_active, ddp_rank, ddp_world_size
from veriscope.core.jsonutil import atomic_append_jsonl, canonical_json_sha256, sanitize_json_value


GOVERNANCE_LOG_SCHEMA_VERSION = 1
MANUAL_GOVERNANCE_EVENT_TYPES = (
    "manual_judgement_set",
    "manual_judgement_cleared",
    "artifact_note",
    "recompute_summary",
)
RUN_GOVERNANCE_EVENT_TYPES = (
    "run_started_v1",
    "capsule_opened_v1",
    "run_overrides_applied_v1",
    "gate_decision_v1",
)
GOVERNANCE_EVENT_TYPES = MANUAL_GOVERNANCE_EVENT_TYPES + RUN_GOVERNANCE_EVENT_TYPES


@dataclass(frozen=True)
class GovernanceLogEntry:
    schema_version: int
    rev: int
    ts_utc: str
    actor: Optional[str]
    event: str
    payload: dict[str, Any]
    prev_hash: Optional[str]
    entry_hash: Optional[str]
    line_no: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class GovernanceLogResult:
    entry: Optional[GovernanceLogEntry]
    rev: Optional[int]
    entry_hash: Optional[str]
    warnings: tuple[str, ...]
    event_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernanceLogValidation:
    ok: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class GovernanceStatus:
    present: bool
    ok: bool
    rev: Optional[int]
    entry_hash: Optional[str]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    legacy_missing_entry_hash: bool
    event_counts: dict[str, int]


def _warn_governance_invalid(line_no: int, message: str) -> str:
    return f"WARNING:GOVERNANCE_LOG_INVALID line={line_no} {message}"


def _warn_governance_ts_decrease(line_no: int) -> str:
    return f"WARNING:GOVERNANCE_LOG_TS_DECREASE line={line_no}"


def _warn_governance_rev_nonmonotone(line_no: int, rev: int, prev_rev: int) -> str:
    return f"WARNING:GOVERNANCE_LOG_REV_NONMONOTONE line={line_no} rev={rev} prev_rev={prev_rev}"


def _warn_governance_hash_mismatch(line_no: int) -> str:
    return f"WARNING:GOVERNANCE_LOG_HASH_MISMATCH line={line_no}"


def _warn_governance_prev_hash_mismatch(line_no: int) -> str:
    return f"WARNING:GOVERNANCE_LOG_PREV_HASH_MISMATCH line={line_no}"


def _warn_governance_entry_hash_missing(line_no: int) -> str:
    return f"WARNING:GOVERNANCE_LOG_ENTRY_HASH_MISSING line={line_no}"


def _warn_governance_event_dual_present(line_no: int) -> str:
    return f"WARNING:GOVERNANCE_LOG_EVENT_DUAL_PRESENT line={line_no}"


def _warn_governance_payload_required_fields(
    line_no: int, *, event: str, missing: list[str], invalid: list[str]
) -> str:
    missing_part = ",".join(sorted(missing)) if missing else "-"
    invalid_part = ",".join(sorted(invalid)) if invalid else "-"
    return _warn_governance_invalid(
        line_no,
        f"payload_required_fields event={event} missing={missing_part} invalid={invalid_part}",
    )


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_required_payload_fields(event: str, payload: dict[str, Any], *, line_no: int) -> Optional[str]:
    """
    Validate event-specific governance payload keys required by docs/contract_v1.md.
    """
    if event == "run_started_v1":
        required: dict[str, type] = {
            "run_id": str,
            "outdir": str,
            "argv": list,
            "code_identity": dict,
            "window_signature_ref": dict,
            "entrypoint": dict,
        }
    elif event == "run_overrides_applied_v1":
        required = {
            "run_id": str,
            "outdir": str,
            "overrides": dict,
            "profile": dict,
            "entrypoint": dict,
        }
    elif event == "gate_decision_v1":
        required = {
            "run_id": str,
            "outdir": str,
            "iter": int,
            "decision": str,
            "audit": dict,
        }
    else:
        return None

    missing: list[str] = []
    invalid: list[str] = []

    for key, expected_type in required.items():
        if key not in payload:
            missing.append(key)
            continue
        value = payload.get(key)
        if expected_type is str:
            if not _is_non_empty_str(value):
                invalid.append(key)
            continue
        if expected_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                invalid.append(key)
            continue
        if not isinstance(value, expected_type):
            invalid.append(key)

    if event == "run_started_v1":
        ws_ref = payload.get("window_signature_ref")
        if isinstance(ws_ref, dict):
            if not _is_non_empty_str(ws_ref.get("hash")):
                invalid.append("window_signature_ref.hash")
            if not _is_non_empty_str(ws_ref.get("path")):
                invalid.append("window_signature_ref.path")
        entrypoint = payload.get("entrypoint")
        if isinstance(entrypoint, dict):
            if not _is_non_empty_str(entrypoint.get("kind")):
                invalid.append("entrypoint.kind")
            if not _is_non_empty_str(entrypoint.get("name")):
                invalid.append("entrypoint.name")
        distributed = payload.get("distributed")
        if distributed is not None:
            if not isinstance(distributed, dict):
                invalid.append("distributed")
            else:
                try:
                    DistributedRunConfigV1.model_validate(distributed)
                except Exception:
                    invalid.append("distributed")

    if event == "run_overrides_applied_v1":
        if "argv" in payload and not isinstance(payload.get("argv"), list):
            invalid.append("argv")
        entrypoint = payload.get("entrypoint")
        if isinstance(entrypoint, dict):
            if not _is_non_empty_str(entrypoint.get("kind")):
                invalid.append("entrypoint.kind")
            if not _is_non_empty_str(entrypoint.get("name")):
                invalid.append("entrypoint.name")
        profile = payload.get("profile")
        if isinstance(profile, dict) and not _is_non_empty_str(profile.get("gate_preset")):
            invalid.append("profile.gate_preset")

    if event == "gate_decision_v1":
        decision = payload.get("decision")
        decision_norm: Optional[str] = None
        if isinstance(decision, str):
            decision_norm = decision.strip().lower()
            if decision_norm not in {"pass", "warn", "fail", "skip"}:
                invalid.append("decision")
        if "ok" in payload and not isinstance(payload.get("ok"), bool):
            invalid.append("ok")
        if "warn" in payload and not isinstance(payload.get("warn"), bool):
            invalid.append("warn")
        audit_obj = payload.get("audit")
        if isinstance(audit_obj, dict):
            if "per_metric_tv" not in audit_obj:
                invalid.append("audit.per_metric_tv")
            try:
                audit_model = AuditV1.model_validate(audit_obj)
            except Exception:
                invalid.append("audit")
            else:
                if decision_norm == "skip" and bool(audit_model.evaluated):
                    invalid.append("audit.evaluated")
                if decision_norm in {"pass", "warn", "fail"} and (not bool(audit_model.evaluated)):
                    invalid.append("audit.evaluated")

    if missing or invalid:
        missing = sorted(set(missing))
        invalid = sorted(set(invalid))
        return _warn_governance_payload_required_fields(
            line_no,
            event=event,
            missing=missing,
            invalid=invalid,
        )
    return None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pkg_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("veriscope")
    except Exception:
        return "unknown"


def _git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        sha = (result.stdout or "").strip()
        return sha or None
    except Exception:
        return None


def _parse_ts_utc_z(ts_utc: str) -> datetime:
    raw = ts_utc.strip()
    if not raw:
        raise ValueError("ts_utc cannot be empty")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def governance_entry_hash(entry: dict[str, Any]) -> str:
    payload = dict(entry)
    payload.pop("entry_hash", None)
    payload.pop("line_no", None)
    return canonical_json_sha256(payload)


def _parse_governance_log_line(line: str, *, line_no: int) -> tuple[Optional[GovernanceLogEntry], Optional[str]]:
    raw = line.strip()
    if not raw:
        return None, None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, _warn_governance_invalid(line_no, f"json_error={exc.msg}")
    if not isinstance(obj, dict):
        return None, _warn_governance_invalid(line_no, "not_object")

    try:
        schema_version = int(obj.get("schema_version"))
    except Exception:
        return None, _warn_governance_invalid(line_no, "schema_version_missing_or_invalid")
    if schema_version != GOVERNANCE_LOG_SCHEMA_VERSION:
        return None, _warn_governance_invalid(line_no, f"schema_version={schema_version}")

    try:
        rev_int = int(obj.get("rev"))
    except Exception:
        return None, _warn_governance_invalid(line_no, "rev_missing_or_invalid")

    event = obj.get("event") or obj.get("event_type")
    if event not in GOVERNANCE_EVENT_TYPES:
        return None, _warn_governance_invalid(line_no, f"event_invalid={event}")

    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return None, _warn_governance_invalid(line_no, "payload_missing_or_invalid")
    payload_validation = _validate_required_payload_fields(str(event), payload, line_no=line_no)

    ts_utc = obj.get("ts_utc")
    if not isinstance(ts_utc, str):
        return None, _warn_governance_invalid(line_no, "ts_utc_missing_or_invalid")
    try:
        _ = _parse_ts_utc_z(ts_utc)
    except Exception as exc:
        return None, _warn_governance_invalid(line_no, f"ts_utc_invalid={exc}")

    actor = obj.get("actor")
    if actor is not None and not isinstance(actor, str):
        return None, _warn_governance_invalid(line_no, "actor_invalid")

    prev_hash = obj.get("prev_hash")
    if prev_hash is not None and not isinstance(prev_hash, str):
        return None, _warn_governance_invalid(line_no, "prev_hash_invalid")

    entry_hash = obj.get("entry_hash")
    if entry_hash is not None and not isinstance(entry_hash, str):
        return None, _warn_governance_invalid(line_no, "entry_hash_invalid")

    return (
        GovernanceLogEntry(
            schema_version=schema_version,
            rev=rev_int,
            ts_utc=ts_utc,
            actor=actor,
            event=str(event),
            payload=payload,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            line_no=line_no,
            raw=obj,
        ),
        payload_validation,
    )


def read_governance_log(path: Path) -> GovernanceLogResult:
    warnings: List[str] = []
    last_entry: Optional[GovernanceLogEntry] = None
    last_ts: Optional[datetime] = None
    last_rev: Optional[int] = None
    last_hash: Optional[str] = None
    event_counts: dict[str, int] = {}

    if not path.exists():
        return GovernanceLogResult(entry=None, rev=None, entry_hash=None, warnings=tuple(warnings), event_counts={})

    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry, warning = _parse_governance_log_line(line, line_no=idx)
        if warning:
            warnings.append(warning)
        if entry is None:
            continue

        if last_rev is None:
            if entry.rev != 1:
                warnings.append(_warn_governance_rev_nonmonotone(entry.line_no, entry.rev, 0))
        elif entry.rev != last_rev + 1:
            warnings.append(_warn_governance_rev_nonmonotone(entry.line_no, entry.rev, last_rev))

        try:
            ts = _parse_ts_utc_z(entry.ts_utc)
            if last_ts is not None and ts < last_ts:
                warnings.append(_warn_governance_ts_decrease(entry.line_no))
            last_ts = ts
        except Exception:
            pass

        if "event" in entry.raw and "event_type" in entry.raw:
            warnings.append(_warn_governance_event_dual_present(entry.line_no))

        expected_hash = governance_entry_hash(entry.raw)

        if entry.entry_hash is None:
            warnings.append(_warn_governance_entry_hash_missing(entry.line_no))
            effective_hash = expected_hash
        else:
            if entry.entry_hash != expected_hash:
                warnings.append(_warn_governance_hash_mismatch(entry.line_no))
            effective_hash = entry.entry_hash

        if last_hash is None:
            if entry.prev_hash is not None:
                warnings.append(_warn_governance_prev_hash_mismatch(entry.line_no))
        elif entry.prev_hash != last_hash:
            warnings.append(_warn_governance_prev_hash_mismatch(entry.line_no))

        event_counts[entry.event] = event_counts.get(entry.event, 0) + 1
        last_entry = entry
        last_rev = entry.rev
        last_hash = effective_hash

    if last_entry is None:
        return GovernanceLogResult(entry=None, rev=None, entry_hash=None, warnings=tuple(warnings), event_counts={})
    return GovernanceLogResult(
        entry=last_entry,
        rev=last_entry.rev,
        entry_hash=last_hash,
        warnings=tuple(warnings),
        event_counts=event_counts,
    )


def validate_governance_log(path: Path, *, allow_legacy_governance: bool = False) -> GovernanceLogValidation:
    result = read_governance_log(path)
    errors: List[str] = []
    for warning in result.warnings:
        if "GOVERNANCE_LOG_HASH_MISMATCH" in warning or "GOVERNANCE_LOG_PREV_HASH_MISMATCH" in warning:
            errors.append(warning)
        if "GOVERNANCE_LOG_INVALID" in warning or "payload_required_fields" in warning:
            errors.append(warning)
        if "GOVERNANCE_LOG_ENTRY_HASH_MISSING" in warning and not allow_legacy_governance:
            errors.append(warning)
        if "GOVERNANCE_LOG_REV_NONMONOTONE" in warning:
            errors.append(warning)
    ok = not errors
    return GovernanceLogValidation(ok=ok, warnings=result.warnings, errors=tuple(errors))


def get_governance_status(outdir: Path, *, allow_legacy_governance: bool) -> GovernanceStatus:
    outdir = Path(outdir)
    gov_path = outdir / "governance_log.jsonl"
    if not gov_path.exists():
        return GovernanceStatus(
            present=False,
            ok=True,
            rev=None,
            entry_hash=None,
            errors=tuple(),
            warnings=tuple(),
            legacy_missing_entry_hash=False,
            event_counts={},
        )

    result = read_governance_log(gov_path)
    validation = validate_governance_log(gov_path, allow_legacy_governance=allow_legacy_governance)
    legacy_missing_entry_hash = any("GOVERNANCE_LOG_ENTRY_HASH_MISSING" in w for w in result.warnings)
    return GovernanceStatus(
        present=True,
        ok=validation.ok,
        rev=result.rev,
        entry_hash=result.entry_hash,
        errors=validation.errors,
        warnings=result.warnings,
        legacy_missing_entry_hash=legacy_missing_entry_hash,
        event_counts=result.event_counts,
    )


def _append_governance_entry(outdir: Path, entry: dict[str, Any]) -> tuple[Path, tuple[str, ...]]:
    outdir = Path(outdir)
    log_path = outdir / "governance_log.jsonl"

    last_rev = 0
    last_ts: Optional[datetime] = None
    last_hash: Optional[str] = None
    warnings: List[str] = []
    if log_path.exists():
        validation = validate_governance_log(log_path)
        if not validation.ok:
            raise RuntimeError(f"Cannot append to invalid governance_log.jsonl: {validation.errors}")
        result = read_governance_log(log_path)
        if result.rev is not None:
            last_rev = result.rev
        if result.entry is not None:
            try:
                last_ts = _parse_ts_utc_z(result.entry.ts_utc)
            except Exception:
                last_ts = None
        last_hash = result.entry_hash
        warnings.extend(result.warnings)

    has_event = "event" in entry and entry.get("event") is not None
    has_event_type = "event_type" in entry and entry.get("event_type") is not None
    if has_event and has_event_type:
        raise ValueError("governance entry must not include both event and event_type")
    if not has_event and not has_event_type:
        raise ValueError("governance entry must include event or event_type")

    event_name = entry.get("event") if has_event else entry.get("event_type")
    payload_obj = entry.get("payload")
    if not isinstance(payload_obj, dict):
        raise ValueError("payload missing or invalid for governance entry append")
    if isinstance(event_name, str):
        payload_warning = _validate_required_payload_fields(event_name, payload_obj, line_no=0)
        if payload_warning is not None:
            raise ValueError(payload_warning.replace("WARNING:GOVERNANCE_LOG_INVALID line=0 ", ""))

    ts_utc = entry.get("ts_utc")
    if not isinstance(ts_utc, str):
        raise ValueError("ts_utc missing or invalid for governance entry append")
    ts_dt = _parse_ts_utc_z(ts_utc)
    if last_ts is not None and ts_dt < last_ts:
        warnings.append(_warn_governance_ts_decrease(0))

    entry_payload = sanitize_json_value(dict(entry))
    entry_payload["schema_version"] = GOVERNANCE_LOG_SCHEMA_VERSION
    entry_payload["rev"] = last_rev + 1
    entry_payload["prev_hash"] = last_hash
    entry_payload["entry_hash"] = governance_entry_hash(entry_payload)

    atomic_append_jsonl(log_path, entry_payload, fsync=True)
    return log_path, tuple(warnings)


def build_code_identity(*, git_sha: Optional[str] = None) -> dict[str, Any]:
    code_identity: dict[str, Any] = {"package_version": _pkg_version()}
    if git_sha:
        code_identity["git_sha"] = str(git_sha)
    return code_identity


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_distributed_context() -> dict[str, Any]:
    world_size = ddp_world_size()
    rank = ddp_rank()
    ddp_active = ddp_is_active()
    ddp_backend = None
    if ddp_active:
        try:
            import torch.distributed as dist  # local import

            if getattr(dist, "is_available", lambda: False)() and dist.is_initialized():
                ddp_backend = dist.get_backend()
        except Exception:
            ddp_backend = None

    if world_size <= 1:
        distributed_mode = "single_process"
    elif ddp_active and world_size > 1:
        distributed_mode = "ddp_wrapped"
    else:
        distributed_mode = "replicated_single_chief_emit"

    local_rank = None
    if distributed_mode != "single_process":
        local_rank = _parse_optional_int(os.environ.get("LOCAL_RANK"))

    ddp_wrapped = distributed_mode == "ddp_wrapped"
    return {
        "distributed_mode": distributed_mode,
        "world_size_observed": int(world_size),
        "backend": ddp_backend,
        "rank": int(rank),
        "local_rank": local_rank,
        "ddp_wrapped": ddp_wrapped,
        # Legacy aliases kept for backward compatibility in older consumers.
        "rank_observed": int(rank),
        "local_rank_observed": local_rank,
        "ddp_backend": ddp_backend,
        "ddp_active": bool(ddp_active),
    }


def append_governance_log(
    outdir: Path,
    *,
    event_type: str,
    payload: dict[str, Any],
    ts_utc: str,
    actor: Optional[str] = None,
) -> tuple[Path, tuple[str, ...]]:
    if event_type not in MANUAL_GOVERNANCE_EVENT_TYPES:
        raise ValueError(f"Unsupported governance event_type={event_type!r}")

    entry: dict[str, Any] = {
        "ts_utc": ts_utc,
        "actor": actor,
        "event": event_type,
        "payload": payload,
    }
    return _append_governance_entry(outdir, entry)


def append_run_started(
    outdir: Path,
    *,
    run_id: str,
    outdir_path: Path,
    argv: Iterable[str],
    code_identity: dict[str, Any],
    window_signature_ref: dict[str, Any],
    entrypoint: dict[str, Any],
    distributed: Optional[dict[str, Any]] = None,
    ts_utc: Optional[str] = None,
    actor: Optional[str] = None,
) -> tuple[Path, tuple[str, ...]]:
    payload = {
        "run_id": str(run_id),
        "outdir": str(outdir_path),
        "argv": list(argv),
        "code_identity": dict(code_identity),
        "window_signature_ref": dict(window_signature_ref),
        "entrypoint": dict(entrypoint),
    }
    if distributed is not None:
        payload["distributed"] = dict(distributed)
    entry = {
        "ts_utc": ts_utc or _now_utc_iso(),
        "actor": actor,
        "event": "run_started_v1",
        "payload": payload,
    }
    return _append_governance_entry(outdir, entry)


def append_overrides(
    outdir: Path,
    *,
    run_id: str,
    overrides: dict[str, Any],
    profile: dict[str, Any],
    entrypoint: dict[str, Any],
    argv: Optional[Iterable[str]] = None,
    ts_utc: Optional[str] = None,
    actor: Optional[str] = None,
) -> tuple[Path, tuple[str, ...]]:
    payload = {
        "run_id": str(run_id),
        "outdir": str(outdir),
        "overrides": dict(overrides),
        "profile": dict(profile),
        "entrypoint": dict(entrypoint),
    }
    if argv is not None:
        payload["argv"] = list(argv)
    entry = {
        "ts_utc": ts_utc or _now_utc_iso(),
        "actor": actor,
        "event": "run_overrides_applied_v1",
        "payload": payload,
    }
    return _append_governance_entry(outdir, entry)


def append_gate_decision(
    outdir: Path,
    *,
    run_id: str,
    iter_num: int,
    decision: str,
    ok: Optional[bool],
    warn: Optional[bool],
    audit: dict[str, Any],
    ts_utc: Optional[str] = None,
    actor: Optional[str] = None,
) -> tuple[Path, tuple[str, ...]]:
    decision_norm = str(decision).strip().lower()
    if decision_norm not in {"pass", "warn", "fail", "skip"}:
        raise ValueError(f"Invalid gate decision={decision!r}")

    audit_payload = dict(audit)
    if decision_norm == "skip":
        ok_value = bool(ok) if ok is not None else True
        warn_value = False
    else:
        if ok is None:
            raise ValueError("gate_decision_v1 payload requires ok for non-skip decisions")
        ok_value = bool(ok)
        warn_value = bool(warn)
        if not ok_value:
            warn_value = False

    evaluated = bool(audit_payload.get("evaluated", decision_norm != "skip"))
    expected = derive_gate_decision(evaluated=evaluated, ok=ok_value, warn=warn_value)
    if decision_norm != expected:
        raise ValueError(
            "Inconsistent gate decision payload: "
            f"decision={decision_norm!r} expected={expected!r} evaluated={evaluated!r} ok={ok_value!r} warn={warn_value!r}"
        )

    entry = {
        "ts_utc": ts_utc or _now_utc_iso(),
        "actor": actor,
        "event": "gate_decision_v1",
        "payload": {
            "run_id": str(run_id),
            "outdir": str(outdir),
            "iter": int(iter_num),
            "decision": decision_norm,
            "ok": ok_value,
            "warn": warn_value,
            "audit": audit_payload,
        },
    }
    return _append_governance_entry(outdir, entry)
