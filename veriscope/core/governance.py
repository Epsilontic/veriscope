# veriscope/core/governance.py
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional

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
        None,
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
            continue
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
        if "GOVERNANCE_LOG_INVALID" in warning:
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
    ts_utc: Optional[str] = None,
    actor: Optional[str] = None,
) -> tuple[Path, tuple[str, ...]]:
    entry = {
        "ts_utc": ts_utc or _now_utc_iso(),
        "actor": actor,
        "event": "run_started_v1",
        "payload": {
            "run_id": str(run_id),
            "outdir": str(outdir_path),
            "argv": list(argv),
            "code_identity": dict(code_identity),
            "window_signature_ref": dict(window_signature_ref),
            "entrypoint": dict(entrypoint),
        },
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
    entry = {
        "ts_utc": ts_utc or _now_utc_iso(),
        "actor": actor,
        "event": "gate_decision_v1",
        "payload": {
            "run_id": str(run_id),
            "outdir": str(outdir),
            "iter": int(iter_num),
            "decision": str(decision),
            "ok": ok,
            "warn": warn,
            "audit": dict(audit),
        },
    }
    return _append_governance_entry(outdir, entry)
