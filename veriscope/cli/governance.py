# veriscope/cli/governance.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as PydanticCoreValidationError

from veriscope.core.artifacts import ManualJudgementV1
from veriscope.core.jsonutil import canonical_dumps, canonical_json_sha256


@dataclass(frozen=True)
class ManualJudgementLogEntry:
    judgement: ManualJudgementV1
    rev: int
    line_no: int


@dataclass(frozen=True)
class ManualJudgementLogResult:
    judgement: Optional[ManualJudgementV1]
    rev: Optional[int]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ManualJudgementEffective:
    judgement: Optional[ManualJudgementV1]
    source: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GovernanceLogEntry:
    schema_version: int
    rev: int
    ts_utc: str
    actor: Optional[str]
    event_type: str
    payload: dict[str, Any]
    prev_hash: Optional[str]
    entry_hash: Optional[str]
    line_no: int


@dataclass(frozen=True)
class GovernanceLogResult:
    entry: Optional[GovernanceLogEntry]
    rev: Optional[int]
    entry_hash: Optional[str]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GovernanceLogValidation:
    ok: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class DisplayStatus:
    status: str
    source: str
    manual: Optional[ManualJudgementV1]
    warnings: tuple[str, ...]


GOVERNANCE_LOG_SCHEMA_VERSION = 1
GOVERNANCE_EVENT_TYPES = (
    "manual_judgement_set",
    "manual_judgement_cleared",
    "artifact_note",
    "recompute_summary",
)


def _warn_invalid_log_line(line_no: int, message: str) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_LOG_INVALID line={line_no} {message}"


def _warn_ts_decrease(line_no: int) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_TS_DECREASE line={line_no}"


def _warn_manual_run_id_mismatch(line_no: int, *, expected: str, got: str) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_RUN_ID_MISMATCH line={line_no} expected={expected} got={got}"


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


def _parse_log_line(line: str, *, line_no: int) -> tuple[Optional[ManualJudgementLogEntry], Optional[str]]:
    raw = line.strip()
    if not raw:
        return None, None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, _warn_invalid_log_line(line_no, f"json_error={exc.msg}")
    if not isinstance(obj, dict):
        return None, _warn_invalid_log_line(line_no, "not_object")
    rev = obj.get("rev")
    try:
        rev_int = int(rev)
    except Exception:
        return None, _warn_invalid_log_line(line_no, "rev_missing_or_invalid")
    try:
        judgement = ManualJudgementV1.model_validate(obj)
    except (PydanticValidationError, PydanticCoreValidationError) as exc:
        return None, _warn_invalid_log_line(line_no, f"schema_error={exc}")
    except Exception as exc:  # pragma: no cover
        return None, _warn_invalid_log_line(line_no, f"error={exc}")
    return ManualJudgementLogEntry(judgement=judgement, rev=rev_int, line_no=line_no), None


def read_manual_judgement_log(path: Path, *, run_id: Optional[str] = None) -> ManualJudgementLogResult:
    warnings: List[str] = []
    last_match: Optional[ManualJudgementLogEntry] = None
    last_ts_match: Optional[datetime] = None
    last_ts_any: Optional[datetime] = None
    max_rev: Optional[int] = None  # Max rev observed across valid entries (used for append sequencing).

    if not path.exists():
        return ManualJudgementLogResult(judgement=None, rev=None, warnings=tuple(warnings))

    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry, warning = _parse_log_line(line, line_no=idx)
        if warning:
            warnings.append(warning)
            continue
        if entry is None:
            continue

        max_rev = entry.rev if max_rev is None else max(max_rev, entry.rev)

        ts = entry.judgement.ts_utc
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)

        if run_id is None:
            if last_ts_any is not None and ts < last_ts_any:
                warnings.append(_warn_ts_decrease(entry.line_no))
            last_ts_any = ts
            last_match = entry
            continue

        if entry.judgement.run_id != run_id:
            warnings.append(_warn_manual_run_id_mismatch(entry.line_no, expected=run_id, got=entry.judgement.run_id))
            continue
        if last_ts_match is not None and ts < last_ts_match:
            warnings.append(_warn_ts_decrease(entry.line_no))
        last_ts_match = ts
        last_match = entry

    if last_match is None:
        return ManualJudgementLogResult(judgement=None, rev=max_rev, warnings=tuple(warnings))

    return ManualJudgementLogResult(judgement=last_match.judgement, rev=max_rev, warnings=tuple(warnings))


def resolve_manual_judgement(outdir: Path, *, prefer_jsonl: bool = True) -> ManualJudgementEffective:
    outdir = Path(outdir)
    log_path = outdir / "manual_judgement.jsonl"
    log_result: Optional[ManualJudgementLogResult] = None
    if prefer_jsonl and log_path.exists():
        log_result = read_manual_judgement_log(log_path, run_id=None)
        if log_result.judgement is not None:
            return ManualJudgementEffective(
                judgement=log_result.judgement, source="jsonl", warnings=log_result.warnings
            )

    json_path = outdir / "manual_judgement.json"
    if json_path.exists():
        try:
            judgement = ManualJudgementV1.model_validate_json(json_path.read_text(encoding="utf-8"))
            warnings = log_result.warnings if log_result is not None else tuple()
            return ManualJudgementEffective(judgement=judgement, source="json", warnings=warnings)
        except Exception as exc:
            warning = _warn_invalid_log_line(0, f"manual_judgement_json_invalid={exc}")
            warnings = (warning,)
            if log_result is not None:
                warnings = (*log_result.warnings, warning)
            return ManualJudgementEffective(judgement=None, source="json", warnings=warnings)

    warnings = log_result.warnings if log_result is not None else tuple()
    return ManualJudgementEffective(judgement=None, source="-", warnings=warnings)


def resolve_manual_overlay(outdir: Path, run_id: str, *, prefer_jsonl: bool = True) -> ManualJudgementEffective:
    outdir = Path(outdir)
    warnings: List[str] = []

    if prefer_jsonl:
        log_path = outdir / "manual_judgement.jsonl"
        if log_path.exists():
            log_result = read_manual_judgement_log(log_path, run_id=run_id)
            warnings.extend(log_result.warnings)
            if log_result.judgement is not None:
                return ManualJudgementEffective(
                    judgement=log_result.judgement, source="jsonl", warnings=tuple(warnings)
                )

    json_path = outdir / "manual_judgement.json"
    if json_path.exists():
        try:
            judgement = ManualJudgementV1.model_validate_json(json_path.read_text(encoding="utf-8"))
            if judgement.run_id != run_id:
                warnings.append(_warn_manual_run_id_mismatch(0, expected=run_id, got=judgement.run_id))
                return ManualJudgementEffective(judgement=None, source="json", warnings=tuple(warnings))
            return ManualJudgementEffective(judgement=judgement, source="json", warnings=tuple(warnings))
        except Exception as exc:
            warnings.append(_warn_invalid_log_line(0, f"manual_judgement_json_invalid={exc}"))
            return ManualJudgementEffective(judgement=None, source="json", warnings=tuple(warnings))

    return ManualJudgementEffective(judgement=None, source="-", warnings=tuple(warnings))


def resolve_display_status(automated_decision: str, manual: ManualJudgementEffective) -> DisplayStatus:
    if manual.judgement is not None:
        return DisplayStatus(
            status=manual.judgement.status, source="manual", manual=manual.judgement, warnings=manual.warnings
        )
    return DisplayStatus(status=automated_decision, source="automated", manual=None, warnings=manual.warnings)


def resolve_effective_status(
    outdir: Path,
    *,
    run_id: str,
    automated_decision: str,
    prefer_jsonl: bool = True,
) -> DisplayStatus:
    manual = resolve_manual_overlay(outdir, run_id, prefer_jsonl=prefer_jsonl)
    return resolve_display_status(automated_decision, manual)


def append_manual_judgement_log(outdir: Path, judgement: ManualJudgementV1) -> tuple[Path, tuple[str, ...]]:
    outdir = Path(outdir)
    log_path = outdir / "manual_judgement.jsonl"

    last_rev = 0
    last_ts = None
    warnings: List[str] = []
    if log_path.exists():
        result = read_manual_judgement_log(log_path, run_id=None)
        if result.rev is not None:
            last_rev = result.rev
        last_ts = result.judgement.ts_utc if result.judgement is not None else None
        warnings.extend(result.warnings)

    if last_ts is not None and judgement.ts_utc < last_ts:
        warnings.append(_warn_ts_decrease(0))

    entry = judgement.model_dump(mode="json", by_alias=True)
    entry["rev"] = last_rev + 1
    line = canonical_dumps(entry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

    return log_path, tuple(warnings)


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


def _hash_governance_entry(entry: dict[str, Any]) -> str:
    payload = {k: v for k, v in entry.items() if k not in {"entry_hash", "line_no"}}
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

    event_type = obj.get("event_type")
    if event_type not in GOVERNANCE_EVENT_TYPES:
        return None, _warn_governance_invalid(line_no, f"event_type_invalid={event_type}")

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
            event_type=event_type,
            payload=payload,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
            line_no=line_no,
        ),
        None,
    )


def read_governance_log(path: Path) -> GovernanceLogResult:
    warnings: List[str] = []
    last_entry: Optional[GovernanceLogEntry] = None
    last_ts: Optional[datetime] = None
    last_rev: Optional[int] = None
    last_hash: Optional[str] = None

    if not path.exists():
        return GovernanceLogResult(entry=None, rev=None, entry_hash=None, warnings=tuple(warnings))

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

        expected_hash = _hash_governance_entry(
            {
                "schema_version": entry.schema_version,
                "rev": entry.rev,
                "ts_utc": entry.ts_utc,
                "actor": entry.actor,
                "event_type": entry.event_type,
                "payload": entry.payload,
                "prev_hash": entry.prev_hash,
            }
        )

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

        last_entry = entry
        last_rev = entry.rev
        last_hash = effective_hash

    if last_entry is None:
        return GovernanceLogResult(entry=None, rev=None, entry_hash=None, warnings=tuple(warnings))
    return GovernanceLogResult(entry=last_entry, rev=last_entry.rev, entry_hash=last_hash, warnings=tuple(warnings))


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


def append_governance_log(
    outdir: Path,
    *,
    event_type: str,
    payload: dict[str, Any],
    ts_utc: str,
    actor: Optional[str] = None,
) -> tuple[Path, tuple[str, ...]]:
    if event_type not in GOVERNANCE_EVENT_TYPES:
        raise ValueError(f"Unsupported governance event_type={event_type!r}")

    outdir = Path(outdir)
    log_path = outdir / "governance_log.jsonl"

    last_rev = 0
    last_ts = None
    last_hash = None
    warnings: List[str] = []
    if log_path.exists():
        # Legacy governance logs (missing entry_hash) are invalid for appends.
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

    ts_dt = _parse_ts_utc_z(ts_utc)
    if last_ts is not None and ts_dt < last_ts:
        warnings.append(_warn_governance_ts_decrease(0))

    entry: dict[str, Any] = {
        "schema_version": GOVERNANCE_LOG_SCHEMA_VERSION,
        "rev": last_rev + 1,
        "ts_utc": ts_utc,
        "actor": actor,
        "event_type": event_type,
        "payload": payload,
        "prev_hash": last_hash,
    }
    entry_hash = _hash_governance_entry(entry)
    entry["entry_hash"] = entry_hash

    line = canonical_dumps(entry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

    return log_path, tuple(warnings)
