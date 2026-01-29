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
from veriscope.core.governance import (
    GOVERNANCE_EVENT_TYPES,
    GOVERNANCE_LOG_SCHEMA_VERSION,
    MANUAL_GOVERNANCE_EVENT_TYPES,
    GovernanceStatus,
    append_governance_log,
    append_gate_decision,
    append_overrides,
    append_run_started,
    build_code_identity,
    get_governance_status,
    read_governance_log,
    validate_governance_log,
)
from veriscope.core.jsonutil import canonical_dumps


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
class DisplayStatus:
    status: str
    source: str
    manual: Optional[ManualJudgementV1]
    warnings: tuple[str, ...]


def _warn_invalid_log_line(line_no: int, message: str) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_LOG_INVALID line={line_no} {message}"


def _warn_ts_decrease(line_no: int) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_TS_DECREASE line={line_no}"


def _warn_manual_run_id_mismatch(line_no: int, *, expected: str, got: str) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_RUN_ID_MISMATCH line={line_no} expected={expected} got={got}"


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
