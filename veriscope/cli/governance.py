# veriscope/cli/governance.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError as PydanticValidationError
from pydantic_core import ValidationError as PydanticCoreValidationError

from veriscope.core.artifacts import ManualJudgementV1
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


def _warn_invalid_log_line(line_no: int, message: str) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_LOG_INVALID line={line_no} {message}"


def _warn_ts_decrease(line_no: int) -> str:
    return f"WARNING:MANUAL_JUDGEMENT_TS_DECREASE line={line_no}"


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
    except Exception as exc:  # pragma: no cover - defensive
        return None, _warn_invalid_log_line(line_no, f"error={exc}")
    return ManualJudgementLogEntry(judgement=judgement, rev=rev_int, line_no=line_no), None


def read_manual_judgement_log(path: Path) -> ManualJudgementLogResult:
    warnings: List[str] = []
    last_entry: Optional[ManualJudgementLogEntry] = None
    last_ts = None

    if not path.exists():
        return ManualJudgementLogResult(judgement=None, rev=None, warnings=tuple(warnings))

    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry, warning = _parse_log_line(line, line_no=idx)
        if warning:
            warnings.append(warning)
            continue
        if entry is None:
            continue
        if last_ts is not None and entry.judgement.ts_utc < last_ts:
            warnings.append(_warn_ts_decrease(entry.line_no))
        last_entry = entry
        last_ts = entry.judgement.ts_utc

    if last_entry is None:
        return ManualJudgementLogResult(judgement=None, rev=None, warnings=tuple(warnings))

    return ManualJudgementLogResult(
        judgement=last_entry.judgement,
        rev=last_entry.rev,
        warnings=tuple(warnings),
    )


def resolve_manual_judgement(outdir: Path, *, prefer_jsonl: bool = True) -> ManualJudgementEffective:
    outdir = Path(outdir)
    log_path = outdir / "manual_judgement.jsonl"
    log_result: Optional[ManualJudgementLogResult] = None
    if prefer_jsonl and log_path.exists():
        log_result = read_manual_judgement_log(log_path)
        if log_result.judgement is not None:
            return ManualJudgementEffective(
                judgement=log_result.judgement,
                source="jsonl",
                warnings=log_result.warnings,
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

    warnings = tuple()
    if log_result is not None:
        warnings = log_result.warnings
    return ManualJudgementEffective(judgement=None, source="-", warnings=warnings)


def append_manual_judgement_log(outdir: Path, judgement: ManualJudgementV1) -> tuple[Path, tuple[str, ...]]:
    outdir = Path(outdir)
    log_path = outdir / "manual_judgement.jsonl"

    last_rev = 0
    last_ts = None
    warnings: List[str] = []
    if log_path.exists():
        result = read_manual_judgement_log(log_path)
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
