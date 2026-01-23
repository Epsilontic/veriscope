# veriscope/cli/override.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from veriscope.cli.governance import append_manual_judgement_log
from veriscope.core.artifacts import ManualJudgementV1, ResultsV1
from veriscope.core.jsonutil import atomic_write_pydantic_json


def _parse_ts_utc(ts_utc: Optional[str]) -> datetime:
    if ts_utc is None:
        return datetime.now(timezone.utc)

    raw = ts_utc.strip()
    if not raw:
        raise ValueError("ts_utc cannot be empty")

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ts_utc format: {ts_utc!r}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def write_manual_judgement(
    outdir: Path,
    *,
    status: str,
    reason: str,
    reviewer: Optional[str] = None,
    ts_utc: Optional[str] = None,
    force: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_path = outdir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found. Run results must exist before override.")
    results = ResultsV1.model_validate_json(results_path.read_text(encoding="utf-8"))

    path = outdir / "manual_judgement.json"
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")

    judgement = ManualJudgementV1(
        run_id=results.run_id,
        status=status,
        reason=reason,
        reviewer=reviewer,
        ts_utc=_parse_ts_utc(ts_utc),
    )
    atomic_write_pydantic_json(path, judgement, by_alias=True, exclude_none=True)
    _, warnings = append_manual_judgement_log(outdir, judgement)
    return path, warnings
