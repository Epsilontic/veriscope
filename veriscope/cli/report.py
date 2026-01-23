# veriscope/cli/report.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from veriscope.cli.comparability import comparable, load_run_metadata
from veriscope.cli.governance import get_governance_status, resolve_effective_status
from veriscope.cli.validate import validate_outdir
from veriscope.core.artifacts import ResultsSummaryV1, ResultsV1


def _read_json_obj(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_dt(dt: Any) -> str:
    if dt is None:
        return ""
    try:
        if hasattr(dt, "isoformat"):
            return dt.isoformat().replace("+00:00", "Z")
        return str(dt)
    except Exception:
        return str(dt)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _extract_pass_count(counts: Any) -> int:
    """
    Be defensive against schema drift:
      - Prefer explicit pass_/pass_count if present
      - Otherwise derive pass = evaluated - warn - fail (never negative)
    """
    evaluated = _safe_int(getattr(counts, "evaluated", None))
    warn = _safe_int(getattr(counts, "warn", None))
    fail = _safe_int(getattr(counts, "fail", None))

    explicit = getattr(counts, "pass_", None)
    if explicit is None:
        explicit = getattr(counts, "pass_count", None)

    if explicit is not None:
        return max(0, _safe_int(explicit))

    return max(0, evaluated - warn - fail)


def _escape_md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.splitlines())
    return text.replace("|", r"\|")


def _artifact_rows(outdir: Path) -> List[Tuple[str, bool]]:
    names = (
        "run_config_resolved.json",
        "window_signature.json",
        "results.json",
        "results_summary.json",
        "manual_judgement.json",
        "manual_judgement.jsonl",
        "governance_log.jsonl",
    )
    return [(name, (outdir / name).exists()) for name in names]


def render_report_md(outdir: Path, *, fmt: str = "md") -> str:
    """
    Renders a report. Despite the name, supports fmt="md" and fmt="text".
    """
    outdir = Path(outdir)

    # Report is best-effort; allow partial artifact capsules by default.
    v = validate_outdir(outdir, allow_partial=True)
    if not v.ok:
        raise ValueError(f"Cannot report: {v.message}")

    # Validation passed; re-load for reporting.
    res_path = outdir / "results.json"
    res = ResultsV1.model_validate_json(res_path.read_text("utf-8")) if res_path.exists() else None
    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))
    run_id = res.run_id if res is not None else summ.run_id
    manual_status = resolve_effective_status(
        outdir,
        run_id=run_id,
        automated_decision=str(summ.final_decision),
        prefer_jsonl=True,
    )
    gov_status = get_governance_status(outdir, allow_legacy_governance=True)

    run_cfg_path = outdir / "run_config_resolved.json"
    run_cfg: Optional[Dict[str, Any]] = None
    if run_cfg_path.exists():
        try:
            obj = _read_json_obj(run_cfg_path)
            run_cfg = (
                obj
                if isinstance(obj, dict)
                else {"error": {"message": "run_config_resolved.json is not a JSON object"}}
            )
        except Exception:
            run_cfg = {"error": {"message": "Could not parse run_config_resolved.json"}}

    wrapper_exit = run_cfg.get("wrapper_exit_code") if isinstance(run_cfg, dict) else None
    runner_exit = getattr(summ, "runner_exit_code", None)
    ws_hash = v.window_signature_hash or (res.window_signature_ref.hash if res else summ.window_signature_ref.hash)

    c = summ.counts
    evaluated = _safe_int(getattr(c, "evaluated", None))
    skip = _safe_int(getattr(c, "skip", None))
    warn = _safe_int(getattr(c, "warn", None))
    fail = _safe_int(getattr(c, "fail", None))
    passed = _extract_pass_count(c)
    total = max(0, evaluated + skip)
    final_status = (
        f"MANUAL {manual_status.status.upper()}"
        if manual_status.source == "manual"
        else str(summ.final_decision).upper()
    )
    gate_preset = res.profile.gate_preset if res is not None else summ.profile.gate_preset
    started_ts = res.started_ts_utc if res is not None else summ.started_ts_utc
    ended_ts = res.ended_ts_utc if res is not None else summ.ended_ts_utc

    if fmt == "text":
        lines: List[str] = []
        lines.append(f"Veriscope Report: {outdir}")
        lines.append("")
        lines.append(f"Run ID: {run_id}")
        lines.append(f"Status: {str(summ.run_status).upper()}")
        lines.append(f"Final Status: {final_status}")
        lines.append(f"Wrapper Exit: {wrapper_exit if wrapper_exit is not None else '-'}")
        lines.append(f"Runner Exit: {runner_exit if runner_exit is not None else '-'}")
        lines.append(f"Gate Preset: {gate_preset}")
        lines.append(f"Started: {_fmt_dt(started_ts)}")
        lines.append(f"Ended: {_fmt_dt(ended_ts)}")
        lines.append(f"Window Signature: {ws_hash}")
        lines.append("")
        if manual_status.manual is not None:
            lines.append("MANUAL JUDGEMENT:")
            lines.append(f"  Status: {manual_status.manual.status.upper()}")
            lines.append(f"  Reason: {manual_status.manual.reason}")
            lines.append(f"  Reviewer: {manual_status.manual.reviewer or '-'}")
            lines.append(f"  Timestamp: {_fmt_dt(manual_status.manual.ts_utc)}")
            lines.append("")
        lines.append("GOVERNANCE:")
        lines.append(f"  Log: {'YES' if gov_status.present else 'NO'}")
        if gov_status.present:
            lines.append(f"  Valid: {'YES' if gov_status.ok else 'NO'}")
            lines.append(f"  Last Rev: {gov_status.rev if gov_status.rev is not None else '-'}")
            lines.append(f"  Last Hash: {_short_hash(gov_status.entry_hash or '')}")
            lines.append(f"  Legacy (missing entry_hash): {'YES' if gov_status.legacy_missing_entry_hash else 'NO'}")
            if not gov_status.ok and gov_status.errors:
                lines.append("  ⚠ Governance log invalid:")
                for err in gov_status.errors:
                    lines.append(f"    {err}")
            if gov_status.warnings:
                lines.append(f"  Warnings: {len(gov_status.warnings)}")
        lines.append("")
        lines.append("Gate Summary:")
        lines.append(f"  Total: {total}")
        lines.append(f"  Evaluated: {evaluated}")
        lines.append(f"  PASS: {passed}")
        lines.append(f"  SKIP: {skip}")
        lines.append(f"  WARN: {warn}")
        lines.append(f"  FAIL: {fail}")
        lines.append("")
        lines.append("Artifacts:")
        for name, exists in _artifact_rows(outdir):
            lines.append(f"  {name}: {'YES' if exists else 'NO'}")
        lines.append("")

        if isinstance(run_cfg, dict) and isinstance(run_cfg.get("error"), dict):
            err = run_cfg["error"]
            lines.append("Error Capsule:")
            lines.append(f"  Type: {err.get('type', 'UnknownError')}")
            lines.append(f"  Message: {err.get('message', 'No message provided')}")
            lines.append("")

        return "\n".join(lines)

    # Default: Markdown
    lines = []
    lines.append(f"# Veriscope Report: `{outdir}`")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Run ID | `{run_id}` |")
    lines.append(f"| Status | **{str(summ.run_status).upper()}** |")
    lines.append(f"| Final Status | **{final_status}** |")
    lines.append(f"| Wrapper Exit | {wrapper_exit if wrapper_exit is not None else '-'} |")
    lines.append(f"| Runner Exit | {runner_exit if runner_exit is not None else '-'} |")
    lines.append(f"| Gate Preset | `{gate_preset}` |")
    lines.append(f"| Started | {_fmt_dt(started_ts)} |")
    lines.append(f"| Ended | {_fmt_dt(ended_ts)} |")
    lines.append(f"| Window Signature | `{ws_hash}` |")
    lines.append("")

    if manual_status.manual is not None:
        lines.append("## ⚠ Manual Judgement")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Status | **{manual_status.manual.status.upper()}** |")
        lines.append(f"| Reason | {_escape_md_cell(manual_status.manual.reason)} |")
        reviewer = _escape_md_cell(manual_status.manual.reviewer)
        lines.append(f"| Reviewer | {reviewer or '-'} |")
        lines.append(f"| Timestamp | {_fmt_dt(manual_status.manual.ts_utc)} |")
        lines.append("")

    lines.append("## Governance")
    lines.append("")
    if gov_status.present and not gov_status.ok and gov_status.errors:
        banner = _escape_md_cell(gov_status.errors[0])
        lines.append(f"> ⚠ **Governance log invalid:** `{banner}`")
        lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Log | {'YES' if gov_status.present else 'NO'} |")
    if gov_status.present:
        lines.append(f"| Valid | {'YES' if gov_status.ok else 'NO'} |")
        lines.append(f"| Last Rev | {gov_status.rev if gov_status.rev is not None else '-'} |")
        lines.append(f"| Last Hash | `{_short_hash(gov_status.entry_hash or '')}` |")
        lines.append(f"| Legacy (missing entry_hash) | {'YES' if gov_status.legacy_missing_entry_hash else 'NO'} |")
        if gov_status.warnings:
            lines.append(f"| Warnings | {len(gov_status.warnings)} |")
    lines.append("")
    if gov_status.present and not gov_status.ok and gov_status.errors:
        lines.append("**Errors:**")
        lines.append("")
        for err in gov_status.errors:
            lines.append(f"- `{_escape_md_cell(err)}`")
        lines.append("")

    lines.append("## Gate Summary")
    lines.append("")
    lines.append("| Total | Evaluated | PASS | SKIP | WARN | FAIL |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| **{total}** | {evaluated} | **{passed}** | {skip} | {warn} | {fail} |")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("| Artifact | Found |")
    lines.append("|---|---|")
    for name, exists in _artifact_rows(outdir):
        lines.append(f"| `{name}` | {'✅' if exists else '❌'} |")
    lines.append("")

    if isinstance(run_cfg, dict) and isinstance(run_cfg.get("error"), dict):
        err = run_cfg["error"]
        lines.append("## Error Capsule")
        lines.append("")
        lines.append(f"- **Type**: `{err.get('type', 'UnknownError')}`")
        lines.append(f"- **Message**: {err.get('message', 'No message provided')}")
        lines.append("")

    return "\n".join(lines)


@dataclass(frozen=True)
class ReportCompareOutput:
    exit_code: int
    stdout: str
    stderr: str


def _short_hash(value: str, *, width: int = 12) -> str:
    return value[:width] if value else "-"


def _fmt_cell(value: Any) -> str:
    return "-" if value is None else str(value)


def render_report_compare(
    outdirs: List[Path],
    *,
    fmt: str = "text",
    allow_incompatible: bool = False,
    allow_gate_preset_mismatch: bool = False,
) -> ReportCompareOutput:
    if not outdirs:
        return ReportCompareOutput(exit_code=2, stdout="", stderr="INVALID: no OUTDIRs provided")

    validations = []
    for outdir in outdirs:
        v = validate_outdir(Path(outdir), allow_partial=True)
        if not v.ok:
            return ReportCompareOutput(exit_code=2, stdout="", stderr=f"INVALID: {v.message}")
        validations.append(v)

    runs = [load_run_metadata(Path(outdir), v, prefer_jsonl=True) for outdir, v in zip(outdirs, validations)]
    warnings = [f"run={run.run_id}:{w}" for run in runs for w in run.manual.warnings]

    reference = runs[0]
    incompatible: List[tuple[int, str]] = []
    for idx, run in enumerate(runs[1:], start=1):
        ok, reason = comparable(reference, run, allow_gate_preset_mismatch=allow_gate_preset_mismatch)
        if not ok:
            incompatible.append((idx, reason or "INCOMPARABLE"))

    if incompatible and not allow_incompatible:
        reason = incompatible[0][1]
        stderr = "\n".join([*warnings, f"INCOMPARABLE: {reason}"]) if warnings else f"INCOMPARABLE: {reason}"
        return ReportCompareOutput(exit_code=2, stdout="", stderr=stderr)

    header = [
        "run_id",
        "run_status",
        "wrapper_exit_code",
        "runner_exit_code",
        "final_decision",
        "manual_effective_status",
        "window_signature_hash",
        "partial",
    ]
    if allow_incompatible:
        header.extend(["comparable", "reason"])

    rows: List[List[str]] = []
    for idx, run in enumerate(runs):
        manual_status = run.manual_status or "-"
        row = [
            run.run_id,
            run.run_status,
            _fmt_cell(run.wrapper_exit_code),
            _fmt_cell(run.runner_exit_code),
            run.final_decision,
            manual_status,
            _short_hash(run.window_signature_hash),
            str(run.partial).lower(),
        ]
        if allow_incompatible:
            reason = "-"
            comparable_flag = "yes"
            if idx != 0:
                match = next((r for r in incompatible if r[0] == idx), None)
                if match:
                    comparable_flag = "no"
                    reason = match[1]
            row.extend([comparable_flag, reason])
        rows.append(row)

    lines: List[str] = []
    if fmt == "md":
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

    group_ok = not incompatible
    reason = "OK" if group_ok else incompatible[0][1]
    lines.append("")
    lines.append(f"Comparable group: {'yes' if group_ok else 'no'}, reason={reason}")

    stderr = "\n".join(warnings)
    return ReportCompareOutput(exit_code=0, stdout="\n".join(lines), stderr=stderr)
