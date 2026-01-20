# veriscope/cli/report.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _artifact_rows(outdir: Path) -> List[Tuple[str, bool]]:
    names = (
        "run_config_resolved.json",
        "window_signature.json",
        "results.json",
        "results_summary.json",
    )
    return [(name, (outdir / name).exists()) for name in names]


def render_report_md(outdir: Path, *, fmt: str = "md") -> str:
    """
    Renders a report. Despite the name, supports fmt="md" and fmt="text".
    """
    outdir = Path(outdir)

    v = validate_outdir(outdir)
    if not v.ok:
        raise ValueError(f"Cannot report: {v.message}")

    # Validation passed; re-load for reporting.
    res = ResultsV1.model_validate_json((outdir / "results.json").read_text("utf-8"))
    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))

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
    ws_hash = v.window_signature_hash or res.window_signature_ref.hash

    c = summ.counts
    evaluated = _safe_int(getattr(c, "evaluated", None))
    skip = _safe_int(getattr(c, "skip", None))
    warn = _safe_int(getattr(c, "warn", None))
    fail = _safe_int(getattr(c, "fail", None))
    passed = _extract_pass_count(c)
    total = max(0, evaluated + skip)

    if fmt == "text":
        lines: List[str] = []
        lines.append(f"Veriscope Report: {outdir}")
        lines.append("")
        lines.append(f"Run ID: {res.run_id}")
        lines.append(f"Status: {str(summ.run_status).upper()}")
        lines.append(f"Wrapper Exit: {wrapper_exit if wrapper_exit is not None else '-'}")
        lines.append(f"Runner Exit: {runner_exit if runner_exit is not None else '-'}")
        lines.append(f"Gate Preset: {res.profile.gate_preset}")
        lines.append(f"Started: {_fmt_dt(res.started_ts_utc)}")
        lines.append(f"Ended: {_fmt_dt(res.ended_ts_utc)}")
        lines.append(f"Window Signature: {ws_hash}")
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
    lines.append(f"| Run ID | `{res.run_id}` |")
    lines.append(f"| Status | **{str(summ.run_status).upper()}** |")
    lines.append(f"| Wrapper Exit | {wrapper_exit if wrapper_exit is not None else '-'} |")
    lines.append(f"| Runner Exit | {runner_exit if runner_exit is not None else '-'} |")
    lines.append(f"| Gate Preset | `{res.profile.gate_preset}` |")
    lines.append(f"| Started | {_fmt_dt(res.started_ts_utc)} |")
    lines.append(f"| Ended | {_fmt_dt(res.ended_ts_utc)} |")
    lines.append(f"| Window Signature | `{ws_hash}` |")
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
