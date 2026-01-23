# veriscope/cli/diff.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from veriscope.cli.comparability import RunMetadata, comparable, load_run_metadata
from veriscope.cli.validate import validate_outdir


@dataclass(frozen=True)
class DiffOutput:
    exit_code: int
    stdout: str
    stderr: str


def _fmt_value(value: Optional[object]) -> str:
    return "-" if value is None else str(value)


def _short_hash(value: str, *, width: int = 12) -> str:
    return value[:width] if value else "-"


def _render_header(label: str, run: RunMetadata) -> List[str]:
    manual_status = run.manual_status or "-"
    return [
        f"{label}:",
        f"  outdir: {run.outdir}",
        f"  run_id: {run.run_id}",
        f"  run_status: {run.run_status}",
        f"  wrapper_exit_code: {_fmt_value(run.wrapper_exit_code)}",
        f"  runner_exit_code: {_fmt_value(run.runner_exit_code)}",
        f"  final_decision: {run.final_decision}",
        f"  manual_status: {manual_status}",
        f"  window_signature_hash: {_short_hash(run.window_signature_hash)}",
        f"  partial: {str(run.partial).lower()}",
    ]


def _diff_flag(a: object, b: object) -> str:
    return "yes" if a != b else "no"


def diff_outdirs(
    outdir_a: Path,
    outdir_b: Path,
    *,
    allow_gate_preset_mismatch: bool = False,
) -> DiffOutput:
    outdir_a = Path(outdir_a)
    outdir_b = Path(outdir_b)

    v_a = validate_outdir(outdir_a, allow_partial=True)
    if not v_a.ok:
        return DiffOutput(exit_code=2, stdout="", stderr=f"INVALID: {v_a.message}")
    v_b = validate_outdir(outdir_b, allow_partial=True)
    if not v_b.ok:
        return DiffOutput(exit_code=2, stdout="", stderr=f"INVALID: {v_b.message}")

    run_a = load_run_metadata(outdir_a, v_a, prefer_jsonl=True)
    run_b = load_run_metadata(outdir_b, v_b, prefer_jsonl=True)

    ok, reason = comparable(run_a, run_b, allow_gate_preset_mismatch=allow_gate_preset_mismatch)
    if not ok:
        token = reason or "INCOMPARABLE"
        return DiffOutput(exit_code=2, stdout="", stderr=f"INCOMPARABLE: {token}")

    warnings = [f"RUN_A:{w}" for w in run_a.manual.warnings] + [f"RUN_B:{w}" for w in run_b.manual.warnings]
    stderr = "\n".join(warnings)

    lines: List[str] = []
    lines.extend(_render_header("Run A", run_a))
    lines.append("")
    lines.extend(_render_header("Run B", run_b))
    lines.append("")
    lines.append("Comparison:")
    lines.append(f"  run_id_differs: {_diff_flag(run_a.run_id, run_b.run_id)}")
    lines.append(f"  run_status_differs: {_diff_flag(run_a.run_status, run_b.run_status)}")
    lines.append(f"  wrapper_exit_code_differs: {_diff_flag(run_a.wrapper_exit_code, run_b.wrapper_exit_code)}")
    lines.append(f"  runner_exit_code_differs: {_diff_flag(run_a.runner_exit_code, run_b.runner_exit_code)}")
    lines.append(f"  final_decision_differs: {_diff_flag(run_a.final_decision, run_b.final_decision)}")
    lines.append(f"  manual_status_differs: {_diff_flag(run_a.manual_status or '-', run_b.manual_status or '-')}")
    lines.append(
        f"  window_signature_hash_differs: {_diff_flag(run_a.window_signature_hash, run_b.window_signature_hash)}"
    )

    if run_a.partial or run_b.partial:
        lines.append("")
        lines.append("NOTE:PARTIAL_COMPARISON_LIMITED")

    return DiffOutput(exit_code=0, stdout="\n".join(lines), stderr=stderr)
