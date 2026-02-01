# veriscope/cli/diff.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from veriscope.cli.comparability import ComparableResult, RunMetadata, comparable_explain, load_run_metadata
from veriscope.cli.governance import load_distributed_meta
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


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def _coerce_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _distributed_meta_fields(
    meta: Optional[dict[str, object]],
) -> Optional[tuple[Optional[str], Optional[int], Optional[str]]]:
    if meta is None:
        return None
    mode = meta.get("distributed_mode")
    world_size = _coerce_int(meta.get("world_size_observed"))
    backend = meta.get("ddp_backend")
    mode_value = str(mode) if mode is not None else None
    backend_value = str(backend) if backend is not None else None
    return mode_value, world_size, backend_value


def _format_distributed_meta(meta: Optional[tuple[Optional[str], Optional[int], Optional[str]]]) -> str:
    if meta is None:
        return "mode=unknown,world_size=unknown,backend=unknown"
    mode, world_size, backend = meta
    mode_value = mode if mode is not None else "unknown"
    world_value = str(world_size) if world_size is not None else "unknown"
    backend_value = backend if backend is not None else "unknown"
    return f"mode={mode_value},world_size={world_value},backend={backend_value}"


def _distributed_mismatch_warning(
    meta_a: Optional[dict[str, object]],
    meta_b: Optional[dict[str, object]],
) -> Optional[str]:
    if meta_a is None or meta_b is None:
        return None
    fields_a = _distributed_meta_fields(meta_a)
    fields_b = _distributed_meta_fields(meta_b)
    if fields_a is None or fields_b is None:
        return None
    if fields_a == fields_b:
        return None
    return (
        "WARNING:DISTRIBUTED_METADATA_MISMATCH "
        f"A({_format_distributed_meta(fields_a)}) B({_format_distributed_meta(fields_b)})"
    )


def _format_comparable(result: ComparableResult) -> List[str]:
    lines = [
        "Comparability:",
        f"  ok: {_fmt_bool(result.ok)}",
        f"  reason: {result.reason or '-'}",
    ]
    if result.details:
        lines.append("  details:")
        for key, detail in result.details.items():
            lines.append(f"    {key}:")
            lines.append(f"      expected: {detail.get('expected')}")
            lines.append(f"      got: {detail.get('got')}")
    else:
        lines.append("  details: -")
    if result.policy:
        lines.append(f"  policy: {result.policy}")
    if result.warnings:
        lines.append(f"  warnings: {len(result.warnings)}")
    return lines


def _extract_pass_count(counts: object) -> int:
    try:
        evaluated = int(getattr(counts, "evaluated", 0) or 0)
    except Exception:
        evaluated = 0
    try:
        warn = int(getattr(counts, "warn", 0) or 0)
    except Exception:
        warn = 0
    try:
        fail = int(getattr(counts, "fail", 0) or 0)
    except Exception:
        fail = 0

    explicit = getattr(counts, "pass_", None)
    if explicit is None:
        explicit = getattr(counts, "pass_count", None)
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except Exception:
            return 0
    return max(0, evaluated - warn - fail)


def _render_header(label: str, run: RunMetadata) -> List[str]:
    manual_status = run.manual_status or "-"
    return [
        f"{label}:",
        f"  outdir: {run.outdir}",
        f"  run_id: {run.run_id}",
        f"  run_status: {run.run_status}",
        f"  schema_version: {run.schema_version}",
        f"  wrapper_exit_code: {_fmt_value(run.wrapper_exit_code)}",
        f"  runner_exit_code: {_fmt_value(run.runner_exit_code)}",
        f"  final_decision: {run.final_decision}",
        f"  effective_decision: {run.display_status}",
        f"  decision_source: {run.decision_source}",
        f"  manual_status: {manual_status}",
        f"  gate_preset: {run.gate_preset}",
        f"  window_signature_hash: {_short_hash(run.window_signature_hash)}",
        f"  partial: {str(run.partial).lower()}",
        f"  governance_present: {_fmt_bool(run.governance.present)}",
        f"  governance_ok: {_fmt_bool(run.governance.ok)}",
        f"  governance_rev: {_fmt_value(run.governance.rev)}",
        f"  governance_legacy_missing_hash: {_fmt_bool(run.governance.legacy_missing_entry_hash)}",
    ]


def _header_payload(run: RunMetadata) -> dict[str, object]:
    return {
        "outdir": str(run.outdir),
        "run_id": run.run_id,
        "run_status": run.run_status,
        "schema_version": run.schema_version,
        "wrapper_exit_code": run.wrapper_exit_code,
        "runner_exit_code": run.runner_exit_code,
        "final_decision": run.final_decision,
        "effective_decision": run.display_status,
        "decision_source": run.decision_source,
        "manual_status": run.manual_status,
        "gate_preset": run.gate_preset,
        "window_signature_hash": run.window_signature_hash,
        "partial": run.partial,
        "governance_present": run.governance.present,
        "governance_ok": run.governance.ok,
        "governance_rev": run.governance.rev,
        "governance_legacy_missing_hash": run.governance.legacy_missing_entry_hash,
    }


def _diff_flag(a: object, b: object) -> str:
    return "yes" if a != b else "no"


def diff_outdirs(
    outdir_a: Path,
    outdir_b: Path,
    *,
    allow_gate_preset_mismatch: bool = False,
    fmt: str = "text",
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

    comparable_result = comparable_explain(run_a, run_b, allow_gate_preset_mismatch=allow_gate_preset_mismatch)
    warnings = [f"RUN_A:{w}" for w in run_a.manual.warnings] + [f"RUN_B:{w}" for w in run_b.manual.warnings]
    distributed_warning = _distributed_mismatch_warning(
        load_distributed_meta(outdir_a),
        load_distributed_meta(outdir_b),
    )
    if distributed_warning:
        warnings.append(distributed_warning)

    fmt = fmt.strip().lower()
    if fmt == "json":
        payload = json.dumps(
            {
                "comparability": comparable_result.to_dict(),
                "run_a": _header_payload(run_a),
                "run_b": _header_payload(run_b),
                "partial_mode": run_a.partial or run_b.partial,
            },
            indent=2,
            sort_keys=True,
        )
        return DiffOutput(exit_code=0 if comparable_result.ok else 2, stdout=payload, stderr="\n".join(warnings))
    if not comparable_result.ok:
        token = comparable_result.reason or "INCOMPARABLE"
        details = "\n".join(_format_comparable(comparable_result))
        return DiffOutput(exit_code=2, stdout="", stderr="\n".join([f"INCOMPARABLE: {token}", details]))

    stderr = "\n".join(warnings)

    lines: List[str] = []
    lines.extend(_render_header("Run A", run_a))
    lines.append("")
    lines.extend(_render_header("Run B", run_b))
    lines.append("")
    lines.append("Comparison:")
    lines.append(f"  run_id_differs: {_diff_flag(run_a.run_id, run_b.run_id)}")
    lines.append(f"  run_status_differs: {_diff_flag(run_a.run_status, run_b.run_status)}")
    lines.append(f"  schema_version_differs: {_diff_flag(run_a.schema_version, run_b.schema_version)}")
    lines.append(f"  wrapper_exit_code_differs: {_diff_flag(run_a.wrapper_exit_code, run_b.wrapper_exit_code)}")
    lines.append(f"  runner_exit_code_differs: {_diff_flag(run_a.runner_exit_code, run_b.runner_exit_code)}")
    lines.append(f"  final_decision_differs: {_diff_flag(run_a.final_decision, run_b.final_decision)}")
    lines.append(f"  effective_decision_differs: {_diff_flag(run_a.display_status, run_b.display_status)}")
    lines.append(f"  decision_source_differs: {_diff_flag(run_a.decision_source, run_b.decision_source)}")
    lines.append(f"  manual_status_differs: {_diff_flag(run_a.manual_status or '-', run_b.manual_status or '-')}")
    lines.append(f"  gate_preset_differs: {_diff_flag(run_a.gate_preset, run_b.gate_preset)}")
    lines.append(
        f"  window_signature_hash_differs: {_diff_flag(run_a.window_signature_hash, run_b.window_signature_hash)}"
    )
    lines.append(f"  governance_present_differs: {_diff_flag(run_a.governance.present, run_b.governance.present)}")
    lines.append(f"  governance_ok_differs: {_diff_flag(run_a.governance.ok, run_b.governance.ok)}")
    lines.append(f"  governance_rev_differs: {_diff_flag(run_a.governance.rev, run_b.governance.rev)}")

    if run_a.governance.present and not run_a.governance.ok:
        lines.append("  NOTE:GOVERNANCE_INVALID run=A")
    if run_b.governance.present and not run_b.governance.ok:
        lines.append("  NOTE:GOVERNANCE_INVALID run=B")

    if run_a.partial or run_b.partial:
        lines.append("")
        lines.append("NOTE:PARTIAL_MODE decision-only")
    else:
        counts_a = run_a.summary.counts
        counts_b = run_b.summary.counts
        lines.append(f"  counts_differs: {_diff_flag(counts_a.model_dump(), counts_b.model_dump())}")
        lines.append(
            "  counts_a:"
            f" evaluated={getattr(counts_a, 'evaluated', 0)}"
            f" skip={getattr(counts_a, 'skip', 0)}"
            f" pass={_extract_pass_count(counts_a)}"
            f" warn={getattr(counts_a, 'warn', 0)}"
            f" fail={getattr(counts_a, 'fail', 0)}"
        )
        lines.append(
            "  counts_b:"
            f" evaluated={getattr(counts_b, 'evaluated', 0)}"
            f" skip={getattr(counts_b, 'skip', 0)}"
            f" pass={_extract_pass_count(counts_b)}"
            f" warn={getattr(counts_b, 'warn', 0)}"
            f" fail={getattr(counts_b, 'fail', 0)}"
        )

    lines.append("")
    lines.extend(_format_comparable(comparable_result))

    return DiffOutput(exit_code=0, stdout="\n".join(lines), stderr=stderr)
