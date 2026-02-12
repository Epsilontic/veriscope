# veriscope/core/pilot_calibration.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, NoReturn, Optional, Tuple, Union

from .jsonutil import window_signature_sha256


class CalibrationError(Exception):
    def __init__(self, token: str, message: str) -> None:
        super().__init__(f"{token}: {message}")
        self.token = token
        self.message = message


def _die(token: str, message: str) -> NoReturn:
    raise CalibrationError(token, message)


def _load_json(path: Path, token: str) -> Dict[str, Any]:
    if not path.exists():
        _die(token, f"Missing {path.name} at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(token, f"Invalid JSON in {path.name}: {exc}")
    if not isinstance(data, dict):
        _die(token, f"{path.name} must be a JSON object")
    return data


def _extract_summary_window_hash(summary: Dict[str, Any], *, label: str) -> str:
    window_signature_ref = summary.get("window_signature_ref")
    if not isinstance(window_signature_ref, dict):
        _die(
            "MISSING_WINDOW_SIGNATURE_REF",
            f"{label} results_summary.json missing window_signature_ref",
        )
    hash_value = window_signature_ref.get("hash")
    if not isinstance(hash_value, str) or not hash_value.strip():
        _die(
            "MISSING_WINDOW_SIGNATURE_REF_HASH",
            f"{label} results_summary.json missing window_signature_ref.hash",
        )
    return hash_value.strip()


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_gate_window(
    window_signature: Dict[str, Any],
    control_run_config: Optional[Dict[str, Any]],
    injected_run_config: Optional[Dict[str, Any]],
) -> Tuple[int, str]:
    gate_controls = window_signature.get("gate_controls")
    if isinstance(gate_controls, dict) and "gate_window" in gate_controls:
        return int(gate_controls["gate_window"]), "window_signature.gate_controls.gate_window"
    for run_config, label in (
        (control_run_config, "control"),
        (injected_run_config, "injected"),
    ):
        if not isinstance(run_config, dict):
            continue
        resolved_gate_cfg = run_config.get("resolved_gate_cfg")
        if isinstance(resolved_gate_cfg, dict) and "gate_window" in resolved_gate_cfg:
            return int(resolved_gate_cfg["gate_window"]), f"{label}.run_config_resolved.resolved_gate_cfg.gate_window"
        if "gate_window" in run_config:
            return int(run_config["gate_window"]), f"{label}.run_config_resolved.gate_window"
    _die("MISSING_GATE_WINDOW", "gate_window not found in window_signature.json or run_config_resolved.json")
    raise CalibrationError("MISSING_GATE_WINDOW", "unreachable")


def _resolve_warmup(
    window_signature: Dict[str, Any],
    control_run_config: Optional[Dict[str, Any]],
    injected_run_config: Optional[Dict[str, Any]],
) -> Tuple[Optional[int], str]:
    for run_config, label in (
        (control_run_config, "control"),
        (injected_run_config, "injected"),
    ):
        if not isinstance(run_config, dict):
            continue
        resolved_gate_cfg = run_config.get("resolved_gate_cfg")
        if isinstance(resolved_gate_cfg, dict) and "gate_warmup" in resolved_gate_cfg:
            return int(resolved_gate_cfg["gate_warmup"]), f"{label}.run_config_resolved.resolved_gate_cfg.gate_warmup"
        if "gate_warmup" in run_config:
            return int(run_config["gate_warmup"]), f"{label}.run_config_resolved.gate_warmup"
    gate_controls = window_signature.get("gate_controls")
    if isinstance(gate_controls, dict) and "gate_warmup" in gate_controls:
        return int(gate_controls["gate_warmup"]), "window_signature.gate_controls.gate_warmup"
    return None, "missing"


ResultsGateStatus = Union[List[Dict[str, Any]], Literal["missing", "invalid"]]


def _load_results_gates(results_path: Path) -> ResultsGateStatus:
    if not results_path.exists():
        return "missing"
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid"
    if not isinstance(data, dict):
        return "invalid"
    gates = data.get("gates")
    if isinstance(gates, list):
        return [g for g in gates if isinstance(g, dict)]
    return "invalid"


def _normalize_decision(value: Any) -> Optional[str]:
    raw = str(value).strip().lower()
    if raw in {"pass", "warn", "fail"}:
        return raw
    return None


def _build_gate_event(
    iter_value: Any,
    decision_value: Any,
    audit_value: Any,
    *,
    source: str,
    context: str,
) -> Dict[str, Any]:
    if iter_value is None:
        _die("MISSING_GATE_ITER", f"{source}: payload.iter missing ({context})")
    try:
        iter_i = int(iter_value)
    except (TypeError, ValueError):
        _die("INVALID_GATE_ITER", f"{source}: payload.iter invalid ({context})")
    decision = _normalize_decision(decision_value)
    if decision is None:
        _die("INVALID_GATE_DECISION", f"{source}: decision invalid ({context})")
    audit = audit_value if isinstance(audit_value, dict) else {}
    reason = audit.get("reason") or audit.get("base_reason") or audit.get("change_reason") or None
    worst_dw = audit.get("worst_DW")
    if worst_dw is None:
        worst_dw = audit.get("worst_dw")
    eps_eff = audit.get("eps_eff")
    if eps_eff is None:
        eps_eff = audit.get("eps_eff_value")
    return {
        "iter": iter_i,
        "decision": decision,
        "audit": audit,
        "reason": reason,
        "worst_DW": _parse_optional_float(worst_dw),
        "eps_eff": _parse_optional_float(eps_eff),
        "evidence_total": _parse_optional_int(audit.get("evidence_total")),
        "min_evidence": _parse_optional_int(audit.get("min_evidence")),
    }


def _read_governance_gate_events(gov_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    events: List[Dict[str, Any]] = []
    for line_no, line in enumerate(gov_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            warnings.append(f"WARNING:governance_log_invalid line={line_no} json_error={exc.msg}")
            continue
        if not isinstance(obj, dict):
            warnings.append(f"WARNING:governance_log_invalid line={line_no} not_object")
            continue
        event_type = obj.get("event") or obj.get("event_type")
        if event_type != "gate_decision_v1":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            warnings.append(f"WARNING:gate_decision_payload_invalid line={line_no}")
            continue
        event = _build_gate_event(
            payload.get("iter"),
            payload.get("decision"),
            payload.get("audit"),
            source="governance_log.jsonl",
            context=f"line={line_no}",
        )
        events.append(event)
    events.sort(key=lambda item: item["iter"])
    return events, warnings


def _read_results_gate_events(results_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    events: List[Dict[str, Any]] = []
    gates = _load_results_gates(results_path)
    if gates == "missing":
        warnings.append("WARNING:results.json_missing")
        return events, warnings
    if gates == "invalid":
        warnings.append("WARNING:results.json_invalid")
        return events, warnings
    for idx, gate in enumerate(gates):
        event = _build_gate_event(
            gate.get("iter"),
            gate.get("decision"),
            gate.get("audit"),
            source="results.json",
            context=f"index={idx}",
        )
        events.append(event)
    events.sort(key=lambda item: item["iter"])
    return events, warnings


def _read_gate_events(outdir: Path) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    gov_path = outdir / "governance_log.jsonl"
    if gov_path.exists():
        events, warnings = _read_governance_gate_events(gov_path)
        if not events:
            return events, "governance_log_empty", warnings
        return events, "governance_log.jsonl", warnings
    results_path = outdir / "results.json"
    events, warnings = _read_results_gate_events(results_path)
    if events:
        return events, "results.json", warnings
    if any("results.json_invalid" in warning for warning in warnings):
        return events, "results.json_invalid", warnings
    if results_path.exists():
        return events, "results.json_empty", warnings
    return events, "missing", warnings


def _post_warmup(events: List[Dict[str, Any]], warmup_iters: int) -> List[Dict[str, Any]]:
    return [event for event in events if int(event.get("iter", -1)) >= warmup_iters]


def _compute_far(
    events: List[Dict[str, Any]], warmup_iters: int
) -> Tuple[Optional[float], Optional[float], int, int, int]:
    post = _post_warmup(events, warmup_iters)
    total = len(post)
    fail = sum(1 for event in post if event.get("decision") == "fail")
    warn_fail = sum(1 for event in post if event.get("decision") in {"warn", "fail"})
    if total == 0:
        return None, None, fail, warn_fail, total
    return fail / float(total), warn_fail / float(total), fail, warn_fail, total


def _compute_delay(
    events: List[Dict[str, Any]],
    injection_onset_iter: Optional[int],
    warmup_iters: int,
    gate_window: int,
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
    if injection_onset_iter is None:
        return None, None, None, None
    t0 = max(int(injection_onset_iter), int(warmup_iters))
    first_warn: Optional[int] = None
    first_fail: Optional[int] = None
    for event in events:
        it = int(event.get("iter", -1))
        if it < t0:
            continue
        decision = event.get("decision")
        if decision in {"warn", "fail"} and first_warn is None:
            first_warn = it
        if decision == "fail" and first_fail is None:
            first_fail = it
        if first_warn is not None and first_fail is not None:
            break
    delay_warn_iters = None if first_warn is None else int(first_warn - t0)
    delay_fail_iters = None if first_fail is None else int(first_fail - t0)
    delay_warn_w = None
    delay_fail_w = None
    if delay_warn_iters is not None:
        delay_warn_w = delay_warn_iters / float(gate_window)
    if delay_fail_iters is not None:
        delay_fail_w = delay_fail_iters / float(gate_window)
    return delay_fail_iters, delay_fail_w, delay_warn_iters, delay_warn_w


def _derive_final_decision(events: List[Dict[str, Any]]) -> str:
    decisions = {event.get("decision") for event in events}
    if "fail" in decisions:
        return "fail"
    if "warn" in decisions:
        return "warn"
    if "pass" in decisions:
        return "pass"
    return "skip"


def _derive_effective_decision(events: List[Dict[str, Any]], warmup_iters: int) -> Tuple[str, str]:
    post = _post_warmup(events, warmup_iters)
    if post:
        return str(post[-1].get("decision")), "post_warmup_last"
    if events:
        return str(events[-1].get("decision")), "last_gate"
    return "skip", "no_gates"


def _split_gate_preview(
    events: List[Dict[str, Any]],
    *,
    limit: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    total = len(events)
    if total <= limit * 2:
        return events, [], False
    return events[:limit], events[-limit:], True


def _preview_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "iter": event.get("iter"),
            "decision": event.get("decision"),
            "reason": event.get("reason"),
            "worst_DW": event.get("worst_DW"),
        }
        for event in events
    ]


def _extract_overhead(run_config: Optional[Dict[str, Any]], missing_fields: List[str]) -> Optional[float]:
    if not isinstance(run_config, dict):
        missing_fields.append("run_config_resolved")
        return None
    timing = run_config.get("timing")
    if not isinstance(timing, dict):
        missing_fields.append("timing")
        return None
    runner_wall = timing.get("runner_wall_s")
    veriscope_wall = timing.get("veriscope_wall_s")
    if runner_wall is None:
        missing_fields.append("timing.runner_wall_s")
    if veriscope_wall is None:
        missing_fields.append("timing.veriscope_wall_s")
    if runner_wall is None or veriscope_wall is None:
        return None
    try:
        runner_wall_f = float(runner_wall)
        veriscope_wall_f = float(veriscope_wall)
    except (TypeError, ValueError):
        missing_fields.append("timing.parse_error")
        return None
    if runner_wall_f <= 0:
        missing_fields.append("timing.runner_wall_s_nonpositive")
        return None
    return veriscope_wall_f / runner_wall_f


def _read_redactions(run_config: Optional[Dict[str, Any]]) -> Union[str, bool]:
    if not isinstance(run_config, dict):
        return "unknown"
    env_capture = run_config.get("env_capture")
    if isinstance(env_capture, dict) and "redactions_applied" in env_capture:
        return bool(env_capture.get("redactions_applied"))
    return "unknown"


def _resolve_injection_onset(
    injected_run_config: Optional[Dict[str, Any]],
    injection_onset_iter: Optional[int],
) -> Tuple[Optional[int], str]:
    if isinstance(injected_run_config, dict):
        onset = _parse_optional_int(injected_run_config.get("data_corrupt_at"))
        if onset is not None and onset >= 0:
            return onset, "run_config_resolved.data_corrupt_at"
    if injection_onset_iter is not None and int(injection_onset_iter) >= 0:
        return int(injection_onset_iter), "cli.injection_onset_iter"
    return None, "missing"


def calibrate_pilot(
    control_dir: Path,
    injected_dir: Path,
    *,
    injection_onset_iter: Optional[int] = None,
) -> Dict[str, Any]:
    control_dir = Path(control_dir)
    injected_dir = Path(injected_dir)

    control_summary = _load_json(control_dir / "results_summary.json", "MISSING_RESULTS_SUMMARY")
    injected_summary = _load_json(injected_dir / "results_summary.json", "MISSING_RESULTS_SUMMARY")
    if control_summary.get("run_status") != "success":
        _die("RUN_STATUS_NOT_SUCCESS", "control run_status != success")
    if injected_summary.get("run_status") != "success":
        _die("RUN_STATUS_NOT_SUCCESS", "injected run_status != success")

    control_window_signature = _load_json(control_dir / "window_signature.json", "MISSING_WINDOW_SIGNATURE")
    injected_window_signature = _load_json(injected_dir / "window_signature.json", "MISSING_WINDOW_SIGNATURE")
    try:
        control_window_hash = window_signature_sha256(control_window_signature)
    except Exception as exc:
        _die("INVALID_WINDOW_SIGNATURE", f"control window_signature.json is invalid for hashing: {exc}")
    try:
        injected_window_hash = window_signature_sha256(injected_window_signature)
    except Exception as exc:
        _die("INVALID_WINDOW_SIGNATURE", f"injected window_signature.json is invalid for hashing: {exc}")

    control_summary_window_hash = _extract_summary_window_hash(control_summary, label="control")
    injected_summary_window_hash = _extract_summary_window_hash(injected_summary, label="injected")
    if control_summary_window_hash != control_window_hash:
        _die(
            "WINDOW_SIGNATURE_HASH_MISMATCH",
            "control results_summary.window_signature_ref.hash does not match control window_signature.json",
        )
    if injected_summary_window_hash != injected_window_hash:
        _die(
            "WINDOW_SIGNATURE_HASH_MISMATCH",
            "injected results_summary.window_signature_ref.hash does not match injected window_signature.json",
        )
    if control_window_hash != injected_window_hash:
        _die(
            "WINDOW_SIGNATURE_HASH_MISMATCH",
            f"control/injected window signature hash mismatch: {control_window_hash} != {injected_window_hash}",
        )

    control_run_cfg = None
    injected_run_cfg = None
    control_run_cfg_path = control_dir / "run_config_resolved.json"
    injected_run_cfg_path = injected_dir / "run_config_resolved.json"
    if control_run_cfg_path.exists():
        control_run_cfg = _load_json(control_run_cfg_path, "INVALID_RUN_CONFIG")
    if injected_run_cfg_path.exists():
        injected_run_cfg = _load_json(injected_run_cfg_path, "INVALID_RUN_CONFIG")

    window_signature = control_window_signature
    gate_window, gate_window_source = _resolve_gate_window(window_signature, control_run_cfg, injected_run_cfg)

    control_events, control_events_source, control_warnings = _read_gate_events(control_dir)
    injected_events, injected_events_source, injected_warnings = _read_gate_events(injected_dir)

    missing_fields: List[str] = []

    warmup_k, warmup_source = _resolve_warmup(window_signature, control_run_cfg, injected_run_cfg)
    notes: List[str] = []
    if warmup_k is None:
        all_events = control_events + injected_events
        if all_events:
            warmup_k = min(event["iter"] for event in all_events)
            warmup_source = "inferred_min_gate_iter"
            notes.append("NOTE:warmup_inferred_from_gate_events")
            missing_fields.append("global.warmup_missing_inferred")
        else:
            warmup_k = 0
            warmup_source = "default_0_no_gates"
            notes.append("NOTE:warmup_defaulted_no_gate_events")
            missing_fields.append("global.warmup_missing_defaulted")

    if control_events_source in {"missing", "governance_log_empty", "results.json_empty"}:
        missing_fields.append("control.gate_events_missing")
    if injected_events_source in {"missing", "governance_log_empty", "results.json_empty"}:
        missing_fields.append("injected.gate_events_missing")

    if control_events_source != "governance_log.jsonl":
        notes.append(f"NOTE:control_gate_events_source={control_events_source}")
    if injected_events_source != "governance_log.jsonl":
        notes.append(f"NOTE:injected_gate_events_source={injected_events_source}")
    notes.extend(control_warnings)
    notes.extend(injected_warnings)

    far_fail, far_warnfail, far_fail_count, far_warnfail_count, far_total = _compute_far(
        control_events,
        warmup_k,
    )
    if far_warnfail is None:
        missing_fields.append("control.post_warmup_events_missing")
    far_source = control_events_source

    injection_onset, injection_onset_source = _resolve_injection_onset(injected_run_cfg, injection_onset_iter)
    if injection_onset is None:
        missing_fields.append("injected.injection_onset_missing")
    delay_fail_iters, delay_fail_w, delay_warn_iters, delay_warn_w = _compute_delay(
        injected_events,
        injection_onset,
        warmup_k,
        gate_window,
    )
    if delay_warn_w is None:
        missing_fields.append("injected.delay_warn_missing")

    control_first_raw, control_last_raw, control_truncated = _split_gate_preview(control_events)
    injected_first_raw, injected_last_raw, injected_truncated = _split_gate_preview(injected_events)
    control_first = _preview_rows(control_first_raw)
    control_last = _preview_rows(control_last_raw)
    injected_first = _preview_rows(injected_first_raw)
    injected_last = _preview_rows(injected_last_raw)

    overhead_missing_fields: List[str] = []
    overhead = _extract_overhead(injected_run_cfg, overhead_missing_fields)
    if overhead is None and overhead_missing_fields:
        notes.extend([f"NOTE:{field}" for field in overhead_missing_fields])
    if control_run_cfg is not None:
        redactions_applied = _read_redactions(control_run_cfg)
    else:
        redactions_applied = _read_redactions(injected_run_cfg)

    policy_rev = "unknown"
    if isinstance(control_run_cfg, dict):
        policy_rev = (
            control_run_cfg.get("provenance", {}).get("policy_rev")
            or control_run_cfg.get("env_capture", {}).get("policy_rev")
            or policy_rev
        )
    if policy_rev == "unknown" and isinstance(injected_run_cfg, dict):
        policy_rev = (
            injected_run_cfg.get("provenance", {}).get("policy_rev")
            or injected_run_cfg.get("env_capture", {}).get("policy_rev")
            or policy_rev
        )

    control_effective, control_effective_source = _derive_effective_decision(control_events, warmup_k)
    injected_effective, injected_effective_source = _derive_effective_decision(injected_events, warmup_k)

    calibration_status = "complete"
    if (
        far_warnfail is None
        or delay_warn_w is None
        or warmup_source in {"inferred_min_gate_iter", "default_0_no_gates"}
    ):
        calibration_status = "incomplete"

    return {
        "policy_rev": policy_rev,
        "W": gate_window,
        "W_source": gate_window_source,
        "K": warmup_k,
        "warmup_source": warmup_source,
        "FAR": far_warnfail,
        "FAR_fail": far_fail,
        "FAR_warnfail": far_warnfail,
        "FAR_counts": {
            "post_warmup_total": far_total,
            "post_warmup_fail": far_fail_count,
            "post_warmup_warnfail": far_warnfail_count,
        },
        "FAR_source": far_source,
        "Delay_W": delay_warn_w,
        "Delay_fail_iters": delay_fail_iters,
        "Delay_fail_W": delay_fail_w,
        "Delay_warn_iters": delay_warn_iters,
        "Delay_warn_W": delay_warn_w,
        "injection_onset_iter": injection_onset,
        "injection_onset_source": injection_onset_source,
        "control_gate_events_source": control_events_source,
        "injected_gate_events_source": injected_events_source,
        "control_gate_events_total": len(control_events),
        "control_gate_events_post_warmup": len(_post_warmup(control_events, warmup_k)),
        "injected_gate_events_total": len(injected_events),
        "injected_gate_events_post_warmup": len(_post_warmup(injected_events, warmup_k)),
        "control_gate_events_first": control_first,
        "control_gate_events_last": control_last,
        "control_gate_events_truncated": control_truncated,
        "injected_gate_events_first": injected_first,
        "injected_gate_events_last": injected_last,
        "injected_gate_events_truncated": injected_truncated,
        "control_final_decision": _derive_final_decision(control_events),
        "control_effective_decision": control_effective,
        "control_effective_decision_source": control_effective_source,
        "injected_final_decision": _derive_final_decision(injected_events),
        "injected_effective_decision": injected_effective,
        "injected_effective_decision_source": injected_effective_source,
        "overhead": overhead,
        "missing_fields": missing_fields,
        "notes": notes,
        "redactions_applied": redactions_applied,
        "calibration_status": calibration_status,
    }


def render_calibration_md(output: Dict[str, Any]) -> str:
    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _fmt_reason(value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("|", "\\|")

    def _render_gate_table(
        title: str,
        first: List[Dict[str, Any]],
        last: List[Dict[str, Any]],
        truncated: bool,
        total: int,
    ) -> List[str]:
        lines = [f"### {title}", "", f"(showing {'first/last 5' if truncated else 'all'} of {total})", ""]
        lines.append("| iter | decision | reason | worst_DW |")
        lines.append("| --- | --- | --- | --- |")
        for row in first:
            lines.append(
                f"| {_fmt(row.get('iter'))} | {_fmt(row.get('decision'))} | {_fmt_reason(row.get('reason'))} | {_fmt(row.get('worst_DW'))} |"
            )
        if truncated:
            lines.append("| ... | ... | ... | ... |")
        for row in last:
            lines.append(
                f"| {_fmt(row.get('iter'))} | {_fmt(row.get('decision'))} | {_fmt_reason(row.get('reason'))} | {_fmt(row.get('worst_DW'))} |"
            )
        return lines

    far_fail = output.get("FAR_fail")
    far_warnfail = output.get("FAR_warnfail")
    far_counts = output.get("FAR_counts") or {}
    delay_fail_iters = output.get("Delay_fail_iters")
    delay_fail_w = output.get("Delay_fail_W")
    delay_warn_iters = output.get("Delay_warn_iters")
    delay_warn_w = output.get("Delay_warn_W")
    overhead = output.get("overhead")

    md_lines: List[str] = [
        "# Pilot Calibration Summary",
        "",
        "## Summary",
        f"- Control FAR_fail (post-warmup): {_fmt(far_fail)}",
        f"- Control FAR_warnfail (post-warmup): {_fmt(far_warnfail)}",
        f"- Injected delay_fail_iters: {_fmt(delay_fail_iters)} (delay_fail_W={_fmt(delay_fail_w)})",
        f"- Injected delay_warn_iters: {_fmt(delay_warn_iters)} (delay_warn_W={_fmt(delay_warn_w)})",
        f"- Overhead (wall-clock): {_fmt(overhead)}",
        f"- Calibration status: {_fmt(output.get('calibration_status'))}",
        "",
        "## Control",
        f"- gate_window: {_fmt(output.get('W'))} (source={_fmt(output.get('W_source'))})",
        f"- warmup_iters: {_fmt(output.get('K'))} (source={_fmt(output.get('warmup_source'))})",
        f"- gate_events_source: {_fmt(output.get('control_gate_events_source'))}",
        f"- n_gate_events_post_warmup: {_fmt(output.get('control_gate_events_post_warmup'))}",
        f"- FAR_fail: {_fmt(far_fail)} ({_fmt(far_counts.get('post_warmup_fail'))}/{_fmt(far_counts.get('post_warmup_total'))})",
        f"- FAR_warnfail: {_fmt(far_warnfail)} ({_fmt(far_counts.get('post_warmup_warnfail'))}/{_fmt(far_counts.get('post_warmup_total'))})",
        f"- final_decision: {_fmt(output.get('control_final_decision'))}",
        f"- effective_decision: {_fmt(output.get('control_effective_decision'))} (source={_fmt(output.get('control_effective_decision_source'))})",
        "",
    ]

    md_lines.extend(
        _render_gate_table(
            "Control Gate Events",
            output.get("control_gate_events_first") or [],
            output.get("control_gate_events_last") or [],
            bool(output.get("control_gate_events_truncated")),
            int(output.get("control_gate_events_total") or 0),
        )
    )

    md_lines.extend(
        [
            "",
            "## Injected",
            f"- gate_window: {_fmt(output.get('W'))} (source={_fmt(output.get('W_source'))})",
            f"- warmup_iters: {_fmt(output.get('K'))} (source={_fmt(output.get('warmup_source'))})",
            f"- gate_events_source: {_fmt(output.get('injected_gate_events_source'))}",
            f"- injection_onset_iter: {_fmt(output.get('injection_onset_iter'))} (source={_fmt(output.get('injection_onset_source'))})",
            f"- n_gate_events_post_warmup: {_fmt(output.get('injected_gate_events_post_warmup'))}",
            f"- delay_fail_iters: {_fmt(delay_fail_iters)} (delay_fail_W={_fmt(delay_fail_w)})",
            f"- delay_warn_iters: {_fmt(delay_warn_iters)} (delay_warn_W={_fmt(delay_warn_w)})",
            f"- final_decision: {_fmt(output.get('injected_final_decision'))}",
            f"- effective_decision: {_fmt(output.get('injected_effective_decision'))} (source={_fmt(output.get('injected_effective_decision_source'))})",
            "",
        ]
    )

    md_lines.extend(
        _render_gate_table(
            "Injected Gate Events",
            output.get("injected_gate_events_first") or [],
            output.get("injected_gate_events_last") or [],
            bool(output.get("injected_gate_events_truncated")),
            int(output.get("injected_gate_events_total") or 0),
        )
    )

    missing_fields = output.get("missing_fields") or []
    notes = output.get("notes") or []
    if missing_fields:
        md_lines.extend(["", "## Missing Fields"])
        md_lines.extend([f"- {item}" for item in missing_fields])
    if notes:
        md_lines.extend(["", "## Notes"])
        md_lines.extend([f"- {item}" for item in notes])

    md_lines.extend(
        [
            "",
            "## Thresholds (from docs/productization.md)",
            "- FAR ≤ 5% post-warmup (control).",
            "- Detection delay ≤ 2W iterations (injected).",
            "- Overhead ≤ 5% wall-clock (when timing fields exist).",
        ]
    )
    return "\n".join(md_lines) + "\n"
