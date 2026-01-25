#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _die(token: str, message: str) -> None:
    sys.stderr.write(f"{token}: {message}\n")
    raise SystemExit(2)


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


def _resolve_gate_window(
    window_signature: Dict[str, Any],
    run_config: Optional[Dict[str, Any]],
) -> Tuple[int, str]:
    gate_controls = window_signature.get("gate_controls")
    if isinstance(gate_controls, dict) and "gate_window" in gate_controls:
        return int(gate_controls["gate_window"]), "window_signature.gate_controls.gate_window"
    if isinstance(run_config, dict):
        resolved_gate_cfg = run_config.get("resolved_gate_cfg")
        if isinstance(resolved_gate_cfg, dict) and "gate_window" in resolved_gate_cfg:
            return int(resolved_gate_cfg["gate_window"]), "run_config_resolved.resolved_gate_cfg.gate_window"
        if "gate_window" in run_config:
            return int(run_config["gate_window"]), "run_config_resolved.gate_window"
    _die("MISSING_GATE_WINDOW", "gate_window not found in window_signature.json or run_config_resolved.json")
    raise SystemExit(2)


def _resolve_warmup(run_config: Optional[Dict[str, Any]]) -> Tuple[int, str]:
    if isinstance(run_config, dict):
        resolved_gate_cfg = run_config.get("resolved_gate_cfg")
        if isinstance(resolved_gate_cfg, dict) and "gate_warmup" in resolved_gate_cfg:
            return int(resolved_gate_cfg["gate_warmup"]), "run_config_resolved.resolved_gate_cfg.gate_warmup"
        if "gate_warmup" in run_config:
            return int(run_config["gate_warmup"]), "run_config_resolved.gate_warmup"
    return 0, "default_0"


def _load_results_gates(results_path: Path) -> Optional[List[Dict[str, Any]]]:
    if not results_path.exists():
        return None
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    gates = data.get("gates")
    if isinstance(gates, list):
        return [g for g in gates if isinstance(g, dict)]
    return None


def _compute_far_from_gates(gates: List[Dict[str, Any]], warmup: int) -> Tuple[Optional[float], int, int]:
    evaluated = 0
    warn_fail = 0
    for idx, gate in enumerate(gates):
        if idx < warmup:
            continue
        decision = str(gate.get("decision", "")).lower()
        if decision in {"pass", "warn", "fail"}:
            evaluated += 1
            if decision in {"warn", "fail"}:
                warn_fail += 1
    if evaluated == 0:
        return None, warn_fail, evaluated
    return warn_fail / evaluated, warn_fail, evaluated


def _compute_delay_w(gates: List[Dict[str, Any]], warmup: int, gate_window: int) -> Optional[float]:
    for idx, gate in enumerate(gates):
        if idx < warmup:
            continue
        decision = str(gate.get("decision", "")).lower()
        if decision in {"warn", "fail"}:
            return (idx - warmup) / float(gate_window)
    return None


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score pilot control/injected runs.")
    parser.add_argument("--control-dir", required=True, help="Control capsule OUTDIR")
    parser.add_argument("--injected-dir", required=True, help="Injected capsule OUTDIR")
    parser.add_argument("--out", default="calibration.json", help="Output JSON path")
    parser.add_argument("--out-md", default="calibration.md", help="Output Markdown path")
    args = parser.parse_args(argv)

    control_dir = Path(args.control_dir)
    injected_dir = Path(args.injected_dir)

    control_summary = _load_json(control_dir / "results_summary.json", "MISSING_RESULTS_SUMMARY")
    injected_summary = _load_json(injected_dir / "results_summary.json", "MISSING_RESULTS_SUMMARY")
    if control_summary.get("run_status") != "success":
        _die("RUN_STATUS_NOT_SUCCESS", "control run_status != success")
    if injected_summary.get("run_status") != "success":
        _die("RUN_STATUS_NOT_SUCCESS", "injected run_status != success")

    control_run_cfg = None
    injected_run_cfg = None
    control_run_cfg_path = control_dir / "run_config_resolved.json"
    injected_run_cfg_path = injected_dir / "run_config_resolved.json"
    if control_run_cfg_path.exists():
        control_run_cfg = _load_json(control_run_cfg_path, "INVALID_RUN_CONFIG")
    if injected_run_cfg_path.exists():
        injected_run_cfg = _load_json(injected_run_cfg_path, "INVALID_RUN_CONFIG")

    window_signature = _load_json(control_dir / "window_signature.json", "MISSING_WINDOW_SIGNATURE")
    gate_window, gate_window_source = _resolve_gate_window(window_signature, control_run_cfg)
    warmup_k, warmup_source = _resolve_warmup(control_run_cfg)

    missing_fields: List[str] = []

    control_gates = _load_results_gates(control_dir / "results.json")
    if control_gates is None:
        counts = control_summary.get("counts", {})
        evaluated = int(counts.get("evaluated", 0) or 0)
        warn = int(counts.get("warn", 0) or 0)
        fail = int(counts.get("fail", 0) or 0)
        far = None if evaluated == 0 else (warn + fail) / float(evaluated)
        far_source = "summary_fallback"
    else:
        far, _, _ = _compute_far_from_gates(control_gates, warmup_k)
        far_source = "results.json"
        if far is None:
            missing_fields.append("control.post_warmup_evaluated")

    injected_gates = _load_results_gates(injected_dir / "results.json")
    delay_w = None
    if injected_gates is None:
        missing_fields.append("injected.results.json")
    else:
        delay_w = _compute_delay_w(injected_gates, warmup_k, gate_window)
        if delay_w is None:
            missing_fields.append("injected.first_warn_fail")

    overhead = _extract_overhead(injected_run_cfg, missing_fields)
    redactions_applied = (
        _read_redactions(control_run_cfg) if control_run_cfg is not None else _read_redactions(injected_run_cfg)
    )

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

    calibration_status = "complete"
    if missing_fields or far is None or delay_w is None:
        calibration_status = "incomplete"

    output = {
        "policy_rev": policy_rev,
        "W": gate_window,
        "W_source": gate_window_source,
        "K": warmup_k,
        "warmup_source": warmup_source,
        "FAR": far,
        "FAR_source": far_source,
        "Delay_W": delay_w,
        "overhead": overhead,
        "missing_fields": missing_fields,
        "redactions_applied": redactions_applied,
        "calibration_status": calibration_status,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = [
        "# Pilot Calibration Summary",
        "",
        f"- Control FAR (post-warmup): {far if far is not None else 'n/a'}",
        f"- Injected Delay/W: {delay_w if delay_w is not None else 'n/a'}",
        f"- Overhead (wall-clock): {overhead if overhead is not None else 'n/a'}",
        "",
        "## Thresholds (from docs/productization.md)",
        "- FAR ≤ 5% post-warmup (control).",
        "- Detection delay ≤ 2W iterations (injected).",
        "- Overhead ≤ 5% wall-clock (when timing fields exist).",
    ]
    Path(args.out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
