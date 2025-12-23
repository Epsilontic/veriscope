# veriscope/runners/gpt/analyze_gates.py
"""Gate recall/precision analysis for veriscope GPT experiments.

Usage:
  python -m veriscope.runners.gpt.analyze_gates \
      --results /mnt/work/out/veriscope_gpt_inject_MI2_W100_20251210_010157.json \
      --spike_start 2500 \
      --spike_len 400 \
      --gate_window 100 \
      --metric_interval 2
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Strict bool coercion helper to avoid NumPy-bool identity traps.
def _as_opt_bool(x: Any) -> Optional[bool]:
    """Coerce bool/np.bool_ and integer-like 0/1 to Optional[bool].

    Returns None for anything else (including floats, strings, etc.).
    """
    if x is None:
        return None
    # bool is a subclass of int, so check bool first
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        if int(x) == 0:
            return False
        if int(x) == 1:
            return True
        return None
    return None


def _recent_window_bounds_iter_space(
    gate_iter: int,
    gate_window: int,
    metric_interval: int,
) -> Tuple[int, int]:
    """Compute the iteration range covered by the *recent half-window* for a gate check.

    In train_nanogpt.py:
      Wm = gate_window // metric_interval (snapshots per half-window)
      recent_slice spans Wm snapshots => W_iter = Wm * metric_interval iterations

    We approximate recent window as [gate_iter - W_iter, gate_iter).
    """
    interval = max(1, int(metric_interval))
    wm = max(1, int(gate_window) // interval)
    w_iter = int(wm * interval)
    return int(gate_iter) - w_iter, int(gate_iter)


def _windows_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Check if two half-open intervals [a0,a1) and [b0,b1) overlap."""
    return a0 < b1 and b0 < a1


def _is_onset_gate(
    recent_start: int,
    recent_end: int,
    gate_window: int,
    metric_interval: int,
    spike_start: int,
    spike_end: int,
) -> bool:
    """True iff past-half does NOT overlap spike, but recent-half DOES.

    This scores whether a *change detector* caught the corruption onset.

    past_half  = [recent_start - W_iter, recent_start)
    recent_half= [recent_start, recent_end)

    We reuse the same W_iter implied by gate_window and metric_interval.
    """
    interval = max(1, int(metric_interval))
    wm = max(1, int(gate_window) // interval)
    w_iter = int(wm * interval)

    past_start = int(recent_start) - int(w_iter)
    past_end = int(recent_start)

    past_overlaps = _windows_overlap(past_start, past_end, int(spike_start), int(spike_end))
    recent_overlaps = _windows_overlap(int(recent_start), int(recent_end), int(spike_start), int(spike_end))
    return (not past_overlaps) and recent_overlaps


def _parse_gate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    audit = row.get("audit", {}) or {}

    # Prefer namespaced base_reason, fallback to legacy reason, then gate_reason, then row-level reason
    reason = str(
        audit.get(
            "base_reason",
            audit.get(
                "reason",
                row.get(
                    "gate_reason",
                    row.get("reason", ""),
                ),
            ),
        )
    ).strip()

    # Prefer explicit evaluated flag when present; otherwise infer from reason.
    evaluated_flag = _as_opt_bool(audit.get("evaluated", None))
    not_eval = (
        reason.startswith("not_evaluated") or ("insufficient_evidence" in reason) or ("insufficient_history" in reason)
    )
    evaluated = (not not_eval) if evaluated_flag is None else bool(evaluated_flag)

    chg = _as_opt_bool(audit.get("change_dw_ok", None))  # Optional[bool]

    change_ok_bool = None
    if evaluated and (chg is not None):
        change_ok_bool = bool(chg)

    return {
        "reason": reason or None,
        "base_reason": audit.get("base_reason", reason or None),
        "evaluated": bool(evaluated),
        "change_dw_ok": chg,
        "change_ok_bool": change_ok_bool,
        "base_ok": bool(audit.get("base_ok", True)),
        "change_gain_ok": _as_opt_bool(audit.get("change_gain_ok", None)),
    }


def _change_valid_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if bool(r.get("evaluated", False)) and (r.get("change_dw_ok", None) is not None)]


def _compute_change_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(rows)
    excluded_not_evaluated = sum(1 for r in rows if not bool(r.get("evaluated", False)))
    excluded_unknown_stability = sum(
        1 for r in rows if bool(r.get("evaluated", False)) and (r.get("change_dw_ok", None) is None)
    )

    valid = _change_valid_rows(rows)

    tp = sum(1 for r in valid if (r.get("change_dw_ok") is False) and bool(r.get("overlaps_spike", False)))
    fp = sum(1 for r in valid if (r.get("change_dw_ok") is False) and (not bool(r.get("overlaps_spike", False))))
    fn = sum(1 for r in valid if (r.get("change_dw_ok") is True) and bool(r.get("overlaps_spike", False)))
    tn = sum(1 for r in valid if (r.get("change_dw_ok") is True) and (not bool(r.get("overlaps_spike", False))))

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "n_total": int(n_total),
        "excluded_not_evaluated": int(excluded_not_evaluated),
        "excluded_unknown_stability": int(excluded_unknown_stability),
        "n_valid": int(len(valid)),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def _count_gain_fps(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gain_fp = [
        r
        for r in rows
        if bool(r.get("evaluated", False)) and (r.get("change_dw_ok") is True) and (not bool(r.get("base_ok", True)))
    ]

    dist: Dict[str, int] = {}
    for r in gain_fp:
        v_raw = r.get("change_gain_ok", None)
        v = _as_opt_bool(v_raw)
        k = "None" if v is None else ("True" if bool(v) else "False")
        dist[k] = dist.get(k, 0) + 1

    return {"n_gain_fp": int(len(gain_fp)), "change_gain_ok_distribution": dist}


def _expand_results_args(results_args: List[str]) -> List[Path]:
    """
    Expand a list of --results tokens into a de-duped list of Paths.

    Supports comma-separated entries within a token:
      --results a.json,b.json c.json

    Dedupe key normalizes without requiring existence:
      - expanduser (~)
      - resolve(strict=False) (collapses ./, ../, etc.)
    """
    out: List[Path] = []
    seen: set[str] = set()
    for token in results_args or []:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            p2 = Path(part).expanduser()
            try:
                key = str(p2.resolve(strict=False))
            except Exception:
                key = str(p2)
            if key in seen:
                continue
            seen.add(key)
            out.append(p2)
    return out


def analyze_gates(
    results_path: Path,
    spike_start: int,
    spike_len: int,
    gate_window: int,
    metric_interval: int,
    control: bool = False,
) -> Dict[str, Any]:
    spike_end = int(spike_start) + int(spike_len)
    data = json.loads(results_path.read_text())
    gates: List[Dict[str, Any]] = list(data.get("gates", []) or [])

    per_gate: List[Dict[str, Any]] = []
    for g in gates:
        gate_iter = int(g.get("iter", 0) or 0)
        ok = bool(g.get("ok", True))
        audit = g.get("audit", {}) or {}
        parsed = _parse_gate_row(g)

        recent_start, recent_end = _recent_window_bounds_iter_space(
            gate_iter=gate_iter,
            gate_window=gate_window,
            metric_interval=metric_interval,
        )

        overlaps = _windows_overlap(
            recent_start,
            recent_end,
            int(spike_start),
            int(spike_end),
        )

        ov0 = max(recent_start, int(spike_start))
        ov1 = min(recent_end, int(spike_end))
        overlap_iters = max(0, ov1 - ov0)

        # Normalize by the actual iteration-span of the recent window
        interval = max(1, int(metric_interval))
        wm = max(1, int(gate_window) // interval)
        w_iter = int(wm * interval)
        overlap_frac = float(overlap_iters) / float(w_iter) if w_iter > 0 else 0.0

        onset_gate = _is_onset_gate(
            recent_start=recent_start,
            recent_end=recent_end,
            gate_window=gate_window,
            metric_interval=metric_interval,
            spike_start=int(spike_start),
            spike_end=int(spike_end),
        )

        per_gate.append(
            {
                "iter": gate_iter,
                "ok": ok,
                "fail": (not ok),
                "overlaps_spike": overlaps,
                "onset_gate": bool(onset_gate),
                "overlap_frac": overlap_frac,
                "is_control": bool(control),
                "recent_window": (recent_start, recent_end),
                # Change detector fields
                "worst_DW": audit.get("worst_DW", float("nan")),
                "eps_eff": audit.get("eps_eff", float("nan")),
                "gain_bits": audit.get("gain_bits", float("nan")),
                # Regime fields (defaults are conservative: assume OK if missing)
                # "change_ok": bool(audit.get("change_ok", True)),
                "regime_ok": bool(audit.get("regime_ok", True)),
                "regime_active": bool(audit.get("regime_active", False)),
                "regime_enabled": bool(audit.get("regime_enabled", False)),
                "regime_worst_DW": audit.get("regime_worst_DW", float("nan")),
                "ref_established_at": audit.get("ref_established_at"),
                "ref_just_established": bool(audit.get("ref_just_established", False)),
                # New: parsed change detector fields
                "base_reason": parsed.get("base_reason"),
                "evaluated": bool(parsed.get("evaluated", bool((g.get("audit") or {}).get("evaluated", False)))),
                "change_dw_ok": parsed.get("change_dw_ok", None),
                "change_ok_bool": parsed.get("change_ok_bool", None),
                "base_ok": bool(parsed.get("base_ok", True)),
                "change_gain_ok": parsed.get("change_gain_ok", None),
            }
        )

    # ---- Union gate confusion matrix (existing behavior) ----
    tp = sum(1 for r in per_gate if r["fail"] and r["overlaps_spike"])
    fp = sum(1 for r in per_gate if r["fail"] and not r["overlaps_spike"])
    fn = sum(1 for r in per_gate if r["ok"] and r["overlaps_spike"])
    tn = sum(1 for r in per_gate if r["ok"] and not r["overlaps_spike"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    # ---- Union gate confusion matrix (evaluated-only) ----
    # Prevents ok=True but evaluated=False rows from inflating TN/FN.
    union_eval = [r for r in per_gate if bool(r.get("evaluated", False))]
    union_excluded_not_evaluated = int(len(per_gate) - len(union_eval))
    tp_e = sum(1 for r in union_eval if r["fail"] and r["overlaps_spike"])
    fp_e = sum(1 for r in union_eval if r["fail"] and not r["overlaps_spike"])
    fn_e = sum(1 for r in union_eval if r["ok"] and r["overlaps_spike"])
    tn_e = sum(1 for r in union_eval if r["ok"] and not r["overlaps_spike"])

    precision_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else float("nan")
    recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else float("nan")
    specificity_e = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else float("nan")

    # ---- Change detection breakdown ----
    change_metrics = _compute_change_metrics(per_gate)
    cc = change_metrics["confusion"]
    change_precision = change_metrics["precision"]
    change_recall = change_metrics["recall"]
    change_specificity = change_metrics["specificity"]

    # ---- Regime detection breakdown (only when active) ----
    regime_active_gates = [r for r in per_gate if r["regime_active"]]

    regime_tp = sum(1 for r in regime_active_gates if (not r["regime_ok"]) and r["overlaps_spike"])
    regime_fp = sum(1 for r in regime_active_gates if (not r["regime_ok"]) and (not r["overlaps_spike"]))
    regime_fn = sum(1 for r in regime_active_gates if r["regime_ok"] and r["overlaps_spike"])
    regime_tn = sum(1 for r in regime_active_gates if r["regime_ok"] and (not r["overlaps_spike"]))

    regime_precision = regime_tp / (regime_tp + regime_fp) if (regime_tp + regime_fp) > 0 else float("nan")
    regime_recall = regime_tp / (regime_tp + regime_fn) if (regime_tp + regime_fn) > 0 else float("nan")
    regime_specificity = regime_tn / (regime_tn + regime_fp) if (regime_tn + regime_fp) > 0 else float("nan")

    # ---- Onset scoring (transition into spike): change detector only ----
    # IMPORTANT: in control runs (no injection), "onset_gate" is purely a time-window boundary,
    # not a corruption onset ground-truth. Disable onset scoring under --control.
    if control:
        onset_total = 0
        onset_change_tp = 0
        onset_change_fn = 0
        onset_change_recall = float("nan")
    else:
        onset_candidates = [
            r
            for r in per_gate
            if r.get("onset_gate") and bool(r.get("evaluated", False)) and (r.get("change_dw_ok", None) is not None)
        ]
        onset_total = len(onset_candidates)
        onset_change_tp = sum(1 for r in onset_candidates if (r.get("change_dw_ok") is False))
        onset_change_fn = sum(1 for r in onset_candidates if (r.get("change_dw_ok") is True))
        onset_change_recall = (
            onset_change_tp / (onset_change_tp + onset_change_fn)
            if (onset_change_tp + onset_change_fn) > 0
            else float("nan")
        )

    # ---- Attribution: who drives union failures in spike window? ----
    valid_change = _change_valid_rows(per_gate)

    both_fail_in_spike = sum(
        1
        for r in valid_change
        if (r.get("change_dw_ok") is False)
        and bool(r.get("regime_active", False))
        and (not bool(r.get("regime_ok", True)))
        and bool(r.get("overlaps_spike", False))
    )
    change_only_in_spike = sum(
        1
        for r in valid_change
        if (r.get("change_dw_ok") is False)
        and (not bool(r.get("regime_active", False)) or bool(r.get("regime_ok", True)))
        and bool(r.get("overlaps_spike", False))
    )
    regime_only_in_spike = sum(
        1
        for r in valid_change
        if (r.get("change_dw_ok") is True)
        and bool(r.get("regime_active", False))
        and (not bool(r.get("regime_ok", True)))
        and bool(r.get("overlaps_spike", False))
    )

    return {
        "results_file": str(results_path),
        "spike": {"start": int(spike_start), "end": int(spike_end), "len": int(spike_len)},
        # Union gate (existing)
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "union_evaluated_only": {
            "excluded_not_evaluated": union_excluded_not_evaluated,
            "confusion": {"tp": int(tp_e), "fp": int(fp_e), "fn": int(fn_e), "tn": int(tn_e)},
            "precision": float(precision_e),
            "recall": float(recall_e),
            "specificity": float(specificity_e),
        },
        # Change detection decomposition
        "change_confusion": cc,
        "change_precision": change_precision,
        "change_recall": change_recall,
        "change_specificity": change_specificity,
        # Regime detection decomposition
        "regime_confusion": {"tp": regime_tp, "fp": regime_fp, "fn": regime_fn, "tn": regime_tn},
        "regime_precision": regime_precision,
        "regime_recall": regime_recall,
        "regime_specificity": regime_specificity,
        "regime_active_gates": len(regime_active_gates),
        # Onset scoring (transition)
        "onset": {
            "total": onset_total,
            "change_tp": onset_change_tp,
            "change_fn": onset_change_fn,
            "change_recall": onset_change_recall,
        },
        # Attribution breakdown
        "spike_attribution": {
            "both_fail": both_fail_in_spike,
            "change_only": change_only_in_spike,
            "regime_only": regime_only_in_spike,
        },
        # Totals
        "total_gates": len(per_gate),
        "total_overlap": tp + fn,
        "total_nonoverlap": tn + fp,
        "per_gate": per_gate,
        "change_metrics": change_metrics,
        "gain_fps": _count_gain_fps(per_gate),
    }


def _fmt_metric(val: float, width: int = 5) -> str:
    """Format a metric value, handling nan gracefully."""
    if math.isnan(val):
        return "n/a".center(width)
    return f"{val:.3f}"


def print_analysis(a: Dict[str, Any], verbose: bool) -> None:
    # Extract per_gate once at the top; reuse throughout
    per_gate = a.get("per_gate", [])
    cc = a["change_confusion"]
    change_metrics = a.get("change_metrics", {}) or {}
    n_valid = int(change_metrics.get("n_valid", 0) or 0)
    ex_ne = int(change_metrics.get("excluded_not_evaluated", 0) or 0)
    ex_unk = int(change_metrics.get("excluded_unknown_stability", 0) or 0)
    rc = a["regime_confusion"]
    onset = a.get("onset", {}) or {}
    onset_total = int(onset.get("total", 0) or 0)
    onset_tp = int(onset.get("change_tp", 0) or 0)

    # =========================================================================
    # EXECUTIVE SUMMARY (what you actually care about)
    # =========================================================================
    print("\n" + "=" * 65)
    print(" EXECUTIVE SUMMARY")
    print("=" * 65)

    # Corruption detection (change-only)
    print("\n┌─ CORRUPTION DETECTION (use this for spike experiments) ──────┐")
    print(f"│  Precision:    {_fmt_metric(a['change_precision']):>5}  (when alarmed, was it real?)             │")
    print(f"│  Recall:       {_fmt_metric(a['change_recall']):>5}  (did we catch the corruption?)           │")
    if onset_total > 0:
        print(
            f"│  Onset recall: {_fmt_metric(float(onset.get('change_recall', float('nan')))):>5}  "
            f"(caught onset: {onset_tp}/{onset_total})               │"
        )
    else:
        print("│  Onset gates:  0     (no transition in window — control or mismatch) │")
    print(f"│  Specificity:  {_fmt_metric(a['change_specificity']):>5}  (quiet when clean?)                      │")
    print(f"│  Change FAILs: {cc['tp']:>3} in-spike, {cc['fp']:>3} off-spike (valid n={n_valid})               │")
    print(f"│  Exclusions:   {ex_ne:>3} not-evaluated, {ex_unk:>3} unknown-stability                              │")
    print("└─────────────────────────────────────────────────────────────┘")

    # Baseline drift (regime)
    if a["regime_active_gates"] > 0:
        regime_total_fail = rc["tp"] + rc["fp"]
        print("\n┌─ BASELINE DRIFT (separate concern, not for spike scoring) ──┐")
        print(
            f"│  Regime active:  {a['regime_active_gates']:>3}/{a['total_gates']:<3} gates "
            f"(reference was established)        │"
        )
        print(
            f"│  Regime FAILs:   {regime_total_fail:>3} total ({rc['tp']} in spike, {rc['fp']} outside)              │"
        )
        print(
            f"│  Regime Prec:    {_fmt_metric(a['regime_precision']):>5}  "
            f"Recall: {_fmt_metric(a['regime_recall']):>5}  "
            f"Spec: {_fmt_metric(a['regime_specificity']):>5}       │"
        )
        print("└─────────────────────────────────────────────────────────────┘")
    else:
        print("\n┌─ BASELINE DRIFT ────────────────────────────────────────────┐")
        print("│  Regime: NOT ACTIVE (reference never established)          │")
        print("│  → All gate decisions driven by change detector alone      │")
        print("└─────────────────────────────────────────────────────────────┘")

    # Quick verdict (heuristic; interpret cc['tp'] as "true detection" in corrupted runs)
    print("\n" + "-" * 65)
    low_fp = cc["fp"] <= 2

    # If there are no onset gates, there is no transition-to-detect in this window.
    # This is expected for control runs, or indicates a spike window mismatch.
    if onset_total == 0:
        if low_fp:
            verdict = "✓ CONTROL/NO-ONSET: No transition to detect, quiet baseline"
        else:
            verdict = "⚠ CONTROL/NO-ONSET: No transition, but noisy (consider raising ε)"
    else:
        # For spike experiments, success is primarily about catching the onset.
        change_detected = onset_tp > 0
        if change_detected and low_fp:
            verdict = "✓ GOOD: Caught onset with low false alarms"
        elif change_detected and not low_fp:
            verdict = "⚠ MIXED: Caught onset but noisy (consider raising ε)"
        elif not change_detected and low_fp:
            verdict = "⚠ MISSED: Missed onset, but quiet (consider lowering ε)"
        else:
            verdict = "✗ BAD: Missed onset AND noisy"
    print(f" VERDICT: {verdict}")
    print("-" * 65)

    # =========================================================================
    # DETAILED ANALYSIS
    # =========================================================================
    print("\n" + "=" * 65)
    print(" DETAILED ANALYSIS")
    print("=" * 65)

    # Union gate confusion matrix
    conf = a["confusion"]
    print("\n--- UNION GATE (change OR regime) ---")
    print("Note: Not recommended for spike scoring when regime is active.\n")
    print("Confusion Matrix:")
    print("                    | Spike Overlap | No Overlap |")
    print(f"  Gate FAIL         |      {conf['tp']:3d}      |     {conf['fp']:3d}      |")
    print(f"  Gate OK           |      {conf['fn']:3d}      |     {conf['tn']:3d}      |")

    print(f"\nPrecision:   {a['precision']:.3f}")
    print(f"Recall:      {a['recall']:.3f}")
    print(f"Specificity: {a['specificity']:.3f}")

    print(f"\nTotal gates: {a['total_gates']}")
    print(f"Gates with spike overlap: {a['total_overlap']}")
    print(f"Gates without overlap:    {a['total_nonoverlap']}")

    # ---- Policy Decomposition ----
    print("\n" + "=" * 65)
    print(" POLICY DECOMPOSITION: CHANGE vs REGIME")
    print("=" * 65)

    # Change detection
    print("\n--- CHANGE DETECTION (past vs recent) ---")
    print("                    | Spike Overlap | No Overlap |")
    print(f"  Change FAIL       |      {cc['tp']:3d}      |     {cc['fp']:3d}      |")
    print(f"  Change OK         |      {cc['fn']:3d}      |     {cc['tn']:3d}      |")
    print(f"\n  Precision:   {a['change_precision']:.3f}")
    print(f"  Recall:      {a['change_recall']:.3f}")
    if onset_total > 0:
        print(f"  Onset recall: {float(onset.get('change_recall', float('nan'))):.3f} ({onset_tp}/{onset_total})")
    else:
        print("  Onset gates:  0 (no transition in window — control or mismatch)")
    print(f"  Specificity: {a['change_specificity']:.3f}")

    print(f"  Valid rows: {n_valid}  | excluded_not_evaluated={ex_ne}, excluded_unknown_stability={ex_unk}")

    # Regime detection
    print(f"\n--- REGIME DETECTION (ref vs recent) [n={a['regime_active_gates']} active gates] ---")
    print("                    | Spike Overlap | No Overlap |")
    print(f"  Regime FAIL       |      {rc['tp']:3d}      |     {rc['fp']:3d}      |")
    print(f"  Regime OK         |      {rc['fn']:3d}      |     {rc['tn']:3d}      |")
    print(f"\n  Precision:   {a['regime_precision']:.3f}")
    print(f"  Recall:      {a['regime_recall']:.3f}")
    print(f"  Specificity: {a['regime_specificity']:.3f}")

    # ---- Attribution breakdown ----
    print("\n" + "-" * 65)
    print(" SPIKE OVERLAP FAILURE ATTRIBUTION")
    print("-" * 65)

    change_total_fail = cc["tp"] + cc["fp"]
    regime_total_fail = rc["tp"] + rc["fp"]

    print(f"Change detector fired {change_total_fail} times total")
    print(f"Regime detector fired {regime_total_fail} times total")

    attr = a.get("spike_attribution", {})
    print(f"\nWithin spike overlap windows ({a['total_overlap']} gates):")
    print(f"  Both detectors failed:    {attr.get('both_fail', 0)}")
    print(f"  Change-only failed:       {attr.get('change_only', 0)}")
    print(f"  Regime-only failed:       {attr.get('regime_only', 0)}")

    # ---- Regime state summary ----
    regime_enabled_count = sum(1 for g in per_gate if g.get("regime_enabled", False))
    regime_active_count = sum(1 for g in per_gate if g.get("regime_active", False))

    print(f"\n{'=' * 65}")
    print(" REGIME STATE SUMMARY")
    print("=" * 65)
    print(f"Gates with regime enabled: {regime_enabled_count}/{len(per_gate)}")
    print(f"Gates with regime active:  {regime_active_count}/{len(per_gate)}")

    # Find reference establishment point
    ref_established = None
    for g in per_gate:
        if g.get("ref_just_established"):
            ref_established = g.get("iter")
            break

    if ref_established is not None:
        print(f"Reference established at:  iter {ref_established}")
    else:
        print("Reference: NOT ESTABLISHED")

    if not verbose:
        return

    # =========================================================================
    # VERBOSE: DETAILED TABLE
    # =========================================================================
    overlap = [r for r in per_gate if r["overlaps_spike"]]
    overlap_sorted = sorted(overlap, key=lambda x: x["iter"])

    print("\n" + "-" * 65)
    print(" GATES WITH SPIKE OVERLAP (sorted by iter)")
    print("-" * 65)
    print(f"{'iter':>6}*| {'union':^6} | {'change':^6} | {'regime':^6} | {'ov%':>5} | {'chg_DW':>8} | {'reg_DW':>8}")
    print("-" * 65)

    def _fmt(x: Any) -> str:
        """Format a numeric value, handling nan/None gracefully."""
        if x is None:
            return "n/a".rjust(8)
        try:
            f = float(x)
            return f"{f:.4f}" if math.isfinite(f) else "nan".rjust(8)
        except (ValueError, TypeError):
            return "n/a".rjust(8)

    for r in overlap_sorted:
        union_status = "FAIL" if r["fail"] else "ok"
        if r.get("change_dw_ok") is True:
            change_status = "ok"
        elif r.get("change_dw_ok") is False:
            change_status = "FAIL"
        else:
            change_status = "n/a"

        # Regime status: n/a if not active, otherwise FAIL/ok
        if r["regime_active"]:
            regime_status = "FAIL" if not r["regime_ok"] else "ok"
        else:
            regime_status = "n/a"

        ovp = 100.0 * float(r["overlap_frac"])
        chg_dw = _fmt(r["worst_DW"])
        reg_dw = _fmt(r["regime_worst_DW"])

        onset_marker = "*" if r.get("onset_gate") else " "
        print(
            f"{r['iter']:>6}{onset_marker}| {union_status:^6} | {change_status:^6} | {regime_status:^6} | "
            f"{ovp:>4.0f}% | {chg_dw:>8} | {reg_dw:>8}"
        )

    print("(* = onset gate)")

    # ---- D_W distribution for spike gates ----
    spike_dws = [r["worst_DW"] for r in overlap_sorted if math.isfinite(r.get("worst_DW", float("nan")))]
    if spike_dws:
        print(
            f"\nSpike-overlap D_W stats: "
            f"min={min(spike_dws):.4f}, max={max(spike_dws):.4f}, "
            f"mean={sum(spike_dws) / len(spike_dws):.4f}"
        )
        # Try to show epsilon for context
        first_eps = next(
            (r["eps_eff"] for r in overlap_sorted if math.isfinite(r.get("eps_eff", float("nan")))),
            None,
        )
        if first_eps is not None:
            print(f"Effective epsilon (ε_eff): {first_eps:.4f}")
            above_eps = sum(1 for d in spike_dws if d > first_eps)
            print(f"D_W > ε_eff: {above_eps}/{len(spike_dws)} spike gates")

    # ---- False negatives detail ----
    fns = [r for r in overlap_sorted if r["ok"] and r["overlaps_spike"]]
    if fns:
        print("\n" + "-" * 65)
        print(" FALSE NEGATIVES (overlap but union OK)")
        print("-" * 65)
        for r in fns:
            rs, re = r["recent_window"]
            print(f"\niter {r['iter']}: recent=[{rs},{re}) overlap={100.0 * r['overlap_frac']:.0f}%")
            print(
                f"  change_dw_ok={r.get('change_dw_ok')}, evaluated={r.get('evaluated')}, base_reason={r.get('base_reason')}, "
                f"regime_ok={r['regime_ok']}, regime_active={r['regime_active']}"
            )
            print(f"  change_DW={_fmt(r['worst_DW'])}, regime_DW={_fmt(r['regime_worst_DW'])}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze veriscope gate precision/recall vs spike overlap.")
    p.add_argument("--results", required=True, nargs="+", type=str)
    p.add_argument("--spike_start", required=True, type=int)
    p.add_argument("--spike_len", required=True, type=int)
    p.add_argument("--gate_window", default=100, type=int)
    p.add_argument("--metric_interval", default=2, type=int)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--control", action="store_true", help="Treat this results file as a control run (disable onset scoring)."
    )
    args = p.parse_args()

    results_paths = _expand_results_args(args.results)
    if not results_paths:
        raise SystemExit("No --results paths provided after parsing.")

    analyses: List[Dict[str, Any]] = [
        analyze_gates(
            results_path=rp,
            spike_start=args.spike_start,
            spike_len=args.spike_len,
            gate_window=args.gate_window,
            metric_interval=args.metric_interval,
            control=bool(args.control),
        )
        for rp in results_paths
    ]

    if args.json:
        out_obj: Any = analyses[0] if len(analyses) == 1 else analyses
        if args.quiet:
            if isinstance(out_obj, list):
                out_obj = [{k: v for (k, v) in a.items() if k != "per_gate"} for a in out_obj]
            else:
                out_obj = {k: v for (k, v) in out_obj.items() if k != "per_gate"}
        print(json.dumps(out_obj, indent=2, default=str))
    else:
        for a in analyses:
            if len(analyses) > 1:
                print("\n" + "=" * 80)
                print(f"RESULTS: {a.get('results_file', '')}")
                print("=" * 80)
            print_analysis(a, verbose=not args.quiet)


if __name__ == "__main__":
    main()
