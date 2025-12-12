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
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def analyze_gates(
    results_path: Path,
    spike_start: int,
    spike_len: int,
    gate_window: int,
    metric_interval: int,
) -> Dict[str, Any]:
    spike_end = int(spike_start) + int(spike_len)
    data = json.loads(results_path.read_text())
    gates: List[Dict[str, Any]] = list(data.get("gates", []) or [])

    per_gate: List[Dict[str, Any]] = []
    for g in gates:
        gate_iter = int(g.get("iter", 0) or 0)
        ok = bool(g.get("ok", True))
        audit = g.get("audit", {}) or {}

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

        # NOTE: Audit key names must match trainer output in train_nanogpt.py.
        # The trainer's _compute_gate_check() and RegimeAnchoredGateEngine.check()
        # populate: change_ok, regime_ok, regime_active, regime_enabled,
        # regime_worst_DW, worst_DW (change detector), eps_eff, gain_bits.
        per_gate.append(
            {
                "iter": gate_iter,
                "ok": ok,
                "fail": (not ok),
                "overlaps_spike": overlaps,
                "overlap_frac": overlap_frac,
                "recent_window": (recent_start, recent_end),
                # Change detector fields
                "worst_DW": audit.get("worst_DW", float("nan")),
                "eps_eff": audit.get("eps_eff", float("nan")),
                "gain_bits": audit.get("gain_bits", float("nan")),
                # Regime fields (defaults are conservative: assume OK if missing)
                "change_ok": bool(audit.get("change_ok", True)),
                "regime_ok": bool(audit.get("regime_ok", True)),
                "regime_active": bool(audit.get("regime_active", False)),
                "regime_enabled": bool(audit.get("regime_enabled", False)),
                "regime_worst_DW": audit.get("regime_worst_DW", float("nan")),
                "ref_established_at": audit.get("ref_established_at"),
                "ref_just_established": bool(audit.get("ref_just_established", False)),
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

    # ---- Change detection breakdown ----
    change_tp = sum(1 for r in per_gate if (not r["change_ok"]) and r["overlaps_spike"])
    change_fp = sum(1 for r in per_gate if (not r["change_ok"]) and (not r["overlaps_spike"]))
    change_fn = sum(1 for r in per_gate if r["change_ok"] and r["overlaps_spike"])
    change_tn = sum(1 for r in per_gate if r["change_ok"] and (not r["overlaps_spike"]))

    change_precision = change_tp / (change_tp + change_fp) if (change_tp + change_fp) > 0 else float("nan")
    change_recall = change_tp / (change_tp + change_fn) if (change_tp + change_fn) > 0 else float("nan")
    change_specificity = change_tn / (change_tn + change_fp) if (change_tn + change_fp) > 0 else float("nan")

    # ---- Regime detection breakdown (only when active) ----
    regime_active_gates = [r for r in per_gate if r["regime_active"]]

    regime_tp = sum(1 for r in regime_active_gates if (not r["regime_ok"]) and r["overlaps_spike"])
    regime_fp = sum(1 for r in regime_active_gates if (not r["regime_ok"]) and (not r["overlaps_spike"]))
    regime_fn = sum(1 for r in regime_active_gates if r["regime_ok"] and r["overlaps_spike"])
    regime_tn = sum(1 for r in regime_active_gates if r["regime_ok"] and (not r["overlaps_spike"]))

    regime_precision = regime_tp / (regime_tp + regime_fp) if (regime_tp + regime_fp) > 0 else float("nan")
    regime_recall = regime_tp / (regime_tp + regime_fn) if (regime_tp + regime_fn) > 0 else float("nan")
    regime_specificity = regime_tn / (regime_tn + regime_fp) if (regime_tn + regime_fp) > 0 else float("nan")

    # ---- Attribution: who drives union failures in spike window? ----
    both_fail_in_spike = sum(
        1
        for r in per_gate
        if (not r["change_ok"])
        and r["regime_active"]
        and (not r["regime_ok"])
        and r["overlaps_spike"]
    )
    change_only_in_spike = sum(
        1
        for r in per_gate
        if (not r["change_ok"]) and (not r["regime_active"] or r["regime_ok"]) and r["overlaps_spike"]
    )
    regime_only_in_spike = sum(
        1
        for r in per_gate
        if r["change_ok"] and r["regime_active"] and (not r["regime_ok"]) and r["overlaps_spike"]
    )

    return {
        "results_file": str(results_path),
        "spike": {"start": int(spike_start), "end": int(spike_end), "len": int(spike_len)},
        # Union gate (existing)
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        # Change detection decomposition
        "change_confusion": {"tp": change_tp, "fp": change_fp, "fn": change_fn, "tn": change_tn},
        "change_precision": change_precision,
        "change_recall": change_recall,
        "change_specificity": change_specificity,
        # Regime detection decomposition
        "regime_confusion": {"tp": regime_tp, "fp": regime_fp, "fn": regime_fn, "tn": regime_tn},
        "regime_precision": regime_precision,
        "regime_recall": regime_recall,
        "regime_specificity": regime_specificity,
        "regime_active_gates": len(regime_active_gates),
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
    rc = a["regime_confusion"]

    # =========================================================================
    # EXECUTIVE SUMMARY (what you actually care about)
    # =========================================================================
    print("\n" + "=" * 65)
    print(" EXECUTIVE SUMMARY")
    print("=" * 65)

    # Corruption detection (change-only)
    print("\n┌─ CORRUPTION DETECTION (use this for spike experiments) ──────┐")
    print(
        f"│  Precision:    {_fmt_metric(a['change_precision']):>5}  "
        f"(when alarmed, was it real?)             │"
    )
    print(
        f"│  Recall:       {_fmt_metric(a['change_recall']):>5}  "
        f"(did we catch the corruption?)           │"
    )
    print(
        f"│  Specificity:  {_fmt_metric(a['change_specificity']):>5}  "
        f"(quiet when clean?)                      │"
    )
    print(
        f"│  Change FAILs: {cc['tp']:>3} in-spike, "
        f"{cc['fp']:>3} off-spike (of {a['total_overlap']} overlap gates)  │"
    )
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
            f"│  Regime FAILs:   {regime_total_fail:>3} total "
            f"({rc['tp']} in spike, {rc['fp']} outside)              │"
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
    change_detected = cc["tp"] > 0
    low_fp = cc["fp"] <= 2
    if change_detected and low_fp:
        verdict = "✓ GOOD: Detected corruption with low false alarms"
    elif change_detected and not low_fp:
        verdict = "⚠ MIXED: Detected corruption but noisy (consider raising ε)"
    elif not change_detected and low_fp:
        verdict = "⚠ MISSED: No detection, but quiet (consider lowering ε)"
    else:
        verdict = "✗ BAD: No detection AND noisy"
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
    print(f"  Specificity: {a['change_specificity']:.3f}")

    # Regime detection
    print(
        f"\n--- REGIME DETECTION (ref vs recent) [n={a['regime_active_gates']} active gates] ---"
    )
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
    print(
        f"{'iter':>6} | {'union':^6} | {'change':^6} | {'regime':^6} | "
        f"{'ov%':>5} | {'chg_DW':>8} | {'reg_DW':>8}"
    )
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
        change_status = "FAIL" if not r["change_ok"] else "ok"

        # Regime status: n/a if not active, otherwise FAIL/ok
        if r["regime_active"]:
            regime_status = "FAIL" if not r["regime_ok"] else "ok"
        else:
            regime_status = "n/a"

        ovp = 100.0 * float(r["overlap_frac"])
        chg_dw = _fmt(r["worst_DW"])
        reg_dw = _fmt(r["regime_worst_DW"])

        print(
            f"{r['iter']:>6} | {union_status:^6} | {change_status:^6} | {regime_status:^6} | "
            f"{ovp:>4.0f}% | {chg_dw:>8} | {reg_dw:>8}"
        )

    # ---- D_W distribution for spike gates ----
    spike_dws = [
        r["worst_DW"]
        for r in overlap_sorted
        if math.isfinite(r.get("worst_DW", float("nan")))
    ]
    if spike_dws:
        print(
            f"\nSpike-overlap D_W stats: "
            f"min={min(spike_dws):.4f}, max={max(spike_dws):.4f}, "
            f"mean={sum(spike_dws)/len(spike_dws):.4f}"
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
                f"  change_ok={r['change_ok']}, regime_ok={r['regime_ok']}, "
                f"regime_active={r['regime_active']}"
            )
            print(
                f"  change_DW={_fmt(r['worst_DW'])}, "
                f"regime_DW={_fmt(r['regime_worst_DW'])}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze veriscope gate precision/recall vs spike overlap."
    )
    p.add_argument("--results", required=True, type=str)
    p.add_argument("--spike_start", required=True, type=int)
    p.add_argument("--spike_len", required=True, type=int)
    p.add_argument("--gate_window", default=100, type=int)
    p.add_argument("--metric_interval", default=2, type=int)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    analysis = analyze_gates(
        results_path=Path(args.results),
        spike_start=args.spike_start,
        spike_len=args.spike_len,
        gate_window=args.gate_window,
        metric_interval=args.metric_interval,
    )

    if args.json:
        if args.quiet:
            analysis = {k: v for (k, v) in analysis.items() if k != "per_gate"}
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print_analysis(analysis, verbose=not args.quiet)


if __name__ == "__main__":
    main()
