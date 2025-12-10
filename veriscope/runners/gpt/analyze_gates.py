# veriscope/runners/gpt/analyze_gates.py
"""
Gate recall/precision analysis for veriscope GPT experiments.

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
    """
    Compute the iteration range covered by the *recent half-window* for a gate check.

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

        # normalize by the actual iteration-span of the recent window
        interval = max(1, int(metric_interval))
        wm = max(1, int(gate_window) // interval)
        w_iter = int(wm * interval)
        overlap_frac = float(overlap_iters) / float(w_iter) if w_iter > 0 else 0.0

        per_gate.append(
            {
                "iter": gate_iter,
                "ok": ok,
                "fail": (not ok),
                "overlaps_spike": overlaps,
                "overlap_frac": overlap_frac,
                "recent_window": (recent_start, recent_end),
                "worst_DW": audit.get("worst_DW", float("nan")),
                "eps_eff": audit.get("eps_eff", float("nan")),
                "gain_bits": audit.get("gain_bits", float("nan")),
            }
        )

    tp = sum(1 for r in per_gate if r["fail"] and r["overlaps_spike"])
    fp = sum(1 for r in per_gate if r["fail"] and not r["overlaps_spike"])
    fn = sum(1 for r in per_gate if r["ok"] and r["overlaps_spike"])
    tn = sum(1 for r in per_gate if r["ok"] and not r["overlaps_spike"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "results_file": str(results_path),
        "spike": {"start": int(spike_start), "end": int(spike_end), "len": int(spike_len)},
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "total_gates": len(per_gate),
        "total_overlap": tp + fn,
        "total_nonoverlap": tn + fp,
        "per_gate": per_gate,
    }


def print_analysis(a: Dict[str, Any], verbose: bool) -> None:
    conf = a["confusion"]

    print("\n" + "=" * 64)
    print("GATE RECALL/PRECISION ANALYSIS")
    print("=" * 64)

    print("\nConfusion Matrix:")
    print("                    | Spike Overlap | No Overlap |")
    print(f"  Gate FAIL         |      {conf['tp']:3d}      |     {conf['fp']:3d}      |")
    print(f"  Gate OK           |      {conf['fn']:3d}      |     {conf['tn']:3d}      |")

    print(f"\nPrecision:   {a['precision']:.3f}")
    print(f"Recall:      {a['recall']:.3f}")
    print(f"Specificity: {a['specificity']:.3f}")

    print(f"\nTotal gates: {a['total_gates']}")
    print(f"Gates with spike overlap: {a['total_overlap']}")
    print(f"Gates without overlap:    {a['total_nonoverlap']}")

    if not verbose:
        return

    overlap = [r for r in a["per_gate"] if r["overlaps_spike"]]
    overlap_sorted = sorted(overlap, key=lambda x: x["iter"])

    print("\n" + "-" * 64)
    print("GATES WITH SPIKE OVERLAP (sorted by iter)")
    print("-" * 64)
    print(f"{'iter':>6} | {'status':^8} | {'ov%':>6} | {'worst_DW':>9} | {'eps_eff':>7} | {'gain':>9}")
    print("-" * 64)

    for r in overlap_sorted:
        status = "FAIL" if r["fail"] else "ok"
        ovp = 100.0 * float(r["overlap_frac"])
        worst = r["worst_DW"]
        eps = r["eps_eff"]
        gain = r["gain_bits"]

        def _fmt(x: Any) -> str:
            return f"{float(x):.4f}" if isinstance(x, (int, float)) and math.isfinite(float(x)) else "nan"

        print(f"{r['iter']:>6} | {status:^8} | {ovp:>5.0f}% | {_fmt(worst):>9} | {_fmt(eps):>7} | {_fmt(gain):>9}")

    fns = [r for r in overlap_sorted if (r["ok"] and r["overlaps_spike"])]
    if fns:
        print("\n" + "-" * 64)
        print("FALSE NEGATIVES (overlap but OK)")
        print("-" * 64)
        for r in fns:
            rs, re = r["recent_window"]
            print(f"\niter {r['iter']}: recent=[{rs},{re}) overlap={100.0 * r['overlap_frac']:.0f}%")
            print(f"  worst_DW={r['worst_DW']}, eps_eff={r['eps_eff']}, gain_bits={r['gain_bits']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze veriscope gate precision/recall vs spike overlap.")
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
