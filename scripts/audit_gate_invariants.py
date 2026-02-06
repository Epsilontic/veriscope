#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


DECISIONS = {"pass", "warn", "fail", "skip"}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[audit] failed to parse {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"[audit] expected JSON object at {path}")
    return obj


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _derive_final(pass_n: int, warn_n: int, fail_n: int) -> str:
    if fail_n > 0:
        return "fail"
    if warn_n > 0:
        return "warn"
    if pass_n > 0:
        return "pass"
    return "skip"


def _metric_count(gate: Dict[str, Any], window_sig: Dict[str, Any]) -> Optional[int]:
    audit = gate.get("audit")
    if isinstance(audit, dict):
        cbm = audit.get("counts_by_metric")
        if isinstance(cbm, dict) and cbm:
            return len(cbm)
    evidence = window_sig.get("evidence")
    if isinstance(evidence, dict):
        metrics = evidence.get("metrics")
        if isinstance(metrics, list):
            return len(metrics)
    return None


def _max_half_window_samples(window_sig: Dict[str, Any]) -> Optional[int]:
    gate_controls = window_sig.get("gate_controls")
    if not isinstance(gate_controls, dict):
        return None
    gate_window = _coerce_int(gate_controls.get("gate_window"))
    if gate_window is None or gate_window <= 0:
        return None
    metric_interval = _coerce_int(gate_controls.get("metric_interval"))
    if metric_interval is not None and metric_interval > 0:
        return max(1, gate_window // metric_interval)
    return gate_window


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/audit_gate_invariants.py OUTDIR", file=sys.stderr)
        return 2

    outdir = Path(sys.argv[1]).expanduser().resolve()
    results_path = outdir / "results.json"
    if not results_path.exists():
        print(f"[audit] missing file: {results_path}", file=sys.stderr)
        return 2

    results = _load_json(results_path)
    gates = results.get("gates")
    if not isinstance(gates, list):
        print(f"[audit] results.json missing array 'gates': {results_path}", file=sys.stderr)
        return 2

    window_sig = {}
    ws_path = outdir / "window_signature.json"
    if ws_path.exists():
        window_sig = _load_json(ws_path)
    max_half = _max_half_window_samples(window_sig)

    errors: List[str] = []
    pass_n = warn_n = fail_n = skip_n = 0
    prev_iter: Optional[int] = None

    for idx, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(f"gate[{idx}] must be an object")
            continue
        it = _coerce_int(gate.get("iter"))
        if it is None:
            errors.append(f"gate[{idx}] missing integer iter")
        elif prev_iter is not None and it <= prev_iter:
            errors.append(f"gate[{idx}] iter={it} not strictly increasing (prev={prev_iter})")
        prev_iter = it if it is not None else prev_iter

        decision = gate.get("decision")
        if decision not in DECISIONS:
            errors.append(f"gate[{idx}] invalid decision={decision!r}")
            continue

        if decision == "pass":
            pass_n += 1
        elif decision == "warn":
            warn_n += 1
        elif decision == "fail":
            fail_n += 1
        else:
            skip_n += 1

        audit = gate.get("audit")
        if not isinstance(audit, dict):
            errors.append(f"gate[{idx}] audit must be an object")
            continue

        evaluated = bool(audit.get("evaluated", True))
        ok = gate.get("ok")
        warn = gate.get("warn")

        if not evaluated:
            if decision != "skip":
                errors.append(f"gate[{idx}] evaluated=False requires decision='skip' (got {decision!r})")
            if warn is not False:
                errors.append(f"gate[{idx}] evaluated=False requires warn=False (got {warn!r})")
        if decision in {"pass", "warn"} and ok is not True:
            errors.append(f"gate[{idx}] decision={decision!r} requires ok=True (got {ok!r})")
        if decision == "fail":
            if ok is not False:
                errors.append(f"gate[{idx}] decision='fail' requires ok=False (got {ok!r})")
            if warn is not False:
                errors.append(f"gate[{idx}] decision='fail' requires warn=False (got {warn!r})")
        if bool(warn) != (decision == "warn"):
            errors.append(f"gate[{idx}] warn must be true iff decision='warn' (decision={decision!r}, warn={warn!r})")

        if evaluated:
            if not audit.get("reason"):
                errors.append(f"gate[{idx}] evaluated=True missing audit.reason")
            if not audit.get("policy"):
                errors.append(f"gate[{idx}] evaluated=True missing audit.policy")
            ev_total = _coerce_int(audit.get("evidence_total"))
            min_ev = _coerce_int(audit.get("min_evidence"))
            if ev_total is None or ev_total < 0:
                errors.append(f"gate[{idx}] evaluated=True invalid evidence_total={audit.get('evidence_total')!r}")
            if min_ev is None or min_ev < 0:
                errors.append(f"gate[{idx}] evaluated=True invalid min_evidence={audit.get('min_evidence')!r}")

            metric_n = _metric_count(gate, window_sig)
            if max_half is not None and metric_n is not None and ev_total is not None:
                max_expected = max_half * metric_n
                if ev_total > max_expected:
                    errors.append(
                        f"gate[{idx}] evidence_total={ev_total} exceeds window bound={max_expected} "
                        f"(half_window={max_half}, metric_count={metric_n})"
                    )

        per_metric = audit.get("per_metric_tv", {})
        if isinstance(per_metric, dict):
            for name, value in per_metric.items():
                try:
                    fv = float(value["tv"] if isinstance(value, dict) and "tv" in value else value)
                except Exception:
                    continue
                if not math.isfinite(fv):
                    continue
                if fv < 0.0 or fv > 1.0:
                    errors.append(f"gate[{idx}] per_metric_tv[{name!r}]={fv} outside [0,1]")

    summary_path = outdir / "results_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        counts = summary.get("counts")
        if isinstance(counts, dict):
            c_eval = _coerce_int(counts.get("evaluated"))
            c_skip = _coerce_int(counts.get("skip"))
            c_pass = _coerce_int(counts.get("pass"))
            c_warn = _coerce_int(counts.get("warn"))
            c_fail = _coerce_int(counts.get("fail"))
            if None in {c_eval, c_skip, c_pass, c_warn, c_fail}:
                errors.append("results_summary.json has non-integer counts")
            else:
                if c_eval != (c_pass + c_warn + c_fail):
                    errors.append("results_summary counts violate evaluated == pass+warn+fail")
                if c_eval != (pass_n + warn_n + fail_n) or c_skip != skip_n:
                    errors.append(
                        "results_summary counts do not match results.json gates "
                        f"(summary eval/skip={c_eval}/{c_skip}, derived={pass_n + warn_n + fail_n}/{skip_n})"
                    )
                expected_final = _derive_final(pass_n, warn_n, fail_n)
                if summary.get("final_decision") != expected_final:
                    errors.append(
                        f"results_summary final_decision={summary.get('final_decision')!r} expected={expected_final!r}"
                    )

    if errors:
        print("[audit] FAILED", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    print(f"[audit] OK: {len(gates)} gates checked in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
