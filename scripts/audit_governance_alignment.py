#!/usr/bin/env python3
from __future__ import annotations

import json
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


def _read_governance_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception as exc:
            raise SystemExit(f"[audit] failed to parse {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SystemExit(f"[audit] expected object at {path}:{line_no}")
        events.append(obj)
    return events


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _validate_tuple(tag: str, *, decision: Any, ok: Any, warn: Any, evaluated: Any, errors: List[str]) -> None:
    if decision not in DECISIONS:
        errors.append(f"{tag}: invalid decision={decision!r}")
        return
    ev = bool(evaluated)
    if not ev:
        if decision != "skip":
            errors.append(f"{tag}: evaluated=False requires decision='skip'")
        if warn is not False:
            errors.append(f"{tag}: evaluated=False requires warn=False (got {warn!r})")
    if decision in {"pass", "warn"} and ok is not True:
        errors.append(f"{tag}: decision={decision!r} requires ok=True (got {ok!r})")
    if decision == "fail":
        if ok is not False:
            errors.append(f"{tag}: decision='fail' requires ok=False (got {ok!r})")
        if warn is not False:
            errors.append(f"{tag}: decision='fail' requires warn=False (got {warn!r})")
    if bool(warn) != (decision == "warn"):
        errors.append(f"{tag}: warn must be true iff decision='warn' (decision={decision!r}, warn={warn!r})")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/audit_governance_alignment.py OUTDIR", file=sys.stderr)
        return 2

    outdir = Path(sys.argv[1]).expanduser().resolve()
    results_path = outdir / "results.json"
    gov_path = outdir / "governance_log.jsonl"

    if not results_path.exists():
        print(f"[audit] missing file: {results_path}", file=sys.stderr)
        return 2
    if not gov_path.exists():
        print(f"[audit] missing file: {gov_path}", file=sys.stderr)
        return 2

    results = _load_json(results_path)
    run_id = results.get("run_id")
    gates = results.get("gates")
    if not isinstance(gates, list):
        print("[audit] results.json missing array 'gates'", file=sys.stderr)
        return 2

    events = _read_governance_events(gov_path)
    gate_events: List[Dict[str, Any]] = []
    prev_rev: Optional[int] = None
    errors: List[str] = []

    for idx, event in enumerate(events):
        rev = _as_int(event.get("rev"))
        if rev is None:
            errors.append(f"governance[{idx}] missing integer rev")
        elif prev_rev is not None and rev != prev_rev + 1:
            errors.append(f"governance[{idx}] rev={rev} not consecutive after {prev_rev}")
        prev_rev = rev if rev is not None else prev_rev

        ev_name = event.get("event") or event.get("event_type")
        if ev_name == "gate_decision_v1":
            payload = event.get("payload")
            if not isinstance(payload, dict):
                errors.append(f"governance[{idx}] gate_decision_v1 missing object payload")
                continue
            gate_events.append(payload)

    if len(gate_events) != len(gates):
        errors.append(f"gate_decision_v1 count={len(gate_events)} does not match results gates={len(gates)}")

    compare_n = min(len(gate_events), len(gates))
    prev_iter: Optional[int] = None
    for i in range(compare_n):
        gp = gate_events[i]
        gr = gates[i]
        tag = f"gate[{i}]"

        if gp.get("run_id") != run_id:
            errors.append(f"{tag}: governance run_id={gp.get('run_id')!r} != results run_id={run_id!r}")

        g_iter = _as_int(gp.get("iter"))
        r_iter = _as_int(gr.get("iter"))
        if g_iter is None or r_iter is None:
            errors.append(f"{tag}: missing integer iter in governance/results")
        elif g_iter != r_iter:
            errors.append(f"{tag}: iter mismatch governance={g_iter} results={r_iter}")
        if g_iter is not None and prev_iter is not None and g_iter <= prev_iter:
            errors.append(f"{tag}: governance iter={g_iter} not strictly increasing (prev={prev_iter})")
        if g_iter is not None:
            prev_iter = g_iter

        g_decision = gp.get("decision")
        r_decision = gr.get("decision")
        if g_decision != r_decision:
            errors.append(f"{tag}: decision mismatch governance={g_decision!r} results={r_decision!r}")
        if gp.get("ok") != gr.get("ok"):
            errors.append(f"{tag}: ok mismatch governance={gp.get('ok')!r} results={gr.get('ok')!r}")
        if gp.get("warn") != gr.get("warn"):
            errors.append(f"{tag}: warn mismatch governance={gp.get('warn')!r} results={gr.get('warn')!r}")

        g_audit = gp.get("audit")
        r_audit = gr.get("audit")
        if not isinstance(g_audit, dict) or not isinstance(r_audit, dict):
            errors.append(f"{tag}: missing audit object in governance/results")
            continue

        g_eval = g_audit.get("evaluated")
        r_eval = r_audit.get("evaluated")
        if bool(g_eval) != bool(r_eval):
            errors.append(f"{tag}: evaluated mismatch governance={g_eval!r} results={r_eval!r}")

        _validate_tuple(
            f"{tag}/governance",
            decision=g_decision,
            ok=gp.get("ok"),
            warn=gp.get("warn"),
            evaluated=g_eval,
            errors=errors,
        )
        _validate_tuple(
            f"{tag}/results",
            decision=r_decision,
            ok=gr.get("ok"),
            warn=gr.get("warn"),
            evaluated=r_eval,
            errors=errors,
        )

        for key in ("reason", "policy"):
            if g_audit.get(key) != r_audit.get(key):
                errors.append(
                    f"{tag}: audit.{key} mismatch governance={g_audit.get(key)!r} results={r_audit.get(key)!r}"
                )

    if errors:
        print("[audit] FAILED", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        return 1

    print(f"[audit] OK: {compare_n} gate rows aligned between results and governance in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
