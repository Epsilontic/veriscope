# veriscope/runners/gpt/emit_artifacts.py
from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import veriscope
from veriscope.core.governance import (
    append_gate_decision,
    append_overrides,
    append_run_started,
    build_distributed_context,
)
from veriscope.core.artifacts import (
    AuditV1,
    CountsV1,
    GateRecordV1,
    ProfileV1,
    ResultsSummaryV1,
    ResultsV1,
    WindowSignatureRefV1,
    derive_gate_decision,
    derive_final_decision,
)
from veriscope.core.jsonutil import (
    atomic_write_json,
    atomic_write_pydantic_json,
    atomic_write_text,
    window_signature_sha256,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmittedArtifactsV1:
    window_signature_path: Path
    results_path: Path
    results_summary_path: Path
    window_signature_hash: str


def _iso_z(dt: datetime) -> str:
    """
    Serialize datetime as a stable ISO-8601 UTC string (Z-suffixed, no microseconds).

    Policy:
      - naive datetime => treated as UTC (internal convention)
      - tz-aware datetime => converted to UTC
    """
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
    dt_utc = dt_utc.replace(microsecond=0)
    return dt_utc.isoformat().replace("+00:00", "Z")


def _needs_fill(x: Any) -> bool:
    """True if value is None, empty string, or the string 'none' (case-insensitive)."""
    if x is None:
        return True
    if isinstance(x, str):
        s = x.strip()
        return (s == "") or (s.lower() == "none")
    if isinstance(x, (int, float, bool)):
        return False
    try:
        return str(x).strip().lower() == "none"
    except Exception:
        return False


def _coerce_int(val: Any, *, field: str) -> int:
    """
    Coerce a messy iteration value to int with clear errors.
    Accepts:
      - int
      - float that is integral (e.g., 10.0)
      - str of int (e.g., "0010")
      - str of integral float (e.g., "10.0")
    """
    if isinstance(val, bool):
        # avoid bool subclass-of-int footgun
        raise ValueError(f"{field} must be an integer-like value, got bool={val!r}")

    if isinstance(val, int):
        return val

    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        raise ValueError(f"{field} must be an integer-like value, got float={val!r}")

    if isinstance(val, str):
        s = val.strip()
        if s == "":
            raise ValueError(f"{field} must be an integer-like value, got empty string")
        # "10" or "0010"
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        # "10.0"
        try:
            f = float(s)
        except Exception as e:
            raise ValueError(f"{field} must be an integer-like value, got str={val!r}") from e
        if f.is_integer():
            return int(f)
        raise ValueError(f"{field} must be an integer-like value, got str(float)={val!r}")

    raise ValueError(f"{field} must be an integer-like value, got {type(val).__name__}={val!r}")


def _get_event_iter(ev: Mapping[str, Any]) -> int:
    """Extract iteration number from varied keys with contextual errors."""
    for k in ("iter", "iter_num", "it", "step", "t"):
        if k in ev and ev.get(k) is not None:
            try:
                return _coerce_int(ev.get(k), field=f"gate_history.{k}")
            except Exception as e:
                snippet = repr(dict(ev))[:200]
                raise ValueError(f"Invalid iteration value for key '{k}': {ev.get(k)!r}. snippet={snippet}") from e

    snippet = repr(dict(ev))[:200]
    raise KeyError(f"gate_history event missing iteration key. keys={list(ev.keys())} snippet={snippet}")


def _normalize_policy(policy_val: Any) -> str:
    """
    Policy normalization:
      - accept non-empty strings
      - None/empty/'none' => 'unknown'
      - non-strings => error (avoid str(dict) instability)
    """
    if _needs_fill(policy_val):
        return "unknown"
    if isinstance(policy_val, str):
        s = policy_val.strip()
        return s if s else "unknown"
    raise ValueError(f"audit.policy must be a string (or missing); got {type(policy_val).__name__}={policy_val!r}")


def _normalize_audit_dict(ev: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build an audit dict compatible with AuditV1.

    Hardening:
      - do NOT mutate source event
      - ensure per_metric_tv exists
      - evaluated default True
      - policy: only accept str; None/empty => 'unknown'
      - reason fallback: audit.reason -> event.reason -> derived string
      - normalize worst_dw -> worst_DW
      - evidence defaults
    """
    a: Dict[str, Any] = dict(ev.get("audit") or {})

    # Required by artifact contract
    a.setdefault("per_metric_tv", {})

    # evaluated
    if "evaluated" not in a:
        a["evaluated"] = True
    else:
        a["evaluated"] = bool(a["evaluated"])

    # policy
    policy_val = a.get("policy", ev.get("policy"))
    a["policy"] = _normalize_policy(policy_val)

    # reason fallback (CRITICAL: runner may place row_reason at event root)
    if _needs_fill(a.get("reason")):
        parent_reason = ev.get("reason")
        if not _needs_fill(parent_reason):
            a["reason"] = str(parent_reason)
        else:
            a["reason"] = "evaluated_unknown" if bool(a.get("evaluated", True)) else "not_evaluated"

    # Optional extras copied from event root if audit missing them
    for k in ("base_reason", "change_reason"):
        if k in ev and _needs_fill(a.get(k)):
            a[k] = ev.get(k)

    # Legacy key normalization
    if "worst_DW" not in a and "worst_dw" in a:
        a["worst_DW"] = a.get("worst_dw")

    # Evidence defaults
    for k in ("evidence_total", "min_evidence"):
        if k not in a:
            a[k] = ev.get(k, 0)

    return a


def _sanitize_audit_nonfinite(audit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize non-finite numeric values when audit.evaluated is False.

    For skip decisions, replace any non-finite floats with 0.0 while preserving structure.
    """
    if audit.get("evaluated", True) is not False:
        return audit

    def _is_nonfinite(value: Any) -> bool:
        return isinstance(value, float) and not math.isfinite(value)

    def _sanitize_value(value: Any) -> tuple[Any, int]:
        if _is_nonfinite(value):
            return 0.0, 1
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            replaced = 0
            for key, item in value.items():
                sanitized, n = _sanitize_value(item)
                out[str(key)] = sanitized
                replaced += n
            return out, replaced
        if isinstance(value, list):
            out_list: List[Any] = []
            replaced = 0
            for item in value:
                sanitized, n = _sanitize_value(item)
                out_list.append(sanitized)
                replaced += n
            return out_list, replaced
        return value, 0

    sanitized, replaced = _sanitize_value(dict(audit))
    if replaced > 0 and isinstance(sanitized, dict):
        sanitized.setdefault("sanitized_nonfinite", True)
        sanitized.setdefault("sanitized_nonfinite_count", int(replaced))
        logger.warning("Sanitized %d non-finite audit value(s) for unevaluated gate row", replaced)
    return sanitized


def _derive_decision(ev: Mapping[str, Any], audit: AuditV1) -> str:
    """
    Deterministic gate decision derivation.

    Contract:
      - if audit.evaluated == False => skip
      - prefer explicit ev["decision"] if present
      - else fallback to legacy booleans via canonical fail-dominant precedence

    Observability:
      - warn once per event when legacy booleans are used (no explicit decision).
    """
    if audit.evaluated is False:
        return "skip"

    explicit = ev.get("decision")
    if explicit is not None:
        explicit_decision = str(explicit).strip().lower()
        if explicit_decision not in {"pass", "warn", "fail", "skip"}:
            raise ValueError(f"Invalid explicit gate decision={explicit!r}")
        if ("warn" in ev) or ("ok" in ev):
            derived = derive_gate_decision(
                evaluated=True,
                ok=bool(ev.get("ok", explicit_decision in {"pass", "warn"})),
                warn=bool(ev.get("warn", explicit_decision == "warn")),
            )
            if explicit_decision != derived:
                logger.warning(
                    "Explicit gate decision=%r disagrees with legacy booleans (ok=%r warn=%r) at iter_hint=%r; "
                    "using canonical decision=%r",
                    explicit_decision,
                    ev.get("ok", None),
                    ev.get("warn", None),
                    ev.get("iter", ev.get("iter_num", None)),
                    derived,
                )
                return derived
        return explicit_decision

    warn = bool(ev.get("warn", False))
    ok = bool(ev.get("ok", False))

    # Legacy path used: surface it so we can migrate producers toward explicit `decision`.
    if ("warn" in ev) or ("ok" in ev):
        logger.debug(
            "Gate decision derived from legacy booleans (ok=%r warn=%r) at iter_hint=%r",
            ev.get("ok", None),
            ev.get("warn", None),
            ev.get("iter", ev.get("iter_num", None)),
        )

    return derive_gate_decision(evaluated=True, ok=ok, warn=warn)


def _counts_from_gate_records(gates: Sequence[GateRecordV1]) -> CountsV1:
    """
    CountsV1 semantics: evaluated excludes skip.
    We also enforce that decisions are from the canonical set.
    """
    skip = 0
    warn = 0
    fail = 0
    pass_n = 0

    for r in gates:
        d = r.decision
        if d == "skip":
            skip += 1
        elif d == "warn":
            warn += 1
        elif d == "fail":
            fail += 1
        elif d == "pass":
            pass_n += 1
        else:
            raise ValueError(f"Unexpected gate decision={d!r} at iter={getattr(r, 'iter', None)!r}")

    evaluated = len(gates) - skip
    # CountsV1 requires pass_ (aliased to "pass" in JSON) and enforces pass_+warn+fail==evaluated.
    return CountsV1(evaluated=evaluated, skip=skip, pass_=pass_n, warn=warn, fail=fail)


def _flatten_gate_events(raw: Sequence[Any]) -> List[Any]:
    """Unwrap top-level or per-element {"gates": [...]} wrappers, one level deep."""
    source: Sequence[Any] = raw
    if isinstance(raw, Mapping) and "gates" in raw:
        candidate = raw["gates"]
        if isinstance(candidate, (list, tuple)):
            source = candidate
    out: List[Any] = []
    for item in source:
        if isinstance(item, Mapping) and "gates" in item:
            nested = item["gates"]
            if isinstance(nested, (list, tuple)):
                out.extend(nested)
                continue
        out.append(item)
    return out


def _first_fail_iter_from_events(events: Sequence[Any]) -> Optional[int]:
    """
    Compute earliest fail iteration from a heterogeneous legacy event stream.

    Rules:
      - ignore non-object entries
      - decision precedence:
          event["decision"] OR event["audit"]["decision"] OR event["audit"]["final_decision"]
      - iter precedence:
          event["iter"] OR event["audit"]["iter"] OR event["audit"]["step"]
      - accept iter only when it is an int (bool excluded)
      - return min(iter) for decision=="fail", else None

    Callers should pre-flatten via _flatten_gate_events if the input may contain
    {"gates": [...]} wrappers.
    """
    first_fail_iter: Optional[int] = None

    for event in events:
        if not isinstance(event, Mapping):
            continue

        audit_raw = event.get("audit", {})
        audit = audit_raw if isinstance(audit_raw, Mapping) else {}

        decision_raw = event.get("decision")
        if decision_raw is None:
            decision_raw = audit.get("decision")
        if decision_raw is None:
            decision_raw = audit.get("final_decision")
        if decision_raw is None:
            continue

        decision = str(decision_raw).strip().lower()
        if decision != "fail":
            continue

        iter_raw = event.get("iter")
        if iter_raw is None:
            iter_raw = audit.get("iter")
        if iter_raw is None:
            iter_raw = audit.get("step")

        if isinstance(iter_raw, bool) or not isinstance(iter_raw, int):
            continue

        if first_fail_iter is None or iter_raw < first_fail_iter:
            first_fail_iter = int(iter_raw)

    return first_fail_iter


def _read_json_obj(path: Path) -> Dict[str, Any]:
    """Read JSON file and parse to an object, with a tighter error message."""
    try:
        txt = path.read_text(encoding="utf-8")
        return json.loads(txt)
    except Exception as e:
        raise ValueError(f"Failed to read/parse JSON at {path}") from e


def emit_gpt_artifacts_v1(
    *,
    outdir: Path,
    run_id: str,
    started_ts_utc: datetime,
    ended_ts_utc: Optional[datetime],
    gate_preset: str,
    overrides: Optional[Dict[str, Any]],
    resolved_gate_cfg: Dict[str, Any],
    gate_history: Sequence[Any],
    metric_interval: Optional[int] = None,
    metric_pipeline: Optional[Dict[str, Any]] = None,
    signature_metrics: Optional[Dict[str, Any]] = None,
    dw_aggregator: Optional[Dict[str, Any]] = None,
    run_status: str = "success",
    runner_exit_code: Optional[int] = None,
    runner_signal: Optional[str] = None,
    argv: Optional[Iterable[str]] = None,
    metrics_ref: Optional[Mapping[str, Any]] = None,
) -> EmittedArtifactsV1:
    """
    Emit standardized Veriscope artifacts (V1):
      - window_signature.json
      - results.json
      - results_summary.json

    This function is intentionally CPU-light and does not import nanoGPT/torch.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Emitting GPT artifacts to %s (run_id=%s)", outdir, run_id)

    # --- 1) window_signature.json ---
    code_identity: Dict[str, Any] = {"package_version": veriscope.__version__}

    git_sha = (metric_pipeline or {}).get("git_sha") if metric_pipeline else None
    if not _needs_fill(git_sha):
        code_identity["git_sha"] = str(git_sha)

    window_sig: Dict[str, Any] = {
        "schema_version": 1,
        "code_identity": code_identity,
        "gate_controls": dict(resolved_gate_cfg),
    }
    if metric_interval is not None:
        window_sig["metric_interval"] = int(metric_interval)
    if metric_pipeline is not None:
        window_sig["metric_pipeline"] = dict(metric_pipeline)
    if signature_metrics is not None:
        window_sig["metrics"] = deepcopy(signature_metrics)
    if dw_aggregator is not None:
        window_sig["dw_aggregator"] = deepcopy(dw_aggregator)

    window_signature_path = outdir / "window_signature.json"
    atomic_write_json(window_signature_path, window_sig)

    # Hash consistency: hash what is actually on disk (parsed object), not the in-memory dict.
    window_sig_on_disk = _read_json_obj(window_signature_path)
    ws_hash = window_signature_sha256(window_sig_on_disk)
    ws_ref = WindowSignatureRefV1(hash=ws_hash, path="window_signature.json")
    argv_list = list(argv) if argv is not None else list(sys.argv)
    try:
        append_run_started(
            outdir,
            run_id=run_id,
            outdir_path=outdir,
            argv=argv_list,
            code_identity=dict(code_identity),
            window_signature_ref={"hash": ws_hash, "path": "window_signature.json"},
            entrypoint={"kind": "runner", "name": "veriscope.runners.gpt.train_nanogpt"},
            distributed=build_distributed_context(),
        )
    except Exception as exc:
        logger.warning("Failed to append governance run_started entry: %s", exc)
    if overrides:
        try:
            append_overrides(
                outdir,
                run_id=run_id,
                overrides=dict(overrides),
                profile={"gate_preset": gate_preset, "overrides": dict(overrides)},
                entrypoint={"kind": "runner", "name": "veriscope.runners.gpt.emit_artifacts"},
                argv=argv_list,
            )
        except Exception as exc:
            logger.warning("Failed to append governance overrides entry: %s", exc)

    # --- 2) gate_history -> GateRecordV1 ---
    n_events = len(gate_history)
    logger.debug("Processing %d gate_history events", n_events)

    gate_events = _flatten_gate_events(gate_history)
    gate_records: List[GateRecordV1] = []
    prev_iter: Optional[int] = None
    for i, raw_event in enumerate(gate_events):
        if not isinstance(raw_event, Mapping):
            logger.debug("Skipping non-object gate_history entry at index=%d (type=%s)", i, type(raw_event).__name__)
            continue
        ev = raw_event
        try:
            it = _get_event_iter(ev)
            if prev_iter is not None and it <= prev_iter:
                raise ValueError(
                    f"gate_history iterations must be strictly increasing: index={i} iter={it} prev_iter={prev_iter}"
                )
            prev_iter = it
            audit_dict = _normalize_audit_dict(ev)
            audit_dict = _sanitize_audit_nonfinite(audit_dict)

            # Validation boundary (Pydantic)
            audit = AuditV1(**audit_dict)
            decision = _derive_decision(ev, audit)

            if decision == "skip":
                ok_value = bool(ev.get("ok", True))
                warn_value = False
            else:
                ok_value = bool(ev.get("ok", decision in {"pass", "warn"}))
                warn_value = bool(ev.get("warn", decision == "warn"))
                if not ok_value:
                    warn_value = False

            rec_kwargs: Dict[str, Any] = {
                "iter": it,
                "decision": decision,
                "audit": audit,
                "ok": ok_value,
                "warn": warn_value,
            }

            record = GateRecordV1(**rec_kwargs)
            gate_records.append(record)
            try:
                audit_payload = record.audit.model_dump(mode="json", by_alias=True, exclude_none=True)
                append_gate_decision(
                    outdir,
                    run_id=run_id,
                    iter_num=int(record.iter),
                    decision=str(record.decision),
                    ok=record.ok,
                    warn=record.warn,
                    audit=audit_payload,
                )
            except Exception as exc:
                logger.warning("Failed to append governance gate decision: %s", exc)

        except Exception as e:
            iter_hint = ev.get("iter", ev.get("iter_num", "MISSING"))
            snippet = repr(dict(ev))[:400]
            # includes stack trace in logs:
            logger.exception(
                "Failed to process gate_history event index=%d iter_hint=%r snippet=%s",
                i,
                iter_hint,
                snippet,
            )
            raise ValueError(
                f"Artifact generation failed at gate_history index={i} iter_hint={iter_hint!r}: {e}"
            ) from e

    # Optional: ensure monotonic order (if desired). We preserve input order unless it’s clearly out-of-order.
    # If you want hard enforcement, switch this to a ValueError on any inversion.
    # gate_records.sort(key=lambda r: int(r.iter))

    # --- 3) Aggregate Results ---
    # ProfileV1.overrides is a mapping with a default_factory; do not pass None.
    profile = ProfileV1(gate_preset=gate_preset, overrides=(overrides or {}))

    results_kwargs: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "window_signature_ref": ws_ref,
        "profile": profile,
        "run_status": run_status,
        "runner_exit_code": runner_exit_code,
        "runner_signal": runner_signal,
        "started_ts_utc": _iso_z(started_ts_utc),
        "ended_ts_utc": _iso_z(ended_ts_utc) if ended_ts_utc is not None else None,
        "gates": gate_records,  # list is the natural JSON shape
        # Canonical V1 is intentionally gates-first; metric snapshots stay in legacy out_json.
        "metrics": [],
    }
    if metrics_ref is not None:
        ref_path = str(metrics_ref.get("path", "")).strip()
        if not ref_path:
            raise ValueError("metrics_ref.path must be a non-empty string when metrics_ref is provided")
        ref_payload: Dict[str, Any] = {
            "path": ref_path,
            "format": str(metrics_ref.get("format", "legacy_v0")).strip() or "legacy_v0",
        }
        if metrics_ref.get("count", None) is not None:
            ref_count = _coerce_int(metrics_ref.get("count"), field="metrics_ref.count")
            if ref_count < 0:
                raise ValueError("metrics_ref.count must be >= 0")
            ref_payload["count"] = ref_count
        results_kwargs["metrics_ref"] = ref_payload

    results = ResultsV1(
        **results_kwargs,
    )

    counts = _counts_from_gate_records(gate_records)
    final_decision = derive_final_decision(counts)
    first_fail_iter = _first_fail_iter_from_events(gate_events)

    if counts.fail == 0:
        if first_fail_iter is not None:
            logger.warning(
                "Computed first_fail_iter=%s but counts.fail=0; forcing first_fail_iter to null",
                first_fail_iter,
            )
        first_fail_iter = None
    elif first_fail_iter is None:
        msg = (
            "Invariant violation: counts.fail > 0 but first_fail_iter could not be derived "
            "from gate_history events."
        )
        logger.error(msg)
        raise ValueError(msg)

    summary = ResultsSummaryV1(
        schema_version=1,
        run_id=run_id,
        window_signature_ref=ws_ref,
        profile=profile,
        run_status=run_status,
        runner_exit_code=runner_exit_code,
        runner_signal=runner_signal,
        started_ts_utc=_iso_z(started_ts_utc),
        ended_ts_utc=_iso_z(ended_ts_utc) if ended_ts_utc is not None else None,
        counts=counts,
        final_decision=final_decision,
        first_fail_iter=first_fail_iter,
    )

    # --- 4) Write output ---
    results_path = outdir / "results.json"
    results_summary_path = outdir / "results_summary.json"

    logger.debug("Writing %s", results_path)
    atomic_write_pydantic_json(results_path, results, by_alias=True, exclude_none=True, fsync=True)

    logger.debug("Writing %s", results_summary_path)
    summary_payload = summary.model_dump(mode="json", by_alias=True, exclude_none=True)
    # Keep the key explicit so downstream finite-window checks can distinguish null vs missing.
    summary_payload["first_fail_iter"] = first_fail_iter
    atomic_write_json(results_summary_path, summary_payload, fsync=True)

    first_fail_iter_path = outdir / "first_fail_iter.txt"
    if first_fail_iter is None:
        if first_fail_iter_path.exists():
            first_fail_iter_path.unlink()
    else:
        atomic_write_text(first_fail_iter_path, f"{int(first_fail_iter)}\n", fsync=True)

    logger.info("Artifacts generated successfully (final_decision=%s)", final_decision)

    return EmittedArtifactsV1(
        window_signature_path=window_signature_path,
        results_path=results_path,
        results_summary_path=results_summary_path,
        window_signature_hash=ws_hash,
    )
