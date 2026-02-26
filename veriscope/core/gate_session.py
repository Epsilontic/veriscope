from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from veriscope.core.artifacts import AuditV1, GateRecordV1, MetricRecordV1, derive_gate_decision
from veriscope.core.calibration import aggregate_epsilon_stat
from veriscope.core.evidence import compute_evidence_counts, extract_metric_array, metric_snapshot
from veriscope.core.gate import GateEngine
from veriscope.core.governance import append_gate_decision, append_run_started, build_code_identity
from veriscope.core.jsonutil import atomic_write_json, window_signature_sha256
from veriscope.core.window import WindowDecl


class GateSession:
    """Manage gate lifecycle, governance, and canonical capsule emission."""

    def __init__(
        self,
        *,
        outdir: Path,
        run_id: str,
        window_decl: WindowDecl,
        gate_engine: GateEngine,
        gate_window: int,
        gate_policy: str,
        gate_min_evidence: int,
        gate_preset: str,
        window_signature: Dict[str, Any],
        cadence: int = 1,
        observer_mode: bool = False,
        emit_governance: bool = True,
        started_ts_utc: Optional[datetime] = None,
        argv: Optional[List[str]] = None,
        code_identity: Optional[Dict[str, Any]] = None,
        entrypoint: Optional[Dict[str, Any]] = None,
        distributed: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._outdir = Path(outdir)
        self._outdir.mkdir(parents=True, exist_ok=True)
        self._run_id = str(run_id)
        self._window_decl = window_decl
        self._gate_engine = gate_engine
        self._gate_window = int(gate_window)
        self._gate_policy = str(gate_policy)
        self._gate_min_evidence = int(gate_min_evidence)
        self._gate_preset = str(gate_preset)
        self._cadence = int(cadence)
        self._observer_mode = bool(observer_mode)
        self._emit_governance = bool(emit_governance)
        self._started_ts = started_ts_utc or datetime.now(timezone.utc)
        self._history: List[Dict[str, Any]] = []
        self._gate_records: List[GateRecordV1] = []
        self._metric_records: List[MetricRecordV1] = []

        # Single-owner write of window_signature.json.
        ws_path = self._outdir / "window_signature.json"
        atomic_write_json(ws_path, window_signature, fsync=True)

        # Freeze hash from disk content (mirrors downstream consumers).
        ws_on_disk = json.loads(ws_path.read_text(encoding="utf-8"))
        self._ws_hash = window_signature_sha256(ws_on_disk)

        if self._emit_governance:
            append_run_started(
                self._outdir,
                run_id=self._run_id,
                outdir_path=self._outdir,
                argv=list(argv) if argv is not None else ["gate_session"],
                code_identity=dict(code_identity) if code_identity is not None else build_code_identity(),
                window_signature_ref={"hash": self._ws_hash, "path": "window_signature.json"},
                entrypoint=dict(entrypoint)
                if entrypoint is not None
                else {"kind": "session", "name": "veriscope.core.gate_session"},
                distributed=dict(distributed) if distributed is not None else None,
            )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def window_signature_hash(self) -> str:
        return self._ws_hash

    def record_step(self, step: int, **metrics: float) -> None:
        row: Dict[str, Any] = {"iter": int(step)}
        row.update(metrics)
        self._history.append(row)

        for name, value in metrics.items():
            safe_val: Any = None
            try:
                f = float(value)
            except Exception:
                f = float("nan")
            if math.isfinite(f):
                safe_val = f
            self._metric_records.append(MetricRecordV1(name=str(name), iter=int(step), value=safe_val))

    def evaluate(self, step: int, *, gain_bits: float = 0.0) -> GateRecordV1:
        metrics = list(self._window_decl.weights.keys())

        if self._observer_mode:
            record = self._make_skip_record(
                int(step),
                reason="not_evaluated_observer_mode",
                evidence_total=len(self._history),
            )
            self._gate_records.append(record)
            self._emit_gate_governance(record)
            return record

        past_slice, recent_slice = metric_snapshot(self._history, self._gate_window)
        if not past_slice or not recent_slice:
            record = self._make_skip_record(
                int(step),
                reason="not_evaluated_insufficient_evidence",
                evidence_total=len(self._history),
            )
            self._gate_records.append(record)
            self._emit_gate_governance(record)
            return record

        past_dict = {m: extract_metric_array(past_slice, m) for m in metrics}
        recent_dict = {m: extract_metric_array(recent_slice, m) for m in metrics}
        counts, evidence_total = compute_evidence_counts(past_dict, recent_dict, metrics)

        if evidence_total < self._gate_min_evidence:
            record = self._make_skip_record(
                int(step),
                reason="not_evaluated_insufficient_finite_evidence",
                evidence_total=evidence_total,
            )
            self._gate_records.append(record)
            self._emit_gate_governance(record)
            return record

        eps_stat = aggregate_epsilon_stat(self._window_decl, counts, alpha=0.05)
        result = self._gate_engine.check(
            past=past_dict,
            recent=recent_dict,
            counts_by_metric=counts,
            gain_bits=float(gain_bits),
            kappa_sens=0.0,
            eps_stat_value=eps_stat,
            iter_num=int(step),
        )

        audit_payload = dict(result.audit or {})
        audit_payload.setdefault("per_metric_tv", {})
        audit_payload.setdefault("policy", self._gate_policy)
        audit_payload.setdefault("evidence_total", int(evidence_total))
        audit_payload.setdefault("min_evidence", int(self._gate_min_evidence))
        evaluated = bool(audit_payload.get("evaluated", True))
        audit_payload["evaluated"] = evaluated
        if evaluated:
            audit_payload.setdefault("reason", "evaluated")
        else:
            audit_payload.setdefault("reason", "not_evaluated")
        audit = AuditV1(**audit_payload)

        ok = bool(result.ok)
        warn = bool(getattr(result, "warn", False))
        if not ok:
            warn = False
        decision = derive_gate_decision(evaluated=audit.evaluated, ok=ok, warn=warn)
        record = GateRecordV1(iter=int(step), decision=decision, audit=audit, ok=ok, warn=warn)
        self._gate_records.append(record)
        self._emit_gate_governance(record)
        return record

    def close(
        self,
        *,
        run_status: str = "success",
        ended_ts_utc: Optional[datetime] = None,
        runner_exit_code: Optional[int] = None,
        runner_signal: Optional[str] = None,
    ) -> Path:
        ended = ended_ts_utc or datetime.now(timezone.utc)

        # Defense-in-depth immutability verification before emission.
        ws_path = self._outdir / "window_signature.json"
        if not ws_path.exists():
            raise RuntimeError("window_signature.json was deleted after session init")
        ws_on_disk = json.loads(ws_path.read_text(encoding="utf-8"))
        disk_hash = window_signature_sha256(ws_on_disk)
        if disk_hash != self._ws_hash:
            raise RuntimeError(
                "window_signature.json was modified after run_started_v1: "
                f"init_hash={self._ws_hash} current_hash={disk_hash}"
            )

        if not self._gate_records:
            record = self._make_skip_record(
                0,
                reason="not_evaluated_no_steps",
                evidence_total=0,
            )
            self._gate_records.append(record)
            self._emit_gate_governance(record)

        from veriscope.core.emit import emit_capsule_v1

        emit_capsule_v1(
            outdir=self._outdir,
            run_id=self._run_id,
            started_ts_utc=self._started_ts,
            ended_ts_utc=ended,
            gate_preset=self._gate_preset,
            ws_hash=self._ws_hash,
            gate_records=self._gate_records,
            run_status=str(run_status),
            runner_exit_code=runner_exit_code,
            runner_signal=runner_signal,
            metrics=self._metric_records if self._metric_records else None,
        )
        return self._outdir

    def _make_skip_record(
        self,
        step: int,
        *,
        reason: str,
        evidence_total: int,
    ) -> GateRecordV1:
        audit = AuditV1(
            evaluated=False,
            reason=str(reason),
            policy=self._gate_policy,
            per_metric_tv={},
            evidence_total=int(evidence_total),
            min_evidence=int(self._gate_min_evidence),
        )
        return GateRecordV1(
            iter=int(step),
            decision="skip",
            audit=audit,
            ok=True,
            warn=False,
        )

    def _emit_gate_governance(self, record: GateRecordV1) -> None:
        if not self._emit_governance:
            return
        audit_payload = record.audit.model_dump(mode="json", by_alias=True, exclude_none=True)
        append_gate_decision(
            self._outdir,
            run_id=self._run_id,
            iter_num=int(record.iter),
            decision=str(record.decision),
            ok=record.ok,
            warn=record.warn,
            audit=audit_payload,
        )
