# veriscope/cli/comparability.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from veriscope.cli.governance import (
    GovernanceStatus,
    ManualJudgementEffective,
    get_governance_status,
    resolve_display_status,
    resolve_manual_overlay,
)
from veriscope.cli.validate import ValidationResult
from veriscope.core.artifacts import ResultsSummaryV1, ResultsV1


@dataclass(frozen=True)
class ComparableResult:
    ok: bool
    reason: Optional[str]
    details: Dict[str, Dict[str, Any]]
    warnings: Tuple[str, ...]
    policy: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "details": self.details,
            "warnings": list(self.warnings),
            "policy": self.policy,
        }


@dataclass(frozen=True)
class RunMetadata:
    outdir: Path
    validation: ValidationResult
    summary: ResultsSummaryV1
    results: Optional[ResultsV1]
    run_config: Optional[Dict[str, Any]]
    manual: ManualJudgementEffective
    schema_version: int
    gate_preset: str
    governance: GovernanceStatus

    @property
    def run_id(self) -> str:
        return self.results.run_id if self.results is not None else self.summary.run_id

    @property
    def run_status(self) -> str:
        return str(self.summary.run_status)

    @property
    def runner_exit_code(self) -> Optional[int]:
        return getattr(self.summary, "runner_exit_code", None)

    @property
    def final_decision(self) -> str:
        return str(self.summary.final_decision)

    @property
    def window_signature_hash(self) -> str:
        return self.validation.window_signature_hash or ""

    @property
    def partial(self) -> bool:
        return bool(self.validation.partial)

    @property
    def wrapper_exit_code(self) -> Optional[int]:
        if isinstance(self.run_config, dict):
            raw = self.run_config.get("wrapper_exit_code")
            try:
                return int(raw) if raw is not None else None
            except Exception:
                return None
        return None

    @property
    def manual_status(self) -> Optional[str]:
        return self.manual.judgement.status if self.manual.judgement is not None else None

    @property
    def display_status(self) -> str:
        return resolve_display_status(str(self.summary.final_decision), self.manual).status

    @property
    def decision_source(self) -> str:
        return resolve_display_status(str(self.summary.final_decision), self.manual).source


def _read_json_obj(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def load_run_metadata(outdir: Path, validation: ValidationResult, *, prefer_jsonl: bool = True) -> RunMetadata:
    outdir = Path(outdir)
    res_path = outdir / "results.json"
    res = ResultsV1.model_validate_json(res_path.read_text("utf-8")) if res_path.exists() else None
    summ = ResultsSummaryV1.model_validate_json((outdir / "results_summary.json").read_text("utf-8"))
    run_cfg = _read_json_obj(outdir / "run_config_resolved.json")

    gate_preset = res.profile.gate_preset if res is not None else summ.profile.gate_preset
    schema_version = int(summ.schema_version)
    run_id = res.run_id if res is not None else summ.run_id

    # Contract-aware overlay resolution: prefer last matching jsonl entry; fall back to snapshot; else none.
    manual = resolve_manual_overlay(outdir, run_id=run_id, prefer_jsonl=prefer_jsonl)
    governance = get_governance_status(outdir, allow_legacy_governance=True)

    return RunMetadata(
        outdir=outdir,
        validation=validation,
        summary=summ,
        results=res,
        run_config=run_cfg,
        manual=manual,
        schema_version=schema_version,
        gate_preset=gate_preset,
        governance=governance,
    )


def comparable_explain(
    a: RunMetadata,
    b: RunMetadata,
    *,
    allow_gate_preset_mismatch: bool = False,
) -> ComparableResult:
    policy = {"allow_gate_preset_mismatch": allow_gate_preset_mismatch}

    if a.partial or b.partial:
        return ComparableResult(
            ok=False,
            reason="PARTIAL_CAPSULE",
            details={
                "partial": {
                    "expected": False,
                    "got": {
                        "run_a": bool(a.partial),
                        "run_b": bool(b.partial),
                    },
                }
            },
            warnings=tuple(),
            policy=policy,
        )
    if not a.window_signature_hash or not b.window_signature_hash:
        return ComparableResult(
            ok=False,
            reason="WINDOW_HASH_MISSING",
            details={
                "window_signature_hash": {
                    "expected": a.window_signature_hash or None,
                    "got": b.window_signature_hash or None,
                }
            },
            warnings=tuple(),
            policy=policy,
        )
    if int(a.schema_version) != int(b.schema_version):
        return ComparableResult(
            ok=False,
            reason="SCHEMA_MISMATCH",
            details={
                "schema_version": {
                    "expected": int(a.schema_version),
                    "got": int(b.schema_version),
                }
            },
            warnings=tuple(),
            policy=policy,
        )
    if a.window_signature_hash != b.window_signature_hash:
        return ComparableResult(
            ok=False,
            reason="WINDOW_HASH_MISMATCH",
            details={
                "window_signature_hash": {
                    "expected": a.window_signature_hash,
                    "got": b.window_signature_hash,
                }
            },
            warnings=tuple(),
            policy=policy,
        )
    if not allow_gate_preset_mismatch and a.gate_preset != b.gate_preset:
        return ComparableResult(
            ok=False,
            reason="GATE_PRESET_MISMATCH",
            details={
                "gate_preset": {
                    "expected": a.gate_preset,
                    "got": b.gate_preset,
                }
            },
            warnings=tuple(),
            policy=policy,
        )
    return ComparableResult(ok=True, reason=None, details={}, warnings=tuple(), policy=policy)


def comparable(
    a: RunMetadata,
    b: RunMetadata,
    *,
    allow_gate_preset_mismatch: bool = False,
) -> Tuple[bool, Optional[str]]:
    result = comparable_explain(a, b, allow_gate_preset_mismatch=allow_gate_preset_mismatch)
    return result.ok, result.reason
