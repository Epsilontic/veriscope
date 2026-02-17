from __future__ import annotations
import math

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Annotated, Final, Literal, Optional

try:
    from typing import TypeAlias  # py>=3.10
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias  # py3.9
from collections.abc import Mapping

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

# Version-tolerant import of Pydantic's named-recursive JsonValue.
try:
    # Pydantic v2 (recent) exposes JsonValue at top-level
    from pydantic import JsonValue as PydanticJsonValue
except ImportError:  # pragma: no cover
    # Older Pydantic v2 layouts
    from pydantic.types import JsonValue as PydanticJsonValue

# ---- Public contract exports ----

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SchemaVersion",
    "Decision",
    "RunStatus",
    "JudgementStatus",
    "DistributedMode",
    "JsonValue",
    "VSModel",
    "WindowSignatureRefV1",
    "DistributedRunConfigV1",
    "ProfileV1",
    "AuditV1",
    "GateRecordV1",
    "MetricRecordV1",
    "MetricsRefV1",
    "ResultsV1",
    "CountsV1",
    "ResultsSummaryV1",
    "ManualJudgementV1",
    "derive_gate_decision",
    "derive_final_decision",
    "decision_from_audit",
    "export_schemas_v1",
]

# ---- Common helpers/types (v0 contract surface) ----

CURRENT_SCHEMA_VERSION: Final[int] = 1
SchemaVersion = Literal[1]

Decision = Literal["pass", "warn", "fail", "skip"]
RunStatus = Literal["success", "user_code_failure", "veriscope_failure"]

# Manual judgements are binary to avoid conflating with Decision states.
JudgementStatus = Literal["pass", "fail"]
DistributedMode = Literal["single_process", "replicated_single_chief_emit", "ddp_wrapped"]

# JSON-compatible values (artifact hygiene for external-facing contracts).
# Use Pydantic's named-recursive JsonValue to avoid schema-generation recursion.
JsonValue: TypeAlias = PydanticJsonValue

NonNegInt = Annotated[StrictInt, Field(ge=0)]
PosInt = Annotated[StrictInt, Field(ge=1)]
# Prefer non-strict float for JSON-producer friendliness; still enforce >= 0.
NonNegFloat = Annotated[float, Field(ge=0)]


class VSModel(BaseModel):
    """
    Shared base model for Veriscope artifact contracts.

    Immutability notes:
    - frozen=True prevents attribute rebinding.
    - This module uses tuples for large collections and freezes selected mappings to
      MappingProxyType, but extra="allow" means extra fields may still be mutable internally.
      Treat immutability as “best effort” for the core contract fields.
    """

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)


def _dt_to_utc_z(dt: datetime) -> str:
    """
    Deterministic UTC serialization.

    - If naive, treat as UTC (artifact fields are explicitly *_utc).
    - If tz-aware, convert to UTC.
    - Emit ISO8601 seconds precision with trailing 'Z'.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    s = dt.isoformat(timespec="seconds")
    if s.endswith("+00:00"):
        s = s[:-6] + "Z"
    return s


def _ser_opt_dt(v: Optional[datetime]) -> Optional[str]:
    return None if v is None else _dt_to_utc_z(v)


def _freeze_mapping(m: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Freeze a mapping to prevent post-construction mutation via shared references.
    """
    return MappingProxyType(dict(m))


# ---- Core models ----


class WindowSignatureRefV1(VSModel):
    # Spec describes SHA256 hex; enforce and normalize to lowercase.
    hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    path: str = Field(min_length=1)

    @field_validator("hash")
    @classmethod
    def _normalize_hash_lower(cls, v: str) -> str:
        return v.lower()


class ProfileV1(VSModel):
    gate_preset: str = Field(min_length=1)
    overrides: Mapping[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _freeze_overrides(self) -> "ProfileV1":
        object.__setattr__(self, "overrides", _freeze_mapping(self.overrides))
        return self

    @field_serializer("overrides")
    def _ser_overrides(self, v: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        # Ensure downstream JSON tooling sees a plain dict, not MappingProxyType.
        return dict(v)


class DistributedRunConfigV1(VSModel):
    """
    Optional distributed execution metadata recorded in run governance payloads.

    Canonical keys are the short names (`backend`, `rank`, `local_rank`, `ddp_wrapped`).
    Legacy aliases remain accepted for backward compatibility.
    """

    # Intentionally permissive at governance-ingest time; strict allowed-value
    # checks are enforced in validate_outdir so user-facing token behavior is stable.
    distributed_mode: Optional[str] = Field(default=None, min_length=1)
    world_size_observed: Optional[PosInt] = None
    backend: Optional[str] = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("backend", "ddp_backend"),
        serialization_alias="backend",
    )
    rank: Optional[NonNegInt] = Field(
        default=None,
        validation_alias=AliasChoices("rank", "rank_observed"),
        serialization_alias="rank",
    )
    local_rank: Optional[NonNegInt] = Field(
        default=None,
        validation_alias=AliasChoices("local_rank", "local_rank_observed"),
        serialization_alias="local_rank",
    )
    ddp_wrapped: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("ddp_wrapped", "ddp_active"),
        serialization_alias="ddp_wrapped",
    )


class AuditV1(VSModel):
    evaluated: bool

    # Conditional requirements: present when evaluated=True, otherwise may be None.
    reason: Optional[str] = Field(default=None, min_length=1)
    policy: Optional[str] = Field(default=None, min_length=1)

    worst_DW: Optional[NonNegFloat] = Field(
        default=None,
        description="Worst-case D_W / TV-like distance reported by the gate audit (when available).",
    )
    eps_eff: Optional[NonNegFloat] = Field(
        default=None,
        description="Effective epsilon (or equivalent) used/derived by the gate audit (when available).",
    )

    per_metric_tv: Mapping[str, NonNegFloat] = Field(
        default_factory=dict,
        description="Per-metric TV / D_W distances (subset of metrics may be present).",
    )

    ddp_agg: Optional[str] = None
    ddp_barrier_status: Optional[str] = None

    evidence_total: Optional[NonNegInt] = None
    min_evidence: Optional[NonNegInt] = None

    @model_validator(mode="after")
    def _audit_conditional_requirements(self) -> "AuditV1":
        # Freeze mapping for best-effort immutability.
        object.__setattr__(self, "per_metric_tv", _freeze_mapping(self.per_metric_tv))

        if self.evaluated:
            if self.reason is None or self.policy is None:
                raise ValueError("audit.reason and audit.policy are required when audit.evaluated=True")
            if self.evidence_total is None or self.min_evidence is None:
                raise ValueError("audit.evidence_total and audit.min_evidence are required when audit.evaluated=True")
        return self

    @field_serializer("per_metric_tv")
    def _ser_per_metric_tv(self, v: Mapping[str, float]) -> dict[str, float]:
        # Ensure downstream JSON tooling sees a plain dict, not MappingProxyType.
        return dict(v)


class GateRecordV1(VSModel):
    iter: NonNegInt
    decision: Decision
    audit: AuditV1

    # Optional legacy booleans (non-canonical). Kept for transitional compatibility.
    ok: Optional[bool] = None
    warn: Optional[bool] = None

    @model_validator(mode="after")
    def _legacy_bool_consistency(self) -> "GateRecordV1":
        """
        Transitional consistency checks for legacy bools.

        Intentional scope:
        - Forbid clear contradictions.
        - Enforce fail-dominant decision implications when legacy bools are present.
        """
        if self.audit.evaluated and self.decision == "skip":
            raise ValueError("gate.decision cannot be 'skip' when audit.evaluated=True")
        if (not self.audit.evaluated) and self.decision != "skip":
            raise ValueError("gate.decision must be 'skip' when audit.evaluated=False")
        if self.decision in {"pass", "warn"} and self.ok is False:
            raise ValueError(f"gate.ok cannot be False when decision=='{self.decision}'")
        if self.decision == "fail" and self.ok is True:
            raise ValueError("gate.ok cannot be True when decision=='fail'")
        if self.warn is True and self.decision != "warn":
            raise ValueError("gate.warn cannot be True when decision!='warn'")
        return self


class MetricRecordV1(VSModel):
    """
    Placeholder metric record until formalized.

    Constrained to JSON-compatible values to avoid serializing arbitrary objects in artifacts.
    Extra keys are allowed (VSModel.extra="allow") for forward compatibility.
    """

    name: str = Field(min_length=1)
    iter: Optional[NonNegInt] = None
    value: JsonValue = None

    @field_validator("value")
    @classmethod
    def _no_nan_inf(cls, v: JsonValue) -> JsonValue:
        # Robustness: forbid non-JSON floats at the contract boundary.
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v


class MetricsRefV1(VSModel):
    """
    Pointer to an external metrics stream for gates-first results payloads.

    Canonical V1 results remain lightweight (`metrics=[]`) while this reference indicates
    where the legacy snapshot stream is stored.
    """

    path: str = Field(min_length=1)
    format: str = Field(default="legacy_v0", min_length=1)
    count: Optional[NonNegInt] = None
    # Optional integrity binding for external metrics payloads (non-breaking extension).
    sha256: Optional[str] = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")


class _ResultsHeaderV1(VSModel):
    """Shared header fields for results artifacts (DRY base)."""

    schema_version: SchemaVersion = 1

    run_id: str = Field(min_length=1)
    window_signature_ref: WindowSignatureRefV1
    profile: ProfileV1

    run_status: RunStatus
    runner_exit_code: Optional[Annotated[StrictInt, Field(ge=0, le=255)]] = None
    runner_signal: Optional[str] = Field(default=None, min_length=1)

    started_ts_utc: datetime
    ended_ts_utc: Optional[datetime] = None

    @field_serializer("started_ts_utc", "ended_ts_utc")
    def _ser_ts_utc(self, v: Optional[datetime]) -> Optional[str]:
        return _ser_opt_dt(v)

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> "_ResultsHeaderV1":
        if self.ended_ts_utc is not None and self.ended_ts_utc < self.started_ts_utc:
            raise ValueError("ended_ts_utc cannot precede started_ts_utc")
        return self


class ResultsV1(_ResultsHeaderV1):
    # Tuples reduce accidental mutation even under frozen=True.
    gates: tuple[GateRecordV1, ...] = Field(default_factory=tuple, repr=False)
    metrics: tuple[MetricRecordV1, ...] = Field(default_factory=tuple, repr=False)
    metrics_ref: Optional[MetricsRefV1] = Field(default=None)


class CountsV1(VSModel):
    """
    Gate-decision counts for a run summary.

    Semantics:
    - evaluated counts gates that were actually evaluated (i.e., non-skip).
    - skip counts gates that were not evaluated.
    - pass/warn/fail partition evaluated.
    """

    evaluated: NonNegInt
    skip: NonNegInt

    # Use Python-safe name but serialize/validate with key "pass" in JSON.
    pass_: NonNegInt = Field(
        validation_alias=AliasChoices("pass", "pass_"),
        serialization_alias="pass",
    )
    warn: NonNegInt
    fail: NonNegInt

    @model_validator(mode="after")
    def _counts_sanity(self) -> "CountsV1":
        total_eval = int(self.pass_ + self.warn + self.fail)
        if total_eval != int(self.evaluated):
            raise ValueError(f"counts must satisfy pass+warn+fail == evaluated: {total_eval} != {self.evaluated}")
        return self


class ResultsSummaryV1(_ResultsHeaderV1):
    counts: CountsV1
    final_decision: Decision
    first_fail_iter: Optional[NonNegInt] = None
    partial: Optional[bool] = None

    @model_validator(mode="after")
    def _summary_decision_and_fail_iter_invariants(self) -> "ResultsSummaryV1":
        expected_final = derive_final_decision(self.counts)
        if self.final_decision != expected_final:
            raise ValueError(
                "final_decision must match derive_final_decision(counts): "
                f"{self.final_decision!r} != {expected_final!r}"
            )
        if int(self.counts.fail) > 0 and self.first_fail_iter is None:
            raise ValueError("first_fail_iter is required when counts.fail > 0")
        if int(self.counts.fail) == 0 and self.first_fail_iter is not None:
            raise ValueError("first_fail_iter must be null when counts.fail == 0")
        return self


class ManualJudgementV1(VSModel):
    schema_version: SchemaVersion = 1

    run_id: str = Field(min_length=1)
    status: JudgementStatus
    reason: str = Field(min_length=1)
    reviewer: Optional[str] = Field(default=None, min_length=1)
    ts_utc: datetime

    @field_serializer("ts_utc")
    def _ser_ts_utc(self, v: datetime) -> str:
        return _dt_to_utc_z(v)


# ---- Enforcement helpers (unit-testable, runner-agnostic) ----


def derive_gate_decision(*, evaluated: bool, ok: bool, warn: bool) -> Decision:
    """
    Canonical gate decision derivation for legacy boolean inputs.

    Precedence:
      - not evaluated -> "skip"
      - evaluated and ok=False -> "fail"  (FAIL dominates WARN)
      - evaluated and ok=True and warn=True -> "warn"
      - evaluated and ok=True and warn=False -> "pass"
    """
    if not bool(evaluated):
        return "skip"
    if not bool(ok):
        return "fail"
    if bool(warn):
        return "warn"
    return "pass"


def derive_final_decision(counts: CountsV1) -> Decision:
    """
    Spec decision rule:
      - any fail -> "fail"
      - else any warn -> "warn"
      - else any pass -> "pass"
      - else "skip"
    """
    if counts.fail > 0:
        return "fail"
    if counts.warn > 0:
        return "warn"
    if counts.pass_ > 0:
        return "pass"
    return "skip"


_POLICY_TO_DECISION: dict[str, Decision] = {"fail": "fail", "warn": "warn", "pass": "pass"}


def decision_from_audit(*, evaluated: bool, policy_outcome: Optional[str] = None) -> Decision:
    """
    Mapping from audit state to Decision with conservative defaults.

    - not evaluated -> "skip"
    - evaluated and policy_outcome in {"fail","warn","pass"} -> mapped decision
    - evaluated but missing/unknown policy_outcome -> "warn" (forces attention; avoids silently passing bad data)
    """
    if not evaluated:
        return "skip"
    return _POLICY_TO_DECISION.get(policy_outcome or "", "warn")


def export_schemas_v1() -> dict[str, Any]:
    """
    Export JSON schemas for v1 artifact types.

    Use by_alias=True so consumers see contract keys like "pass" rather than "pass_".
    """
    return {
        "ResultsV1": ResultsV1.model_json_schema(by_alias=True),
        "ResultsSummaryV1": ResultsSummaryV1.model_json_schema(by_alias=True),
        "ManualJudgementV1": ManualJudgementV1.model_json_schema(by_alias=True),
        "CountsV1": CountsV1.model_json_schema(by_alias=True),
        "GateRecordV1": GateRecordV1.model_json_schema(by_alias=True),
        "AuditV1": AuditV1.model_json_schema(by_alias=True),
        "ProfileV1": ProfileV1.model_json_schema(by_alias=True),
        "WindowSignatureRefV1": WindowSignatureRefV1.model_json_schema(by_alias=True),
        "MetricRecordV1": MetricRecordV1.model_json_schema(by_alias=True),
        "MetricsRefV1": MetricsRefV1.model_json_schema(by_alias=True),
    }
