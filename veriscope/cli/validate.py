# veriscope/cli/validate.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# Be robust across Pydantic v2 environments: schema errors may be raised from either package.
try:
    from pydantic import ValidationError as PydanticValidationError
except Exception:  # pragma: no cover
    PydanticValidationError = Exception  # type: ignore

try:
    from pydantic_core import ValidationError as PydanticCoreValidationError
except Exception:  # pragma: no cover
    PydanticCoreValidationError = PydanticValidationError  # type: ignore

from veriscope.core.governance import read_governance_log, validate_governance_log
from veriscope.core.artifacts import ManualJudgementV1, ResultsSummaryV1, ResultsV1
from veriscope.core.jsonutil import canonical_json_sha256


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str
    window_signature_hash: Optional[str] = None
    partial: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)


def _read_text_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json_dict(path: Path) -> Dict[str, Any]:
    """
    Read JSON and enforce a top-level object (dict).

    Note: on malformed JSON we raise ValueError with a concise message (no JSONDecodeError
    re-wrapping that retains the full document on the exception object).
    """
    try:
        raw = _read_text_utf8(path)
    except OSError as e:
        raise OSError(f"Failed to read {path.name}: {e}") from None

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        # Keep message small and avoid retaining huge docs on exception objects.
        raise ValueError(f"JSON parse error in {path.name}: {e.msg} (line {e.lineno}, col {e.colno})") from None

    if not isinstance(obj, dict):
        raise TypeError(f"{path.name} must be a JSON object (dict), got {type(obj).__name__}")

    return obj


def _extract_profile_overrides(results: Optional[ResultsV1], summary: ResultsSummaryV1) -> Dict[str, Any]:
    if results is not None:
        return dict(results.profile.overrides or {})
    return dict(summary.profile.overrides or {})


def _run_config_overrides_present(run_cfg: Dict[str, Any]) -> bool:
    overrides = run_cfg.get("overrides")
    return isinstance(overrides, dict) and bool(overrides)


def _format_pydantic_error(e: Exception) -> str:
    errors = getattr(e, "errors", None)
    if callable(errors):
        try:
            items = errors()
            if items and isinstance(items, list):
                first = items[0] if isinstance(items[0], dict) else None
                if first:
                    msg = first.get("msg")
                    loc = first.get("loc")
                    if msg and loc:
                        return f"{msg} at {loc}"
                    if msg:
                        return str(msg)
        except Exception:
            pass
    return str(e)


def _format_identity_token(level: str, field: str, expected: Any, got: Any) -> str:
    return f"{level}:ARTIFACT_IDENTITY_MISMATCH field={field} expected={expected!r} got={got!r}"


def _check_identity_consistency(
    outdir: Path,
    results: Optional[ResultsV1],
    summary: ResultsSummaryV1,
    *,
    strict: bool,
    allow_partial: bool,
) -> tuple[bool, tuple[str, ...], tuple[str, ...], bool]:
    _ = outdir
    if results is None:
        return True, tuple(), tuple(), False

    warnings: list[str] = []
    errors: list[str] = []
    partial = False

    checks = [
        ("schema_version", summary.schema_version, results.schema_version),
        ("run_id", summary.run_id, results.run_id),
        ("window_signature_ref.hash", summary.window_signature_ref.hash, results.window_signature_ref.hash),
        ("window_signature_ref.path", summary.window_signature_ref.path, results.window_signature_ref.path),
        ("profile.gate_preset", summary.profile.gate_preset, results.profile.gate_preset),
    ]

    for field_name, expected, got in checks:
        if expected != got:
            if allow_partial and not strict:
                warnings.append(_format_identity_token("WARNING", field_name, expected, got))
                partial = True
            else:
                errors.append(_format_identity_token("ERROR", field_name, expected, got))

    return not errors, tuple(warnings), tuple(errors), partial


def validate_outdir(
    outdir: Path,
    *,
    strict_version: bool = False,
    allow_partial: bool = False,
    strict_identity: bool = False,
    allow_missing_governance: bool = True,
    allow_invalid_governance: bool = False,
    allow_legacy_governance: bool = False,
) -> ValidationResult:
    """
    Validate the canonical artifact set in OUTDIR.

    Requirements:
      - allow_partial=False: window_signature.json, results.json, results_summary.json exist
      - allow_partial=True: window_signature.json, results_summary.json exist (results.json optional)
      - Pydantic schema validation succeeds for results (if present) + summary
      - manual_judgement.json is optional but must validate if present
      - manual_judgement.jsonl is ignored by validation (governance log)
      - governance_log.jsonl must exist and be parseable when results.json exists (unless allow_missing_governance=True)
      - window_signature_ref.hash matches hash(window_signature.json) in results (if present) + summary
      - window_signature_ref.path equals "window_signature.json" in results (if present) + summary
      - when results.json exists, identity fields match results_summary.json (run_id, schema_version,
        window_signature_ref.hash/path, profile.gate_preset)
      - run_id matches across results (if present) + summary
      - schema_version == 1 (v0 contract)
    """
    _ = strict_version  # forward-compat placeholder
    outdir = Path(outdir)

    ws_path = outdir / "window_signature.json"
    res_path = outdir / "results.json"
    summ_path = outdir / "results_summary.json"
    manual_path = outdir / "manual_judgement.json"

    # 1) Existence check
    required = (ws_path, summ_path) if allow_partial else (ws_path, res_path, summ_path)
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return ValidationResult(False, f"Missing required artifacts: {', '.join(missing)}")

    # 2) Read raw inputs (separate IO/parse errors from schema validation errors)
    try:
        ws_obj = _read_json_dict(ws_path)
        res_text = _read_text_utf8(res_path) if res_path.exists() else None
        summ_text = _read_text_utf8(summ_path)
        manual_text = _read_text_utf8(manual_path) if manual_path.exists() else None
    except (OSError, ValueError, TypeError) as e:
        return ValidationResult(False, str(e))
    except Exception as e:
        return ValidationResult(False, f"Unexpected error reading artifacts: {e}")

    # 3) Pydantic parse/validate
    try:
        res = ResultsV1.model_validate_json(res_text) if res_text is not None else None
        summ = ResultsSummaryV1.model_validate_json(summ_text)
        manual = ManualJudgementV1.model_validate_json(manual_text) if manual_text is not None else None
    except (PydanticValidationError, PydanticCoreValidationError) as e:
        return ValidationResult(False, f"Schema validation failed: {_format_pydantic_error(e)}")
    except Exception as e:
        return ValidationResult(False, f"Unexpected error validating artifacts: {e}")

    # 4) Hash verification (hash what's on disk)
    try:
        ws_hash = canonical_json_sha256(ws_obj)
    except Exception as e:
        return ValidationResult(False, f"Failed to compute canonical hash: {e}")

    # 5) Cross-artifact invariants
    expected_ws_ref_path = "window_signature.json"
    res_ref = res.window_signature_ref if res is not None else None
    summ_ref = summ.window_signature_ref

    if res_ref is not None and getattr(res_ref, "path", None) != expected_ws_ref_path:
        return ValidationResult(
            False,
            f"window_signature_ref.path mismatch in results.json: {getattr(res_ref, 'path', None)!r} != {expected_ws_ref_path!r}",
            window_signature_hash=ws_hash,
        )
    if getattr(summ_ref, "path", None) != expected_ws_ref_path:
        return ValidationResult(
            False,
            f"window_signature_ref.path mismatch in results_summary.json: {getattr(summ_ref, 'path', None)!r} != {expected_ws_ref_path!r}",
            window_signature_hash=ws_hash,
        )

    if res_ref is not None and res_ref.hash != ws_hash:
        return ValidationResult(
            False,
            f"window_signature_ref.hash mismatch in results.json: {res_ref.hash} != {ws_hash}",
            window_signature_hash=ws_hash,
        )
    if summ_ref.hash != ws_hash:
        return ValidationResult(
            False,
            f"window_signature_ref.hash mismatch in results_summary.json: {summ_ref.hash} != {ws_hash}",
            window_signature_hash=ws_hash,
        )

    if manual is not None and res is not None and manual.run_id != res.run_id:
        return ValidationResult(
            False,
            f"run_id mismatch: manual_judgement={manual.run_id!r} results={res.run_id!r}",
            window_signature_hash=ws_hash,
        )
    if manual is not None and res is None and manual.run_id != summ.run_id:
        return ValidationResult(
            False,
            f"run_id mismatch: manual_judgement={manual.run_id!r} summary={summ.run_id!r}",
            window_signature_hash=ws_hash,
        )

    # v0 compatibility: only schema_version=1 accepted
    try:
        res_ver = int(res.schema_version) if res is not None else 1
        summ_ver = int(summ.schema_version)
    except Exception:
        return ValidationResult(
            False,
            "Unsupported schema_version (not an int): "
            f"results={getattr(res, 'schema_version', None)!r} summary={summ.schema_version!r}",
            window_signature_hash=ws_hash,
        )

    if res_ver != 1 or summ_ver != 1:
        return ValidationResult(
            False,
            f"Unsupported schema_version: results={getattr(res, 'schema_version', None)} summary={summ.schema_version}",
            window_signature_hash=ws_hash,
        )

    identity_ok, identity_warnings, identity_errors, identity_partial = _check_identity_consistency(
        outdir,
        res,
        summ,
        strict=strict_identity,
        allow_partial=allow_partial,
    )
    if not identity_ok:
        message = identity_errors[0] if identity_errors else "artifact identity mismatch"
        return ValidationResult(
            False,
            message,
            window_signature_hash=ws_hash,
            partial=False,
            warnings=identity_warnings,
            errors=identity_errors,
        )

    warnings = list(identity_warnings)
    errors = list(identity_errors)
    if res is not None:
        gov_path = outdir / "governance_log.jsonl"
        if not gov_path.exists():
            if allow_missing_governance:
                warnings.append("WARNING:GOVERNANCE_LOG_MISSING governance_log.jsonl not present")
            else:
                return ValidationResult(
                    False,
                    "Missing governance_log.jsonl for results.json",
                    window_signature_hash=ws_hash,
                    partial=False,
                    warnings=tuple(warnings),
                    errors=tuple(errors),
                )
        if not gov_path.exists():
            # Skip further governance checks if missing and allowed.
            gov_result = None
        else:
            gov_validation = validate_governance_log(gov_path, allow_legacy_governance=allow_legacy_governance)
            warnings.extend(gov_validation.warnings)
            if not gov_validation.ok:
                if allow_invalid_governance:
                    warnings.append("WARNING:GOVERNANCE_LOG_INVALID governance_log.jsonl invalid")
                    gov_result = None
                else:
                    errors.extend(gov_validation.errors)
                    return ValidationResult(
                        False,
                        "governance_log.jsonl invalid",
                        window_signature_hash=ws_hash,
                        partial=False,
                        warnings=tuple(warnings),
                        errors=tuple(errors),
                    )
            else:
                gov_result = read_governance_log(gov_path)
        if gov_result is not None:
            overrides_required = bool(_extract_profile_overrides(res, summ))
            run_cfg_path = outdir / "run_config_resolved.json"
            if run_cfg_path.exists():
                try:
                    run_cfg = _read_json_dict(run_cfg_path)
                except Exception:
                    run_cfg = {}
                overrides_required = overrides_required or _run_config_overrides_present(run_cfg)

            run_governance_present = any(
                gov_result.event_counts.get(event_name, 0) > 0
                for event_name in ("run_started_v1", "capsule_opened_v1", "run_overrides_applied_v1", "gate_decision_v1")
            )

            if run_governance_present and overrides_required:
                if gov_result.event_counts.get("run_overrides_applied_v1", 0) < 1:
                    msg = "Missing governance overrides entry for applied overrides"
                    token = "ERROR:GOVERNANCE_OVERRIDES_MISSING run_overrides_applied_v1 entry required"
                    if allow_invalid_governance:
                        warnings.append(token.replace("ERROR", "WARNING"))
                    else:
                        errors.append(token)
                        return ValidationResult(
                            False,
                            msg,
                            window_signature_hash=ws_hash,
                            partial=False,
                            warnings=tuple(warnings),
                            errors=tuple(errors),
                        )

            if run_governance_present and res.gates:
                required_gate_events = len(res.gates)
                gate_events = gov_result.event_counts.get("gate_decision_v1", 0)
                if gate_events < required_gate_events:
                    msg = "Missing governance gate decision entries for results"
                    token = (
                        "ERROR:GOVERNANCE_GATE_DECISIONS_MISSING "
                        f"gate_decision_v1 entries={gate_events} expected={required_gate_events}"
                    )
                    if allow_invalid_governance:
                        warnings.append(token.replace("ERROR", "WARNING"))
                    else:
                        errors.append(token)
                        return ValidationResult(
                            False,
                            msg,
                            window_signature_hash=ws_hash,
                            partial=False,
                            warnings=tuple(warnings),
                            errors=tuple(errors),
                        )

    # Return the derived truth (the computed hash), not just the referenced one.
    summary_partial = bool(getattr(summ, "partial", False))
    return ValidationResult(
        True,
        "OK",
        window_signature_hash=ws_hash,
        partial=(res is None or summary_partial or identity_partial),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
