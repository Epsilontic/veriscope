# Incubation Readiness (v0, Next 1–2 Days)

This page is the short decision memo for AI2-style incubation review.
Normative precedence: `docs/contract_v1.md` is the authoritative contract; this document is derived scope/positioning guidance.

## Scope First: `W_inc = (M_W, I_W, Φ_W, B, ε)`

All claims in this file apply only inside this explicit window.

- `M_W` (metrics in scope): gate evidence represented in capsule artifacts (`results.json.gates[*].audit`, `results_summary.json.counts`) and metrics explicitly declared by `window_signature.json`.
- `I_W` (index/cadence in scope): one capsule run identity (`run_id`) and its emitted gate cadence records.
- `Φ_W` (protocol/profile in scope): `schema_version=1`, fixed `profile.gate_preset`, and matching `window_signature_ref.hash`.
- `B` (boundary conditions): one immutable OUTDIR capsule + append-only governance overlays/logs.
- `ε` (error budget + uncertainty): finite-evidence gate uncertainty (`audit.evidence_total`, `audit.min_evidence`, `audit.eps_eff` when present) and explicit pilot error budgets from calibration protocol.

Anything outside `W_inc` is **non-claimed** and must be treated as invalid/incomparable when implied.

For exact enum names and validator token strings, treat `docs/contract_v1.md` as the source of truth.

## Executive Summary

Veriscope is ready for a narrow incubation lane as a CLI-first reliability gate with auditable capsules. The highest-confidence value today is not "universal detection"; it is contract-enforced decision accounting, reproducible validation, and strict comparability boundaries.

Primary user: ML platform/research engineers who need a deterministic gate + artifact contract around long-running training jobs.

## Claims vs Non-Claims

| Area | Claimed (inside `W_inc`) | Non-Claimed (outside `W_inc`) | Verify |
| --- | --- | --- | --- |
| Capsule integrity | `veriscope validate OUTDIR` is read-only and enforces schema + cross-artifact invariants. | Auto-repair/normalization of broken artifacts. | Command: `veriscope validate OUTDIR`; tests: [`tests/test_cli_validate_report.py`](../tests/test_cli_validate_report.py) |
| Decision accounting | Canonical decision surface is `pass|warn|fail|skip`; `skip` stays neutral. | Inference from legacy `ok`/`warn` booleans as contract truth. | Tests: [`tests/test_cli_validate_report.py`](../tests/test_cli_validate_report.py), [`tests/test_schema.py`](../tests/test_schema.py) |
| Comparability | Runs are comparable only when `window_signature_ref.hash` matches (+ gate preset unless explicitly overridden). | Cross-window or cross-preset claims without explicit override. | Command: `veriscope diff A B`; tests: [`tests/test_cli_diff.py`](../tests/test_cli_diff.py), code: [`veriscope/cli/comparability.py`](../veriscope/cli/comparability.py) |
| Governance/auditability | Governance is append-only and hash-chained; run bindings are checked. | Rewriting governance history or treating missing governance as equivalent evidence. | Tests: [`tests/test_governance_contract.py`](../tests/test_governance_contract.py) |
| Distributed-mode explicitness | Missing distributed identity fields fail loud (`world_size_observed` required when distributed hints are present; invalid/missing mode is rejected when `world_size_observed > 1`). | Silent assumptions about distributed execution when metadata is incomplete. | Test: [`tests/test_distributed_mode_contract.py`](../tests/test_distributed_mode_contract.py); code: [`veriscope/cli/validate.py`](../veriscope/cli/validate.py) |

## Verification Commands (Contract-Oriented)

```bash
# 1) Thin CLI import boundary
python -c "import veriscope.cli.main; print('cli import ok')"

# 2) Golden path smoke + validation + report
bash scripts/run_gpt_smoke.sh
veriscope validate ./out/gpt_smoke_YYYYMMDD_HHMMSS
veriscope report ./out/gpt_smoke_YYYYMMDD_HHMMSS --format text

# 3) Unit tests for contract guarantees
pytest -q tests/test_cli_validate_report.py tests/test_cli_diff.py tests/test_governance_contract.py tests/test_distributed_mode_contract.py
```

## 8–12 Week Partner Pilot Plan

1. Weeks 1–2: Onboard and baseline.
   - Acceptance: partner can run golden path, produce valid capsule, and share `report.md`.
   - Stop condition: cannot produce one valid capsule in target environment within 3 working days.
2. Weeks 3–5: Calibrate within one fixed `window_signature` hash.
   - Acceptance: calibration report artifact produced per [`docs/calibration_protocol_v0.md`](./calibration_protocol_v0.md).
   - Stop condition: no feasible preset meets error-budget constraints for the partner's chosen `W`.
3. Weeks 6–9: Shadow mode on real runs.
   - Acceptance: gate decisions and governance logs are complete, deterministic, and operationally interpretable.
   - Stop condition: repeated ambiguous outcomes due to missing evidence or unstable instrumentation.
4. Weeks 10–12: Decision gate trial.
   - Acceptance: pre-agreed policy actions on `warn`/`fail`, with postmortem traceability from artifacts.
   - Stop condition: unresolved false-positive/false-negative costs exceed partner threshold.

## Immediate Incubation Readiness Decision

Ready for **narrow-scope incubation** if partner agrees that guarantees are limited to `W_inc` and that calibration validity is per-window-signature, not universal.
