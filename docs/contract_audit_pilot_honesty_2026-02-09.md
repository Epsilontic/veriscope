# Veriscope Contract Completeness / Pilot-Honesty Audit (2026-02-09)

## Window W
- Scope: artifact emission + validation + comparability + report paths for runners (`legacy/cifar`, `gpt/nanoGPT`, `hf/transformers`).
- Goal: detect silent contract violations (missing artifacts/fields, inconsistent derivations, nullable-or-missing fields that should be concrete).
- Method: static code-path trace + contract/schema crosswalk + targeted regression tests.
- Environment limits: runtime test execution was blocked by missing dependencies (`pytest`, `numpy`) in available Python environments; syntax checks passed via `py_compile`.

## 1) Contract Matrix (from `docs/contract_v1.md`)

### 1.1 Artifact set
| Artifact | Contract status | Source |
|---|---|---|
| `window_signature.json` | Required (non-partial + partial) | `docs/contract_v1.md:10`, `docs/contract_v1.md:16` |
| `results.json` | Required (non-partial) | `docs/contract_v1.md:12` |
| `results_summary.json` | Required (non-partial + partial, with `partial=true` in partial) | `docs/contract_v1.md:13`, `docs/contract_v1.md:17` |
| `run_config_resolved.json` | Recommended | `docs/contract_v1.md:20` |
| `first_fail_iter.txt` | Recommended when `results_summary.first_fail_iter` is non-null | `docs/contract_v1.md:21` |
| `manual_judgement.json` | Optional governance overlay | `docs/contract_v1.md:24` |
| `manual_judgement.jsonl` | Optional append-only governance overlay | `docs/contract_v1.md:25` |
| `governance_log.jsonl` | Optional append-only governance journal | `docs/contract_v1.md:26` |

### 1.2 Required field rules called out by contract text
| File | Required field rules | Source |
|---|---|---|
| `window_signature.json` | Recomputed canonical hash (volatile keys removed) must match refs; ref path must be `window_signature.json` | `docs/contract_v1.md:37`, `docs/contract_v1.md:38` |
| `results_summary.json` | `final_decision` is authoritative automated outcome; `first_fail_iter` obeys fail/no-fail invariant; partial capsule must set `partial=true` | `docs/contract_v1.md:17`, `docs/contract_v1.md:99`, `docs/contract_v1.md:106`, `docs/contract_v1.md:110` |
| `governance_log.jsonl` | Per-entry fields (`schema_version`, `rev`, `ts_utc`, `event|event_type`, `payload`, `prev_hash`, `entry_hash`) + hash chain semantics | `docs/contract_v1.md:48`, `docs/contract_v1.md:53`, `docs/contract_v1.md:62`, `docs/contract_v1.md:66`, `docs/contract_v1.md:67` |
| `results.json` | Required artifact in non-partial capsules; detailed wire spine enforced in schema (`ResultsV1`) | `docs/contract_v1.md:12`, `veriscope/core/artifacts.py:289` |

## 2) Golden-Path Emission Trace
- `legacy/cifar` wrapper: `veriscope/cli/main.py:704` (`_cmd_run_cifar`) -> delegates to `veriscope-legacy`; wrapper repairs missing/invalid summary via `_write_partial_summary` at `veriscope/cli/main.py:814`.
- `gpt/nanoGPT`: `veriscope/runners/gpt/train_nanogpt.py:2761` -> `veriscope/runners/gpt/emit_artifacts.py:414`; writes all three artifacts + fail marker (`first_fail_iter.txt`) at `veriscope/runners/gpt/emit_artifacts.py:663`.
- `hf/transformers`: `veriscope/runners/hf/train_hf.py:1634` -> `veriscope/runners/hf/emit_artifacts.py:79`; writes all three artifacts + fail marker at `veriscope/runners/hf/emit_artifacts.py:165`.

## 3) `results_summary.json` Invariant Enforcement Status
- `counts.*` coherence with `results.gates`: enforced in validator (`veriscope/cli/validate.py:358` to `veriscope/cli/validate.py:378`).
- `final_decision` derived canonically from counts: enforced in schema model validator (`veriscope/core/artifacts.py:331` to `veriscope/core/artifacts.py:338`).
- `counts.fail > 0` requires `first_fail_iter` and marker file: schema + validator enforce both (`veriscope/core/artifacts.py:339` to `veriscope/core/artifacts.py:342`, `veriscope/cli/validate.py:392` to `veriscope/cli/validate.py:435`).
- Partial summary must be explicit: validator enforces `partial=true` if `results.json` missing (`veriscope/cli/validate.py:347` to `veriscope/cli/validate.py:356`).
- Partial comparability claim: still open (see Issue 6).

## 4) Schema / Validator / Emitter Mismatch Check
- `ResultsSummaryV1.partial` exists and is emitted only for wrapper partial fallbacks (`veriscope/cli/main.py:301` to `veriscope/cli/main.py:313`). Complete runner summaries omit it (expected).
- Extra summary keys emitted by wrappers (`note`, `wrapper_emitted`) are not in schema but are allowed by `extra="allow"` (`veriscope/core/artifacts.py:90`).
- Governance payload contract now checked for required run-event keys (`veriscope/core/governance.py:116` to `veriscope/core/governance.py:215`), but nested audit schema under `gate_decision_v1` is still not fully validated (Issue 10).

## 5) Top 10 Findings (ranked by pilot risk)

### 1. [P0][FIXED] CIFAR wrapper could return success with invalid `results_summary.json`
- Evidence: legacy path now validates summary quality before deciding exit (`veriscope/cli/main.py:779` to `veriscope/cli/main.py:792`).
- Failure mode: before fix, malformed summary file could leave wrapper exit as success.
- Repro/test: `tests/test_cli_run_cifar_wrapper.py:29` to `tests/test_cli_run_cifar_wrapper.py:65`.
- Minimal fix: treat missing/invalid summary as wrapper failure and emit partial fallback.

### 2. [P0][FIXED] `diff` accepted identity-mismatched capsules in allow-partial mode
- Evidence: `diff` now validates with `strict_identity=True` (`veriscope/cli/diff.py:189`, `veriscope/cli/diff.py:192`).
- Failure mode: summary/results identity drift could pass into comparability.
- Repro/test: `tests/test_cli_diff.py:197` to `tests/test_cli_diff.py:209`.
- Minimal fix: strict identity gate at entry to diff path.

### 3. [P0][FIXED] `report --compare` accepted identity-mismatched capsules
- Evidence: compare report now validates with `strict_identity=True` (`veriscope/cli/report.py:405` to `veriscope/cli/report.py:409`).
- Failure mode: multi-run compare could proceed on internally inconsistent capsules.
- Repro/test: `tests/test_cli_diff.py:212` to `tests/test_cli_diff.py:224`.
- Minimal fix: strict identity gate for each compared outdir.

### 4. [P0][FIXED] single-run `report` rendered despite identity mismatch
- Evidence: report path now enforces strict identity at validation (`veriscope/cli/report.py:114` to `veriscope/cli/report.py:118`).
- Failure mode: report appeared healthy while cross-artifact identity was broken.
- Repro/test: `tests/test_cli_validate_report.py:351` to `tests/test_cli_validate_report.py:358`.
- Minimal fix: strict identity in report validation entrypoint.

### 5. [P0][FIXED] governance validator allowed run-event payloads missing required fields
- Evidence: required payload validation added (`veriscope/core/governance.py:116` to `veriscope/core/governance.py:215`, wired at `veriscope/core/governance.py:297` to `veriscope/core/governance.py:299`).
- Failure mode: structurally malformed `run_started_v1` / `run_overrides_applied_v1` / `gate_decision_v1` lines could pass validation.
- Repro/test: `tests/test_governance_contract.py:233` to `tests/test_governance_contract.py:252`.
- Minimal fix: event-specific required key/type checks.

### 6. [P1][OPEN] partial capsules are still reported/computed as comparable
- Evidence: no partial guard in comparability predicate (`veriscope/cli/comparability.py:137` to `veriscope/cli/comparability.py:197`); diff returns success in partial mode (`veriscope/cli/diff.py:215`, `veriscope/cli/diff.py:220`, `veriscope/cli/diff.py:256` to `veriscope/cli/diff.py:259`).
- Failure mode: partial summaries can be interpreted as comparable outcomes.
- Repro/test: existing behavior covered by `tests/test_cli_diff.py:269` to `tests/test_cli_diff.py:285`.
- Suggested fix: force `ComparableResult.ok=False` when either run is partial (or hard-disable partial comparisons in diff/report-compare).

### 7. [P1][OPEN] report paths intentionally tolerate invalid governance logs
- Evidence: report validators set `allow_invalid_governance=True` (`veriscope/cli/report.py:119`, `veriscope/cli/report.py:410`).
- Failure mode: report succeeds even when governance hash chain is invalid.
- Repro/test: `tests/test_cli_validate_report.py:422` to `tests/test_cli_validate_report.py:429` (tampered governance still renders).
- Suggested fix: default strict governance for compare mode, or require explicit `--allow-invalid-governance` opt-in.

### 8. [P1][OPEN] HF can emit `run_started_v1` without `window_signature_ref.hash`
- Evidence: hash is optional in pre-write branch (`veriscope/runners/hf/train_hf.py:1284` to `veriscope/runners/hf/train_hf.py:1302`).
- Failure mode: pre-write failure can still append a run-start entry with incomplete comparability reference.
- Reproduction: monkeypatch `atomic_write_json` in HF runner preamble to raise; observe attempted append with hash-less `window_signature_ref`.
- Suggested fix: skip `append_run_started` unless signature hash is available, or compute hash from canonical in-memory signature when file write succeeds.

### 9. [P2][OPEN] dual `event` + `event_type` is warning-only, not invalid
- Evidence: dual key emits warning (`veriscope/core/governance.py:371` to `veriscope/core/governance.py:372`), and `validate_governance_log` does not promote it to error (`veriscope/core/governance.py:409` to `veriscope/core/governance.py:417`).
- Failure mode: new lines can violate exclusivity contract while still validating.
- Reproduction: write a governance line with both keys and valid hash chain; validation returns `ok=True` with warning.
- Suggested fix: escalate `GOVERNANCE_LOG_EVENT_DUAL_PRESENT` to validation error.

### 10. [P2][OPEN] `gate_decision_v1.audit` nested contract is not schema-validated in governance parser
- Evidence: parser only requires `audit` be a dict (`veriscope/core/governance.py:137` to `veriscope/core/governance.py:144`, `veriscope/core/governance.py:197` to `veriscope/core/governance.py:206`).
- Failure mode: governance gate events can omit required audit semantics (e.g., evaluated/policy/reason) yet still pass log validation.
- Reproduction: append a `gate_decision_v1` line with `audit={}` and valid hashes; validator accepts line shape.
- Suggested fix: nested audit field checks aligned to `AuditV1` contract.

## 6) Implemented Fixes in This Diff (Top 3+)
1. CIFAR wrapper now fails closed on missing/invalid summary and emits valid partial fallback.
2. `diff` and `report --compare` now enforce strict cross-artifact identity.
3. Governance parser now enforces required payload fields for run events.
4. Single-run `report` now also enforces strict identity.

## 7) Added Tests
- `tests/test_cli_run_cifar_wrapper.py:29` (invalid CIFAR summary replacement).
- `tests/test_cli_diff.py:197` (diff identity mismatch rejection).
- `tests/test_cli_diff.py:212` (report-compare identity mismatch rejection).
- `tests/test_cli_validate_report.py:351` (single report identity mismatch rejection).
- `tests/test_governance_contract.py:233` (run_started required payload validation).

## 8) Verification Notes
- Could not execute pytest due missing `pytest`/`numpy` in available environments.
- Syntax checks passed:
  - `PYTHONPYCACHEPREFIX=/tmp/veriscope_pycache /usr/bin/python3 -m py_compile ...` on all modified Python files.
