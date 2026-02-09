# Veriscope Contract Completeness / Pilot-Honesty Audit (2026-02-09)

## Scope and Method
- Scope audited: artifact emission + validation + comparability + report paths for `legacy/cifar`, `gpt/nanoGPT`, `hf/transformers`.
- Contract source: `docs/contract_v1.md`.
- Code paths traced: `veriscope/runners/gpt/*`, `veriscope/runners/hf/*`, `veriscope/cli/validate.py`, `veriscope/cli/diff.py`, `veriscope/cli/report.py`, `veriscope/cli/main.py`, `veriscope/core/artifacts.py`, `veriscope/core/governance.py`.
- Validation status in this environment: could not run pytest end-to-end because the available Python environments are missing `pytest`/`numpy`; syntax checks passed via `py_compile`.

## Contract Matrix (from `docs/contract_v1.md`)

### Artifacts
| Artifact | Contract requirement | Evidence |
|---|---|---|
| `window_signature.json` | Required for non-partial and partial capsules | `docs/contract_v1.md:10`, `docs/contract_v1.md:16` |
| `results.json` | Required for non-partial capsules | `docs/contract_v1.md:12` |
| `results_summary.json` | Required for non-partial and partial capsules; partial must have `partial=true` | `docs/contract_v1.md:13`, `docs/contract_v1.md:17` |
| `run_config_resolved.json` | Recommended | `docs/contract_v1.md:20` |
| `first_fail_iter.txt` | Recommended when `results_summary.first_fail_iter` is non-null | `docs/contract_v1.md:21` |
| `manual_judgement.json` | Optional governance overlay | `docs/contract_v1.md:24` |
| `manual_judgement.jsonl` | Optional append-only governance overlay | `docs/contract_v1.md:25` |
| `governance_log.jsonl` | Optional append-only governance journal | `docs/contract_v1.md:26` |

### Required/MUST field-level rules explicitly stated in contract doc
| File | Required rule(s) from contract text | Evidence |
|---|---|---|
| `window_signature.json` | Hash comparability root; recomputed canonical hash (excluding volatile keys) must match `window_signature_ref.hash` and path must be `window_signature.json` | `docs/contract_v1.md:37`, `docs/contract_v1.md:38` |
| `results.json` | Contract doc requires artifact presence for non-partial capsules; explicit field spine is not fully restated in this file (implemented in schema code) | `docs/contract_v1.md:12` |
| `results_summary.json` | `final_decision` is authoritative automated outcome; optional `first_fail_iter` obeys fail/no-fail invariant; partial capsules must be marked `partial=true` | `docs/contract_v1.md:17`, `docs/contract_v1.md:99`, `docs/contract_v1.md:106`, `docs/contract_v1.md:110` |
| `governance_log.jsonl` | Per-line object with `schema_version`, `rev`, `ts_utc`, `event/event_type`, `payload`; hash chain semantics (`prev_hash`, `entry_hash`) | `docs/contract_v1.md:48`, `docs/contract_v1.md:53`, `docs/contract_v1.md:62`, `docs/contract_v1.md:66`, `docs/contract_v1.md:67` |

## Runner Emission Trace (Golden Paths)
- GPT path: `veriscope/runners/gpt/train_nanogpt.py:2761` -> `veriscope/runners/gpt/emit_artifacts.py:406`
- HF path: `veriscope/runners/hf/train_hf.py:1634` -> `veriscope/runners/hf/emit_artifacts.py:79`
- CIFAR legacy wrapper path: `veriscope/cli/main.py:697` delegates to `veriscope-legacy`, wrapper fills partial summary fallback in `veriscope/cli/main.py:807`

## Top 10 Pilot-Risk Findings (ranked)

### 1) [P0] Summary counts could diverge from gate events without validation failure (FIXED)
- Evidence: invariant now enforced in `veriscope/cli/validate.py:358`; regression test in `tests/test_cli_validate_report.py:340`.
- Failure mode: tampered or stale `results_summary.json` can claim different pass/warn/fail counts than `results.json`.
- Reproduction/unit test: `tests/test_cli_validate_report.py:340`.
- Minimal patch: compare summary counts against counts derived from `results.gates`; fail validation on mismatch.
- Status: fixed in this diff.

### 2) [P0] `final_decision` and `first_fail_iter` invariants were not schema-enforced (FIXED)
- Evidence: invariants now in `veriscope/core/artifacts.py:331`; regression tests at `tests/test_artifacts_roundtrip.py:183` and `tests/test_artifacts_roundtrip.py:199`.
- Failure mode: summary could carry a contradictory final decision or missing `first_fail_iter` under fail counts.
- Reproduction/unit test: `tests/test_artifacts_roundtrip.py:183`, `tests/test_artifacts_roundtrip.py:199`.
- Minimal patch: model-level validator on `ResultsSummaryV1`.
- Status: fixed in this diff.

### 3) [P0] HF emitter did not emit `first_fail_iter`/marker under failures (FIXED)
- Evidence: now computed and emitted in `veriscope/runners/hf/emit_artifacts.py:145`, `veriscope/runners/hf/emit_artifacts.py:165`; regression test at `tests/test_hf_emitter_parity.py:82`.
- Failure mode: HF fail capsules could not satisfy finite-window fail diagnostics parity with GPT.
- Reproduction/unit test: `tests/test_hf_emitter_parity.py:82`.
- Minimal patch: derive earliest fail iteration from gate records; emit `first_fail_iter` and `first_fail_iter.txt`.
- Status: fixed in this diff.

### 4) [P0] Partial capsules could pass allow-partial flows without explicit `partial=true` (FIXED)
- Evidence: check added in `veriscope/cli/validate.py:347`; regression test at `tests/test_cli_validate_report.py:361`.
- Failure mode: missing `results.json` could be silently treated as partial in report/diff/inspect even when summary was not marked partial.
- Reproduction/unit test: `tests/test_cli_validate_report.py:361`.
- Minimal patch: require `results_summary.partial==true` when `results.json` is absent in allow-partial mode.
- Status: fixed in this diff.

### 5) [P1] CIFAR wrapper fallback only checks summary existence, not summary validity (OPEN)
- Evidence: `veriscope/cli/main.py:807` checks only `exists()`.
- Failure mode: malformed `results_summary.json` can remain in capsule with no wrapper repair path.
- Reproduction: run CIFAR path with an invalid pre-existing summary file; wrapper does not rewrite fallback unless file is absent.
- Minimal patch: gate fallback on `exists() and _summary_is_valid(...)` (same policy used in GPT/HF wrapper path at `veriscope/cli/main.py:636`).
- Test plan: add wrapper integration test with fake legacy runner and pre-seeded invalid summary.

### 6) [P1] Identity mismatches are downgraded to warnings in allow-partial mode (OPEN)
- Evidence: `veriscope/cli/validate.py:187` (warning path).
- Failure mode: structurally inconsistent capsules can still flow through report/inspect as “partial”.
- Reproduction: existing behavior locked in `tests/test_cli_inspect.py:123` (exit 0 with `WARNING:ARTIFACT_IDENTITY_MISMATCH`).
- Minimal patch: keep warning in `inspect`, but make `diff`/`report --compare` require strict identity.
- Test plan: add diff/report tests expecting hard-fail on identity mismatch.

### 7) [P1] Report commands intentionally allow invalid governance logs (OPEN)
- Evidence: `veriscope/cli/report.py:114` and `veriscope/cli/report.py:405` set `allow_invalid_governance=True`.
- Failure mode: reports can render from capsules with invalid governance hash chains.
- Reproduction: malformed governance log still renders report with warning banner.
- Minimal patch: add strict mode default for compare/report in CI contexts or a hard-fail flag defaulted on for `report --compare`.
- Test plan: add report compare test that fails on invalid governance by default.

### 8) [P1] Governance validator does not enforce event-specific payload schema requirements (OPEN)
- Evidence: parser only checks `payload` is a dict in `veriscope/core/governance.py:179`; no per-event required-key validation.
- Failure mode: governance lines can validate while missing required keys like `run_id`, `iter`, or `window_signature_ref`.
- Reproduction: craft `gate_decision_v1` entry with `payload={}` and valid hash chain; `validate_governance_log` accepts structural line.
- Minimal patch: add per-event payload validators (required key/type checks).
- Test plan: unit tests for each event type with missing required payload fields.

### 9) [P2] Governance event dual-key (`event` + `event_type`) only warns, never fails (OPEN)
- Evidence: warning emitted in `veriscope/core/governance.py:253`; not promoted to error in `veriscope/core/governance.py:291`.
- Failure mode: new entries can violate contract exclusivity and still pass non-strict validation.
- Reproduction: append a line with both keys; governance validation returns ok if hashes/rev are otherwise valid.
- Minimal patch: treat dual-key presence as invalid for schema v1 writes.
- Test plan: add governance unit test expecting validation failure for dual-key entries.

### 10) [P2] HF run-start governance can be emitted with hash-less `window_signature_ref` (OPEN)
- Evidence: default `{"path": "window_signature.json"}` at `veriscope/runners/hf/train_hf.py:1284`; hash filled only inside try at `veriscope/runners/hf/train_hf.py:1287`.
- Failure mode: governance provenance may omit comparability hash under pre-write failures.
- Reproduction: induce write failure before governance append (e.g., path permissions); inspect logged run_started payload.
- Minimal patch: fail run_started append if hash missing, or defer append until hash is guaranteed.
- Test plan: simulate write failure and assert run_started is not appended with incomplete signature ref.

## Implemented Top-3 Fix Set
- `veriscope/cli/validate.py`
  - Added summary-vs-results count coherence checks.
  - Added explicit partial marker requirement when `results.json` is absent.
  - Added `first_fail_iter.txt` enforcement for fail summaries.
- `veriscope/core/artifacts.py`
  - Added `ResultsSummaryV1.partial`.
  - Added model invariants for canonical `final_decision` and `first_fail_iter` semantics.
- `veriscope/runners/hf/emit_artifacts.py`
  - Added fail-iteration derivation and marker emission/removal logic.
- `veriscope/cli/main.py`
  - Partial summary now derives final decision via canonical function and populates `partial` in-model.

## Added/Updated Tests
- `tests/test_cli_validate_report.py:340`
- `tests/test_cli_validate_report.py:353`
- `tests/test_cli_validate_report.py:361`
- `tests/test_hf_emitter_parity.py:82`
- `tests/test_artifacts_roundtrip.py:183`
- Fixture updates for counts/gates consistency:
  - `tests/test_cli_validate_report.py:148`
  - `tests/test_cli_diff.py:73`
  - `tests/test_cli_inspect.py:64`
  - `tests/test_phase5c_diff_json_parse.py:67`

## Notes
- A pre-existing unrelated worktree change was present in `veriscope/runners/gpt/emit_artifacts.py`; this audit did not modify or revert it.
