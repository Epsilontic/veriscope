# Pilot FAQ (operators)

Normative precedence: `docs/contract_v1.md` is the authoritative contract. This page is derived pilot guidance.

## How do I run the pilot?
Use the harness in `scripts/pilot/run.sh`:

```bash
bash scripts/pilot/run.sh [OUTDIR] -- <runner args>
```

This runs `veriscope run gpt`, then `validate`, `inspect`, and `report` while capturing outputs to the OUTDIR. `scripts/pilot/run.sh` is strict by default (`VERISCOPE_PILOT_STRICT=1`) and exits non-zero if any of those checks fail; set `VERISCOPE_PILOT_STRICT=0` only when you explicitly want lenient capture.

## What do the exit codes mean?
`veriscope run` returns non-zero on runner failure or a Veriscope internal failure. Decision outcomes (pass/warn/fail/skip) are recorded in artifacts rather than as exit codes, per operational guidance in `docs/productization.md` (derived from `docs/contract_v1.md`).

## What does decision=skip mean?
`skip` is neutral: the gate was not evaluated due to missing evidence or policy, and it should not be treated as pass/fail.

## What do WARN vs FAIL mean?
WARN means investigate; FAIL means stop or rollback under the configured preset. Use manual overrides to document false positives without mutating artifacts.

## What should I share?
At minimum:
- `window_signature.json`
- for non-partial capsules: `results.json` and `results_summary.json`
- for partial capsules: `results_summary.json` with `partial=true`

Recommended:
- `run_config_resolved.json`
- `report.md`

If allowed, share the full OUTDIR capsule for reproducibility.

## Why did compare mode fail even though report works?
`veriscope diff` and `veriscope report --compare` are stricter than single-capsule report mode: they require valid `governance_log.jsonl` files and reject capsules marked `partial=true`.
