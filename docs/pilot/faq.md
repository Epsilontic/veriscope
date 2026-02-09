# Pilot FAQ (operators)

## How do I run the pilot?
Use the harness in `scripts/pilot/run.sh`:

```bash
bash scripts/pilot/run.sh [OUTDIR] -- <runner args>
```

This runs `veriscope run gpt`, then `validate`, `inspect`, and `report` while capturing outputs to the OUTDIR.

## What do the exit codes mean?
`veriscope run` returns non-zero on runner failure or a Veriscope internal failure. Decision outcomes (pass/warn/fail/skip) are recorded in artifacts rather than as exit codes, per the productization contract.

## What does decision=skip mean?
`skip` is neutral: the gate was not evaluated due to missing evidence or policy, and it should not be treated as pass/fail.

## What do WARN vs FAIL mean?
WARN means investigate; FAIL means stop or rollback under the configured preset. Use manual overrides to document false positives without mutating artifacts.

## What should I share?
At minimum:
- `report.md`
- `results_summary.json`
- `window_signature.json`
- `run_config_resolved.json`

If allowed, share the full OUTDIR capsule for reproducibility.

## Why did compare mode fail even though report works?
`veriscope diff` and `veriscope report --compare` are stricter than single-capsule report mode: they require valid `governance_log.jsonl` files and reject capsules marked `partial=true`.
