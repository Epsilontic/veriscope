# Pilot kit overview

## What this is
The pilot kit provides a partner-facing workflow for running a control and an injected-pathology run, validating artifacts, and generating shareable outputs. It follows the v0 contract in `docs/productization.md` and uses the pilot harness scripts in `scripts/pilot/`.

## Success criteria (v0)
Measured on `gate_preset=tuned_v0` with control + injected runs:
- False Alarm Rate (post-warmup) ≤ 5% on control runs.
- Detection delay ≤ 2W iterations, where W is the gate window size.
- Overhead ≤ 5% wall-clock when timing fields exist.

## What to share
- `report.md` (from `veriscope report OUTDIR --format md`)
- `calibration.json` + `calibration.md` (from `scripts/pilot/score.py`)
- At minimum (non-partial capsules): `results.json`, `results_summary.json`, `window_signature.json`
- For partial capsules: `window_signature.json` + `results_summary.json` with `partial=true`
- `run_config_resolved.json` is recommended for reproducibility context

## Redaction policy (write-time enforced)
Provenance capture applies allowlist/denylist redaction **at write time** when emitting `run_config_resolved.json`, so redaction is enforced at the boundary, not as a post-run scan. See `veriscope/core/redaction_policy.py` for the shared policy and `veriscope/cli/main.py` for the write-time sanitizer.

## How to run the scripts
See `scripts/pilot/README.md` for step-by-step commands, outputs, and troubleshooting.
