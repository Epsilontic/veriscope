# Pilot kit overview

Normative precedence: `docs/contract_v1.md` is the authoritative contract. This page is derived pilot guidance.

## What this is
The pilot kit provides the GPT-first MVP path: a GPT control/injected run pair, validation, and shareable report/calibration outputs. It follows operational guidance in `docs/productization.md` (derived from `docs/contract_v1.md`) and uses the GPT pilot harness scripts in `scripts/pilot/`.

Use this path when evaluating Veriscope today. Other runners and helper surfaces exist, but this GPT control/injected workflow is the current design-partner path.

## GPT-first MVP path

Run the GPT pilot harness directly:

```bash
# Control run
bash scripts/pilot/run.sh ./out/pilot_control -- --dataset shakespeare_char --nanogpt_dir ./nanoGPT

# Injected run
bash scripts/pilot/run.sh ./out/pilot_injected -- --dataset shakespeare_char --nanogpt_dir ./nanoGPT \
  --data_corrupt_at 2500 --data_corrupt_len 400 --data_corrupt_frac 0.15 --data_corrupt_mode permute

# Calibration outputs
python scripts/pilot/score.py \
  --control-dir ./out/pilot_control \
  --injected-dir ./out/pilot_injected \
  --out calibration.json \
  --out-md calibration.md
```

Primary outputs from this path:
- `report.md` from the harness
- `calibration.json`
- `calibration.md`
- the capsule artifacts under each OUTDIR

## Success criteria (v0)
Measured on `gate_preset=tuned_v0` with control + injected runs:
- False Alarm Rate (post-warmup) ≤ 5% on control runs.
- Detection delay ≤ 2W iterations, where W is the gate window size.
- Overhead ≤ 5% wall-clock when timing fields exist.

## What to share
- `report.md` (from `veriscope report OUTDIR --format md`)
- `calibration.json` + `calibration.md` (from `scripts/pilot/score.py`)
- At minimum (non-partial capsules): `window_signature.json`, `results.json`, `results_summary.json`
- At minimum (partial capsules): `window_signature.json` and `results_summary.json` with `partial=true`
- `run_config_resolved.json` is recommended for reproducibility context
- If you plan to run `veriscope diff` or `veriscope report --compare`, include a valid `governance_log.jsonl`; compare mode rejects missing/invalid governance and rejects `partial=true` capsules.

## Redaction policy (write-time enforced)
Provenance capture applies allowlist/denylist redaction **at write time** when emitting `run_config_resolved.json`, so redaction is enforced at the boundary, not as a post-run scan. See `veriscope/core/redaction_policy.py` for the shared policy and `veriscope/cli/main.py` for the write-time sanitizer.

## How to run the scripts
See `scripts/pilot/README.md` for step-by-step commands, outputs, and troubleshooting. Treat that GPT harness path as the current MVP workflow.
