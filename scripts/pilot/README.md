# Pilot harness scripts

For pilot goals and success criteria, see `docs/pilot/README.md`.

## run.sh

Run a pilot capsule end-to-end with a tuned gate preset by default:

```bash
bash scripts/pilot/run.sh [OUTDIR] -- <runner args>
```

Behavior:
- Writes artifacts into `OUTDIR` (default: `./out/pilot_<ts>`).
- Injects `--gate_preset tuned_v0` **unless** you already provided `--gate_preset`.
- Captures `validate`, `inspect`, and `report` outputs in the OUTDIR.

Outputs in OUTDIR:
- `validate.txt`
- `inspect.txt`
- `report.md`
- `report_stderr.txt`
- `git_sha.txt`
- `version.txt`

Exit code:
- Propagates the `veriscope run gpt` exit code if nonzero; otherwise exits 0.

## score.py

Generate calibration summaries from control + injected runs:

```bash
python scripts/pilot/score.py \
  --control-dir OUTDIR_CONTROL \
  --injected-dir OUTDIR_INJECTED \
  --out calibration.json \
  --out-md calibration.md
```

Delay semantics: `Delay_W` is measured in gate-evaluation steps after warmup, normalized by the gate window size.

## negative_controls.sh

Run negative controls on a completed capsule:

```bash
bash scripts/pilot/negative_controls.sh OUTDIR
```

This script creates copies of the capsule to validate:
- Governance tamper detection (`veriscope validate --strict-governance` must fail).
- Window signature mismatch detection (`veriscope diff` must report `WINDOW_HASH_MISMATCH`).
- Partial-mode diff behavior (no `counts_` lines, with `NOTE:PARTIAL_MODE`).

## Troubleshooting
- If `veriscope diff` fails early, validate the base capsule first with `veriscope validate OUTDIR`.
- If `score.py` exits with `MISSING_GATE_WINDOW`, ensure `window_signature.json` or `run_config_resolved.json` is present.
