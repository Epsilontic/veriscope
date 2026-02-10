# Pilot harness scripts

For pilot goals and success criteria, see `docs/pilot/README.md`.
Normative precedence for artifacts/comparability is `docs/contract_v1.md`; this page is workflow guidance.

## run.sh

Run a pilot capsule end-to-end with a tuned gate preset by default:

```bash
bash scripts/pilot/run.sh [OUTDIR] -- <runner args>
```

Behavior:
- Writes artifacts into `OUTDIR` (default: `./out/pilot_<ts>`).
- Injects `--gate_preset tuned_v0` **unless** you already provided `--gate_preset`.
- Comparability canonicalizes the legacy alias `tuned` to `tuned_v0`.
- Captures `validate`, `inspect`, and `report` outputs in the OUTDIR.
- Resolves the effective capsule directory into `OUTDIR/capdir.txt` (important when `veriscope run ... --force` emits into a fresh subdirectory).
- Strict by default: fails if `run`, `validate`, `inspect`, or `report` fail.
- Lenient mode is explicit: set `VERISCOPE_PILOT_STRICT=0` to keep old "run-only exit status" behavior.

Outputs in OUTDIR:
- `validate.txt`
- `inspect.txt`
- `report.md`
- `report_stderr.txt`
- `capdir.txt` (resolved capsule directory validated/reported by the harness)
- `git_sha.txt`
- `version.txt`

Exit code:
- Default (`VERISCOPE_PILOT_STRICT=1`): nonzero if any of `run`, `validate`, `inspect`, or `report` fail.
- Lenient (`VERISCOPE_PILOT_STRICT=0`): propagates `veriscope run gpt` exit code if nonzero; otherwise exits 0.

## score.py

Generate calibration summaries from control + injected runs:

```bash
python scripts/pilot/score.py \
  --control-dir OUTDIR_CONTROL \
  --injected-dir OUTDIR_INJECTED \
  --out calibration.json \
  --out-md calibration.md
```

Delay semantics: delays are measured in iterations after t0=max(injection onset, warmup) where onset is `data_corrupt_at` from `run_config_resolved.json` (or `--injection-onset-iter` when missing). `Delay_W` reports delay in units of `gate_window`. Gate events are read from `governance_log.jsonl` (`gate_decision_v1`) with a fallback to `results.json`. FAR is computed as warn+fail rate post-warmup (FAR_warnfail).

## negative_controls.sh

Run negative controls on a completed capsule:

```bash
bash scripts/pilot/negative_controls.sh OUTDIR
```

This script creates copies of the capsule to validate:
- Governance tamper detection (`veriscope validate --strict-governance` must fail).
- Window signature mismatch detection (`veriscope diff` must report `WINDOW_HASH_MISMATCH`).
- Partial-capsule rejection (`veriscope diff` must fail with a partial/non-partial indicator).
- Full-capsule comparability sanity check (`veriscope diff` on two full capsules must succeed).

## Troubleshooting
- If `veriscope diff` fails early, validate the base capsule first with `veriscope validate OUTDIR`.
- If `score.py` exits with `MISSING_GATE_WINDOW`, ensure `window_signature.json` or `run_config_resolved.json` is present.
