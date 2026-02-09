# Calibration Protocol v0 (Window-Scoped)

Calibration in Veriscope is a **protocol**, not a constant lookup.

Given an explicit window
`W=(M_W, I_W, Φ_W, B, ε)`,
this protocol produces:

1. a candidate gate preset (or explicit rejection), and
2. a calibration report artifact (JSON) documenting validity limits.

See example artifact: [`docs/examples/calibration_report_v0.example.json`](./examples/calibration_report_v0.example.json).

## Domain of Validity (Hard Boundary)

Calibration results are valid only for the exact `window_signature_ref.hash` used during calibration.

- Claimed: performance estimates for runs comparable under the same window-signature hash (+ matching preset policy assumptions).
- Non-claimed: portability to different hashes, metrics, transports, or gate presets without recalibration.

## Inputs

Required inputs:

- Control capsules (expected healthy behavior) under one window-signature hash.
- Chosen search space for gate preset parameters.
- Operational constraint targets / error budgets.

Optional inputs:

- Injected-pathology capsules for delay/sensitivity estimation.

## Outputs

Required outputs:

- `preset_candidate` (parameterization of gate policy) or explicit `status="rejected"`.
- `calibration_report_v0` JSON artifact with:
  - window identity (`window_signature_ref.hash`, preset family),
  - sample counts,
  - estimated FAR/delay/overhead,
  - constraint checks,
  - failure reasons if rejected.

## Error Budgets and Constraints

Track these named budgets explicitly in the report:

- `δ_FAR`: tolerated false alarm rate on controls.
- `ε_det`: tolerated detection-delay envelope.
- `κ_sens`: tolerated sensitivity shortfall.
- `κ_cal`: tolerated calibration instability (across splits/seeds).

A preset is admissible only if all enabled constraints pass.

## Minimal Method (v0)

1. Build per-metric reference distributions from control runs (post-warmup cadence subset).
2. Convert each evaluated cadence to per-metric exceedance scores relative to references.
3. Aggregate exceedances with an explicit policy (`max`, weighted sum, or declared equivalent).
4. Grid-search preset parameters under constraints (`δ_FAR`, `ε_det`, `κ_sens`, `κ_cal`).
5. Select the simplest admissible preset (tie-break documented), else reject with reasons.
6. Emit report with all assumptions and non-claims.

## Failure Modes (Must Be Explicit)

- Hash mismatch across candidate runs.
- Insufficient control evidence after warmup.
- Missing or invalid governance/events needed for audit reconstruction.
- No preset satisfies constraints.
- Unstable estimates across seeds/splits beyond `κ_cal`.

Any failure must produce `status="rejected"` and a non-empty `failure_modes` list.

## Verification Hooks

```bash
# Validate candidate capsules before calibration
veriscope validate CONTROL_OUTDIR
veriscope validate INJECTED_OUTDIR

# Contract checks for comparability preconditions
veriscope diff CONTROL_OUTDIR INJECTED_OUTDIR

# Existing calibration-related tests
pytest -q tests/test_calibration.py tests/test_cli_calibrate.py tests/test_scoring.py
```

## Protocol Contract Notes

- This file defines the **protocol contract** only.
- Runtime model/CLI extensions for calibration reports are optional and may be added later.
- If runtime support is absent, teams should still emit the report JSON artifact following the schema example.
