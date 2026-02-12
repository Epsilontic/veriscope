# Veriscope Core Audit (2026-02-12)

## Scope and method
- Audit objective: semantic torsion between declared FR/window/comparability/calibration semantics and core implementation.
- Bounded focus: contract surfaces + `veriscope/core/*` algorithm paths + comparability enforcement.
- Deep-read anchors: `docs/contract_v1.md`, `docs/productization.md`, `docs/calibration_protocol_v0.md`, `README.md`, `veriscope/core/ipm.py`, `veriscope/core/gate.py`, `veriscope/core/window.py`, `veriscope/core/artifacts.py`, `veriscope/core/jsonutil.py`, `veriscope/core/calibration.py`, `veriscope/core/pilot_calibration.py`, `veriscope/cli/comparability.py` (plus targeted validation/tests for confirmation).

## A) Canonical algorithmic claims extracted from spec surfaces
- Contract says comparability is hash-gated by canonical `window_signature.json` hash and gate preset policy (`Comparable(A,B)`), with partial capsules non-comparable.
- Productization/README say gate computes a finite-window discrepancy (`worst_DW`) and compares against effective threshold (`eps_eff`), emitting canonical decision enum (`pass|warn|fail|skip`).
- Productization says calibration is read-only scoring over control/injected capsules and reports FAR/delay/overhead fields.
- Calibration protocol says validity is scoped to exact `window_signature_ref.hash`; mismatch/failure modes must be explicit.

## B) What is actually computed (code-level)
### 1) D_W / discrepancy path
- Per-metric discrepancy primitive: `tv_hist_fixed()` in `veriscope/core/ipm.py`.
  - Filters non-finite values, clips to `[0,1]`, uses fixed-bin histogram TV.
  - Returns `0` when both sides empty after filtering; returns `NaN` when exactly one side is empty.
- Gate discrepancy used for decisions: `GateEngine.check()` in `veriscope/core/gate.py`.
  - Applies transport + interventions per metric.
  - Computes per-metric TV via `tv_hist_fixed`.
  - Aggregates by normalized weighted sum (`sum(abs(w_i))/w_sum`) for each intervention.
  - Chooses the worst intervention aggregate as `audit.worst_DW`.
  - Includes zero-TV rescue path (`_rescue_zero_tv_from_bin_collapse`) for non-identical windows collapsing to zero under coarse bins.

### 2) Epsilon / thresholding path
- In `GateEngine.check()`:
  - `eps_cap = epsilon * clamp(eps_stat_max_frac, 0, 1)`
  - `eps_stat = clamp(eps_stat_value, 0, eps_cap)`
  - `eps_eff = max(0, epsilon + eps_sens - eps_stat)`
  - Stability exceedance: `dw_exceeds_threshold = (worst_DW > eps_eff)` when finite.
- Multi-metric consensus filter may override raw exceedance (`min_metrics_exceeding`).

### 3) Evidence and budget entry points
- Gate evidence sufficiency (`GateEngine.check()`):
  - `evidence_total = sum(counts_by_metric.values())`.
  - If `min_evidence > 0` and `evidence_total < min_evidence`: gate is not evaluated (`audit.evaluated=false`, `decision` should be `skip`).
  - If no finite metric TVs exist: also not evaluated.
- Calibration scoring (`veriscope/core/pilot_calibration.py`):
  - FAR computed on post-warmup gate events from control.
  - Delay computed from first warn/fail after `max(warmup, injection_onset)` in injected run.
  - `calibration_status` is `incomplete` on missing core inputs/signals.
  - No grid search / admissibility optimization logic is implemented in runtime path.

### 4) Decision derivation
- Per-gate canonical decision mapping exists in `derive_gate_decision()` (`veriscope/core/artifacts.py`):
  - not evaluated -> `skip`; else `ok=false` -> `fail`; else `warn=true` -> `warn`; else `pass`.
- Summary decision is fail-dominant via `derive_final_decision()`.
- Comparability predicate implementation (`veriscope/cli/comparability.py`) enforces:
  - non-partial,
  - schema match,
  - matching window hash,
  - matching gate preset unless override flag is set.

### 5) Calibration control/injected coupling
- Prior to this audit patch, `calibrate_pilot()` did **not** enforce cross-capsule window hash comparability.
- After patch in this audit:
  - both control/injected `window_signature.json` are loaded,
  - canonical hashes are recomputed,
  - each summary ref hash must match its local recomputed hash,
  - control/injected hashes must match each other,
  - mismatch fails loudly via `CalibrationError` token `WINDOW_SIGNATURE_HASH_MISMATCH`.

## C) Ranked mismatch list (semantic torsion)
1. **[High, fixed in this patch] Calibration domain-of-validity hash gate was not enforced in runtime calibration.**
   - Claim surface: calibration validity is window-hash scoped.
   - Prior behavior: `calibrate_pilot()` could score control/injected pairs from different window signatures without error.
   - Risk: false trust in FAR/delay numbers across non-comparable capsules.

2. **[Medium] `D_W` naming is overloaded: `core/ipm.py::D_W` is max-TV; gate’s `audit.worst_DW` is weighted aggregate.**
   - Claim surface often refers to `D_W` singularly.
   - Implementation splits semantics between helper API and decision-driving gate path.
   - Risk: reviewer confusion, incorrect downstream theoretical interpretation (max vs weighted-IPM proxy).

3. **[Medium] Calibration protocol document describes admissibility budgets (`δ_FAR`, `ε_det`, `κ_sens`, `κ_cal`) and preset selection; runtime calibrate path is descriptive scoring only.**
   - Partially acknowledged in protocol notes as optional runtime support.
   - Risk: users may assume `veriscope calibrate` is full protocol execution rather than pilot-score summary.

4. **[Low/Medium] Calibration event parsing rejects `skip` decisions (`pass|warn|fail` only).**
   - Contract makes `skip` canonical/neutral.
   - Current calibration path hard-fails on `skip` gate events instead of treating them as unevaluated.
   - Risk: calibration operational fragility for runs with expected skip gates.

## D) Silent-hazard checks requested
- Unscoped NaN/Inf serialization:
  - Canonical JSON hashing path uses `allow_nan=False` (`veriscope/core/jsonutil.py`).
  - Gate internals carry NaN placeholders in audit memory; artifact boundary must sanitize or validate accordingly.
- Implicit randomness without stable seed:
  - No randomness in gate/IPM core path; regime sampling path uses deterministic SHA256-derived seeds.
- Window signature used for gating/comparability:
  - Comparability predicate is hash + preset gated in CLI compare path.
  - Calibration now enforces same window-hash scope before computing FAR/delay.
- Threshold/budget enforcement:
  - `min_evidence` and threshold math are enforced in gate.
  - Calibration protocol budgets are not runtime-enforced as an optimization/search loop.
- Aggregation/scaling checks:
  - Gate uses abs-weight normalized sum; this differs from max-TV helper naming in `ipm.D_W`.

## E) Risk to pilot/reviewer
- Highest risk to pilot trust (now addressed): calibration scores computed on non-comparable windows.
- Remaining trust risk: `D_W` naming ambiguity can mislead reviewers about what thresholding actually evaluates.
- Remaining reproducibility risk: protocol-vs-runtime calibration expectation gap may cause over-interpretation of pilot score outputs.

## Containment plan for remaining open issues (no further code changes in this bounded pass)
- For `D_W` ambiguity:
  - Treat `GateEngine.audit.worst_DW` as the decision-authoritative quantity in pilot/reviewer materials.
  - Avoid citing `core.ipm.D_W` as equivalent to gate discrepancy until naming/semantics are unified.
- For calibration protocol gap:
  - Frame `veriscope calibrate` output as pilot score summary, not preset-admissibility proof.
  - Require explicit mention of absent constraint-search execution in reviewer packets.
- For `skip` parsing fragility:
  - Keep calibration inputs to evaluated gate events where possible; prioritize follow-up fix to handle `skip` as neutral rather than invalid.

