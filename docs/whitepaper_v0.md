# Veriscope v0 — Capsule Contract Whitepaper (Contract v1)

**Purpose.** This document describes Veriscope v0 as a *contract-oriented* system: what it emits, what it validates, and what downstream consumers may rely on. It is intentionally conservative about claims: it describes implemented behavior and contract guarantees, without asserting workload-universal statistical or model-quality guarantees.

## Norm hierarchy (how to resolve “what is normative”)

1. **Frozen capsule contract (primary):** `docs/contract_v1.md`
   Normative for artifact schemas, canonicalization, hashing, comparability, and partial-capsule rules.
2. **Operational guide (secondary):** `docs/productization.md`
   Normative for pipeline/ops expectations (atomic writes, canonical decision posture, operational exit codes).
3. **Pilot kit (tertiary):** `docs/pilot/*`, `scripts/pilot/*`
   Normative for pilot harness workflows and scoring definitions (including pilot success criteria stated in pilot docs).

(See **Appendix D — Traceability**.)

---

## 1. Executive Summary

Veriscope v0 is a CLI-operated training monitor that evaluates finite-window discrepancies over *declared* training metrics and emits an auditable capsule directory. A capsule contains:

- a **window signature** (`window_signature.json`): the declared scope/configuration under which results are meaningful,
- per-gate decisions and audit fields (`results.json`) and a rollup (`results_summary.json`), or a partial summary in partial mode,
- optional provenance (`run_config_resolved.json`) and governance overlays (manual judgement + append-only logs).

The core contract property is *scope clarity*: every decision is scoped to a declared window signature, and comparability/diff tooling enforces that capsules are only compared when they share the same identity-defining signature and satisfy the contract’s explicit comparability predicate.

---

## 2. System Overview

### 2.1 Pipeline

1. A runner wrapper (`veriscope run <kind>`) executes training and emits declared metrics.
2. Metrics are transformed by a declared transport/normalization configuration.
3. At gate cadence, the gate engine compares “recent” versus “past” evidence windows and emits decisions.
4. Veriscope writes capsule artifacts into `OUTDIR` (with atomic-write conventions for key JSON outputs).
5. Optional provenance and governance artifacts are written/appended.

### 2.2 The declared evaluation scope W

Conceptually, Veriscope scopes decisions to a declared window:

**W = (M_W, I_W, Φ_W, B, ε)**

- **M_W:** declared metrics (what is measured and how it is interpreted/weighted)
- **I_W:** declared interventions (if present; conceptually part of identity)
- **Φ_W:** declared transport/normalization applied before discrepancy computation
- **B:** binning / histogram configuration used by discrepancy primitives
- **ε:** base tolerance/threshold configuration (often via a preset + optional params)

Operationally, this scope is serialized in `window_signature.json` and participates in the capsule’s comparability hash. Consumers should treat the window signature as the *claims boundary* for interpretation.

---

## 3. The Window Signature: Contract Spine vs Repo-Emitted Identity Keys

The frozen contract defines a canonical **schema spine** for `window_signature.json`, while also allowing **extra keys**. Extra keys are permitted and affect hashing/comparability because the signature hash is computed over the canonicalized object (minus volatile keys).

### 3.1 Contract spine (guaranteed key names)

These are the stable, contract-defined anchors you can count on existing (and remaining named consistently) in `window_signature.json`:

| W component | Contract spine fields in `window_signature.json` | Notes |
|---|---|---|
| M_W | `evidence.metrics` | Declares the metrics being evaluated; identity-defining. |
| Φ_W (+ B when encoded there) | `transport` (including `transport.name` and transport configuration) | Transport config is identity-defining. Histogram/binning parameters, if applicable, are captured as transport config and/or additional allowed keys. |
| ε | `gates.preset` (and optionally `gates.params`) | Preset/params are identity-defining because they are inside the hashed signature spine. |

**Interventions (I_W).** If interventions are identity-defining for a given deployment, they will appear either as contract-defined substructure (if present in schema) or as allowed extra keys. In either case, because hashing is computed over the canonicalized signature object, they become identity-defining whenever present.

### 3.2 Repo-emitted identity keys (allowed by contract; used by ops/pilot tooling)

The repo emits additional fields that are not part of the contract spine, but are commonly relied upon by reporting/scoring and operational reasoning. These keys are still identity-defining in practice because they participate in hashing (unless stripped as volatile). However, they are **not guaranteed by the contract spine**; if absent, pilot/reporting behavior may degrade, fall back to other sources, or become less informative.

| Category | Common repo-emitted fields | Why they matter operationally |
|---|---|---|
| Gate cadence / evidence sufficiency | `gate_controls.gate_window`, `gate_controls.gate_epsilon`, `gate_controls.min_evidence` | Used by pilot scoring/reporting and for interpreting skip/non-evaluation behavior. |
| Metric pipeline identity | `metric_pipeline.transport` | Helps explain how raw metrics were mapped/normalized prior to transport/discrepancy logic. |

**Meaning of “safe to rely on (within this repo).”** Here, “safe” means *identity-preserving and diff-safe*: if present, these keys are hashed into the signature and therefore cannot silently drift without breaking comparability. It does **not** mean these keys are a schema-stable public API across versions.

**Why this is two-layered:** the contract spine provides stable anchors for interop; extra keys reflect operational identity used by this repo today and are safe to rely on *within this repo’s tooling* because they are included in the hashed signature when present.

---

## 4. Capsule Contract: Artifacts

### 4.1 Required artifacts (contract v1)

**Required for a non-partial capsule**
- `window_signature.json`
- `results.json`
- `results_summary.json`

**Required for a partial capsule**
- `window_signature.json`
- `results_summary.json` with `partial=true`

**Recommended by contract**
- `run_config_resolved.json`

**Optional governance overlays**
- `manual_judgement.json` (snapshot)
- `manual_judgement.jsonl` (append-only)
- `governance_log.jsonl` (append-only, hash-chained)

Operational compare note:
- `veriscope diff` and `veriscope report --compare` require valid `governance_log.jsonl` in each compared capsule; missing/invalid governance causes failure.

### 4.2 Operational “golden path” note: `run_config_resolved.json`

The frozen contract calls `run_config_resolved.json` **recommended**. The v0 operational guide and pilot harness treat it as operationally expected for provenance, determinism reporting, warmup/cadence visibility, and scoring convenience. This document follows the contract for interop (recommended) while acknowledging ops/pilot expectations in practice.

### 4.3 Decision semantics are canonical

The canonical decision enum is:

**`pass | warn | fail | skip`**

`skip` is **neutral** and corresponds to a gate that was **not evaluated** (`evaluated=false` in audit). Downstream consumers **must not** treat `skip` as pass/fail.

---

## 5. Canonical JSON, Hashing, and Verification

### 5.1 Canonical JSON serialization (contract v1 MUST form)

Canonical JSON is:

```python
json.dumps(
    obj,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)
```

This is the contract’s **MUST** form for canonicalization used in hashing/verification (see `docs/contract_v1.md`, Canonical JSON serialization (MUST)). Operational implication: non-finite numeric values must not appear in canonicalized objects used for hashing/verification.

### 5.2 Window signature hash and volatile-key stripping

`window_signature_ref.hash` is SHA256 over canonical JSON after removing the volatile fields used by the writer/verifier (the contract gives examples).

**Veriscope v0 behavior (current):** hashing strips `created_ts_utc` (and only that key) as volatile **only when it is a valid ISO8601 UTC string with trailing `Z`**; invalid values fail validation.
**Contract posture:** the contract describes volatile stripping by example (e.g., `created_ts_utc`), and the volatile set may expand; downstream verifiers must apply the same volatile-set rule as the writer/verifier for interoperable recomputation.

### 5.3 Verification rule (avoid the self-reference trap)

When validating or comparing, consumers **must recompute** the canonical hash from the parsed file contents (after applying volatile-key removal). Do **not** trust any embedded hash field as authoritative.

---

## 6. Comparability, Diffing, and Partial Mode

### 6.1 Comparability (contract v1)

Two capsules are comparable if they meet the contract’s explicit predicate, at minimum:

- `schema_version == 1`,
- both capsules are non-partial,
- matching recomputed `window_signature_ref.hash`,
- matching **`profile.gate_preset`** (the gate preset identity recorded in `results_summary.json` / `results.json`, often referred to as `gate_preset`) unless a diff flag explicitly permits a preset mismatch (`--allow-gate-preset-mismatch`),
- valid governance logs for compared capsules.

**Note on “preset in two places.”** `window_signature.json.gates.preset` is identity-defining because it is inside the hashed window signature spine. Separately, the contract’s explicit comparability predicate checks the *results/profile* field (`profile.gate_preset`) unless mismatch is explicitly allowed. Typically these align (`profile.gate_preset == window_signature.gates.preset`), but consumers should treat the profile value as the explicit comparability gate per contract.

### 6.2 Partial capsules: compare-mode rejection rule

Partial mode exists to support workflows where full per-gate evidence is unavailable. In partial mode:

- `veriscope diff` rejects the comparison (`PARTIAL_CAPSULE`),
- `veriscope report --compare` rejects the comparison input,
- single-capsule `veriscope report OUTDIR` can still render the capsule as a standalone report.

This preserves neutral handling for partial evidence without allowing cross-run comparability claims.

---

## 7. CLI Surface and Exit Codes

### 7.1 CLI commands (implementation surface)

The CLI provides:
- `veriscope run <kind>`
- `veriscope validate OUTDIR`
- `veriscope report OUTDIR`
- `veriscope diff A B`
- `veriscope override OUTDIR ...`
- `veriscope inspect OUTDIR ...`

### 7.2 Exit codes: contract vs ops semantics

**Contract v1 public meanings**
- `validate`: `0` valid, `2` invalid
- `report`: `0` ok, `2` failed
- `diff`: `0` comparable, `2` incomparable/invalid (including missing/invalid governance or partial capsules)

**Operational semantics (productization guide)**
- exit code `3` is reserved for **internal error** class conditions (including internal-error codepaths in validate/report),
- `veriscope run` uses operational exit behavior tied to run status, including a fallback rule when `results_summary.json` is missing.

Pipeline recommendation: treat `3` as “tooling/internal failure,” distinct from a contract-level invalid capsule (`2`).

---

## 8. Gate Semantics (Implementation-Accurate, Contract-Scoped)

This section describes what the repo computes and records, without turning those mechanics into workload-universal guarantees.

### 8.1 Discrepancy primitive and aggregation

Veriscope computes per-metric discrepancies using histogram-based comparisons under the declared transport and binning configuration, then aggregates them into gate-level audit values. The implementation explicitly handles missing/non-finite metrics and evidence sufficiency conditions, which can lead to non-evaluation and therefore `skip`.

### 8.2 worst_DW

Audit includes `worst_DW`, representing the worst-case discrepancy across the declared comparison structure (e.g., across interventions/paths as implemented).

### 8.3 Statistical adjustment terms: ε_stat and eps_eff

The gate audit includes statistical adjustment machinery:
- `ε_stat` is computed by calibration utilities and aggregated (weighted) across metrics.
- `eps_eff` is computed as:

`eps_eff = max(0, eps_scaled + eps_sens - eps_stat)`

Gate decisions compare discrepancy audit values against the effective threshold per implementation. This is a transparent “what happened” record; it is not, by itself, a universal statistical guarantee across workloads.

### 8.4 `skip` is neutral

`skip` corresponds to `evaluated=false` and is neutral. Common causes include insufficient evidence (e.g., under `min_evidence`) and non-finite/unavailable metric streams.

---

## 9. Provenance, Determinism, and Redaction

### 9.1 `run_config_resolved.json`

When present, `run_config_resolved.json` captures resolved run configuration and provenance, including sanitized argv/environment and determinism status summaries. This improves reproducibility diagnostics and supports pilot scoring/report interpretations.

### 9.2 Determinism status

Determinism checks are recorded under `run_config_resolved.json.determinism_status` and can be enforced via strict CLI options in workflows that require them.

### 9.3 Redaction policy

Environment capture applies a redaction policy (allowlist + deny rules). Sensitive values are replaced with a redaction token, and the artifact records that redactions were applied.

---

## 10. Governance and Manual Overrides (Auditable Overlays)

Governance artifacts record operator actions without rewriting evidence.

- `manual_judgement.json` provides a current snapshot.
- `manual_judgement.jsonl` and `governance_log.jsonl` provide append-only records.

### 10.1 Hash-chained governance log

`governance_log.jsonl` is append-only and hash-chained:
- `entry_hash` is SHA256 over canonical JSON excluding `entry_hash`.
- `prev_hash` links to the prior entry’s `entry_hash`.

### 10.2 Practical invariants (contract requirements)

- `rev` must be strictly consecutive.
- New entries must not include both `event` and `event_type`.
  **Writer guidance:** new v1 writers SHOULD emit `event` (preferred); `event_type` exists for legacy compatibility.

These invariants make the governance log replayable and verifiable by downstream tools.

---

## 11. Pilot Protocol and Scoring (As Implemented)

### 11.1 Harness

The pilot harness runs end-to-end workflows (run → validate → inspect/report) and includes negative controls (tampering, partial artifacts, diff edge cases).

### 11.2 Scoring metrics (definition is “whatever the script computes”)

Pilot scoring computes:
- **FAR**: post-warmup false-alarm rate on control runs (warn+fail among evaluated gates).
  Implementation uses gate-level parsing from `results.json` when available, with a fallback based on summary counts when gate-level data is absent.
- **Delay_W**: number of gate steps after warmup until first warn/fail on injected runs, normalized by `gate_window`.
- **Overhead**: ratio `veriscope_wall_s / runner_wall_s` derived from timing fields.

The pilot kit also states **pilot success criteria** for these metrics (not product guarantees), e.g. **FAR ≤ 5%**, **Delay ≤ 2W** *(where W denotes the gate cadence, i.e. `gate_window`)*, **Overhead ≤ 5%**; see `docs/pilot/README.md`.

---

## 12. Reporting

`veriscope report` renders capsules into text/Markdown summaries suitable for sharing alongside machine-readable artifacts. `report --compare` additionally enforces comparability preconditions, valid governance logs, and non-partial inputs.

---

## 13. Limitations (v0 posture)

- v0 is primarily oriented around single-node workflows; distributed guarantees are not presented as v0 contract claims.
- Runner maturity and coverage varies by runner kind.
- Statistical adjustment terms are computed and recorded, but the repo posture does not claim workload-universal statistical guarantees without further calibration protocol validation.

---

## 14. Roadmap

The operational guide describes intended expansion areas beyond v0 (e.g., strengthening distributed support and operational hardening). Roadmap items are directional and should not be treated as present-tense guarantees.

---

## 15. Licensing

Veriscope is **AGPL-3.0-only unless you have a commercial license**. See `LICENSE` and `COMMERCIAL_LICENSE.md`.

---

# Appendix A — Glossary

- **Capsule:** Output directory containing contract-defined artifacts.
- **Window signature (W):** Declared scope/configuration serialized in `window_signature.json`.
- **Decision enum:** `pass | warn | fail | skip` (canonical).
- **skip:** Neutral; indicates non-evaluation (`evaluated=false`).
- **Governance overlay:** Manual judgement and append-only logs that record operator actions without rewriting evidence.

---

# Appendix B — Claims Boundary Checklist

**We claim (contract/implementation facts):**
- Artifact sets, schemas, and validation exist per contract v1.
- Decisions use the canonical enum; `skip` is neutral and corresponds to non-evaluation.
- Canonical JSON + hashing rules define comparability; hashes must be recomputed for verification.
- Pilot scoring metrics (FAR, Delay_W, Overhead) are exactly those computed by the pilot scripts; pilot success criteria are stated in the pilot kit.

**We do not claim:**
- Model quality/correctness/safety guarantees.
- Any interpretation outside the declared window signature.
- Universal statistical guarantees across workloads absent a validated calibration protocol.

---

# Appendix C — Operational Handling Notes (Pipeline-Friendly)

- Treat `skip` as “no decision,” not “safe.”
- Validate capsules before ingestion or diffing.
- Treat partial capsules as non-comparable in `diff`/`report --compare`.
- Handle exit code `3` as an internal-error class distinct from invalid capsules (`2`).

---

# Appendix D — Traceability (Repository Pointers)

**Contract and invariants**
- `docs/contract_v1.md`
- `docs/migration_invariants.md`

**Operational semantics**
- `docs/productization.md`

**CLI**
- `veriscope/cli/main.py`

**Gate, discrepancy, calibration**
- `veriscope/core/gate.py`
- `veriscope/core/ipm.py`
- `veriscope/core/calibration.py`

**Redaction and governance**
- `veriscope/core/redaction_policy.py`
- `veriscope/core/governance.py`
- `veriscope/cli/governance.py`

**Pilot kit**
- `docs/pilot/README.md`
- `docs/pilot/one_pager.md`
- `docs/pilot/faq.md`
- `scripts/pilot/run.sh`
- `scripts/pilot/score.py`
- `scripts/pilot/negative_controls.sh`

**Licensing**
- `LICENSE`
- `COMMERCIAL_LICENSE.md`
