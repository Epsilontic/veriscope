# Agents.md — veriscope (agent-facing contract)  
  
This file is the **agent-facing companion** for `veriscope`: invariants, boundary rules, canonical schemas, and golden-path commands. It is optimized for automated agents (Codex-style) so changes can be made safely without repo archaeology.  

**Normative precedence:** `docs/contract_v1.md` is the single normative contract for artifacts, comparability, governance, and exit-code semantics. This file is derived guidance; if there is any conflict, `docs/contract_v1.md` wins.  
  
If this document disagrees with repository behavior/tests, update this file and tests as needed, but keep `docs/contract_v1.md` authoritative. Do not let “tribal knowledge” live only in code.  
  
---  
  
## 0) PR safety checklist (run before/after changes)  
  
1) **CLI import boundary check (must stay thin)**  
```bash  
python -c "import veriscope.cli.main; print('cli import ok')"  
```  
If that becomes slow or imports heavyweight ML deps, you violated the boundary (see §4).  
  
2) **Golden-path smoke**  
```bash  
bash scripts/run_gpt_smoke.sh  
veriscope validate ./out/gpt_smoke_YYYYMMDD_HHMMSS  
veriscope report   ./out/gpt_smoke_YYYYMMDD_HHMMSS > report.md  
```  
  
3) **Schema + cross-artifact consistency**  
- `veriscope validate OUTDIR` must be read-only and deterministic.  
- `results_summary.json` must remain stable and pipeline-safe.  
  
---  
  
## 1) What veriscope is (operationally)  
  
`veriscope` is a **CLI-first training reliability gate**. It observes training-derived evidence (metrics → distributions → divergence checks) and emits **decisions** at a fixed cadence.  
  
**Inputs**  
- Training run telemetry (runner-specific; e.g. nanoGPT metrics/logs)  
- Gate preset + window identity (defines the admissible evidence space)  
  
**Outputs**  
- An **artifact capsule**: a directory containing machine-readable JSON artifacts  
- A **deterministic decision surface**: `pass | warn | fail | skip`  
  
**Posture**  
- This is a *decision system*, not a dashboard.  
- Every decision must be explainable via emitted audit fields and schemas below.  
  
---  
  
## 2) Capsule directory contract (what an OUTDIR is)  
  
A “capsule” is a directory containing artifacts for a single run.  
  
### 2.1 Required files (MUST)  
- Non-partial capsules:
  - `window_signature.json`
  - `results.json`
  - `results_summary.json`
- Partial capsules:
  - `window_signature.json`
  - `results_summary.json` with `partial=true`
  
### 2.2 Recommended / Optional files  
- Recommended:
  - `run_config_resolved.json`
- Optional:
  - `manual_judgement.json` (overlay; does not mutate raw artifacts)
  - `manual_judgement.jsonl` (append-only overlay log)
  - `governance_log.jsonl` (append-only journal)
  - provenance files (runner logs, configs, stdout/stderr captures, etc.)
  
### 2.3 Example layout  
```text  
OUTDIR/  
  window_signature.json  
  results_summary.json  
  results.json                      (required for non-partial capsules)  
  manual_judgement.json             (optional)  
  governance_log.jsonl              (optional)  
  provenance/                       (optional)  
    runner_stdout.txt  
    runner_stderr.txt  
    runner_config.json  
```  
  
### 2.4 Cross-artifact invariants (validator MUST enforce)  
If a referenced file exists, it MUST be consistent:  
  
- `results_summary.run_id` MUST match `results.run_id` (if `results.json` exists)  
- `results_summary.window_signature_ref.hash` MUST equal the recomputed hash of `window_signature.json` (i.e. `sha256(canonical_json(window_signature.json))` after dropping volatile keys like `created_ts_utc`, per §7)  
- `results.window_signature_ref.hash` MUST equal the recomputed hash of `window_signature.json` (i.e. `sha256(canonical_json(window_signature.json))` after dropping volatile keys like `created_ts_utc`, per §7) (if `results.json` exists)  
- `counts.evaluated == counts.pass + counts.warn + counts.fail`  
- Gate emitters MUST emit one `GateRecord` per cadence; non-evaluated cadences MUST be emitted with `decision="skip"` so summary counts match `len(results.gates)`.  
- If `results.json` exists:    
  `counts.evaluated + counts.skip == len(results.gates)`  
- `final_decision` MUST match the derivation rule in §6.5  
- Timestamp compatibility: writers SHOULD emit ISO8601 UTC `...Z` (seconds precision recommended); readers MAY accept `+00:00` or naive ISO8601 and treat them as UTC for backward compatibility.  
  
---  
  
## 3) Golden path (copy/paste)  
  
### 3.1 GPT smoke (recommended)  
```bash  
bash scripts/run_gpt_smoke.sh  
# or choose an explicit outdir:  
bash scripts/run_gpt_smoke.sh ./out/gpt_smoke_manual  
```  
  
The outdir SHOULD contain at least:  
- `window_signature.json`  
- `results_summary.json`  
(and typically also `results.json`, plus provenance artifacts when available)  
  
Validate + report:  
```bash  
veriscope validate ./out/gpt_smoke_YYYYMMDD_HHMMSS  
veriscope report   ./out/gpt_smoke_YYYYMMDD_HHMMSS > report.md  
```  
  
### 3.2 CIFAR smoke (legacy / out-of-scope for v0 semantics)  
```bash  
bash scripts/run_cifar_smoke.sh  
# or choose an explicit outdir:  
bash scripts/run_cifar_smoke.sh ./out/cifar_smoke_manual  
```  
  
---  
  
## 4) Repository layers and hard boundaries  
  
### 4.1 Layers  
- `veriscope/cli/*`    
  User-facing commands, I/O, formatting, subprocess orchestration. **Must remain fast to import.**  
- `veriscope/core/*`    
  Pure logic and contracts: Pydantic models, hashing, decision derivation, gate math. **Avoid side effects.**  
- `veriscope/runners/*`    
  Side-effectful execution (training, log parsing, heavy deps). **May import GPU frameworks.**  
  
### 4.2 Lazy-import and dependency boundary (explicit)  
At import time, `veriscope.cli.main` MUST NOT drag heavyweight ML deps.  
  
**Forbidden at top-level in `veriscope/cli/*.py` (must be lazy or subprocess-only):**  
- `import torch`, `import torchvision`  
- `from veriscope.runners... import ...`  
- legacy CIFAR modules  
  
**Allowed patterns**  
- Import runners inside command handlers (function scope)  
- Spawn subprocesses for heavy work  
- Keep CLI import graph limited to `veriscope.core.*` + standard library  
  
**Why this matters**  
- Keeps `veriscope --help` fast/reliable  
- Prevents CI and packaging regressions  
- Preserves “thin wrapper” product invariant  
  
---  
  
## 5) Non-negotiables (agent constraints)  
  
1) **Decisions are enums (canonical contract)**  
- Always use `decision` (per gate) or `final_decision` (summary).  
- Do not infer from legacy `ok` / `warn` booleans.  
  
2) **`skip` is real and neutral**  
- `decision="skip"` means “not evaluated” (insufficient evidence / missing metrics / policy prevented evaluation).  
- Treating `skip` as pass corrupts accounting and masks broken emitters.  
  
3) **Hashes define comparability**  
- Comparisons/diffs are meaningful only when window identity matches.  
- If `window_signature_ref.hash` differs, runs are incomparable unless an explicit override policy is used.  
  
4) **Governance is append-only**  
- Overlays may affect displayed outcome but MUST NOT mutate raw artifacts.  
- Journals/logs are append-only; never rewrite history.  
  
5) **Validation is read-only**  
- `veriscope validate` MUST NOT “fix” artifacts (no rewrites, no normalization writes).  
  
---  
  
## 6) Canonical schema reference (schema_version = 1)  
  
### 6.1 Conventions  
- Schemas below are **wire-accurate spines**: extra keys are allowed (forward-compat).  
- Types are TypeScript-like for readability.  
- JSON examples are plain JSON (no comments) to enable copy/paste.  
  
### 6.2 Shared enums  
```ts  
type Decision = "pass" | "warn" | "fail" | "skip";  
type RunStatus = "success" | "user_code_failure" | "veriscope_failure";  
```  
  
### 6.3 `window_signature.json` (comparability root)  
This file defines the evidence space. **Any change to its identity-defining contents changes the hash** (excluding volatile metadata like `created_ts_utc`), and therefore comparability.  
  
```ts  
interface WindowSignatureV1 {  
  schema_version: 1;  
  
  // Informational fields (recommended)  
  created_ts_utc?: string;      // ISO8601 UTC (writers emit Z; readers may normalize +00:00/naive to UTC)
  description?: string;  
  
  // Identity-defining content (examples; repo may add more keys)  
  transport?: {  
    name: string;               // e.g. "nanogpt_metrics_v1"  
    cadence?: string;           // e.g. "every_50_iters"  
  };  
  
  evidence?: {  
    metrics: string[];          // metric names expected/used  
    window?: {                  // optional: windowing parameters  
      kind: string;             // e.g. "rolling", "fixed"  
      size?: number;  
      stride?: number;  
    };  
  };  
  
  gates?: {  
    preset: string;             // gate preset name, e.g. "tuned_v0"  
    params?: Record<string, any>;  
  };  
  
  // Anything else allowed; note it will affect hashing/comparability  
  [k: string]: any;  
}  
```  
  
### 6.4 `window_signature_ref` (embedded)  
```ts  
interface WindowSignatureRefV1 {  
  hash: string; // 64 lowercase hex (sha256)  
  path: string; // typically "window_signature.json" (relative path)  
}  
```  
  
### 6.5 `results_summary.json` (authoritative for pipelines)  
Consumers SHOULD rely on this file as the stable interface.  
  
```ts  
interface CountsV1 {  
  evaluated: number; // MUST == pass + warn + fail  
  skip: number;  
  pass: number;  
  warn: number;  
  fail: number;  
}  
  
interface ResultsSummaryV1 {  
  schema_version: 1;  
  
  run_id: string;  
  window_signature_ref: WindowSignatureRefV1;  
  
  profile: {  
    gate_preset: string;  
    overrides?: Record<string, any>;  
  };  
  
  run_status: RunStatus;  
  runner_exit_code?: number | null;  
  runner_signal?: string | null;  
  
  started_ts_utc: string; // ISO8601 UTC (writers emit Z; readers may normalize +00:00/naive to UTC)
  ended_ts_utc?: string | null;  
  
  counts: CountsV1;  
  final_decision: Decision;  
  
  // optional  
  partial?: boolean; // true if run did not complete (may omit results.json)  
}  
```  
  
**Final decision derivation (MUST match validator logic)**  
- If `fail > 0` → `final_decision="fail"`  
- Else if `warn > 0` → `final_decision="warn"`  
- Else if `pass > 0` → `final_decision="pass"`  
- Else (i.e. `evaluated == 0`) → `final_decision="skip"`  
  
### 6.6 `results.json` (detailed audit; one record per cadence)  
```ts  
interface AuditV1 {  
  evaluated: boolean;  
  
  // When evaluated=true, these MUST be present (validator rule):  
  reason?: string;  
  policy?: string;  
  evidence_total?: number;  
  min_evidence?: number;  
  
  // Gate-specific numeric payloads (may be null when not applicable)  
  worst_DW?: number | null;  
  eps_eff?: number | null;  
  
  // Always present (may be empty)  
  per_metric_tv: Record<string, number>;  
}  
  
interface GateRecordV1 {  
  iter: number;  
  decision: Decision;  
  audit: AuditV1;  
  
  // transitional / non-canonical:  
  ok?: boolean | null;  
  warn?: boolean | null;  
}  
  
interface MetricRecordV1 {  
  name: string;  
  value: any; // JSON-compatible only  
}  
  
interface ResultsV1 {  
  schema_version: 1;  
  
  run_id: string;  
  window_signature_ref: WindowSignatureRefV1;  
  
  profile: {  
    gate_preset: string;  
    overrides?: Record<string, any>;  
  };  
  
  run_status: RunStatus;  
  runner_exit_code?: number | null;  
  runner_signal?: string | null;  
  
  started_ts_utc: string;  
  ended_ts_utc?: string | null;  
  
  gates: GateRecordV1[];  
  metrics: MetricRecordV1[];  
}  
```  
  
**Gate-record decision rules (MUST)**  
- If `audit.evaluated == false` → `decision = "skip"`  
- If `audit.evaluated == true` → `decision ∈ {"pass","warn","fail"}` and MUST be explicit  
- If legacy booleans exist, they MUST be consistent with `decision` (but never treated as canonical)  
  
### 6.7 Manual overlay: `manual_judgement.json` (optional)  
Overlay does not mutate raw records.  
  
```ts  
type JudgementStatus = "pass" | "fail";  
  
interface ManualJudgementV1 {  
  schema_version: 1;  
  
  run_id: string;  
  status: JudgementStatus; // intentionally narrower than Decision  
  reason: string;  
  
  reviewer?: string | null;  
  ts_utc: string; // ISO8601 UTC (writers emit Z; readers may normalize +00:00/naive to UTC)
}  
```  
  
### 6.8 Governance journal: `governance_log.jsonl` (optional, append-only)  
Each line is a JSON object. Writers MUST include `entry_hash`; readers MAY warn on legacy lines missing it.  
  
```ts  
type GovernanceEventType =  
  | "manual_judgement_set"  
  | "manual_judgement_cleared"  
  | "artifact_note"  
  | "recompute_summary";  
  
interface GovernanceEntryV1 {  
  schema_version: 1;  
  
  rev: number;        // MUST be strictly consecutive starting at 1  
  ts_utc: string;     // informational ordering (not authoritative)  
  actor?: string;  
  
  event_type: GovernanceEventType;  
  payload: Record<string, any>;  
  
  prev_hash?: string | null;  
  entry_hash: string; // sha256(canonical_json(entry_without_entry_hash))  
}  
```  
  
---  
  
## 7) Hashing and canonicalization rules (agent-ready)  
  
### 7.1 Canonical JSON serialization (MUST)  
Python reference:  
```py  
json.dumps(  
  obj,  
  sort_keys=True,  
  separators=(",", ":"),  
  ensure_ascii=False,  
  allow_nan=False,  
)  
```  
  
### 7.2 Hash primitive (MUST)  
- `sha256(utf8_bytes(canonical_json_string)) -> lowercase hex digest`  
  
### 7.3 Window signature hash (comparability key)  
- `window_signature_ref.hash == sha256(canonical_json(window_signature.json))` with volatile keys (e.g., `created_ts_utc`) removed first.  
  
**Important: avoid self-reference traps**  
- Treat the file hash as authoritative.  
- If `window_signature.json` contains an internal `hash` field, do not trust it; recompute the canonical hash of the file object for verification/comparison.  
  
### 7.4 Governance chaining  
To compute `entry_hash`, serialize the entry with the `entry_hash` key removed entirely (not present and not set to `null`).  
- `entry_hash = sha256(canonical_json(entry_without_entry_hash))`  
- `prev_hash = previous entry_hash` (or `null` for the first entry)  
  
---  
  
## 8) Comparability predicate (Comparable(A,B))  
  
Two runs A and B are comparable iff:  
- both have `schema_version == 1`  
- `A.window_signature_ref.hash == B.window_signature_ref.hash`  
- `A.profile.gate_preset == B.profile.gate_preset`  
- unless an explicit CLI override is used (e.g. `--allow-gate-preset-mismatch`)  
  
If not comparable:  
- `veriscope diff` MUST return non-zero and explain why (hash mismatch, preset mismatch, invalid capsule, etc.)  
  
---  
  
## 9) Outcome resolution (raw vs displayed)  
  
- **Raw outcome**: `results_summary.final_decision` derived from emitted gate results  
- **Displayed outcome** (reporting/UI): raw outcome + overlays (e.g. `manual_judgement.json`)    
  Overlays MUST NOT change raw artifacts.  
  
Agents changing report/diff behavior MUST state clearly whether they operate on raw or displayed outcomes, and MUST NOT silently substitute one for the other.  
  
---  
  
## 10) CLI surface (copy/paste)  
  
### 10.1 Validate a capsule  
```bash  
veriscope validate OUTDIR  
```  
  
### 10.2 Generate a report  
```bash  
veriscope report OUTDIR > report.md  
veriscope report OUTDIR --format text  
```  
  
### 10.3 Compare two runs  
```bash  
veriscope diff OUTDIR_A OUTDIR_B  
```  
  
### 10.4 Override displayed outcome (overlay, not mutation)  
```bash  
veriscope override OUTDIR --status pass --reason "Known infrastructure noise" --reviewer "Windowwright"  
```  
  
### 10.5 Run GPT (arguments after `--` are forwarded to runner)  
```bash  
veriscope run gpt --outdir OUTDIR -- \  
  --dataset shakespeare_char \  
  --nanogpt_dir ./nanoGPT \  
  --device cuda \  
  --max_iters 200 \  
  --no_regime  
```  
  
### 10.6 Run CIFAR legacy smoke  
```bash  
veriscope run cifar --smoke --outdir OUTDIR  
```  
  
### 10.7 Exit code contract (MUST be stable for CI)  
These are the **intended** semantics. If implementation differs, align implementation + tests + this doc.  
  
- `veriscope validate OUTDIR`  
  - `0`: valid  
  - `2`: invalid capsule (schema/cross-artifact errors)  
  
- `veriscope diff A B`  
  - `0`: comparable and diff completed successfully (no structural errors)  
  - `2`: not comparable / invalid input capsules  
  - (If you want “differences found” to be non-zero, document that explicitly and keep it stable.)  
  
- `veriscope run ...`  
  - `0`: run completed and artifacts emitted (even if decision is warn/fail; those are in artifacts)  
  - non-zero: runner execution failure or veriscope internal failure (`run_status != success`)  
  
---  
  
## 11) Common anti-patterns (and consequences)  
  
1) Reading `ok` / `warn` booleans as the contract  
Consequence: you miss `skip` states and treat “not evaluated” as “pass”.  
  
2) Treating `skip` as “pass”  
Consequence: FAR and evaluation metrics become meaningless; broken emitters look healthy.  
  
3) Comparing runs without a window signature match  
Consequence: you mix incomparable evidence spaces (different metrics/bins/transport/gate controls).  
  
4) Letting CLI import runners or ML deps at top-level  
Consequence: `veriscope --help` becomes slow/fragile; breaks thin-CLI invariant.  
  
5) “Fixing” artifacts during validate/report  
Consequence: audits become non-reproducible; governance loses meaning; append-only tests become invalid.  
  
---  
  
## 12) Change-impact matrix (when you change X, update Y)  
  
- If you change any artifact schema:  
  - update Pydantic models (core)  
  - update validators + cross-artifact checks  
  - update smoke scripts/expected outputs (if any)  
  - update `docs/contract_v1.md` (if it mirrors this)  
  
- If you change hashing/canonicalization:  
  - update `veriscope/core/jsonutil.py`  
  - update validators and any comparisons/diffs  
  - update this document (§7) and any migration docs  
  
- If you add/modify a runner:  
  - keep it in `veriscope/runners/*`  
  - ensure CLI imports it lazily (function scope) or via subprocess  
  - update smoke scripts if it’s part of the golden path  
  
- If you change decision semantics:  
  - update derivation rules in one place (core)  
  - update report rendering and diff logic  
  - ensure `skip` semantics remain neutral and explicit  
  
---  
  
## 13) Implementation cues (where to look)  
  
- Schemas and decision derivation:  
  - `veriscope/core/artifacts.py` (Pydantic models + `derive_final_decision`)  
- Canonical JSON + SHA256:  
  - `veriscope/core/jsonutil.py`  
- CLI entrypoint and command routing:  
  - `veriscope/cli/main.py`  
- GPT artifact emission:  
  - `veriscope/runners/gpt/emit_artifacts.py`  
- Frozen contract text (if applicable):  
  - `docs/contract_v1.md`  
- Migration invariants:  
  - `docs/migration_invariants.md`
