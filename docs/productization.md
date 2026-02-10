# Veriscope v0 Productization Guide (Pilot-Ready, Minimal Ambiguity)

This document defines Veriscope v0 as usable infrastructure: contracts, operational semantics, exit codes, and testable artifacts. It is intended to be committed as `docs/productization.md` and shared with design partners.

Companion docs for incubation and boundary clarity:
- `docs/incubation_readiness.md`
- `docs/calibration_protocol_v0.md`
- `docs/distributed_mode.md`

---

## 0) One-page overview (founder + design partner)

### What the product is
Veriscope is a CLI-first training reliability gate that flags representation drift/collapse during ML training using finite-window divergence checks (`D_W`) over declared metrics (e.g., `var_out_k`, `eff_dim`). It emits a machine-readable decision at gate-check cadence (every `gate_window` iterations) via `core/gate.py::GateEngine.check()` (optionally regime-anchored via `core/regime.py`).

**Canonical decision enum (do not infer from booleans):**
- `decision = "pass" | "warn" | "fail" | "skip"`
- `decision="skip"` corresponds to `audit.evaluated=false` and is neutral (must not be treated as pass/fail).

**Known limitation (v0):** single-node, single-GPU runs; distributed training support is v1+.

### Who the first buyer is
ML platform / research engineering teams running expensive fine-tuning or long training jobs (labs, startups, enterprise AI) where wasted GPU time and postmortems are costly, and where a CLI gate + reproducible artifacts can be adopted quickly.

### What the design-partner pilot will look like (v0)
Partner runs Veriscope on 1–2 internal fine-tunes (run gpt shown; run hf uses the same wrapper semantics with HF-specific runner args):

```bash
veriscope run gpt --gate-preset tuned_v0 --outdir OUTDIR -- <their training args>
veriscope validate OUTDIR
veriscope report OUTDIR > report.md

We provide: (1) stable artifact contracts, (2) validators + golden tests, (3) a Markdown report suitable for copy/paste into Slack/Notion/GitHub.

What success looks like (acceptance criteria, not “guarantees”)

Measured on control + injected-pathology runs under the canonical gate_preset=tuned_v0:
	•	FAR ≤ 5% post-warmup on control runs.
	•	Post-warmup definition: gate checks with audit.evaluated=true and iter >= gate_warmup_iters (and after the first evaluated gate).
	•	Detection delay ≤ 2W iterations after onset in injected-pathology runs (W = gate_window).
	•	Overhead ≤ 5% wall-clock (where timing fields exist).
	•	Time to first audit ≤ 2 minutes: from suspicion → shareable veriscope report output.

⸻

Operator FAQ (read this first)
	•	What does decision=skip mean? Gate was neutral due to insufficient evidence or missing metrics; check audit.reason.
	•	What do I do on WARN vs FAIL? WARN = investigate; FAIL = stop/rollback signal under the configured preset.
	•	What does window signature mismatch mean? Metrics/window/transport/code identity or gate controls differ; don’t compare runs without re-baselining.
	•	How do I share results? veriscope report OUTDIR > report.md and paste.
	•	What if a FAIL is a known false positive? Use veriscope override OUTDIR --status pass --reason "..." to create a human decision overlay without editing raw artifacts.

⸻

1) Current state of the codebase (grounded inventory)

Entry points
	•	CLI: veriscope/cli/main.py
	•	veriscope run gpt ... delegates to veriscope/runners/gpt/train_nanogpt.py
	•	veriscope run hf ... delegates to veriscope/runners/hf/train_hf.py
	•	veriscope run cifar ... delegates to legacy subprocess (out-of-scope v0)
	•	Smoke scripts:
	•	scripts/run_gpt_smoke.sh
	•	scripts/run_hf_smoke.sh
	•	scripts/run_cifar_smoke.sh (legacy; out-of-scope v0)

What the GPT runner emits today
	•	run_config_resolved.json (CLI provenance)
	•	a results JSON emitted by runners/gpt/train_nanogpt.py (metrics + gate/audit content; filename may vary today)
	•	optional calibration CSV via core/calibration_recorder.py (CalibrationRecorder.COLUMNS)

Note: v0 productization standardizes output filenames to the required artifact names below; older names are deprecated.

⸻

2) v0 commands (with operational semantics)

2.1 veriscope run gpt (hardened wrapper)

veriscope run gpt \
  --gate-preset tuned_v0 \
  --outdir OUTDIR \
  [--force] \
  -- <training args>

Note: `--strict-determinism` is roadmap and is not currently implemented in `veriscope run gpt`.

Run ID (join key)
	•	run_id is generated once by the CLI and propagated into all emitted artifacts.

Outdir path normalization
	•	In provenance, store absolute paths (e.g., run_config_resolved.json.run.outdir must be absolute).
	•	In human-facing output (report artifact table), use relative paths from OUTDIR.

Idempotency / overwrite policy (v0)
veriscope run treats OUTDIR as “containing Veriscope artifacts” if any of the following sentinel files exist:
	•	run_config_resolved.json
	•	results.json
	•	results_summary.json
	•	window_signature.json

Policy:
	•	If sentinel(s) exist: veriscope run fails unless --force is passed.
	•	--force preserves existing artifacts and starts the run in a fresh subdirectory under OUTDIR.

Process safety requirements (v0 must-have)
	•	Real-time stdout/stderr streaming: child logs must stream live to the console (no end-of-run buffering).
	•	Start a new process group/session for the runner and forward signals to the process group:
	•	On POSIX: spawn runner in a new session and forward SIGINT/SIGTERM to the group.
	•	Best-effort finalization on SIGINT/SIGTERM: wrapper attempts best-effort finalization (write/update summary/status); no guarantee on SIGKILL.
	•	Exit code propagation: see Exit Codes (§2.6). veriscope run must return non-zero on runner crash or veriscope internal failure.

Exit code decision rule (authoritative)
veriscope run determines its own exit code from run_status in results_summary.json when available.

Fallback (consistent, non-brittle):
If results_summary.json is missing/unreadable:
	1.	If the runner exited non-zero → exit code 2 (user_code_failure)
	2.	Else → exit code 3 (veriscope_failure)

⸻

2.2 veriscope run hf (hardened wrapper)

veriscope run hf \
  --gate-preset tuned_v0 \
  --outdir OUTDIR \
  [--force] \
  -- <training args>

Notes (HF runner)
	•	Wrapper inserts --outdir and --run_id if missing.
	•	HF dependency preflight runs before launch and fails fast if datasets/transformers are missing.
	•	Override runner command via VERISCOPE_HF_RUNNER_CMD for CI-safe integration testing.
	•	HF runner inherits the same wrapper process-safety invariants as run gpt (streaming logs, process group, signal forwarding, best-effort finalization).
	•	--force has the same meaning as run gpt (bypass GPU lock; fresh subdirectory when OUTDIR already contains artifacts).

⸻

2.3 veriscope validate OUTDIR (new)

veriscope validate OUTDIR \
  [--strict-governance] \
  [--require-governance] \
  [--strict-identity] \
  [--allow-legacy-governance]

Validates:
	•	schema correctness (required keys + types; extra keys allowed)
	•	cross-artifact compatibility, at minimum:
	•	run_id matches across artifacts
	•	results.json.window_signature_ref.hash and results_summary.json.window_signature_ref.hash must match the recomputed canonical hash of `window_signature.json`
	•	`window_signature.created_ts_utc` is excluded from hashing only when it is a valid ISO8601 UTC timestamp string (trailing `Z`)
	•	results_summary counts must match gate decisions when results.json is present
	•	`first_fail_iter` is required when `counts.fail > 0`, and must be null when `counts.fail == 0`
	•	`first_fail_iter.txt` must exist iff `first_fail_iter` is non-null
	•	if run-governance events exist, governance payload `run_id` and `outdir` must match artifacts
	•	schema_version=1 (v0/v1 contract)

Validation is read-only and does not mutate artifacts.

Schema compatibility rule:
	•	For v0, only `schema_version=1` is accepted.

⸻

2.4 veriscope report OUTDIR (single-capsule + compare modes)

veriscope report OUTDIR \
  [--format md|text]

veriscope report --compare OUTDIR_A OUTDIR_B [OUTDIR_N ...] \
  [--format text|md|json] \
  [--allow-incompatible] \
  [--allow-gate-preset-mismatch]

Compare-mode constraints (enforced):
	•	each OUTDIR must contain a valid `governance_log.jsonl` (missing/invalid governance fails compare mode)
	•	partial capsules (`partial=true`) are non-comparable and are rejected by `diff` and `report --compare`

Default output is text; Markdown is available with `--format md`.

Minimum report skeleton (stable for goldens):
	1.	# Veriscope Report: <OUTDIR>
	2.	Run metadata table:
	•	Run ID
	•	Status
	•	Wrapper Exit (Veriscope wrapper exit code)
	•	Runner Exit (child process exit code)
	•	gate_preset
	•	Started / Ended
	•	Window Signature hash
	3.	## Gate Summary table (Evaluated / SKIP / WARN / FAIL counts)
	4.	Conditional section:
	•	## Manual Judgement (only when manual overlays exist)
	5.	## Artifacts table (relative paths from OUTDIR)
	6.	## Governance summary (log present/validity/rev)

Appendix A provides a golden reference.

⸻

2.5 veriscope override OUTDIR (optional v0+, recommended)

veriscope override OUTDIR --status pass|fail --reason "..." [--reviewer "Name"] [--ts-utc "..."]

Creates manual_judgement.json.

Clarification: manual judgement does not change raw gate records; it is an operator-level overlay that report displays prominently.

⸻

2.6 veriscope calibrate (pilot control/injected, read-only)

veriscope calibrate --control-dir OUTDIR --injected-dir OUTDIR \
  [--out calibration.json] [--out-md calibration.md]

Reads two capsule directories and emits a calibration summary (JSON + Markdown) without mutating artifacts. Output schema matches `scripts/pilot/score.py`:
	•	policy_rev
	•	W, W_source
	•	K, warmup_source
	•	FAR, FAR_source
	•	Delay_W
	•	overhead
	•	missing_fields
	•	redactions_applied
	•	calibration_status

Defaults write to the current working directory (not inside OUTDIRs). The command exits with code 2 on missing/invalid required inputs and produces deterministic, stable JSON (sorted keys, no timestamps).
Exit codes: 0 success, 2 invalid/missing input, 3 internal error.

⸻

2.7 Exit codes (CI-friendly and unambiguous)

veriscope run

Code	Meaning
0	runner exited 0; run_status="success"
2	runner crash (run_status="user_code_failure")
3	veriscope internal error (run_status="veriscope_failure")

Note: WARN does not produce non-zero exit.
Decision outcomes (pass/warn/fail/skip) are recorded in artifacts rather than as exit codes.

veriscope validate

Code	Meaning
0	valid
2	validation failed (schema or cross-artifact mismatch)
3	veriscope internal error

veriscope report

Code	Meaning
0	report generated
2	cannot report due to invalid/missing artifacts
3	veriscope internal error

veriscope diff / veriscope report --compare

Code	Meaning
0	comparison rendered (or incompatible runs included when `--allow-incompatible` is set for report compare)
2	incompatible/invalid inputs, including missing or invalid governance logs, or partial capsules


⸻

3) Artifacts emitted (v0, regular JSON)

Required artifacts (non-partial capsules)
	•	OUTDIR/window_signature.json (v1)
	•	OUTDIR/results.json (v1; detailed; includes gates[]; written atomically)
	•	OUTDIR/results_summary.json (v1; compiled summary; written atomically)

Required artifacts (partial capsules)
	•	OUTDIR/window_signature.json (v1)
	•	OUTDIR/results_summary.json (v1) with `partial=true`

Conditionally required artifact
	•	OUTDIR/first_fail_iter.txt (must exist iff `results_summary.first_fail_iter` is non-null)

Recommended artifacts (best-effort; must not block runs)
	•	OUTDIR/run_config_resolved.json (v1)
	•	Roadmap (not yet implemented): OUTDIR/runner_output.log (captured stdout/stderr)
	•	Roadmap (not yet implemented): OUTDIR/environment.txt (dependency + platform capture; best effort)

Optional artifacts
	•	OUTDIR/calibration.csv (existing contract)
	•	OUTDIR/calibration.meta.json (optional sidecar)
	•	OUTDIR/manual_judgement.json (if override is used)

Atomicity:
	•	results.json and results_summary.json must be written to *.tmp then atomically renamed.
	•	If preempted before finalization, files may be missing; validate/report must handle that case.

⸻

4) Contracts and schemas (v1, Pydantic-first)

Implementation requirement (explicit)

Use Pydantic models:
	•	emit JSON via model_dump_json()
	•	validate via Pydantic parsing
	•	generate JSON Schema via model_json_schema() for docs + tests

Extra keys are allowed unless explicitly forbidden.

Versioning note: v1 schemas are stable within v0; new optional fields may be added without bumping schema_version.

⸻

4A) run_config_resolved.json (v1)

Purpose: provenance, join key anchor, safe environment capture.

Required keys:
	•	schema_version: 1
	•	ts_utc: str (ISO-8601)
	•	package: { name: str, version: str }
	•	git_sha: str | null
	•	null if not a git repo or SHA unavailable.
	•	If dirty is allowed, include notes: ["dirty"].
	•	run: { kind: "gpt", run_id: str, outdir: str }
	•	argv: { veriscope_argv: list[str], runner_cmd?: list[str] }
	•	env: { [key: str]: str } allowlisted + redacted
	•	env_capture: { allowlist: list[str], deny_exact: list[str], deny_regex: list[str], redactions_applied: bool }
	•	determinism_status: { pyhashseed_ok?: bool, cublas_config_ok?: bool, cudnn_deterministic?: bool, notes?: list[str] }
	•	dependencies: { method: "pip_freeze" | "unavailable", path?: str, notes?: list[str] }

Environment capture safety (blocker)
	•	Capture only allowlisted prefixes/keys (e.g. VERISCOPE_*, CUDA_*, SLURM_*, WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT, PYTHONHASHSEED).
	•	Apply redaction using:
	•	exact-key denylist (deny_exact)
	•	regex denylist (deny_regex)
	•	Redacted keys remain present with value "***REDACTED***".
	•	redactions_applied=false only if no deny rules matched any captured key.

Dependency + platform capture (best effort; non-blocking)
	•	Roadmap (not yet implemented): attempt to write OUTDIR/environment.txt. Until implemented, wrappers continue without this file.

Minimum content of environment.txt (best effort):
	•	Roadmap (not yet implemented; target content below when shipped)
	•	Python version (sys.version)
	•	OS/arch (platform.platform(), platform.machine())
	•	pip freeze output (if available)
	•	If torch is installed (best effort):
	•	torch.__version__
	•	torch.cuda.is_available()
	•	torch.version.cuda (if present)

⸻

4B) window_signature.json (v1)

Purpose: compatibility key.

Contract posture:
	•	`window_signature.json` is hashed as an opaque JSON object (after volatile-key handling); comparisons use `results*.window_signature_ref.hash`.
	•	Extra keys are allowed.
	•	Current runners commonly emit fields like `schema_version`, `created_ts_utc`, `transport`, `evidence`, `gates`, and runner metadata.

Canonical JSON hashing:
	•	Canonical JSON means: json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) encoded as UTF-8.
	•	SHA256 is computed over that UTF-8 byte sequence.
	•	When recomputing window-signature hash, `created_ts_utc` is excluded only if it is a valid ISO8601 UTC string with trailing `Z`; invalid values fail validation.
	•	Sub-hashes use the same method on their respective sub-payloads.

Note: v0 assumes hash production and verification are performed by the same implementation; float rendering is therefore implementation-defined within that constraint.

If a runner emits `gate_controls_hash`, canonicalization is:
	•	Hash payload is canonical JSON (sorted keys, UTF-8) of a fixed object including at least:
	•	gate_window: int
	•	gate_warmup_iters: int
	•	min_evidence: int
	•	policy: str
	•	persistence_k: int | null
	•	min_metrics_exceeding: int | null
	•	eps_sens: float | null
	•	cap_frac: float | null
	•	If a control is absent, it must be represented as null (not omitted).

⸻

4C) results.json (v1; detailed)

Purpose: detailed audit record.

Required top-level keys:
	•	schema_version: 1
	•	run_id: str
	•	window_signature_ref: { hash: str, path: str }
	•	profile: { gate_preset: str, overrides?: object }
	•	Standardization: use gate_preset everywhere in artifacts.
	•	run_status: "success" | "user_code_failure" | "veriscope_failure"
	•	runner_exit_code: int | null
	•	runner_signal: str | null
	•	started_ts_utc: str
	•	ended_ts_utc: str | null
	•	gates: list[GateRecordV1]
	•	metrics: list[MetricRecordV1] (may be empty)

Cadence clarification:
	•	gates[] contains one record per gate-check (every gate_window iterations), not every iteration.

GateRecordV1 required fields
	•	iter: int
	•	decision: "pass" | "warn" | "fail" | "skip"
	•	audit: AuditV1

Optional/back-compat fields:
	•	ok: bool and/or warn: bool may exist.
	•	Consumers and validators must treat decision as canonical; if ok/warn are present they must be consistent with decision.

Decision derivation rules (minimal ambiguity):
	•	If audit.evaluated == false → decision = "skip"
	•	Else decision ∈ {"pass","warn","fail"} must be set from the policy outcome (canonical decision).
	•	If optional booleans exist:
	•	decision="fail" ⇒ ok=false
	•	decision="warn" ⇒ ok=true and warn=true
	•	decision="pass" ⇒ ok=true and warn=false (if warn is present)

This ensures implementers do not accidentally depend on ok/warn being present.

AuditV1 required fields
	•	evaluated: bool
	•	reason: str (e.g. insufficient_evidence, metric_unavailable, pass, warn, fail)
	•	policy: str
	•	worst_DW: float | null
	•	eps_eff: float | null
	•	per_metric_tv: { [name: str]: float } (always present; may be {})
	•	evidence_total: int
	•	min_evidence: int

⸻

4D) results_summary.json (v1; compiled summary)

Purpose: stable summary for pipelines; run exit code is derived from `run_status`.

Required keys:
	•	schema_version: 1
	•	run_id: str
	•	window_signature_ref: { hash: str, path: str }
	•	profile: { gate_preset: str, overrides?: object }
	•	run_status: "success" | "user_code_failure" | "veriscope_failure"
	•	runner_exit_code: int | null
	•	runner_signal: str | null
	•	started_ts_utc: str
	•	ended_ts_utc: str | null
	•	counts: { evaluated: int, skip: int, pass: int, warn: int, fail: int }
	•	final_decision: "pass" | "warn" | "fail" | "skip"
	•	first_fail_iter: int | null

final_decision policy:
	•	any fail → fail
	•	else any warn → warn
	•	else any pass → pass
	•	else → skip

first_fail_iter / marker invariants:
	•	if `counts.fail > 0`, `first_fail_iter` must be an integer
	•	if `counts.fail == 0`, `first_fail_iter` must be null
	•	`first_fail_iter.txt` must exist iff `first_fail_iter` is non-null

Run status mapping:
	•	veriscope_failure means the wrapper failed (validation error, schema emission failure, internal exception), not the training.
	•	If runner_exit_code != 0 and wrapper did not fail: run_status="user_code_failure".
	•	If runner exits 0 and wrapper succeeds: run_status="success" (gate FAIL is represented via final_decision; wrapper exit remains 0).

⸻

4E) Manual adjudication overlay (manual_judgement.json, optional)

Required keys:
	•	schema_version: 1
	•	run_id: str
	•	status: "pass" | "fail"
	•	reason: str
	•	reviewer: str | null
	•	ts_utc: str

Manual judgement is an overlay only and does not mutate raw gate records.

⸻

5) Safety rails and configuration (v0)

Presets (required) and discoverability
	•	--gate-preset <name> is required (default allowed but must be recorded as gate_preset).
	•	Canonical preset for pilots: tuned_v0.
	•	Presets documented in docs/presets.md (v0).

Determinism policy
	•	default: warn + record in determinism_status
	•	`--strict-determinism` hard-fail mode is roadmap (not currently implemented in the run wrapper CLI).

Secrets hygiene (explicit blocker)
	•	env capture must be allowlisted + redacted; never dump full env; redacted keys remain present with "***REDACTED***".

⸻

6) Observability

Captured logs (recommended)
	•	Roadmap (not yet implemented): OUTDIR/runner_output.log capture for runner stdout/stderr.
	•	Roadmap (not yet implemented): report linkage to runner_output.log for first-line debugging.

⸻

7) Testing and CI plan (invariants as tests)
	•	Pydantic validation tests for:
	•	run_config_resolved.json (allowlist/denylist; redactions_applied semantics)
	•	window_signature.json (gate_controls_hash canonicalization and null-for-absent)
	•	results.json (decision enum presence; skip semantics; ok/warn consistency if present)
	•	results_summary.json (final_decision policy; run_status mapping)
	•	Golden smoke test (GPT):
	•	run scripts/run_gpt_smoke.sh on GPU CI
	•	assert artifacts exist
	•	veriscope validate OUTDIR exits 0
	•	veriscope report OUTDIR Markdown matches golden skeleton

⸻

8) Roadmap (kept minimal)

v0
	•	process-safe wrapper (streaming, process group, signal forwarding)
	•	regular JSON artifacts + atomic write
	•	window signature + gate controls hash
	•	allowlisted env capture + best-effort dependency/platform capture
	•	validate, Markdown report, optional override
	•	goldens + schema tests

v1+
	•	distributed training support
	•	additional workflows (calibrate/build-reference), streaming artifacts, multi-run comparisons

⸻

9) Implementation checklist (v0)
	1.	Process wrapper: streaming, new session/process group, signal forwarding, best-effort finalization
	2.	CLI-generated run_id propagation to all artifacts
	3.	Emit window_signature.json (+ gate_controls_hash canonicalization)
	4.	Emit results.json + results_summary.json (atomic rename)
	5.	Env allowlist + deny exact/regex redaction (environment.txt remains roadmap)
	6.	Pydantic schemas + veriscope validate (including governance/identity switches)
	7.	Markdown veriscope report with fixed skeleton + goldens
	8.	Optional veriscope override + report overlay

⸻

10) Top v0 pilot regression risks (ranked)
	1.	Secrets leakage risk (env capture not allowlisted/redacted)
	2.	Unsafe wrapper behavior (no streaming / no process-group signal forwarding / wrong exit codes)
	3.	Summary/run-status linkage regressions between emitted artifacts and wrapper exit behavior
	4.	Missing decision derivation alignment (emitter/validator disagree)
	5.	Missing schema validation + goldens (contract drift)

⸻

Appendix A: Sample Markdown report output (golden reference)

# Veriscope Report: `OUTDIR`

| Field            | Value |
|------------------|-------|
| Run ID           | `abc123` |
| Status           | success |
| Wrapper Exit     | 0 |
| Runner Exit      | 0 |
| gate_preset      | `tuned_v0` |
| Started          | 2026-01-15T10:00:00Z |
| Ended            | 2026-01-15T10:45:00Z |
| Window Signature | `sha256:...` |

## Gate Summary

| Total checks | Evaluated | SKIP | WARN | FAIL |
|-------------:|----------:|-----:|-----:|-----:|
| 47           | 44        | 3    | 2    | 0    |

## False Alarm Rate (control)
- FAR (post-warmup): 4.5% (2/44)

## Overhead
- Overhead (wall-clock): 3.1% (from timing fields)

## Artifacts

| Artifact | Path |
|---------|------|
| Provenance | `run_config_resolved.json` |
| Window signature | `window_signature.json` |
| Detailed results | `results.json` |
| Summary | `results_summary.json` |
| Dependencies | `environment.txt` (roadmap; not yet emitted) |
| Runner logs | `runner_output.log` (roadmap; not yet emitted) |

## Manual Judgement
- None
