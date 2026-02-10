# Veriscope Contract v1 (Frozen)

This document is the canonical contract for v1 run capsules. It defines the required artifacts,
hashing rules, comparability predicate, governance artifacts, and precedence rules. Extra keys
are allowed; schema_version is pinned to 1 for the v0/v1 contract. New optional fields may be added
without breaking this contract, but new semantics must be documented here first.

Related operational docs:
- `docs/incubation_readiness.md`
- `docs/calibration_protocol_v0.md`
- `docs/distributed_mode.md`

## Artifact Set

**Required (non-partial run capsules):**
- `window_signature.json`
- `results.json`
- `results_summary.json`

**Required (partial run capsules):**
- `window_signature.json`
- `results_summary.json` (with `partial=true`)

**Recommended:**
- `run_config_resolved.json`

**Conditionally required:**
- `first_fail_iter.txt` (MUST exist iff `results_summary.first_fail_iter` is non-null; contents must be a non-negative integer with trailing newline)

**Governance artifacts (optional for single-capsule validate/report; append-only unless noted):**
- `manual_judgement.json` (snapshot overlay)
- `manual_judgement.jsonl` (append-only)
- `governance_log.jsonl` (append-only, canonical governance journal)
  - `veriscope diff` and `veriscope report --compare` require this file and reject missing or invalid governance logs.

## Hashing Rules

*Canonical JSON* is:
```
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

Hashing requirements:
- `window_signature_ref.hash` equals `sha256(canonical_json(window_signature.json))` with volatile keys removed first.
- `window_signature.created_ts_utc` is excluded from hashing only when it is a valid ISO8601 UTC timestamp string (trailing `Z`); otherwise validation fails.
- `window_signature_ref.path` equals `"window_signature.json"`.
- Governance log hash chaining uses:
  - `entry_hash = sha256(canonical_json(entry_without_entry_hash))`
  - `prev_hash = previous entry_hash` (or null if first entry).
- The governance hash includes whichever of `event` or `event_type` is present (writers should prefer `event`).
- For new entries, `event` and `event_type` are mutually exclusive.
- Non-finite floats MUST NOT appear in governance logs; writers MAY sanitize them to `null` but SHOULD warn.
- Timestamp wire policy for `*_utc` fields:
  - Writers SHOULD emit ISO-8601 UTC strings with trailing `Z` (seconds precision recommended).
  - Readers MAY accept `Z`, `+00:00`, or naive ISO-8601 and treat naive values as UTC.

## Governance Journal: `governance_log.jsonl`

Each line is a JSON object with:
- `schema_version`: `1`
- `rev`: integer (strictly consecutive in append order; `prev_rev + 1`)
- `ts_utc`: ISO-8601 timestamp in UTC (`Z` preferred; `+00:00` and naive accepted; naive is treated as UTC) (informational, not authoritative)
- `actor`: optional string (reviewer/operator)
- `event` (preferred) or `event_type` (legacy), mutually exclusive for new entries: one of:
  - `manual_judgement_set`
  - `manual_judgement_cleared`
  - `artifact_note`
  - `recompute_summary`
  - `run_started_v1`
  - `capsule_opened_v1` (reserved; not currently emitted)
  - `run_overrides_applied_v1`
  - `gate_decision_v1`
- `payload`: event-specific object (required for all events). For run governance:
  - `run_started_v1`: `run_id`, `outdir`, `argv`, `code_identity`, `window_signature_ref`, `entrypoint`, optional `distributed`
  - `run_overrides_applied_v1`: `run_id`, `outdir`, `overrides`, `profile`, `entrypoint`, optional `argv` (wrappers should record default-divergent `gate_preset` values here)
  - `gate_decision_v1`: `run_id`, `outdir`, `iter`, `decision`, optional `ok`/`warn`, `audit`
- Binding invariant (enforced when run-governance events are present): governance payload `run_id`/`outdir` must match artifact identity (`results*.run_id` and capsule OUTDIR).
- `prev_hash`: optional hash of previous entry
- `entry_hash`: hash of this entry (required for canonical v1 journals; legacy missing `entry_hash` is only accepted in explicit legacy-validation modes)

### Distributed execution (run_started_v1 payload)

Distributed execution context is an **optional** object under `payload.distributed`. Writers SHOULD include it when known; readers MUST accept its absence for legacy compatibility.

```
payload.distributed: {
  distributed_mode: "single_process" | "replicated_single_chief_emit" | "ddp_wrapped",
  world_size_observed: int,          # >=1
  backend: "nccl" | "gloo" | string | null,
  rank: int,                         # >=0
  local_rank: int | null,
  ddp_wrapped: bool
}
```

Semantics:
- `ddp_wrapped`: torch.distributed is initialized AND world_size > 1 (use `ddp_is_active()` semantics).
- `replicated_single_chief_emit`: world_size > 1 but no initialized process group; artifacts SHOULD be emitted by rank 0 only (chief = `rank == 0`).
- `single_process`: world_size == 1 (or distributed env absent).
- Legacy aliases are accepted for backward compatibility:
  - `ddp_backend` alias of `backend`
  - `rank_observed` alias of `rank`
  - `local_rank_observed` alias of `local_rank`
  - `ddp_active` alias of `ddp_wrapped`
- Validation fail-loud rule:
  - if distributed-execution hints are present (`rank|local_rank|backend|ddp_wrapped` or legacy aliases) and `world_size_observed` is missing -> `ERROR:DISTRIBUTED_WORLD_SIZE_MISSING`
  - if `world_size_observed > 1` and `distributed_mode` is missing -> `ERROR:DISTRIBUTED_MODE_MISSING`
  - if `distributed_mode` is present but outside allowed enum -> `ERROR:DISTRIBUTED_MODE_INVALID`

## Manual Judgement Artifacts

- `manual_judgement.json`: snapshot overlay (single object, optional).
- `manual_judgement.jsonl`: append-only log of manual judgement events (optional). Each line is `ManualJudgementV1` plus a `rev` integer; entries are not hash-chained.
- Run ID mismatches in manual judgement artifacts are ignored for display (but warned); capsule validation may reject mismatched overlays.

## Precedence Map (Display Contract)

| Concept | Resolution Rule |
| --- | --- |
| **Automated outcome** | `results_summary.final_decision` (always present in non-partial; may be placeholder in partial). |
| **Manual overlay** | Last valid `manual_judgement.jsonl` entry matching run_id; else `manual_judgement.json` if valid and matching run_id; else none. |
| **Displayed Final Status** | Manual overlay if present; else automated outcome. |
| **Comparisons** | Partial capsules are non-comparable and must be rejected by `veriscope diff` and `veriscope report --compare`. Non-partial comparisons require `Comparable(A,B)` plus valid governance logs. |

## Summary Fail Diagnostics

`results_summary.json` includes fail diagnostics used by validation/automation:
- `first_fail_iter`: integer iteration of the earliest fail gate, or `null` when no fail occurred.

Finite-window invariant:
- If `counts.fail > 0`, `first_fail_iter` must be an integer.
- If `counts.fail == 0`, `first_fail_iter` must be `null`.
- `first_fail_iter.txt` must exist iff `first_fail_iter` is non-null.

## Comparable(A,B)

Runs A and B are comparable iff:
- `schema_version` matches (`1`).
- both capsules are non-partial (`partial != true` and full results present).
- `window_signature_ref.hash` matches.
- `gate_preset` matches (unless explicitly overridden).
  - Override is via CLI flag `--allow-gate-preset-mismatch` (or equivalent API argument).

**NOTE:** Runs with different `payload.distributed.distributed_mode` or `world_size_observed` are still comparable per the predicate above, but comparisons may be misleading. Treat distributed metadata as a comparability warning to surface in report/diff output (no behavior change required here).

## Exit Codes

| Command | Exit code | Meaning |
| --- | --- | --- |
| `veriscope validate` | `0` | Valid artifact capsule |
|  | `2` | Invalid/missing artifacts |
| `veriscope report` | `0` | Report rendered |
|  | `2` | Report failed (invalid artifacts or render error) |
| `veriscope diff` | `0` | Comparable (or warnings), with valid governance logs |
|  | `2` | Incomparable or invalid artifacts (including missing/invalid governance or partial capsules) |
| `veriscope report --compare` | `0` | Report rendered (or incompatible rows included when `--allow-incompatible`) |
|  | `2` | Invalid input capsules (including missing/invalid governance or partial capsules), or incompatible group without `--allow-incompatible` |
