# Veriscope Contract v1 (Frozen)

This document is the canonical contract for v1 run capsules. It defines the required artifacts,
hashing rules, comparability predicate, governance artifacts, and precedence rules. Extra keys
are allowed; schema_version is pinned to 1 for the v0/v1 contract. New optional fields may be added
without breaking this contract, but new semantics must be documented here first.

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

**Governance artifacts (optional, append-only unless noted):**
- `manual_judgement.json` (snapshot overlay)
- `manual_judgement.jsonl` (append-only)
- `governance_log.jsonl` (append-only, canonical governance journal)

## Hashing Rules

*Canonical JSON* is:
```
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

Hashing requirements:
- `window_signature_ref.hash` equals `sha256(canonical_json(window_signature.json))`.
- `window_signature_ref.path` equals `"window_signature.json"`.
- Governance log hash chaining uses:
  - `entry_hash = sha256(canonical_json(entry_without_entry_hash))`
  - `prev_hash = previous entry_hash` (or null if first entry).

## Governance Journal: `governance_log.jsonl`

Each line is a JSON object with:
- `schema_version`: `1`
- `rev`: integer (strictly consecutive in append order; `prev_rev + 1`)
- `ts_utc`: ISO-8601 timestamp in UTC (`Z` preferred; `+00:00` accepted) (informational, not authoritative)
- `actor`: optional string (reviewer/operator)
- `event_type`: one of:
  - `manual_judgement_set`
  - `manual_judgement_cleared`
  - `artifact_note`
  - `recompute_summary`
- `payload`: event-specific object (for manual judgement: `status`, `reason`, `reviewer`, `run_id`, `source_path`)
- `prev_hash`: optional hash of previous entry
- `entry_hash`: optional hash of this entry (writers MUST include; readers MAY accept missing entry_hash for legacy journals and must warn)

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
| **Comparisons** | If partial: compare only header-level fields + overlay; never compare counts/gates. If non-partial: compare counts + decisions + overlay, but only when `Comparable(A,B)` holds. |

## Comparable(A,B)

Runs A and B are comparable iff:
- `schema_version` matches (`1`).
- `window_signature_ref.hash` matches.
- `gate_preset` matches (unless explicitly overridden).
  - Override is via CLI flag `--allow-gate-preset-mismatch` (or equivalent API argument).

## Exit Codes

| Command | Exit code | Meaning |
| --- | --- | --- |
| `veriscope validate` | `0` | Valid artifact capsule |
|  | `2` | Invalid/missing artifacts |
| `veriscope report` | `0` | Report rendered |
|  | `2` | Report failed (invalid artifacts or render error) |
| `veriscope diff` | `0` | Comparable (or warnings) |
|  | `2` | Incomparable or invalid artifacts |
| `veriscope report --compare` | `0` | Report rendered |
|  | `2` | Incompatible group (unless `--allow-incompatible`) |

