# Distributed Mode Contract (v0/v1 Bridge)

This document defines how distributed execution is represented in governance and validated.

Normative enum/token names live in `docs/contract_v1.md`; this page is an operator-focused companion.

## Scope and Intent

Distributed metadata is recorded under `governance_log.jsonl` in `run_started_v1.payload.distributed`.

The goal is explicitness, not inference:

- If distributed execution is implied, metadata must say so directly.
- Validation must fail loudly when required fields are missing.

## Supported `distributed_mode` Values

- `single_process`
- `replicated_single_chief_emit`
- `ddp_wrapped`

## Required Recording Fields (Contract Surface)

For distributed payloads, writers should record:

- `distributed_mode` (enum above)
- `world_size_observed` (int)
- `backend` (string or null)
- `rank` (int)
- `local_rank` (int or null)
- `ddp_wrapped` (bool)

Legacy compatibility lane (accepted for backward compatibility):

- `ddp_backend` alias of `backend`
- `rank_observed` alias of `rank`
- `local_rank_observed` alias of `local_rank`
- `ddp_active` alias of `ddp_wrapped`

## Validation Behavior (Fail Loud)

See `docs/contract_v1.md` for exact token strings. Operationally:

- if distributed-execution hints are present but `world_size_observed` is missing, validation fails loud
- missing `distributed_mode` is invalid capsule with token `ERROR:DISTRIBUTED_MODE_MISSING`
- invalid `distributed_mode` value is invalid capsule with token `ERROR:DISTRIBUTED_MODE_INVALID`

Token must be visible in `ValidationResult.message` (not only in `.errors`) so callers that surface `INVALID: {v.message}` show the exact cause.

## Verification

```bash
pytest -q tests/test_distributed_mode_contract.py tests/test_governance_contract.py tests/test_cli_diff.py
```

Relevant implementation anchors:

- [`veriscope/core/governance.py`](../veriscope/core/governance.py)
- [`veriscope/cli/validate.py`](../veriscope/cli/validate.py)
- [`veriscope/cli/governance.py`](../veriscope/cli/governance.py)
