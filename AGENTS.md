# AGENTS.md — Veriscope Agent Operating Guide

This file is a derived agent-facing guide. It is intentionally short.

Normative precedence: [`docs/contract_v1.md`](./docs/contract_v1.md) is the authoritative contract for artifacts, comparability, governance, and CLI exit-code semantics. If anything here conflicts with that file, follow `docs/contract_v1.md`.

## 1) Where To Look First

Start at [`docs/INDEX.md`](./docs/INDEX.md) for the canonical documentation map.

Key anchors:
- Contract: [`docs/contract_v1.md`](./docs/contract_v1.md)
- Product guidance: [`docs/productization.md`](./docs/productization.md)
- Pilot kit: [`docs/pilot/README.md`](./docs/pilot/README.md)
- Reviewer packet: [`docs/examples/reviewer_packet/`](./docs/examples/reviewer_packet/)
- Governance: [`GOVERNANCE.md`](./GOVERNANCE.md)
- Distributed mode: [`docs/distributed_mode.md`](./docs/distributed_mode.md)
- Calibration protocol: [`docs/calibration_protocol_v0.md`](./docs/calibration_protocol_v0.md)

## 2) Invariants Checklist (Must Hold)

- Decisions are canonical enums: use `decision` and `final_decision` (`pass | warn | fail | skip`).
- `skip` means not evaluated; never treat `skip` as `pass`.
- Comparability is hash-gated by `window_signature_ref.hash` (+ gate preset policy).
- `veriscope validate` is read-only and deterministic.
- Governance overlays/logs are append-only; overlays do not mutate raw artifacts.
- CLI imports must stay thin: importing `veriscope.cli.main` must not pull runner-heavy deps.

## 3) Golden Paths

Agent setup bootstrap:
```bash
bash scripts/agent/setup.sh
```

CLI boundary smoke:
```bash
python -c "import veriscope.cli.main; print('cli import ok')"
```

Reviewer packet validation:
```bash
veriscope validate docs/examples/reviewer_packet/run_a
veriscope report docs/examples/reviewer_packet/run_a --format text
veriscope diff docs/examples/reviewer_packet/run_a docs/examples/reviewer_packet/run_b
```

GPT smoke (optional longer path):
```bash
bash scripts/run_gpt_smoke.sh
veriscope validate ./out/gpt_smoke_YYYYMMDD_HHMMSS
```

## 4) Change Rules For Agents

- Keep scope tight and changes reviewable.
- Preserve contract behavior unless explicitly asked to change it.
- Any behavioral change touching artifacts, validation, comparability, governance, or CLI exit codes must include focused tests.
- Prefer fail-loudly behavior over silent coercion.
- Keep top-level CLI imports free of heavy deps (`torch`, `numpy`, `transformers`, runner modules).
- Before finishing, run the smallest relevant checks and report exact commands and results.

## Change-Impact Quick Reference

Artifact schema or hashing → update docs/contract_v1.md + Pydantic models + validator tests
CLI exit codes → update docs/contract_v1.md + integration tests
Runner wrappers → update smoke scripts + reviewer-packet golden path
New veriscope.core.* module → import boundary test auto-discovers; verify with check.sh
New docs page → add to docs/INDEX.md + verify links

## 5) Mechanical Enforcement In Repo

- Import boundary checks: `tests/test_cli_import_boundary.py`
- Markdown fence guard (all tracked `*.md`): `tests/test_markdown_fences.py`
- Docs relative link guard (`docs/**/*.md`): `tests/test_docs_links.py`
- Agent fix loop: `scripts/agent/fixit.md`
- File mode guard for docs/artifacts: `tests/test_file_modes.py`
- Local pre-merge loop: `scripts/agent/check.sh`
- Weekly cheap GC checks: `.github/workflows/gc-weekly.yml`
