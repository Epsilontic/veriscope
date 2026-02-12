# Maintenance GC Checklist

Recurring, cheap checks to keep agent-facing debt low (no GPU required).

## Weekly Checklist

- Documentation fences + links:
  - `python -m pytest -q tests/test_markdown_fences.py`
  - `python -m pytest -q tests/test_docs_links.py`
- Import boundary: run `python -m pytest -q tests/test_cli_import_boundary.py`.
- File mode hygiene: run `python -m pytest -q tests/test_file_modes.py`.
- Reviewer packet validity:
  - `veriscope validate docs/examples/reviewer_packet/run_a`
  - `veriscope report docs/examples/reviewer_packet/run_a --format text`
  - `veriscope diff docs/examples/reviewer_packet/run_a docs/examples/reviewer_packet/run_b`

## One-Command GC Sweep

- `VERISCOPE_AGENT_CHECK_SCOPE=gc scripts/agent/check.sh`

Normative precedence reminder: if any maintenance guidance conflicts with artifact semantics, `docs/contract_v1.md` wins.
