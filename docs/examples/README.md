# Veriscope Examples

Normative precedence: `docs/contract_v1.md` is the authoritative contract. This folder provides derived example artifacts for review and demos.

## Included examples

- `calibration_report_v0.example.json`
  - Example schema-shaped calibration output.
- `reviewer_packet/`
  - Tiny synthetic capsule bundle generated through the existing fake GPT runner path.
  - Contains two comparable runs (`run_a`, `run_b`) plus captured command outputs:
    - `validate_run_a.txt`
    - `report_run_a.txt`
    - `diff_run_a_vs_run_b.txt`

## Reviewer packet scope

What this packet proves:
- Capsule structure and command mechanics: `veriscope validate`, `veriscope report`, and `veriscope diff`.
- Non-partial capsule required artifacts are present (`window_signature.json`, `results.json`, `results_summary.json`).
- Governance log handling for comparison commands (`diff`/compare paths) is exercised.

What this packet does not prove:
- It does not demonstrate `tuned_v0` gate semantics or tuned-v0 performance behavior.
- It is not a calibration or quality claim; it is a synthetic contract/demo packet.
- Single-capsule `veriscope validate` does not require governance; governance is required for compare/diff workflows.

## Reviewer packet generation provenance

Generated on 2026-02-10 from repo commit `e89b96f84660594739661eaaa704964b74327e54` using:

```bash
source ~/venv/bin/activate
export VERISCOPE_GPT_RUNNER_CMD="python scripts/fake_gpt_runner.py --sleep-seconds 0 --emit-artifacts"
veriscope run gpt --outdir docs/examples/reviewer_packet/run_a --force -- --gate_preset tuned_v0 --device cpu --max_iters 1
veriscope run gpt --outdir docs/examples/reviewer_packet/run_b --force -- --gate_preset tuned_v0 --device cpu --max_iters 1
veriscope validate docs/examples/reviewer_packet/run_a > docs/examples/reviewer_packet/validate_run_a.txt
veriscope report docs/examples/reviewer_packet/run_a --format text > docs/examples/reviewer_packet/report_run_a.txt
veriscope diff docs/examples/reviewer_packet/run_a docs/examples/reviewer_packet/run_b > docs/examples/reviewer_packet/diff_run_a_vs_run_b.txt
```

These examples are intentionally small and synthetic; they are for contract/demo inspection rather than performance claims.

Important preset note:
- `scripts/fake_gpt_runner.py` hardcodes emitted artifact preset identity to `gate_preset: "fake_runner"` (in `results.json`, `results_summary.json`, and `window_signature.json`).
- Passing `--gate_preset tuned_v0` in `veriscope run gpt ... -- ...` is still captured in `run_config_resolved.json` argv/provenance, but it does not change the fake runner's emitted artifact preset.
