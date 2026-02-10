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

Note: the fake GPT runner emits `gate_preset: "fake_runner"` regardless of the `--gate_preset` argument; this packet is intended to demonstrate structural contract behavior (artifacts, comparability, governance).
