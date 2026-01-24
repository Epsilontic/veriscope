#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-./out/phase5a_manual_$(date +%Y%m%d_%H%M%S)}"
RUNNER_CMD="${PYTHON:-python} $(pwd)/scripts/fake_gpt_runner.py --emit-artifacts"

export VERISCOPE_GPT_RUNNER_CMD="${RUNNER_CMD}"

veriscope run gpt --outdir "${OUTDIR}" --force -- --sleep-seconds 30 &
WRAPPER_PID=$!

sleep 1
kill -INT "${WRAPPER_PID}" || true
wait "${WRAPPER_PID}" || true

veriscope validate "${OUTDIR}"
veriscope inspect "${OUTDIR}" --format text
