#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${1:-./out/hf_smoke_${ts}}"
force_flag=""
if [ "${VERISCOPE_FORCE:-}" != "" ]; then
  force_flag="--force"
fi

echo "[smoke] outdir=${outdir}"
echo "[smoke] cmd: veriscope run hf --gate-preset tuned_v0 --outdir ${outdir} ${force_flag} -- --model gpt2 --dataset wikitext:wikitext-2-raw-v1 --dataset_split train --max_steps 50 --device cpu"

veriscope run hf --gate-preset tuned_v0 --outdir "${outdir}" ${force_flag} -- \
  --model gpt2 \
  --dataset wikitext:wikitext-2-raw-v1 \
  --dataset_split train \
  --max_steps 50 \
  --device cpu

veriscope validate "${outdir}"
veriscope report "${outdir}" --format md > "${outdir}/report.md"
