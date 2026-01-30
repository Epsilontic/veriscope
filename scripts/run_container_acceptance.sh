#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="/work/out/container_acceptance_${ts}"

mkdir -p "${outdir}"

echo "[acceptance] starting gpt smoke into ${outdir}"

bash scripts/run_gpt_smoke.sh "${outdir}"

capdir="${outdir}"
if [[ ! -f "${capdir}/window_signature.json" ]]; then
  found_sig="$(find "${outdir}" -maxdepth 4 -type f -name window_signature.json -print -quit 2>/dev/null || true)"
  if [[ -n "${found_sig}" ]]; then
    capdir="$(dirname "${found_sig}")"
  fi
fi

veriscope validate "${capdir}"
veriscope report "${capdir}" --format md > "${capdir}/report.md"

echo "[acceptance] outdir=${outdir}"
echo "[acceptance] capdir=${capdir}"
