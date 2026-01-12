#!/usr/bin/env bash  
set -euo pipefail  
  
ts="$(date +%Y%m%d_%H%M%S)"  
outdir="${1:-./out/cifar_smoke_${ts}}"  
  
export SCAR_DATA="${SCAR_DATA:-./data}"  
  
echo "[smoke] outdir=${outdir}"  
echo "[smoke] cmd: veriscope run cifar --smoke --outdir ${outdir}"  
veriscope run cifar --smoke --outdir "${outdir}"  
