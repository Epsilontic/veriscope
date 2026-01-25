#!/usr/bin/env bash  
set -euo pipefail  
  
ts="$(date +%Y%m%d_%H%M%S)"  
outdir="${1:-./out/gpt_smoke_${ts}}"  
nanogpt_dir="${NANOGPT_DIR:-./nanoGPT}"  
force_flag=""  
if [ "${VERISCOPE_FORCE:-}" != "" ]; then  
  force_flag="--force"  
fi  
  
echo "[smoke] outdir=${outdir}"  
echo "[smoke] nanogpt_dir=${nanogpt_dir}"  
echo "[smoke] cmd: veriscope run gpt --outdir ${outdir} ${force_flag} -- --dataset shakespeare_char --nanogpt_dir ${nanogpt_dir} --device cuda --max_iters 200 --no_regime"  
  
veriscope run gpt --outdir "${outdir}" ${force_flag} -- \  
  --dataset shakespeare_char \  
  --nanogpt_dir "${nanogpt_dir}" \  
  --device cuda \  
  --max_iters 200 \  
  --no_regime  
