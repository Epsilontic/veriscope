#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="/work/out/container_acceptance_${ts}"

mkdir -p "${outdir}"

echo "[acceptance] starting gpt smoke into ${outdir}"

# --- Ensure nanoGPT + dataset bins exist in fresh container envs ---
nanogpt_dir="${NANOGPT_DIR:-/work/nanoGPT}"
smoke_dataset="${VERISCOPE_GPT_SMOKE_DATASET:-shakespeare_char}"

if [[ ! -d "${nanogpt_dir}" ]]; then
  echo "[acceptance] cloning nanoGPT into ${nanogpt_dir}"
  git clone --depth 1 https://github.com/karpathy/nanoGPT.git "${nanogpt_dir}"
fi

prep_py="${nanogpt_dir}/data/${smoke_dataset}/prepare.py"
train_bin="${nanogpt_dir}/data/${smoke_dataset}/train.bin"

if [[ ! -f "${prep_py}" ]]; then
  echo "[acceptance] ERROR: nanoGPT prepare script not found: ${prep_py}" >&2
  exit 2
fi

if [[ ! -f "${train_bin}" ]]; then
  echo "[acceptance] preparing nanoGPT dataset bins for ${smoke_dataset}"
  (cd "${nanogpt_dir}" && python "data/${smoke_dataset}/prepare.py")
fi

export NANOGPT_DIR="${nanogpt_dir}"
# GH Actions runners don't have GPUs; keep acceptance deterministic on CPU.
export VERISCOPE_GPT_SMOKE_DEVICE="${VERISCOPE_GPT_SMOKE_DEVICE:-cpu}"

bash scripts/run_gpt_smoke.sh "${outdir}" -- --max_iters 50

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
