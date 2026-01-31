#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="/work/out/container_acceptance_${ts}"

mkdir -p "${outdir}"

echo "[acceptance] starting gpt smoke into ${outdir}"

# --- Ensure nanoGPT + dataset bins exist in fresh container envs ---
smoke_dataset="${VERISCOPE_GPT_SMOKE_DATASET:-shakespeare_char}"
baked_nanogpt_dir="/work/nanoGPT"
baked_train_bin="${baked_nanogpt_dir}/data/${smoke_dataset}/train.bin"
baked_val_bin="${baked_nanogpt_dir}/data/${smoke_dataset}/val.bin"

if [[ -d "${baked_nanogpt_dir}" && -f "${baked_train_bin}" && -f "${baked_val_bin}" ]]; then
  echo "[acceptance] using baked nanoGPT + dataset bins"
  nanogpt_dir="${baked_nanogpt_dir}"
else
  nanogpt_dir="${NANOGPT_DIR:-/work/nanoGPT}"
  if [[ ! -d "${nanogpt_dir}" ]]; then
    echo "[acceptance] cloning nanoGPT into ${nanogpt_dir}"
    git clone --depth 1 https://github.com/karpathy/nanoGPT.git "${nanogpt_dir}"
  fi

  prep_py="${nanogpt_dir}/data/${smoke_dataset}/prepare.py"
  train_bin="${nanogpt_dir}/data/${smoke_dataset}/train.bin"
  val_bin="${nanogpt_dir}/data/${smoke_dataset}/val.bin"

  if [[ ! -f "${prep_py}" ]]; then
    echo "[acceptance] ERROR: nanoGPT prepare script not found: ${prep_py}" >&2
    exit 2
  fi

  if [[ ! -f "${train_bin}" || ! -f "${val_bin}" ]]; then
    echo "[acceptance] preparing nanoGPT dataset bins for ${smoke_dataset}"
    (cd "${nanogpt_dir}" && python "data/${smoke_dataset}/prepare.py")
  fi
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
if veriscope report "${capdir}" --format md > "${capdir}/report.md" 2>"${capdir}/report.stderr"; then
  echo "[acceptance] report: ok"
else
  echo "[acceptance] report: skipped (report deps likely not installed)" >&2
  {
    echo "# Veriscope report (minimal)"
    echo
    echo "The full report renderer was unavailable in this acceptance image."
    echo
    echo "Window: container acceptance"
    echo "Outcome: validate succeeded; smoke succeeded"
    echo
    echo "To generate the full report, install extras:"
    echo "  pip install 'veriscope[report]'"
    echo
    echo "stderr from report attempt saved to: report.stderr"
  } > "${capdir}/report.md"
fi

echo "[acceptance] outdir=${outdir}"
echo "[acceptance] capdir=${capdir}"
