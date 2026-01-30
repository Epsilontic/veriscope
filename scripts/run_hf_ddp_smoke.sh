#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${1:-./out/hf_ddp_smoke_${ts}}"

mkdir -p "${outdir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export VERISCOPE_OUT_BASE="${VERISCOPE_OUT_BASE:-${outdir}}"
export VERISCOPE_FORCE="${VERISCOPE_FORCE:-1}"
# Important: ensure we don't suppress DDP init in the runner.
unset VERISCOPE_FORCE_SINGLE_PROCESS

python_bin="${VERISCOPE_PYTHON_BIN:-python}"

echo "[hf-ddp-smoke] outdir=${outdir}"
export VERISCOPE_HF_RUNNER_CMD="torchrun --standalone --nproc_per_node=2 --module veriscope.runners.hf.train_hf"
echo "[hf-ddp-smoke] runner_cmd=${VERISCOPE_HF_RUNNER_CMD}"
echo "[hf-ddp-smoke] cmd: ${python_bin} -m veriscope.cli.main run hf --outdir ${outdir} --force -- --outdir ${outdir} --run_id hf-ddp-smoke --model sshleifer/tiny-gpt2 --dataset file:${repo_root}/tests/data/hf_micro_smoke.txt --dataset_split train --max_steps 8 --batch_size 1 --block_size 32 --cadence 1 --gate_window 2 --gate_min_evidence 2 --rp_dim 8 --seed 1337 --device cpu"

"${python_bin}" -m veriscope.cli.main run hf \
  --outdir "${outdir}" \
  --force \
  -- \
  --outdir "${outdir}" \
  --run_id "hf-ddp-smoke" \
  --model "sshleifer/tiny-gpt2" \
  --dataset "file:${repo_root}/tests/data/hf_micro_smoke.txt" \
  --dataset_split "train" \
  --max_steps "8" \
  --batch_size "1" \
  --block_size "32" \
  --cadence "1" \
  --gate_window "2" \
  --gate_min_evidence "2" \
  --rp_dim "8" \
  --seed "1337" \
  --device "cpu"

capdir=""
ws_hit="$(find "${outdir}" -type f -name window_signature.json -print -quit 2>/dev/null || true)"
if [[ -n "${ws_hit}" ]]; then
  capdir="$(dirname "${ws_hit}")"
fi
if [[ -z "${capdir}" && -n "${VERISCOPE_OUT_BASE:-}" ]]; then
  ws_hit="$(find "${VERISCOPE_OUT_BASE}" -type f -name window_signature.json -print -quit 2>/dev/null || true)"
  if [[ -n "${ws_hit}" ]]; then
    capdir="$(dirname "${ws_hit}")"
  fi
fi
if [[ -z "${capdir}" ]]; then
  echo "[hf-ddp-smoke] ERROR: could not locate capsule directory for validation/report." >&2
  exit 2
fi

"${python_bin}" -m veriscope.cli.main validate "${capdir}"
"${python_bin}" -m veriscope.cli.main report "${capdir}" --format md > "${capdir}/report.md"
