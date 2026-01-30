#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

ts="$(date +%Y%m%d_%H%M%S)"

# OUTDIR is optional. If the first arg looks like an option (starts with '-') or is '--',
# fall back to the default outdir and treat all args as overrides.
outdir="./out/hf_ddp_smoke_${ts}"
if [[ $# -gt 0 && "${1:-}" != "--" && "${1:0:1}" != "-" ]]; then
  outdir="$1"
  shift
fi

# Allow forwarding extra runner args after an optional `--` sentinel.
# Usage:
#   bash scripts/run_hf_ddp_smoke.sh <outdir> -- --device=cuda --max_steps 8 ...
#   bash scripts/run_hf_ddp_smoke.sh <outdir> --device=cuda --max_steps 8 ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi
extra_args=("$@")

# Must be initialized under `set -u` even when caller provides no extra args.
extra_args_filtered=()

# If caller provides --device, use it and avoid passing --device twice.
# Supports both:
#   --device cuda
#   --device=cuda
smoke_device="cpu"
for ((i=0; i<${#extra_args[@]}; i++)); do
  tok="${extra_args[$i]}"
  if [[ "${tok}" == "--device" ]]; then
    if (( i+1 >= ${#extra_args[@]} )); then
      echo "[hf-ddp-smoke] ERROR: --device provided without a value" >&2
      exit 2
    fi
    smoke_device="${extra_args[$((i+1))]}"   # last one wins
    continue
  fi
  if [[ "${tok}" == --device=* ]]; then
    smoke_device="${tok#--device=}"          # last one wins
    continue
  fi
done

# Filter device overrides out of extra_args so we don't pass --device twice.
# Removes:
#   --device <val>
#   --device=<val>
# (extra_args_filtered already initialized above)
skip_next=0
for ((i=0; i<${#extra_args[@]}; i++)); do
  tok="${extra_args[$i]}"
  if (( skip_next )); then
    skip_next=0
    continue
  fi
  if [[ "${tok}" == "--device" ]]; then
    skip_next=1
    continue
  fi
  if [[ "${tok}" == --device=* ]]; then
    continue
  fi
  extra_args_filtered+=("${tok}")
done

# Safe string form for logging.
extra_args_str=""
if [[ ${#extra_args_filtered[@]} -gt 0 ]]; then
  extra_args_str="${extra_args_filtered[*]}"
fi

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
echo "[hf-ddp-smoke] extra_args=${extra_args_str}"
echo "[hf-ddp-smoke] cmd: ${python_bin} -m veriscope.cli.main run hf --outdir ${outdir} --force -- --outdir ${outdir} --run_id hf-ddp-smoke --model sshleifer/tiny-gpt2 --dataset file:${repo_root}/tests/data/hf_micro_smoke.txt --dataset_split train --max_steps 8 --batch_size 1 --block_size 32 --cadence 1 --gate_window 2 --gate_min_evidence 2 --rp_dim 8 --seed 1337 --device ${smoke_device} ${extra_args_str}"

run_cmd=(
  "${python_bin}" -m veriscope.cli.main run hf
  --outdir "${outdir}"
  --force
  --
  --outdir "${outdir}"
  --run_id "hf-ddp-smoke"
  --model "sshleifer/tiny-gpt2"
  --dataset "file:${repo_root}/tests/data/hf_micro_smoke.txt"
  --dataset_split "train"
  --max_steps "8"
  --batch_size "1"
  --block_size "32"
  --cadence "1"
  --gate_window "2"
  --gate_min_evidence "2"
  --rp_dim "8"
  --seed "1337"
  --device "${smoke_device}"
)
# Bash 3.2 + `set -u`: expanding an empty array as "${arr[@]}" can throw "unbound variable".
if (( ${#extra_args_filtered[@]} )); then
  run_cmd+=( "${extra_args_filtered[@]}" )
fi
"${run_cmd[@]}"

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
