#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${1:-./out/gpt_smoke_${ts}}"
shift || true

# Allow forwarding extra args after optional `--`
# Usage:
#   bash scripts/run_gpt_smoke.sh <outdir> -- --device=cpu --max_iters 50
if [[ "${1:-}" == "--" ]]; then
  shift
fi
extra_args=("$@")

nanogpt_dir="${NANOGPT_DIR:-./nanoGPT}"

# Default dataset for smoke. (If you later add dataset override parsing, update this.)
smoke_dataset="${VERISCOPE_GPT_SMOKE_DATASET:-shakespeare_char}"

dataset_dir="${nanogpt_dir}/data/${smoke_dataset}"
train_bin="${dataset_dir}/train.bin"
prep_py="${dataset_dir}/prepare.py"

force_flag=()
if [[ -n "${VERISCOPE_FORCE:-}" ]]; then
  force_flag=(--force)
fi

# Default smoke device stays "cuda" (historical behavior), but callers may override.
smoke_device="${VERISCOPE_GPT_SMOKE_DEVICE:-cuda}"
default_max_iters="${VERISCOPE_GPT_SMOKE_MAX_ITERS:-50}"

# Detect caller --device / --device=... and avoid passing --device twice.
has_max_iters=0
for ((i=0; i<${#extra_args[@]}; i++)); do
  tok="${extra_args[$i]}"
  if [[ "${tok}" == "--device" ]]; then
    if (( i+1 >= ${#extra_args[@]} )); then
      echo "[smoke] ERROR: --device provided without a value" >&2
      exit 2
    fi
    smoke_device="${extra_args[$((i+1))]}"
    continue
  fi
  if [[ "${tok}" == --device=* ]]; then
    smoke_device="${tok#--device=}"
    continue
  fi
  if [[ "${tok}" == "--max_iters" ]]; then
    if (( i+1 >= ${#extra_args[@]} )); then
      echo "[smoke] ERROR: --max_iters provided without a value" >&2
      exit 2
    fi
    has_max_iters=1
    continue
  fi
  if [[ "${tok}" == --max_iters=* ]]; then
    has_max_iters=1
    continue
  fi
done

extra_args_filtered=()
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

extra_args_str=""
if [[ ${#extra_args_filtered[@]} -gt 0 ]]; then
  extra_args_str="${extra_args_filtered[*]}"
fi

base_args=(--dataset "${smoke_dataset}" --nanogpt_dir "${nanogpt_dir}" --device "${smoke_device}")
if (( ! has_max_iters )); then
  base_args+=(--max_iters "${default_max_iters}")
fi
base_args+=(--no_regime)

base_args_str="${base_args[*]}"

echo "[smoke] outdir=${outdir}"
echo "[smoke] nanogpt_dir=${nanogpt_dir}"
echo "[smoke] extra_args=${extra_args_str}"
echo "[smoke] cmd: veriscope run gpt --outdir ${outdir} ${force_flag[*]:-} -- ${base_args_str} ${extra_args_str}"

# Ensure nanoGPT dataset bins exist inside fresh environments (e.g., containers).
if [[ ! -f "${train_bin}" ]]; then
  if [[ -f "${prep_py}" ]]; then
    echo "[smoke] preparing dataset bins: ${train_bin} missing; running ${prep_py}"
    python "${prep_py}"
  else
    echo "[smoke] ERROR: dataset bins missing and prepare script not found:" >&2
    echo "[smoke]   expected train_bin=${train_bin}" >&2
    echo "[smoke]   expected prep_py=${prep_py}" >&2
    exit 2
  fi
fi

cmd=(veriscope run gpt --outdir "${outdir}")
cmd+=("${force_flag[@]}")
cmd+=(-- "${base_args[@]}")
if [[ ${#extra_args_filtered[@]} -gt 0 ]]; then
  cmd+=("${extra_args_filtered[@]}")
fi

"${cmd[@]}"
