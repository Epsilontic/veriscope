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

use_force=0
if [[ -n "${VERISCOPE_FORCE:-}" ]]; then
  use_force=1
fi

# Default smoke device is CPU for reviewer-safe behavior on macOS/CPU-only hosts.
smoke_device="${VERISCOPE_GPT_SMOKE_DEVICE:-cpu}"
default_max_iters="${VERISCOPE_GPT_SMOKE_MAX_ITERS:-50}"

# Detect caller --device / --device=... and avoid passing --device twice.
has_max_iters=0
has_gate_preset=0
caller_provided_device=0
for ((i=0; i<${#extra_args[@]}; i++)); do
  tok="${extra_args[$i]}"
  if [[ "${tok}" == "--device" ]]; then
    if (( i+1 >= ${#extra_args[@]} )); then
      echo "[smoke] ERROR: --device provided without a value" >&2
      exit 2
    fi
    caller_provided_device=1
    smoke_device="${extra_args[$((i+1))]}"
    continue
  fi
  if [[ "${tok}" == --device=* ]]; then
    caller_provided_device=1
    smoke_device="${tok#--device=}"
    continue
  fi
  if [[ "${tok}" == "--gate_preset" || "${tok}" == "--gate-preset" ]]; then
    if (( i+1 >= ${#extra_args[@]} )); then
      echo "[smoke] ERROR: ${tok} provided without a value" >&2
      exit 2
    fi
    has_gate_preset=1
    continue
  fi
  if [[ "${tok}" == --gate_preset=* || "${tok}" == --gate-preset=* ]]; then
    has_gate_preset=1
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
if (( ! has_gate_preset )); then
  base_args+=(--gate_preset tuned_v0)
fi
if (( ! has_max_iters )); then
  base_args+=(--max_iters "${default_max_iters}")
fi
base_args+=(--no_regime)

base_args_str="${base_args[*]}"

echo "[smoke] outdir=${outdir}"
echo "[smoke] nanogpt_dir=${nanogpt_dir}"
echo "[smoke] extra_args=${extra_args_str}"
if [[ -z "${VERISCOPE_GPT_SMOKE_DEVICE:-}" && ${caller_provided_device} -eq 0 ]]; then
  echo "[smoke] defaulting to --device cpu (override with VERISCOPE_GPT_SMOKE_DEVICE or --device ...)"
fi
force_flag_str=""
if (( use_force )); then
  force_flag_str="--force"
fi
echo "[smoke] cmd: veriscope run gpt --outdir ${outdir} ${force_flag_str} -- ${base_args_str} ${extra_args_str}"

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
if (( use_force )); then
  cmd+=(--force)
fi
cmd+=(-- "${base_args[@]}")
if [[ ${#extra_args_filtered[@]} -gt 0 ]]; then
  cmd+=("${extra_args_filtered[@]}")
fi

"${cmd[@]}"

# Resolve the effective capsule directory and persist it for follow-up commands.
capdir="${outdir}"
if [[ ! -f "${capdir}/window_signature.json" ]]; then
  found_sig="$(find "${outdir}" -maxdepth 4 -type f -name window_signature.json -print -quit 2>/dev/null || true)"
  if [[ -n "${found_sig}" ]]; then
    capdir="$(dirname "${found_sig}")"
  fi
fi
printf "%s\n" "${capdir}" > "${outdir}/capdir.txt"
echo "[smoke] capdir=${capdir} (saved to ${outdir}/capdir.txt)"
