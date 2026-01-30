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

force_flag=()
if [[ -n "${VERISCOPE_FORCE:-}" ]]; then
  force_flag=(--force)
fi

# Default smoke device stays "cuda" (historical behavior), but callers may override.
smoke_device="${VERISCOPE_GPT_SMOKE_DEVICE:-cuda}"

# Detect caller --device / --device=... and avoid passing --device twice.
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

echo "[smoke] outdir=${outdir}"
echo "[smoke] nanogpt_dir=${nanogpt_dir}"
echo "[smoke] extra_args=${extra_args_str}"
echo "[smoke] cmd: veriscope run gpt --outdir ${outdir} ${force_flag[*]:-} -- --dataset shakespeare_char --nanogpt_dir ${nanogpt_dir} --device ${smoke_device} --max_iters 200 --no_regime ${extra_args_str}"

cmd=(veriscope run gpt --outdir "${outdir}")
cmd+=("${force_flag[@]}")
cmd+=(-- \
  --dataset shakespeare_char \
  --nanogpt_dir "${nanogpt_dir}" \
  --device "${smoke_device}" \
  --max_iters 200 \
  --no_regime
)
if [[ ${#extra_args_filtered[@]} -gt 0 ]]; then
  cmd+=("${extra_args_filtered[@]}")
fi

"${cmd[@]}"
