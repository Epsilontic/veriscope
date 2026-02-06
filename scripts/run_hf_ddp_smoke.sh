#!/usr/bin/env bash


set -euo pipefail


script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

ts="$(date +%Y%m%d_%H%M%S)"

if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "[hf-ddp-smoke] ERROR: CUDA_VISIBLE_DEVICES is set but empty. Unset it to preserve default GPU visibility, or set a concrete value like 0." >&2
  exit 2
fi

# Hard bound the DDP smoke so CI/local runs don't hang for 10 minutes on rendezvous.
# Use GNU timeout when available, else run unbounded.
ddp_timeout_secs="${VERISCOPE_HF_DDP_SMOKE_TIMEOUT_SECS:-240}"
timeout_bin=""
if command -v timeout >/dev/null 2>&1; then
  timeout_bin="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  timeout_bin="gtimeout"
fi

echo "[hf-ddp-smoke] timeout=${ddp_timeout_secs}s (timeout_bin=${timeout_bin:-none})"

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

# Force rendezvous to loopback to avoid hostname/IPv6 resolution issues (esp. macOS).
# If tests/CI provided MASTER_ADDR/MASTER_PORT, respect them. Otherwise, pick a free local port
# to avoid collisions when multiple runs happen concurrently.
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -n "${MASTER_PORT:-}" ]]; then
  _master_port_src="env"
else
  _master_port_src="auto"
  _py_for_port="${VERISCOPE_PYTHON_BIN:-python}"
  MASTER_PORT="$("${_py_for_port}" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
finally:
    s.close()
PY
)"
fi
export MASTER_ADDR MASTER_PORT

# Prefer loopback iface for gloo on macOS/Linux.
if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export GLOO_SOCKET_IFNAME="lo0"
  else
    export GLOO_SOCKET_IFNAME="lo"
  fi
fi
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"

if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
export VERISCOPE_OUT_BASE="${VERISCOPE_OUT_BASE:-${outdir}}"
export VERISCOPE_FORCE="${VERISCOPE_FORCE:-1}"
# Important: ensure we don't suppress DDP init in the runner.
unset VERISCOPE_FORCE_SINGLE_PROCESS


python_bin="${VERISCOPE_PYTHON_BIN:-python}"

# Fail-fast rendezvous: avoid 300s+ hangs when TCPStore cannot form.
# Only apply if this torchrun supports the flag.
rdzv_timeout_secs="${VERISCOPE_TORCHRUN_RDZV_TIMEOUT_SECS:-60}"
rdzv_timeout_arg=""
local_addr_arg=""
master_addr_arg=""
master_port_arg=""
if command -v torchrun >/dev/null 2>&1; then
  _torchrun_help="$(torchrun --help 2>/dev/null || true)"
  if echo "${_torchrun_help}" | grep -q "rdzv_timeout"; then
    rdzv_timeout_arg="--rdzv_timeout=${rdzv_timeout_secs}"
  fi
  # Critical on macOS: force torchrun to use loopback for the node's address,
  # otherwise it may advertise hostname (e.g. "Prometheus") and hang on IPv6/DNS.
  if echo "${_torchrun_help}" | grep -q -- "--local_addr"; then
    local_addr_arg="--local_addr=${MASTER_ADDR}"
  fi
  # Some torchrun builds still honor these, and they don't hurt.
  if echo "${_torchrun_help}" | grep -q -- "--master_addr"; then
    master_addr_arg="--master_addr=${MASTER_ADDR}"
  fi
  if echo "${_torchrun_help}" | grep -q -- "--master_port"; then
    master_port_arg="--master_port=${MASTER_PORT}"
  fi
fi

echo "[hf-ddp-smoke] outdir=${outdir}"
export VERISCOPE_HF_RUNNER_CMD="torchrun \
  ${local_addr_arg} \
  ${master_addr_arg} \
  ${master_port_arg} \
  --nnodes=1 \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=veriscope-ddp-smoke-${ts} \
  ${rdzv_timeout_arg} \
  --max_restarts=0 \
  --module veriscope.runners.hf.train_hf"
echo "[hf-ddp-smoke] runner_cmd=${VERISCOPE_HF_RUNNER_CMD}"
echo "[hf-ddp-smoke] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} (src=${_master_port_src}) GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME} rdzv_timeout_arg=${rdzv_timeout_arg:-} local_addr_arg=${local_addr_arg:-} master_addr_arg=${master_addr_arg:-} master_port_arg=${master_port_arg:-}"
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
# Replace the direct invocation with a bounded one.
# NOTE: keep exit code nonzero on timeout so tests can skip based on stderr/stdout.
set +e
if [[ -n "${timeout_bin}" ]]; then
  "${timeout_bin}" "${ddp_timeout_secs}" "${run_cmd[@]}"
  rc=$?
else
  "${run_cmd[@]}"
  rc=$?
fi
set -e

if [[ $rc -ne 0 ]]; then
  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    echo "[hf-ddp-smoke] ERROR: timed out after ${ddp_timeout_secs}s" >&2
  fi
  exit $rc
fi

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
