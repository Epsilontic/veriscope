#!/usr/bin/env bash
set -euo pipefail

# Purpose: pilot harness for HF DDP that runs, validates, inspects, and reports a capsule.
# Expected environment: single node, 2 ranks, GPU available.
# Asserts governance payload.distributed contains DDP metadata (ddp_wrapped, world_size=2, ddp_active=true, ranks).

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${1:-./out/pilot_hf_ddp_${ts}}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "[pilot-hf-ddp] ERROR: CUDA_VISIBLE_DEVICES is set but empty. Unset it to preserve default GPU visibility, or set a concrete value like 0." >&2
  exit 2
fi

python_bin="${VERISCOPE_PYTHON_BIN:-python}"
veriscope_bin="${VERISCOPE_BIN:-veriscope}"

# Allow forwarding extra runner args after an optional `--` sentinel.
# Usage:
#   bash scripts/pilot/run_hf_ddp.sh <outdir> -- --device=cuda --max_steps 16 ...
#   bash scripts/pilot/run_hf_ddp.sh <outdir> --device=cuda --max_steps 16 ...
if [[ "${1:-}" == "--" ]]; then
  shift
fi
extra_args=("$@")

timeout_secs="${VERISCOPE_HF_DDP_PILOT_TIMEOUT_SECS:-300}"
timeout_bin=""
if command -v timeout >/dev/null 2>&1; then
  timeout_bin="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  timeout_bin="gtimeout"
fi

if command -v torchrun >/dev/null 2>&1; then
  launcher_cmd=(torchrun --standalone --nproc_per_node=2 --module veriscope.runners.hf.train_hf)
else
  launcher_cmd=("${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=2 --module veriscope.runners.hf.train_hf)
fi

launcher_cmd_str="${launcher_cmd[*]}"

unset VERISCOPE_FORCE_SINGLE_PROCESS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

mkdir -p "${outdir}"

"${python_bin}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[pilot-hf-ddp] ERROR: CUDA is required for this pilot run, but torch.cuda.is_available() is False.")
print(f"[pilot-hf-ddp] CUDA ok: device_count={torch.cuda.device_count()}")
PY

run_cmd=(
  "${launcher_cmd[@]}"
  --force
  --gate_preset "tuned_v0"
  --outdir "${outdir}"
  --run_id "hf-ddp-pilot"
  --model "sshleifer/tiny-gpt2"
  --dataset "file:/home/ubuntu/work/veriscope/tests/data/hf_micro_smoke.txt"
  --dataset_split "train"
  --max_steps "16"
  --batch_size "1"
  --block_size "32"
  --cadence "1"
  --gate_window "2"
  --gate_min_evidence "2"
  --rp_dim "8"
  --seed "1337"
  --device "cuda"
)

if (( ${#extra_args[@]} )); then
  run_cmd+=( "${extra_args[@]}" )
fi

printf "[pilot-hf-ddp] outdir=%s\n" "${outdir}"
printf "[pilot-hf-ddp] launcher=%s\n" "${launcher_cmd_str}"
printf "[pilot-hf-ddp] runner_args:"
printf " %q" "${run_cmd[@]}"
printf "\n"
if [[ -n "${timeout_bin}" ]]; then
  printf "[pilot-hf-ddp] timeout=%ss (bin=%s)\n" "${timeout_secs}" "${timeout_bin}"
else
  printf "[pilot-hf-ddp] timeout=none\n"
fi

runner_stdout="${outdir}/runner_stdout.txt"
runner_stderr="${outdir}/runner_stderr.txt"

set +e
if [[ -n "${timeout_bin}" ]]; then
  "${timeout_bin}" "${timeout_secs}" "${run_cmd[@]}" >"${runner_stdout}" 2>"${runner_stderr}"
  run_status=$?
else
  "${run_cmd[@]}" >"${runner_stdout}" 2>"${runner_stderr}"
  run_status=$?
fi
"${veriscope_bin}" validate "${outdir}" 2>&1 | tee "${outdir}/validate.txt"
validate_status=${PIPESTATUS[0]}
set -e

timed_out=0
if [[ ${run_status} -eq 124 ]]; then
  timed_out=1
  echo "[pilot-hf-ddp] ERROR: runner timed out after ${timeout_secs}s" | tee -a "${outdir}/validate.txt" >/dev/null
fi

printf "run_status=%s\nvalidate_status=%s\ntimed_out=%s\n" "${run_status}" "${validate_status}" "${timed_out}" > "${outdir}/pilot_status.txt"

if [[ ${validate_status} -eq 0 ]]; then
  "${python_bin}" - "${outdir}" <<'PY'
import json
from pathlib import Path
import sys

outdir = Path(sys.argv[1])
log_path = outdir / "governance_log.jsonl"
if not log_path.exists():
    raise SystemExit(f"Missing governance log: {log_path}")

entries = []
with log_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))

if not entries:
    raise SystemExit(f"Governance log empty: {log_path}")

match = None
for entry in reversed(entries):
    event = entry.get("event") or entry.get("event_type")
    if event == "run_started_v1":
        match = entry
        break

if match is None:
    raise SystemExit("No run_started_v1 entry found in governance_log.jsonl")

payload = match.get("payload")
if not isinstance(payload, dict):
    raise SystemExit("run_started_v1 payload is missing or not an object")

distributed = payload.get("distributed")
if not isinstance(distributed, dict):
    raise SystemExit("payload.distributed is missing or not an object")

if distributed.get("distributed_mode") != "ddp_wrapped":
    raise SystemExit(f"distributed_mode expected ddp_wrapped, got {distributed.get('distributed_mode')!r}")
if distributed.get("world_size_observed") != 2:
    raise SystemExit(f"world_size_observed expected 2, got {distributed.get('world_size_observed')!r}")
if distributed.get("ddp_active") is not True:
    raise SystemExit(f"ddp_active expected True, got {distributed.get('ddp_active')!r}")

rank = distributed.get("rank_observed")
if not isinstance(rank, int) or rank < 0:
    raise SystemExit(f"rank_observed expected int>=0, got {rank!r}")

if "local_rank_observed" not in distributed:
    raise SystemExit("local_rank_observed missing from distributed payload")
if "ddp_backend" not in distributed:
    raise SystemExit("ddp_backend missing from distributed payload")
if distributed.get("ddp_backend") is None:
    print("[pilot-hf-ddp] WARNING: ddp_backend is null for ddp_wrapped")

print(json.dumps(distributed, indent=2, sort_keys=True))
PY

  "${veriscope_bin}" inspect "${outdir}" > "${outdir}/inspect.txt" 2>&1
  "${veriscope_bin}" report "${outdir}" --format md > "${outdir}/report.md" 2> "${outdir}/report_stderr.txt"
fi

if [[ $run_status -ne 0 ]]; then
  exit "$run_status"
fi
if [[ ${validate_status} -ne 0 ]]; then
  exit "${validate_status}"
fi
exit 0
