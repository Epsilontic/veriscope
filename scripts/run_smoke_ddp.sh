#!/usr/bin/env bash
set -euo pipefail

# Local convenience helper (not intended for CI). Requires torchrun-compatible setup.

outdir=${1:-"./out/ddp_smoke_$(date -u +%Y%m%d_%H%M%S)"}

if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "[ddp-smoke] ERROR: CUDA_VISIBLE_DEVICES is set but empty. Unset it to preserve default GPU visibility, or set a concrete value like 0." >&2
  exit 2
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:$PWD"
else
  export PYTHONPATH="$PWD"
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(( ( RANDOM % 10000 )  + 20000 ))}

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  scripts/ddp_smoke_minimal.py --outdir "$outdir"

veriscope validate "$outdir"

echo "DDP smoke artifacts written to $outdir"
