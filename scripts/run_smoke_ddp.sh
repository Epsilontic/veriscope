#!/usr/bin/env bash
set -euo pipefail

# Local convenience helper (not intended for CI). Requires torchrun-compatible setup.

outdir=${1:-"./out/ddp_smoke_$(date -u +%Y%m%d_%H%M%S)"}

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
