#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${1:-./out/pilot_${ts}}"
if [[ $# -gt 0 ]]; then
  shift
fi

runner_args=("$@")
inject_gate_preset=true
for arg in "${runner_args[@]}"; do
  case "$arg" in
    --gate_preset|--gate_preset=*|--gate-preset|--gate-preset=*)
      inject_gate_preset=false
      break
      ;;
  esac
done

gate_args=()
if [[ "$inject_gate_preset" == "true" ]]; then
  run_kind="gpt"
  if [[ "${run_kind}" == "gpt" ]]; then
    gate_args=(--gate_preset tuned)
  else
    gate_args=(--gate_preset tuned_v0)
  fi
fi

mkdir -p "$outdir"
{
  git rev-parse HEAD 2>/dev/null || echo "unknown"
} > "$outdir/git_sha.txt"
python - <<'PY' > "$outdir/version.txt"
import veriscope
print(veriscope.__version__)
PY

set +e
veriscope run gpt --outdir "$outdir" --force -- "${gate_args[@]}" "${runner_args[@]}"
run_status=$?
veriscope validate "$outdir" > "$outdir/validate.txt" 2>&1
veriscope inspect "$outdir" > "$outdir/inspect.txt" 2>&1
veriscope report "$outdir" --format md > "$outdir/report.md" 2> "$outdir/report_stderr.txt"
set -e

if [[ $run_status -ne 0 ]]; then
  exit "$run_status"
fi
exit 0
