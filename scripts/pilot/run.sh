#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${1:-./out/pilot_${ts}}"
if [[ $# -gt 0 ]]; then
  shift
fi

strict_mode_raw="${VERISCOPE_PILOT_STRICT:-1}"
case "${strict_mode_raw}" in
  1|true|TRUE|yes|YES)
    strict_mode=1
    ;;
  0|false|FALSE|no|NO)
    strict_mode=0
    ;;
  *)
    echo "[pilot] ERROR: VERISCOPE_PILOT_STRICT must be 0/1 (or true/false), got: ${strict_mode_raw}" >&2
    exit 2
    ;;
esac

runner_args=("$@")
inject_gate_preset=true
if [[ ${#runner_args[@]} -gt 0 ]]; then
  for arg in "${runner_args[@]}"; do
    case "$arg" in
      --gate_preset|--gate_preset=*|--gate-preset|--gate-preset=*)
        inject_gate_preset=false
        break
        ;;
    esac
  done
fi

gate_args=()
if [[ "$inject_gate_preset" == "true" ]]; then
  run_kind="gpt"
  if [[ "${run_kind}" == "gpt" ]]; then
    gate_args=(--gate_preset tuned_v0)
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
run_cmd=(veriscope run gpt --outdir "$outdir" --force --)
if [[ ${#gate_args[@]} -gt 0 ]]; then
  run_cmd+=("${gate_args[@]}")
fi
if [[ ${#runner_args[@]} -gt 0 ]]; then
  run_cmd+=("${runner_args[@]}")
fi
"${run_cmd[@]}"
run_status=$?
set -e

capdir="$outdir"
if [[ ! -f "${capdir}/window_signature.json" ]]; then
  found_sig="$(find "${outdir}" -maxdepth 4 -type f -name window_signature.json -print -quit 2>/dev/null || true)"
  if [[ -n "${found_sig}" ]]; then
    capdir="$(dirname "${found_sig}")"
  fi
fi
printf "%s\n" "${capdir}" > "${outdir}/capdir.txt"

set +e
veriscope validate "${capdir}" > "${outdir}/validate.txt" 2>&1
validate_status=$?
veriscope inspect "${capdir}" > "${outdir}/inspect.txt" 2>&1
inspect_status=$?
veriscope report "${capdir}" --format md > "${outdir}/report.md" 2> "${outdir}/report_stderr.txt"
report_status=$?
set -e

if [[ $run_status -ne 0 ]]; then
  exit "$run_status"
fi

if [[ $strict_mode -eq 1 ]]; then
  if [[ $validate_status -ne 0 ]]; then
    exit "$validate_status"
  fi
  if [[ $inspect_status -ne 0 ]]; then
    exit "$inspect_status"
  fi
  if [[ $report_status -ne 0 ]]; then
    exit "$report_status"
  fi
else
  if [[ $validate_status -ne 0 || $inspect_status -ne 0 || $report_status -ne 0 ]]; then
    echo "[pilot] VERISCOPE_PILOT_STRICT=0; ignoring validate/inspect/report failure(s)." >&2
  fi
fi
exit 0
