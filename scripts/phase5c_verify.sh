#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${1:-./out/phase5c_golden_${TS}}"
SMOKE_SCRIPT="$(pwd)/scripts/run_gpt_smoke.sh"

if [ -e "${OUTDIR}" ]; then
  echo "[phase5c] ERROR: outdir already exists: ${OUTDIR}" >&2
  exit 1
fi

echo "[phase5c] outdir=${OUTDIR}"

if [ -x "${SMOKE_SCRIPT}" ]; then
  echo "[phase5c] using smoke script: ${SMOKE_SCRIPT}"
  VERISCOPE_FORCE=1 bash "${SMOKE_SCRIPT}" "${OUTDIR}"
else
  echo "[phase5c] smoke script not found; running minimal GPT smoke"
  veriscope run gpt --outdir "${OUTDIR}" --force -- \
    --dataset shakespeare_char \
    --nanogpt_dir "${NANOGPT_DIR:-./nanoGPT}" \
    --device "${VERISCOPE_DEVICE:-cuda}" \
    --max_iters 200 \
    --no_regime
fi

veriscope validate "${OUTDIR}" > "${OUTDIR}/validate.txt" 2>&1
veriscope inspect "${OUTDIR}" --format text > "${OUTDIR}/inspect.txt" 2>&1

if [ -f "${OUTDIR}/governance_log.jsonl" ]; then
  veriscope inspect "${OUTDIR}" --format text --strict-governance > \
    "${OUTDIR}/inspect_strict_governance.txt" 2>&1
fi

veriscope report "${OUTDIR}" --format md > "${OUTDIR}/report.md" 2> "${OUTDIR}/report_stderr.txt"

git rev-parse HEAD > "${OUTDIR}/git_sha.txt"
if veriscope --version > "${OUTDIR}/version.txt" 2>&1; then
  :
else
  "${PYTHON:-python}" -c "import veriscope; print(getattr(veriscope, '__version__', 'unknown'))" \
    > "${OUTDIR}/version.txt"
fi

echo "[phase5c] wrote: ${OUTDIR}/validate.txt"
echo "[phase5c] wrote: ${OUTDIR}/inspect.txt"
echo "[phase5c] wrote: ${OUTDIR}/report.md"
echo "[phase5c] wrote: ${OUTDIR}/report_stderr.txt"
echo "[phase5c] wrote: ${OUTDIR}/git_sha.txt"
echo "[phase5c] wrote: ${OUTDIR}/version.txt"

if [ -f "${OUTDIR}/governance_log.jsonl" ]; then
  OUTDIR_NEG_GOV="${OUTDIR}_neg_gov_tamper"
  cp -a "${OUTDIR}" "${OUTDIR_NEG_GOV}"
  "${PYTHON:-python}" - <<PY
import json
from pathlib import Path

outdir = Path("""${OUTDIR_NEG_GOV}""")
path = outdir / "governance_log.jsonl"
lines = path.read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("governance_log.jsonl is empty")
idx = None
obj = None
for i, line in enumerate(lines):
    if not line.strip():
        continue
    try:
        obj = json.loads(line)
        idx = i
        break
    except Exception:
        continue
if idx is None or obj is None:
    raise SystemExit("governance_log.jsonl has no JSON entries")
if isinstance(obj.get("actor"), str):
    obj["actor"] = obj["actor"] + "x"
else:
    obj["actor"] = "tamper"
lines[idx] = json.dumps(obj, sort_keys=True)
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

  set +e
  veriscope validate "${OUTDIR_NEG_GOV}" --strict-governance \
    > "${OUTDIR_NEG_GOV}/validate_strict_stdout.txt" \
    2> "${OUTDIR_NEG_GOV}/validate_strict_stderr.txt"
  gov_status=$?
  set -e

  if [ "${gov_status}" -eq 0 ]; then
    echo "EXPECTED_FAILURE_MISSING: strict governance validation passed unexpectedly" \
      > "${OUTDIR_NEG_GOV}/RESULT"
    exit 1
  fi

  echo "EXPECTED_FAILURE: strict governance validation should fail (GOVERNANCE_HASH_MISMATCH)" \
    > "${OUTDIR_NEG_GOV}/RESULT"
  echo "[phase5c] wrote: ${OUTDIR_NEG_GOV}/RESULT"
fi

OUTDIR_A="${OUTDIR}_A"
OUTDIR_B="${OUTDIR}_B"
cp -a "${OUTDIR}" "${OUTDIR_A}"
cp -a "${OUTDIR}" "${OUTDIR_B}"

"${PYTHON:-python}" - <<PY
import json
import hashlib
from pathlib import Path

def canonical_dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

outdir = Path("""${OUTDIR_B}""")

ws_path = outdir / "window_signature.json"
ws_obj = json.loads(ws_path.read_text(encoding="utf-8"))
ws_obj["tamper"] = "x"
ws_path.write_text(json.dumps(ws_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

ws_hash = sha256_hex(canonical_dumps(ws_obj).encode("utf-8"))

for name in ("results.json", "results_summary.json"):
    path = outdir / name
    obj = json.loads(path.read_text(encoding="utf-8"))
    ref = obj.get("window_signature_ref") or {}
    ref["hash"] = ws_hash
    ref["path"] = "window_signature.json"
    obj["window_signature_ref"] = ref
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

set +e
veriscope diff "${OUTDIR_A}" "${OUTDIR_B}" \
  > "${OUTDIR}/diff_window_mismatch.txt" \
  2> "${OUTDIR}/diff_window_mismatch_stderr.txt"
diff_status=$?
set -e

if [ "${diff_status}" -eq 0 ]; then
  echo "[phase5c] ERROR: diff succeeded unexpectedly for window mismatch" >&2
  exit 1
fi

set +e
veriscope diff "${OUTDIR_A}" "${OUTDIR_B}" --json \
  > "${OUTDIR}/diff_window_mismatch.json" \
  2> "${OUTDIR}/diff_window_mismatch_json_stderr.txt"
diff_json_status=$?
set -e

"${PYTHON:-python}" - <<PY
import json
from pathlib import Path

path = Path("""${OUTDIR}""") / "diff_window_mismatch.json"
obj = json.load(path.open(encoding="utf-8"))
reason = obj.get("comparability", {}).get("reason")
if reason != "WINDOW_HASH_MISMATCH":
    raise SystemExit(f"Expected WINDOW_HASH_MISMATCH, got {reason!r}")
PY

if [ "${diff_json_status}" -eq 0 ]; then
  echo "[phase5c] ERROR: diff --json exited 0 unexpectedly for window mismatch" >&2
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  GREP_BIN="rg"
  GREP_ARGS="-n"
else
  GREP_BIN="grep"
  GREP_ARGS="-En"
fi

OUTDIR_PARTIAL="${OUTDIR}_partial"
cp -a "${OUTDIR}" "${OUTDIR_PARTIAL}"
rm -f "${OUTDIR_PARTIAL}/results.json"

set +e
veriscope diff "${OUTDIR_A}" "${OUTDIR_PARTIAL}" \
  > "${OUTDIR}/diff_partial.txt" \
  2> "${OUTDIR}/diff_partial_stderr.txt"
partial_status=$?
set -e

if [ "${partial_status}" -ne 0 ]; then
  echo "[phase5c] ERROR: diff failed in partial mode" >&2
  exit 1
fi

if ! "${GREP_BIN}" ${GREP_ARGS} "NOTE:PARTIAL_MODE decision-only" \
  "${OUTDIR}/diff_partial.txt" >/dev/null; then
  echo "[phase5c] ERROR: missing partial mode note" >&2
  exit 1
fi

if "${GREP_BIN}" ${GREP_ARGS} "counts_" "${OUTDIR}/diff_partial.txt" >/dev/null; then
  echo "[phase5c] ERROR: partial diff should not include counts lines" >&2
  exit 1
fi

secrets_match=0
: > "${OUTDIR}/secrets_scan.txt"
if [ -f "${OUTDIR}/run_config_resolved.json" ]; then
  if "${GREP_BIN}" ${GREP_ARGS} "API_KEY|SECRET|TOKEN|PASSWORD" \
    "${OUTDIR}/run_config_resolved.json" \
    >> "${OUTDIR}/secrets_scan.txt"; then
    secrets_match=1
  fi
else
  echo "run_config_resolved.json: missing" >> "${OUTDIR}/secrets_scan.txt"
fi

if [ -f "${OUTDIR}/report.md" ]; then
  if "${GREP_BIN}" ${GREP_ARGS} "API_KEY|SECRET|TOKEN|PASSWORD" \
    "${OUTDIR}/report.md" \
    >> "${OUTDIR}/secrets_scan.txt"; then
    secrets_match=1
  fi
else
  echo "report.md: missing" >> "${OUTDIR}/secrets_scan.txt"
fi

if [ "${secrets_match}" -ne 0 ]; then
  echo "[phase5c] ERROR: secret markers found; see ${OUTDIR}/secrets_scan.txt" >&2
  exit 1
fi

echo "[phase5c] wrote: ${OUTDIR}/secrets_scan.txt"
