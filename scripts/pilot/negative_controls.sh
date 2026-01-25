#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 OUTDIR" >&2
  exit 2
fi

base_outdir="$1"
if [[ ! -d "$base_outdir" ]]; then
  echo "MISSING_BASE_OUTDIR: $base_outdir" >&2
  exit 2
fi

ts="$(date +%Y%m%d_%H%M%S)"
base_name="$(basename "$base_outdir")"
base_parent="$(dirname "$base_outdir")"

if command -v rg >/dev/null 2>&1; then
  GREP="rg"
else
  GREP="grep"
fi

gov_dir="${base_parent}/${base_name}_neg_governance_${ts}"
cp -a "$base_outdir" "$gov_dir"

gov_log="$gov_dir/governance_log.jsonl"
if [[ -f "$gov_log" ]]; then
  python - "$gov_log" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
for idx, line in enumerate(lines):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(obj, dict):
        obj["tampered"] = True
        lines[idx] = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sys.exit(0)
raise SystemExit("NO_PARSEABLE_GOVERNANCE_LINE")
PY
  set +e
  veriscope validate --strict-governance "$gov_dir" > "$gov_dir/governance_tamper_validate.txt" 2>&1
  gov_status=$?
  set -e
  if [[ $gov_status -eq 0 ]]; then
    echo "EXPECTED_GOVERNANCE_FAIL: validate succeeded on tampered log" >&2
    exit 2
  fi
else
  echo "WARNING:GOVERNANCE_LOG_MISSING" > "$gov_dir/governance_tamper_validate.txt"
fi

mismatch_dir="${base_parent}/${base_name}_neg_window_${ts}"
cp -a "$base_outdir" "$mismatch_dir"

python - "$mismatch_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ws_path = root / "window_signature.json"
data = json.loads(ws_path.read_text(encoding="utf-8"))
data["tampered"] = True
ws_path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
new_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _update(path: Path) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ref = obj.get("window_signature_ref")
    if isinstance(ref, dict):
        ref["hash"] = new_hash
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

summary = root / "results_summary.json"
if summary.exists():
    _update(summary)
results = root / "results.json"
if results.exists():
    _update(results)
PY

set +e
diff_json="$(veriscope diff "$base_outdir" "$mismatch_dir" --json 2> "$mismatch_dir/window_mismatch_diff_stderr.txt")"
diff_status=$?
set -e
echo "$diff_json" > "$mismatch_dir/window_mismatch_diff.json"
if [[ $diff_status -eq 0 ]]; then
  echo "EXPECTED_WINDOW_MISMATCH_FAIL: diff succeeded" >&2
  exit 2
fi
python - "$mismatch_dir/window_mismatch_diff.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
reason = payload.get("comparability", {}).get("reason")
if reason != "WINDOW_HASH_MISMATCH":
    raise SystemExit(f"EXPECTED_WINDOW_HASH_MISMATCH: got {reason}")
PY

partial_dir="${base_parent}/${base_name}_neg_partial_${ts}"
cp -a "$base_outdir" "$partial_dir"
rm -f "$partial_dir/results.json"

set +e
partial_out="$(veriscope diff "$base_outdir" "$partial_dir" 2> "$partial_dir/partial_diff_stderr.txt")"
partial_status=$?
set -e
echo "$partial_out" > "$partial_dir/partial_diff.txt"
if [[ $partial_status -ne 0 ]]; then
  echo "EXPECTED_PARTIAL_DIFF_SUCCESS: diff returned $partial_status" >&2
  exit 2
fi
echo "$partial_out" | $GREP -q "NOTE:PARTIAL_MODE" || {
  echo "EXPECTED_PARTIAL_NOTE_MISSING" >&2
  exit 2
}
if echo "$partial_out" | $GREP -q "counts_"; then
  echo "EXPECTED_NO_COUNTS_LINES" >&2
  exit 2
fi
