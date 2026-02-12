#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCOPE="${VERISCOPE_AGENT_CHECK_SCOPE:-fast}"

run() {
  echo "+ $*"
  "$@"
}

vs() {
  if command -v veriscope >/dev/null 2>&1; then
    veriscope "$@"
  else
    python -m veriscope.cli.main "$@"
  fi
}

run ruff check .

if [[ "$SCOPE" == "lint" ]]; then
  echo "agent check completed (scope=$SCOPE)"
  exit 0
elif [[ "$SCOPE" == "full" ]]; then
  run python -m pytest -q
elif [[ "$SCOPE" == "gc" ]]; then
  run python -m pytest -q \
    tests/test_cli_import_boundary.py \
    tests/test_markdown_fences.py \
    tests/test_file_modes.py
else
  run python -m pytest -q \
    tests/test_cli_import_boundary.py \
    tests/test_cli_diff.py \
    tests/test_cli_validate_report.py \
    tests/test_markdown_fences.py \
    tests/test_file_modes.py
fi

run vs validate docs/examples/reviewer_packet/run_a

REPORT_FILE="$(mktemp)"
echo "+ vs report docs/examples/reviewer_packet/run_a --format text > $REPORT_FILE"
vs report docs/examples/reviewer_packet/run_a --format text > "$REPORT_FILE"
if [[ ! -s "$REPORT_FILE" ]]; then
  echo "report output is empty" >&2
  exit 1
fi

run vs diff docs/examples/reviewer_packet/run_a docs/examples/reviewer_packet/run_b

echo "agent check completed (scope=$SCOPE)"
