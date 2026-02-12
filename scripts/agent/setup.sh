#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install ruff pytest
echo "agent setup complete"
