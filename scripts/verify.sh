#!/usr/bin/env bash
# The one command that proves the analysis still works.
#   ./scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# GitHub Actions gives us `python`; a plain macOS/Linux shell usually only has `python3`.
PY="${PYTHON:-$(command -v python || command -v python3)}"

echo "==> unit tests"
"$PY" -m unittest discover -s analysis/tests -t .

echo "==> end-to-end build against the real CSV (into a scratch dir)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
"$PY" -m analysis.report --semester all --out "$scratch/reports"
test -f "$scratch/reports/README.md"

echo "==> ok"
