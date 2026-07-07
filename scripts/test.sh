#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python at $PYTHON_BIN" >&2
  echo "Create the project venv and install dependencies before running tests." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m pytest "$@"
