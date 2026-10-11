#!/usr/bin/env bash
# One command to a running Glassbox on http://localhost:8000, no Kaggle download:
#   ./scripts/demo.sh
# Trains a demo model on synthetic applicants, builds the web app and serves both.
# Needs Python 3.11+ and Node 20+ (on macOS also `brew install libomp` for LightGBM).
# Docker alternative: docker build -t glassbox . && docker run -p 8000:8000 glassbox
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8000}"

"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "Glassbox needs Python 3.11 or newer (found $("$PYTHON" --version))."; exit 1; }

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv ..."
  "$PYTHON" -m venv .venv
fi
echo "Installing Python dependencies ..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e .

.venv/bin/python -c 'import lightgbm' 2>/dev/null \
  || { echo "LightGBM can't load. On macOS, run: brew install libomp"; exit 1; }

.venv/bin/python -m credit_risk.demo

echo "Building the web app ..."
(cd web && npm ci --silent && npm run build --silent)

echo "Glassbox demo on http://localhost:$PORT (Ctrl+C to stop)"
GLASSBOX_ARTIFACTS_DIR=artifacts/demo exec .venv/bin/uvicorn api.serve:app --port "$PORT"
