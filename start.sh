#!/usr/bin/env bash
set -e

# ── Configure your model here ──────────────────────────────────────────────
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.3:latest}"
# export OLLAMA_BASE_URL="http://localhost:11434/v1"
# ───────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
  echo "Python 3 not found. Install from https://python.org"
  exit 1
fi

python3 -m pip install -q -r requirements.txt
python3 main.py "$@"
