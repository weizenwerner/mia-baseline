#!/usr/bin/env bash
set -euo pipefail

# Mia-Chat launcher
# - lädt config.env, aber Launcher-ENV soll gewinnen

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

# Load config.env WITHOUT overriding existing env vars
if [[ -f "${CONFIG_FILE}" ]]; then
  while IFS= read -r line; do
    # strip comments
    line="${line%%#*}"
    line="$(echo "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    [[ "$line" != *=* ]] && continue

    key="${line%%=*}"
    val="${line#*=}"

    key="$(echo "$key" | xargs || true)"
    val="$(echo "$val" | xargs || true)"

    # remove surrounding quotes
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"

    # if already set in env -> keep it
    if [[ -z "${!key-}" ]]; then
      export "${key}=${val}"
    fi
  done < "${CONFIG_FILE}"
fi

PY="${MIA_PYTHON:-/data/mia/venv/bin/python3}"
exec "${PY}" "${SCRIPT_DIR}/main.py"

