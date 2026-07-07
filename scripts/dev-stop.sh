#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/dev-common.sh"
load_env

FORCE_PORTS=0
if [[ "${1:-}" == "--all" ]]; then
  FORCE_PORTS=1
fi

stop_pid_from_file "$API_PID_FILE" "FastAPI"
stop_pid_from_file "$OLLAMA_PID_FILE" "Ollama"

if [[ "$FORCE_PORTS" == "1" ]]; then
  stop_port_listener "$APP_PORT" "FastAPI"
  stop_port_listener "$OLLAMA_PORT" "Ollama"
fi
