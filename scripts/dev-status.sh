#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/dev-common.sh"
load_env

show_service_status() {
  local label="$1"
  local pid_file="$2"
  local port="$3"
  local pid

  pid="$(read_pid_file "$pid_file")"
  if is_pid_running "$pid"; then
    echo "$label: running (pid $pid, port $port)"
    return
  fi

  if port_listening "$port"; then
    echo "$label: listening on port $port (external process, no active pid file)"
    return
  fi

  echo "$label: stopped"
}

show_service_status "Ollama" "$OLLAMA_PID_FILE" "$OLLAMA_PORT"
show_service_status "FastAPI" "$API_PID_FILE" "$APP_PORT"

echo "Dashboard: http://$APP_HOST:$APP_PORT/"
echo "Docs:      http://$APP_HOST:$APP_PORT/docs"
