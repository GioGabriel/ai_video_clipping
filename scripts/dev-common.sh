#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_env() {
  cd "$PROJECT_ROOT"

  if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
  fi

  APP_HOST="${APP_HOST:-127.0.0.1}"
  APP_PORT="${APP_PORT:-8000}"
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
  OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
  OLLAMA_PORT="${OLLAMA_BASE_URL##*:}"

  RUNTIME_DIR="$PROJECT_ROOT/data/runtime"
  LOG_DIR="$PROJECT_ROOT/data/logs"
  mkdir -p "$RUNTIME_DIR" "$LOG_DIR"

  API_PID_FILE="$RUNTIME_DIR/api.pid"
  OLLAMA_PID_FILE="$RUNTIME_DIR/ollama.pid"
  API_LOG_FILE="$LOG_DIR/api.log"
  OLLAMA_LOG_FILE="$LOG_DIR/ollama.log"
}

ensure_binary() {
  local binary="$1"
  if ! command -v "$binary" >/dev/null 2>&1; then
    echo "Missing required binary: $binary" >&2
    exit 1
  fi
}

ensure_file() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    echo "Missing $label at $path" >&2
    exit 1
  fi
}

read_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    tr -d '[:space:]' < "$file"
  fi
}

is_pid_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

port_listening() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for_port() {
  local port="$1"
  local label="$2"

  for _ in $(seq 1 50); do
    if port_listening "$port"; then
      return 0
    fi
    sleep 0.2
  done

  echo "$label did not start listening on port $port." >&2
  return 1
}

terminate_pid() {
  local pid="$1"
  local label="$2"

  if ! is_pid_running "$pid"; then
    return 0
  fi

  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 25); do
    if ! is_pid_running "$pid"; then
      echo "Stopped $label (pid $pid)."
      return 0
    fi
    sleep 0.2
  done

  kill -9 "$pid" 2>/dev/null || true
  if ! is_pid_running "$pid"; then
    echo "Force-stopped $label (pid $pid)."
    return 0
  fi

  echo "Failed to stop $label (pid $pid)." >&2
  return 1
}

stop_pid_from_file() {
  local pid_file="$1"
  local label="$2"
  local pid

  pid="$(read_pid_file "$pid_file")"
  if [[ -z "$pid" ]]; then
    echo "No pid file for $label."
    return 0
  fi

  terminate_pid "$pid" "$label"
  rm -f "$pid_file"
}

stop_port_listener() {
  local port="$1"
  local label="$2"
  local pids

  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "No listener found on port $port for $label."
    return 0
  fi

  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    terminate_pid "$pid" "$label on port $port"
  done <<< "$pids"
}
