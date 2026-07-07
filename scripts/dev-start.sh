#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/dev-common.sh"
load_env

ensure_binary ollama
ensure_binary lsof
ensure_file "$PROJECT_ROOT/.venv/bin/uvicorn" "project uvicorn executable"

if port_listening "$OLLAMA_PORT"; then
  echo "Ollama already listening on port $OLLAMA_PORT."
else
  nohup ollama serve > "$OLLAMA_LOG_FILE" 2>&1 &
  echo $! > "$OLLAMA_PID_FILE"
  wait_for_port "$OLLAMA_PORT" "Ollama"
  echo "Started Ollama on port $OLLAMA_PORT."
fi

if port_listening "$APP_PORT"; then
  echo "FastAPI already listening on port $APP_PORT."
else
  nohup "$PROJECT_ROOT/.venv/bin/uvicorn" src.main:app --host "$APP_HOST" --port "$APP_PORT" > "$API_LOG_FILE" 2>&1 &
  echo $! > "$API_PID_FILE"
  wait_for_port "$APP_PORT" "FastAPI"
  echo "Started FastAPI on http://$APP_HOST:$APP_PORT."
fi

if [[ "$OLLAMA_MODEL" == *:* ]]; then
  MODEL_MATCH_PATTERN="$OLLAMA_MODEL"
else
  MODEL_MATCH_PATTERN="${OLLAMA_MODEL}(:latest)?"
fi

if ollama list 2>/dev/null | awk 'NR > 1 {print $1}' | grep -E -x -q "$MODEL_MATCH_PATTERN"; then
  echo "Ollama model '$OLLAMA_MODEL' is installed."
else
  echo "Warning: Ollama model '$OLLAMA_MODEL' is not installed. Run: ollama pull $OLLAMA_MODEL"
fi

echo
echo "Dashboard: http://$APP_HOST:$APP_PORT/"
echo "Docs:      http://$APP_HOST:$APP_PORT/docs"
echo "Logs:      tail -f \"$API_LOG_FILE\" \"$OLLAMA_LOG_FILE\""
