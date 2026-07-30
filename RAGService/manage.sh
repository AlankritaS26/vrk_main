#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# manage.sh  —  Start / stop / status the RAGService API
# Usage:
#   ./manage.sh start    Start the API (port 8600 by default)
#   ./manage.sh stop     Stop the API
#   ./manage.sh restart  Stop then start
#   ./manage.sh status   Show PID and URL
#   ./manage.sh logs     Tail the log
#   PORT=8700 ./manage.sh start   Run on a custom port
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PID_FILE="$SCRIPT_DIR/.ragservice.pid"
LOG_FILE="$SCRIPT_DIR/.ragservice.log"
PORT="${PORT:-8600}"

_check_venv() {
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "❌  venv not found. Run:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        exit 1
    fi
}

_is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid; pid=$(<"$PID_FILE")
        kill -0 "$pid" 2>/dev/null && return 0
    fi
    local live; live=$(pgrep -f "uvicorn app:app" 2>/dev/null | head -1 || true)
    [[ -n "$live" ]] && { echo "$live" > "$PID_FILE"; return 0; }
    return 1
}

cmd_start() {
    _check_venv
    if _is_running; then
        echo "⚠️   Already running (PID $(<"$PID_FILE")) → http://localhost:$PORT"
        return
    fi
    [[ ! -f "$SCRIPT_DIR/.env" ]] && cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env" && echo "ℹ️   Created .env from .env.example — edit it if needed."
    echo "🚀  Starting RAGService on port $PORT …"
    nohup "$VENV/bin/uvicorn" app:app \
        --host 0.0.0.0 --port "$PORT" --workers 1 \
        > "$LOG_FILE" 2>&1 &
    local pid=$!
    sleep 2
    local actual; actual=$(pgrep -f "uvicorn app:app" 2>/dev/null | head -1 || echo "$pid")
    echo "$actual" > "$PID_FILE"
    if kill -0 "$actual" 2>/dev/null; then
        echo "✅  Started  (PID $actual)"
        echo "    API    : http://localhost:$PORT"
        echo "    Docs   : http://localhost:$PORT/docs"
        echo "    Health : http://localhost:$PORT/health"
        echo "    Logs   : $LOG_FILE"
    else
        rm -f "$PID_FILE"; echo "❌  Failed to start. Check $LOG_FILE"; tail -20 "$LOG_FILE"; exit 1
    fi
}

cmd_stop() {
    if ! _is_running; then echo "ℹ️   Not running."; rm -f "$PID_FILE"; return; fi
    local pid; pid=$(<"$PID_FILE")
    echo "🛑  Stopping (PID $pid) …"
    kill "$pid" 2>/dev/null || true
    for i in {1..10}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "✅  Stopped."
}

cmd_status() {
    if _is_running; then
        echo "✅  Running  (PID $(<"$PID_FILE"))  →  http://localhost:$PORT"
        echo "    Docs: http://localhost:$PORT/docs"
    else
        echo "⛔  Not running."
    fi
}

cmd_logs() {
    [[ ! -f "$LOG_FILE" ]] && { echo "No log file yet."; exit 1; }
    echo "📋  Tailing $LOG_FILE  (Ctrl+C to stop)"
    tail -f "$LOG_FILE"
}

UI_PID_FILE="$SCRIPT_DIR/.ragui.pid"
UI_LOG_FILE="$SCRIPT_DIR/.ragui.log"
UI_PORT="${UI_PORT:-8601}"

cmd_ui_start() {
    _check_venv
    if [[ -f "$UI_PID_FILE" ]]; then
        local pid; pid=$(<"$UI_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "⚠️   UI already running (PID $pid) → http://localhost:$UI_PORT"
            return
        fi
    fi
    echo "🖥   Starting RAGService UI on port $UI_PORT …"
    nohup "$VENV/bin/streamlit" run "$SCRIPT_DIR/ui.py" \
        --server.port "$UI_PORT" --server.headless true \
        > "$UI_LOG_FILE" 2>&1 &
    sleep 2
    local pid; pid=$(pgrep -f "streamlit run.*ui.py" 2>/dev/null | head -1 || true)
    if [[ -n "$pid" ]]; then
        echo "$pid" > "$UI_PID_FILE"
        echo "✅  UI Started  (PID $pid)"
        echo "    URL  : http://localhost:$UI_PORT"
        echo "    Logs : $UI_LOG_FILE"
    else
        echo "❌  UI failed to start. Check $UI_LOG_FILE"
        tail -10 "$UI_LOG_FILE"
    fi
}

cmd_ui_stop() {
    if [[ -f "$UI_PID_FILE" ]]; then
        local pid; pid=$(<"$UI_PID_FILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$UI_PID_FILE"
    fi
    pkill -f "streamlit run.*ui.py" 2>/dev/null || true
    echo "✅  UI stopped."
}

case "${1:-}" in
    start)    cmd_start      ;;
    stop)     cmd_stop       ;;
    restart)  cmd_stop; sleep 1; cmd_start ;;
    status)   cmd_status     ;;
    logs)     cmd_logs       ;;
    ui)
        case "${2:-start}" in
            start)  cmd_ui_start ;;
            stop)   cmd_ui_stop  ;;
            *)      echo "Usage: $0 ui [start|stop]" ;;
        esac
        ;;
    all)
        # Start both API and UI
        cmd_start
        cmd_ui_start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|ui|all}"
        echo ""
        echo "  API control:"
        echo "    start         Start the FastAPI service (port \$PORT, default 8600)"
        echo "    stop          Stop the API"
        echo "    restart       Stop then start the API"
        echo "    status        Show API running state"
        echo "    logs          Tail the API log"
        echo ""
        echo "  Web UI:"
        echo "    ui [start]    Start the Streamlit admin UI (port \$UI_PORT, default 8601)"
        echo "    ui stop       Stop the UI"
        echo ""
        echo "  Shortcuts:"
        echo "    all           Start both API and UI"
        echo ""
        echo "  Examples:"
        echo "    PORT=8700 UI_PORT=8701 ./manage.sh all"
        exit 1 ;;
esac
