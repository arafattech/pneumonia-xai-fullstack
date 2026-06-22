#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

usage() {
  cat <<EOF
Usage: ./run.sh [MODE]

Modes:
  docker      Run full stack via Docker Compose (default)
  local       Run backend + frontend locally (no Docker)
  backend     Run only backend (local)
  frontend    Run only frontend (local)
  stop        Stop Docker containers
  logs        Tail Docker logs
  help        Show this message
EOF
}

run_docker() {
  command -v docker &>/dev/null || error "Docker not installed"
  info "Starting full stack with Docker Compose..."
  docker compose -f "$ROOT/docker-compose.yml" up --build
}

stop_docker() {
  info "Stopping containers..."
  docker compose -f "$ROOT/docker-compose.yml" down
}

logs_docker() {
  docker compose -f "$ROOT/docker-compose.yml" logs -f
}

run_backend() {
  info "Starting backend on http://localhost:8000"
  cd "$BACKEND"
  if [ -d venv ]; then
    source venv/bin/activate
  elif command -v python3 &>/dev/null; then
    warn "No venv found — using system python3"
  else
    error "python3 not found"
  fi
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

run_frontend() {
  info "Starting frontend on http://localhost:5173"
  cd "$FRONTEND"
  command -v npm &>/dev/null || error "npm not installed"
  npm run dev
}

run_local() {
  info "Starting backend and frontend locally..."

  run_backend &
  BACKEND_PID=$!

  sleep 2

  run_frontend &
  FRONTEND_PID=$!

  trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; info 'Stopped.'" EXIT INT TERM
  wait
}

MODE="${1:-local}"

case "$MODE" in
  docker)    run_docker ;;
  local)     run_local ;;
  backend)   run_backend ;;
  frontend)  run_frontend ;;
  stop)      stop_docker ;;
  logs)      logs_docker ;;
  help|-h|--help) usage ;;
  *) error "Unknown mode: $MODE. Use './run.sh help'" ;;
esac
