#!/usr/bin/env bash

# ==============================================================================
# PDF Q&A Assistant - All-in-One Startup Script
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo " Starting PDF Q&A Assistant"
echo "=================================================="

# 1. Ensure .env exists
if [ ! -f ".env" ]; then
    echo "[Config] .env not found. Copying from .env.example..."
    cp .env.example .env
fi

# 2. Check and Start Redis Stack via Docker if available
if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        echo "[Redis] Docker daemon is running. Starting Redis Stack container..."
        docker compose up -d
    else
        echo "[Redis] Docker is installed but daemon is not running."
        echo "        The app will automatically use the high-performance in-memory semantic cache fallback."
    fi
else
    echo "[Redis] Docker not found. Using in-memory semantic cache fallback."
fi

# 3. Ensure Python Virtual Environment exists
if [ ! -d "backend/.venv" ]; then
    echo "[Backend] Creating Python virtual environment..."
    python3 -m venv backend/.venv
    ./backend/.venv/bin/pip install --upgrade pip
    ./backend/.venv/bin/pip install -r backend/requirements.txt
fi

# 4. Ensure Frontend node_modules exist
if [ ! -d "frontend/node_modules" ]; then
    echo "[Frontend] Installing npm dependencies..."
    (cd frontend && npm install)
fi

# 5. Clean shutdown handler for background processes
cleanup() {
    echo ""
    echo "=================================================="
    echo " Shutting down PDF Q&A Assistant..."
    echo "=================================================="
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "All processes stopped. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 6. Start Backend (FastAPI + Uvicorn)
echo "[Backend] Starting FastAPI server on http://localhost:8000..."
PYTHONPATH="$PROJECT_ROOT" ./backend/.venv/bin/uvicorn backend.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload &
BACKEND_PID=$!

# Wait briefly for backend to initialize
sleep 2

# 7. Start Frontend (Vite)
echo "[Frontend] Starting React Vite dev server on http://localhost:5173..."
(cd frontend && npm run dev -- --host 0.0.0.0 --port 5173) &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo " PDF Q&A Assistant is ready!"
echo "   User / Admin UI : http://localhost:5173"
echo "   Backend API Docs: http://localhost:8000/docs"
echo "   RedisInsight UI : http://localhost:8001 (if Docker active)"
echo ""
echo " Press [Ctrl+C] anytime to stop all servers."
echo "=================================================="
echo ""

# Keep script running to monitor children
wait
