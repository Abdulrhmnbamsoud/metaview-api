#!/usr/bin/env bash
# MetaView 3 — Full Stack Intelligence Launcher
# Starts both Backend (FastAPI on :8000) and Frontend (Vite on :5173)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "      MetaView 3 — Global News Intelligence Platform      "
echo "=========================================================="

cleanup() {
    echo ""
    echo "Stopping MetaView 3 services..."
    kill "$BACKEND_PID" 2>/dev/null
    kill "$FRONTEND_PID" 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# 1. Start Backend FastAPI
echo "▶ Starting Backend API on http://127.0.0.1:8000..."
cd "$DIR/backend"
if [ -f "./.venv/bin/uvicorn" ]; then
    ./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
    BACKEND_PID=$!
else
    python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
    BACKEND_PID=$!
fi

# Wait for backend to be ready
sleep 2

# 2. Start Frontend Vite
echo "▶ Starting Frontend Dashboard on http://localhost:5173..."
cd "$DIR"
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "✔ MetaView 3 is running live!"
echo "  - Frontend UI:  http://localhost:5173"
echo "  - Backend API:  http://127.0.0.1:8000"
echo "  - Swagger Docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to terminate both servers."

wait
