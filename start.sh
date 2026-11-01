#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "=========================================================="
echo " Starting DeepMisconception Academic Research Platform "
echo "=========================================================="

# Trap exits to kill background processes cleanly
cleanup() {
  echo ""
  echo "Shutting down servers..."
  kill $(jobs -p) 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Backend FastAPI Server
echo "1. Launching Backend Server on http://127.0.0.1:8000..."
cd "$DIR/backend"
source venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
sleep 1

# 2. Start Frontend Vite Dev Server
echo "2. Launching Frontend Web App on http://localhost:3000..."
cd "$DIR/frontend"
npm run dev
