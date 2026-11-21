#!/usr/bin/env bash
# DeepMisconception: Verification & Evaluation Pipeline Runner
set -e

echo "================================================================================"
echo "DEEPMISCONCEPTION: VERIFICATION AND REPRODUCIBILITY PIPELINE"
echo "================================================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${ROOT_DIR}/backend/venv/bin/python"
PYTEST="${ROOT_DIR}/backend/venv/bin/pytest"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Virtual environment python not found at $PYTHON"
    exit 1
fi

echo -e "\n[STEP 1/3] Running Complete Backend Test Suite (157 tests)..."
PYTHONPATH=backend "$PYTEST" backend/tests/ -q

echo -e "\n[STEP 2/3] Generating Clean Publication-Grade Figures..."
"$PYTHON" scripts/generate_all_clean_figures.py

echo -e "\n[STEP 3/3] Verifying Output Plots..."
echo "  - Generated Plots in DeepMisconception/plots/: $(ls DeepMisconception/plots/*.png | wc -l | tr -d ' ') files"
echo "  - Generated Evaluation Figures in figures/evaluation/: $(ls figures/evaluation/*.png | wc -l | tr -d ' ') files"

echo -e "\n================================================================================"
echo "[SUCCESS] ALL TESTS AND EVALUATION ARTIFACTS VERIFIED SUCCESSFULLY!"
echo "================================================================================"
