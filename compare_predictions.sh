#!/usr/bin/env bash
# Side-by-side predicted (green) vs ground-truth (yellow) lung mask, over
# every annotated DDR frame, as one contact sheet.
#
#   ./compare_predictions.sh                  fine-tuned model (default)
#   ./compare_predictions.sh --model best_model.h5   pre-fine-tune baseline
#   ./compare_predictions.sh --case KaU001_PA_deep   one case only
#
# Any extra args are passed straight through to view_ddr_truth.py.

set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
source ./_focus_window.sh

python view_ddr_truth.py --grid --predict --thumb-dim 220 --cols 10 "$@" &
PY_PID=$!
echo ">>> Window launching -- macOS blocks background processes from grabbing focus"
echo ">>> automatically. Cmd+Tab to it (or click its 'python' Dock icon) once it appears."
activate_pid "$PY_PID"  # best-effort; often blocked by the OS, see message above
wait $PY_PID
