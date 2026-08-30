#!/usr/bin/env bash
# Step through every annotated DDR frame one at a time with arrow keys / scroll
# wheel: predicted mask (green) vs ground truth (yellow), predicted apex point
# (filled dot) vs truth point (hollow circle), connecting line labeled with
# the error in px and mm.
#
#   ./slideshow_predictions.sh                        fine-tuned model (default)
#   ./slideshow_predictions.sh --model best_model.h5   pre-fine-tune baseline
#   ./slideshow_predictions.sh --case KaU001_PA_deep   one case only
#
# Any extra args are passed straight through to view_ddr_truth.py.

set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
source ./_focus_window.sh

python view_ddr_truth.py --predict "$@" &
PY_PID=$!
echo ">>> Window launching -- macOS blocks background processes from grabbing focus"
echo ">>> automatically. Cmd+Tab to it (or click its 'python' Dock icon) once it appears."
activate_pid "$PY_PID"  # best-effort; often blocked by the OS, see message above
wait $PY_PID
