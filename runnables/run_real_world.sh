#!/usr/bin/env bash
# External (real-world or semi-synthetic) panel. Extra arguments go to the runner, e.g.
#   bash runnables/run_real_world.sh --input /path/to/panel.csv --first-post 19 --horizons 1 5 10
set -euo pipefail
cd "$(dirname "$0")/.."
python -m runnables.run_experiment --experiment real-world --config configs/continuous.json \
  --output "${OUTPUT:-results/runs/external}" "$@"
