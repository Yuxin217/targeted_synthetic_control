#!/usr/bin/env bash
# Continuous synthetic experiment. Extra arguments go to the runner, e.g.
#   bash runnables/run_continuous.sh --dgp linear --seeds 7000 --horizons 1
set -euo pipefail
cd "$(dirname "$0")/.."
python -m runnables.run_experiment --config configs/continuous.json \
  --output "${OUTPUT:-results/runs/continuous}" "$@"
