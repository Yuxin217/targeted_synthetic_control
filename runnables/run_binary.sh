#!/usr/bin/env bash
# Binary synthetic experiment. Extra arguments go to the runner, e.g.
#   bash runnables/run_binary.sh --dgp linear --seeds 16000 --horizons 1
set -euo pipefail
cd "$(dirname "$0")/.."
python -m runnables.run_experiment --config configs/binary.json \
  --output "${OUTPUT:-results/runs/binary}" "$@"
