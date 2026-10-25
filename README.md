# Targeted Synthetic Control

This repository contains the implementation of TSC, the SCM, Plug-in estimator, and ASCM baselines, the synthetic data generators, and the saved results for the paper. Run all commands from the repository root.

## Abstract
The *synthetic control method* (SCM) estimates causal effects in panel data with a single treated unit by constructing a counterfactual outcome as a weighted combination of untreated control units that matches the pre-treatment trajectory. In this paper, we introduce the **targeted synthetic control (TSC)** method, a new two-stage estimator that directly estimates the counterfactual outcome. Specifically, our TSC method (1) yields a targeted debiasing estimator, in the sense that the targeted update refines the initial weights to produce more stable weights; and (2) ensures that the final counterfactual estimate is a convex combination of observed control outcomes, which enables direct interpretation of the synthetic control weights. TSC is flexible and can be instantiated with arbitrary machine learning models. Methodologically, TSC starts from an initial set of synthetic control weights and applies a one-dimensional targeted update through a weight-tilting submodel, which calibrates the weights to reduce the bias in weight estimation that arises from pre-treatment fit. Furthermore, TSC avoids key shortcomings of existing methods (e.g., the augmented SCM), which can produce unbounded counterfactual estimates. Across extensive synthetic and real-world experiments, TSC consistently improves estimation accuracy over state-of-the-art SCM baselines.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Experiments run serially with a single numerical thread.

## Organization

```text
tsc/                 TSC estimator, nuisance fitting, and baselines
dgp/                 Synthetic data generators (continuous and binary)
configs/             Experiment settings for each outcome type and DGP
runnables/           Experiment runner and shell scripts
plotting/            Scripts for paper tables and figures
results/             Saved results and figures from the paper
```

## Synthetic experiments

```bash
bash runnables/run_continuous.sh
bash runnables/run_binary.sh
```

Each script evaluates all four methods on the same seeds and writes the output
to a new directory (`results/runs/continuous` or `results/runs/binary` by
default; set `OUTPUT` to change it). Errors are computed against the realized
untreated outcomes. Extra arguments are passed to `runnables/run_experiment.py`,
for example:

```bash
OUTPUT=results/runs/binary_smoke bash runnables/run_binary.sh \
  --dgp linear --seeds 15000 --horizons 1
```

Arguments that are not given (`--dgp`, `--seeds`, `--horizons`) are taken from
the configuration file. In all settings, SCM uses the full pre-treatment
history, and the shrinkage `alpha` is selected on pre-treatment pseudo-post
periods, except for binary Quadratic, which fixes `alpha = 1`. The binary DGPs
use DGP-specific nuisance settings and seeds, as listed in
`configs/binary.json`.

Each run writes `details.csv`, `diagnostics.csv`, `weights.csv`, the generated
panels, and a `manifest.json`. Saved results in `results/` are never
overwritten.

## External panels

For real-world or semi-synthetic data, provide a CSV in the format described in
`data/README.md`:

```bash
bash runnables/run_real_world.sh --input /path/to/panel.csv \
  --first-post 19 --horizons 1 5 10
```

`--first-post` is the first post-treatment period. If the panel has no
`Y0_true` column, only predictions are saved. The external datasets are not
included in this repository.
