"""Serial release runner. New outputs never replace archived results."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import pandas as pd
from threadpoolctl import threadpool_limits
from dgp.dgp import PanelConfig, generate_panel
from tsc import tsc_estimator
from tsc.baselines import estimates
from tsc.utils import validate_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['synthetic','semi-synthetic','real-world'], default='synthetic')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/continuous.json')
    parser.add_argument('--dgp', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--horizons', type=int, nargs='+', default=None)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--first-post', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    binary = cfg['outcome'] == 'binary'
    if cfg['estimator'].get('binary_model') != 'ridge':
        parser.error('Release runner requires Ridge (unbounded for binary outcomes)')
    if args.experiment != 'synthetic' and (args.input is None or args.first_post is None):
        parser.error('External panels require --input and --first-post')
    horizons = args.horizons or cfg['horizons']
    if min(horizons) < 1:
        parser.error('Horizons must be positive')
    dgps = [args.dgp] if args.dgp else list(cfg['dgps'])
    if any(d not in cfg['dgps'] for d in dgps):
        parser.error('DGP not in config')
    args.output.mkdir(parents=True, exist_ok=False)
    panel_cfg = PanelConfig(**cfg['panel'])
    jobs = [(d, s) for d in dgps for s in (args.seeds or cfg['dgps'][d].get('seeds', cfg['seeds']))] if args.experiment == 'synthetic' else [('external', None)]
    rows, diagnostics, weights = [], [], []
    with threadpool_limits(limits=1):
        for dgp, seed in jobs:
            overrides = cfg['dgps'].get(dgp, {})
            options = {**cfg['estimator'], **overrides.get('estimator', {})}
            if options.get('binary_model') != 'ridge':
                raise ValueError('Per-DGP overrides must retain Ridge nuisance regression')
            if args.experiment == 'synthetic':
                if max(horizons) > panel_cfg.max_horizon:
                    raise ValueError('Requested horizon exceeds configured panel length')
                frame = generate_panel(dgp, cfg['outcome'], seed, config=panel_cfg)
                cutoff = panel_cfg.last_pre_time + 1
            else:
                frame = pd.read_csv(args.input, dtype={'abb': str})
                cutoff = args.first_post
            panel_path = args.output/f'{dgp}_{seed}_panel.csv'
            frame.to_csv(panel_path, index=False)
            for h in horizons:
                target = cutoff+h-1
                validate_panel(frame, cutoff, target, binary)
                fit = tsc_estimator(str(panel_path), cutoff, target, binary, **options)
                treated = frame[(frame.A == 1)&(frame['T'] == target)].iloc[0]
                truth = float(treated.Y0_true) if 'Y0_true' in frame else None
                key = dict(dgp=dgp, seed=seed, horizon=h, outcome=cfg['outcome'])
                for method, value in estimates(fit).items():
                    rows.append({**key, 'method': method, 'estimate': value, 'truth': truth,
                                 'estimand': 'realized' if truth is not None else 'unobserved',
                                 'squared_error': (value-truth)**2 if truth is not None else None})
                diagnostics.append({**key, 'alpha': fit.shrinkage_alpha, 'root_exists': fit.root_exists,
                                    'moment_before': fit.moment_before, 'moment_after': fit.moment_after,
                                    'validation_losses': json.dumps(fit.validation_losses)})
                donors = sorted(frame.loc[(frame['T']==target)&(frame.A==0), 'abb'].astype(str))
                for donor, initial, final in zip(donors, fit.w0, fit.w_star):
                    weights.append({**key, 'donor': donor, 'w0': initial, 'w_star': final})
            print(f'Completed {dgp}, seed={seed}, horizons={horizons}', flush=True)
    pd.DataFrame(rows).to_csv(args.output/'details.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(args.output/'diagnostics.csv', index=False)
    pd.DataFrame(weights).to_csv(args.output/'weights.csv', index=False)
    manifest = dict(config=cfg, experiment=args.experiment, jobs=jobs, horizons=horizons,
                    metric='Realized untreated squared error; missing for real data without counterfactual truth',
                    threads=1, selection='All methods evaluated on identical supplied seeds',
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
