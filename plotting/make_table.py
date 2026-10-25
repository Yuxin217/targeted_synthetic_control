"""Create the paper's tabular layout from saved realized-outcome errors."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DGPS = ['linear', 'hinge', 'quadratic', 'time_varying']
METHODS = ['SCM', 'Plugin', 'ASCM', 'TSC']
LABELS = ['Classical SCM', 'Plug-in estimator', 'Augmented SCM', r'\method (\textit{ours})']


def summarize(frame):
    frame = frame[frame.method.isin(METHODS)].copy()
    if 'estimand' in frame:
        frame = frame[frame.estimand == 'realized']
    if frame.empty or frame.squared_error.isna().any():
        raise ValueError('Realized squared errors are required')
    keys = ['dgp','horizon','method','seed']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate cells; filter stage/outcome/settings before aggregation')
    for key, group in frame.groupby(['dgp','horizon']):
        samples = [set(group.loc[group.method == m, 'seed']) for m in METHODS]
        if any(s != samples[0] for s in samples[1:]) or not samples[0]:
            raise ValueError(f'Methods must share evaluation seeds for {key}')
    out = frame.groupby(['dgp','horizon','method']).squared_error.agg(MSE='mean', std='std', n='count').reset_index()
    out['RMSE'] = np.sqrt(out.MSE)
    out['RMSE_SE'] = out['std']/np.sqrt(out.n)/(2*out.RMSE)
    out.loc[(out.RMSE == 0)&(out.n > 1), 'RMSE_SE'] = 0
    return out


def render(summary):
    lines = [r'\begin{tabular}{clcccc}', r'\toprule',
             r'\makecell[c]{Prediction\\horizon} & Methods & Linear & Hinge & Quadratic & Time-varying \\', r'\midrule']
    horizons = sorted(summary.horizon.unique())
    for h in horizons:
        for i, (method, label) in enumerate(zip(METHODS, LABELS)):
            cells = []
            for dgp in DGPS:
                block = summary[(summary.dgp == dgp)&(summary.horizon == h)]
                row = block[block.method == method]
                if len(row) != 1:
                    cells.append('--')
                    continue
                row = row.iloc[0]
                se = f'{row.RMSE_SE:.3f}' if pd.notna(row.RMSE_SE) else r'\mathrm{NA}'
                cell = f'{row.RMSE:.3f} \\pm {se}'
                if abs(row.RMSE-block.RMSE.min()) < 1e-8:
                    cell = r'\mathbf{' + cell + '}'
                cells.append('$'+cell+'$')
            first = rf'\multirow{{4}}{{*}}{{{h}}}' if i == 0 else ''
            lines.append(first+' & '+label+' & '+' & '.join(cells)+r' \\')
        lines.append(r'\bottomrule' if h == horizons[-1] else r'\midrule')
    return '\n'.join(lines+[r'\end{tabular}', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage')
    parser.add_argument('--outcome', choices=['continuous','binary'])
    parser.add_argument('--signal-scale', type=float)
    parser.add_argument('--rho', type=float)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    for column, value in [('stage',args.stage),('outcome',args.outcome),('signal_scale',args.signal_scale),('rho',args.rho)]:
        if value is not None:
            frame = frame[frame[column] == value]
    summary = summarize(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix('.csv').exists():
        raise FileExistsError('Choose fresh output paths to preserve archived assets')
    args.output.write_text(render(summary))
    summary.to_csv(args.output.with_suffix('.csv'), index=False)
    print(args.output)


if __name__ == '__main__':
    main()
