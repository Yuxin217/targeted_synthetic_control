"""Input checks for panels passed to the release runner."""


def validate_panel(frame, cutoff, target, binary):
    required = {'abb', 'T', 'A', 'Y'}
    if not required <= set(frame):
        raise ValueError(f'Required panel columns: {sorted(required)}')
    if frame.duplicated(['abb', 'T']).any() or frame[list(required)].isna().any().any():
        raise ValueError('Duplicate or missing panel observations')
    if cutoff < 7 or target < cutoff:
        raise ValueError('Need at least six pre periods and target >= first post period')
    if not (frame.groupby('T').A.sum() == 1).all():
        raise ValueError('Exactly one treated unit is required at every time')
    if not frame.A.isin([0,1]).all() or (frame.groupby('abb').A.nunique() != 1).any():
        raise ValueError('A must be a fixed 0/1 indicator for each unit')
    if frame.abb.nunique() < 3 or not (frame.groupby('T').size() == frame.abb.nunique()).all():
        raise ValueError('A complete panel with at least two controls is required')
    if target not in set(frame['T']):
        raise ValueError('Target time is absent from the panel')
    if len(frame.loc[frame['T'] < cutoff, 'T'].unique()) < 6:
        raise ValueError('At least six pre-treatment periods are required')
    if binary and not frame.Y.isin([0, 1]).all():
        raise ValueError('Binary panels require Y in {0,1}')
