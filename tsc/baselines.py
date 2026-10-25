"""SCM solver and shared-nuisance baseline predictions."""
from .scm_utils import canonical_simplex_weights


def _simplex_sc_weights(y_treated_pre, y_donor_pre):
    return canonical_simplex_weights(y_donor_pre, y_treated_pre)


def estimates(result):
    """These baseline estimates are computed with the same fit as TSC."""
    return {'SCM': result.sc_estimate, 'Plugin': result.plugin_estimate,
            'ASCM': result.ascm_estimate, 'TSC': result.estimate}
