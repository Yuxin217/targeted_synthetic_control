"""Canonical simplex-constrained SCM weights."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def canonical_simplex_weights(
    donor_matrix: np.ndarray,
    treated_vector: np.ndarray,
    metric: np.ndarray | None = None,
) -> np.ndarray:
    """Solve SCM and use minimum distance to uniform as an explicit tie-break.

    The primary problem is the classical SCM pre-treatment loss on the
    probability simplex. If that loss has multiple minimizers, a second convex
    optimization chooses the minimizer closest to uniform weights. This is a
    deterministic minimum-L2-norm tie-break and does not use post-treatment
    outcomes.
    """
    donors = np.asarray(donor_matrix, dtype=float)
    treated = np.asarray(treated_vector, dtype=float).reshape(-1)
    if donors.ndim != 2 or donors.shape[0] != len(treated):
        raise ValueError("donor_matrix must have shape (periods, donors)")
    n_periods, n_donors = donors.shape
    if n_donors == 0:
        raise ValueError("SCM requires at least one donor")
    if metric is None:
        metric_array = np.eye(n_periods)
    else:
        metric_array = np.asarray(metric, dtype=float)
        if metric_array.ndim == 1:
            metric_array = np.diag(metric_array)
        if metric_array.shape != (n_periods, n_periods):
            raise ValueError("metric has incompatible dimensions")

    outcome_scale = max(float(np.std(donors)), 1e-6)
    donors = donors / outcome_scale
    treated = treated / outcome_scale
    uniform = np.full(n_donors, 1.0 / n_donors)

    def fit_loss(weights: np.ndarray) -> float:
        residual = treated - donors @ weights
        return float(residual @ metric_array @ residual / n_periods)

    def fit_gradient(weights: np.ndarray) -> np.ndarray:
        residual = treated - donors @ weights
        return -2.0 * donors.T @ metric_array @ residual / n_periods

    simplex_constraint = {
        "type": "eq",
        "fun": lambda weights: float(weights.sum() - 1.0),
        "jac": lambda weights: np.ones_like(weights),
    }
    bounds = [(0.0, 1.0)] * n_donors
    primary = minimize(
        fit_loss,
        uniform,
        jac=fit_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=simplex_constraint,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not primary.success:
        raise RuntimeError(f"Simplex SCM optimization failed: {primary.message}")

    optimum = fit_loss(primary.x)
    tolerance = max(1e-12, 1e-10 * max(1.0, abs(optimum)))

    def tie_loss(weights: np.ndarray) -> float:
        difference = weights - uniform
        return float(0.5 * difference @ difference)

    fit_constraint = {
        "type": "ineq",
        "fun": lambda weights: optimum + tolerance - fit_loss(weights),
        "jac": lambda weights: -fit_gradient(weights),
    }
    tied = minimize(
        tie_loss,
        primary.x,
        jac=lambda weights: weights - uniform,
        method="SLSQP",
        bounds=bounds,
        constraints=(simplex_constraint, fit_constraint),
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    selected = tied.x if tied.success else primary.x
    selected = np.clip(selected, 0.0, 1.0)
    return selected / selected.sum()
