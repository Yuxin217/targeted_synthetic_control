"""Stabilized cross-fitted TSC for point counterfactual estimation.

This implementation is deliberately separate from the original TSC:

1. m_t is fitted cross-sectionally on control outcomes at the target time,
   as defined in the paper, rather than extrapolated from pooled pre-periods.
2. Donor residuals are out-of-fold predictions.
3. The exponential tilt uses the residual direction appearing in Eq. (10), so
   the derivative of the log-partition function is the targeted moment.
4. A pre-treatment pseudo-post validation chooses how much of the raw residual
   tilt to apply. Alpha=0 (classical SCM) is always a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .baselines import _simplex_sc_weights
from .nuisance import _target_time_nuisance


@dataclass
class TSCResult:
    estimate: float
    plugin_estimate: float
    ascm_estimate: float
    sc_estimate: float
    w0: np.ndarray
    w_star: np.ndarray
    raw_epsilon: float
    selected_path: str
    shrinkage_alpha: float
    moment_before: float
    moment_after: float
    effective_sample_size_before: float
    effective_sample_size_after: float
    validation_losses: dict[str, float]
    root_exists: bool
    nuisance_oof_rmse: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()










@lru_cache(maxsize=128)
def _cached_history_weights(
    shape: tuple[int, int], donor_bytes: bytes, treated_bytes: bytes, order: str,
) -> np.ndarray:
    """Memoize the deterministic fit by exact pre-outcome arrays, not file IDs.

    Nuisance history/model ablations share identical SCM fits. Neither target
    outcomes nor regression features enter this cache. Callers receive copies.
    """
    # Retain the original memory order: otherwise BLAS summation order can
    # perturb the numerical tie-break even though matrix values are identical.
    donors = np.array(np.frombuffer(donor_bytes, dtype=np.float64).reshape(shape),
                      order=order, copy=True)
    treated = np.frombuffer(treated_bytes, dtype=np.float64)
    return _simplex_sc_weights(treated, donors)


def _initial_weights(
    df: pd.DataFrame, cutoff: int, donor_units: list[str]
) -> tuple[np.ndarray, float, np.ndarray]:
    times = sorted(df.loc[df["T"] < cutoff, "T"].unique())
    donor_history = (
        df[(df["T"] < cutoff) & (df["A"] == 0)]
        .pivot(index="T", columns="abb", values="Y")
        .reindex(index=times, columns=donor_units)
        .to_numpy(dtype=float)
    )
    treated_history = (
        df[(df["T"] < cutoff) & (df["A"] == 1)]
        .set_index("T")
        .reindex(times)["Y"]
        .to_numpy(dtype=float)
    )
    weights = _cached_history_weights(
        donor_history.shape, donor_history.astype(np.float64).tobytes(),
        treated_history.astype(np.float64).tobytes(),
        "F" if donor_history.flags.f_contiguous else "C",
    ).copy()
    return weights, float(weights @ donor_history[-1]), donor_history


def _raw_residual_tilt(
    w0: np.ndarray,
    residual: np.ndarray,
    epsilon_bound: float,
) -> tuple[float, bool, np.ndarray, np.ndarray]:
    floor = 1e-10
    w_base = np.maximum(w0, floor)
    w_base /= w_base.sum()
    # The fluctuation score is exactly S_j = Y_jT - m_hat_T(X_j).
    direction = residual.copy()
    log_base = np.log(w_base)

    def weights(epsilon: float) -> np.ndarray:
        return _softmax(log_base + epsilon * direction)

    def moment(epsilon: float) -> float:
        return float(weights(epsilon) @ residual)

    moment_left = moment(-epsilon_bound)
    moment_right = moment(epsilon_bound)
    root_exists = moment_left <= 0.0 <= moment_right
    if root_exists:
        epsilon = float(
            brentq(moment, -epsilon_bound, epsilon_bound, xtol=1e-10)
        )
    else:
        epsilon = 0.0
    return epsilon, root_exists, w_base, direction


def _weights_at_shrinkage(
    w_base: np.ndarray,
    direction: np.ndarray,
    raw_epsilon: float,
    alpha: float,
) -> np.ndarray:
    return _softmax(np.log(w_base) + alpha * raw_epsilon * direction)


@dataclass
class _CandidateSet:
    estimates: dict[str, float]
    weights: dict[str, np.ndarray]
    donor_units: list[str]
    w0: np.ndarray
    donor_y: np.ndarray
    mu_target: float
    mu_donor_full: np.ndarray
    residual: np.ndarray
    residual_epsilon: float
    residual_root_exists: bool


def _single_time_candidates(
    df: pd.DataFrame,
    target_t: int,
    history_cutoff: int,
    binary: bool,
    ridge_alpha: float,
    n_folds: int,
    epsilon_bound: float,
    alpha_grid: tuple[float, ...],
    history_window: int | None = None,
    binary_model: str = "logistic",
    history_representation: str = "full",
) -> _CandidateSet:
    donor_units, mu_donor, mu_donor_full, mu_target, donor_y = (
        _target_time_nuisance(
            df,
            target_t=target_t,
            history_cutoff=history_cutoff,
            binary=binary,
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            return_full_donor=True,
            history_window=history_window,
            binary_model=binary_model,
            history_representation=history_representation,
        )
    )
    w0, _, _ = _initial_weights(df, history_cutoff, donor_units)
    residual = donor_y - mu_donor
    epsilon, root_exists, w_base, direction = _raw_residual_tilt(
        w0, residual, epsilon_bound
    )
    estimates: dict[str, float] = {}
    candidate_weights: dict[str, np.ndarray] = {}
    for alpha in alpha_grid:
        residual_weights = _weights_at_shrinkage(
            w_base, direction, epsilon, float(alpha)
        )
        key = f"residual:{float(alpha):g}"
        estimates[key] = float(residual_weights @ donor_y)
        candidate_weights[key] = residual_weights
    return _CandidateSet(
        estimates=estimates,
        weights=candidate_weights,
        donor_units=donor_units,
        w0=w0,
        donor_y=donor_y,
        mu_target=mu_target,
        mu_donor_full=mu_donor_full,
        residual=residual,
        residual_epsilon=epsilon,
        residual_root_exists=root_exists,
    )


def tsc_estimator(
    file_path: str,
    T0: int,
    T_lagged: int,
    binary: bool,
    ridge_alpha: float = 5.0,
    n_folds: int = 5,
    n_validation_periods: int = 6,
    epsilon_bound: float = 20.0,
    alpha_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    history_window: int | None = None,
    binary_model: str = "logistic",
    history_representation: str = "full",
) -> TSCResult:
    """Estimate a point counterfactual with stabilized cross-fitted TSC.

    history_window restricts ONLY nuisance features (including all summaries)
    to the last k observed periods before each actual/pseudo history cutoff.
    SCM always uses the complete available pre-treatment history. binary_model
    'ridge' fits 0/1 outcomes by squared loss WITHOUT prediction clipping.
    history_representation='summaries' keeps only the five history summaries;
    the default 'full' preserves historical behavior and all SCM inputs.
    """
    if history_window is not None and history_window < 1:
        raise ValueError("history_window must be positive or None")
    if binary_model not in {"logistic", "ridge"}:
        raise ValueError("binary_model must be 'logistic' or 'ridge'")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2 for cross-fitting")
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one shrinkage value")
    df = pd.read_csv(file_path)
    if "abb" not in df:
        df["abb"] = df.groupby("T").cumcount().astype(str)
    df = df.sort_values(["T", "abb"]).reset_index(drop=True)
    if T_lagged < T0:
        raise ValueError("T_lagged must be a post-treatment period")

    horizon = T_lagged - T0 + 1
    validation_start = max(5 + horizon - 1, T0 - n_validation_periods)
    candidate_keys = [
        f"residual:{float(alpha):g}"
        for alpha in alpha_grid
    ]
    validation_losses = {key: [] for key in candidate_keys}
    for pseudo_t in range(validation_start, T0):
        pseudo_cutoff = pseudo_t - horizon + 1
        candidate_set = _single_time_candidates(
            df,
            target_t=pseudo_t,
            history_cutoff=pseudo_cutoff,
            binary=binary,
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            epsilon_bound=epsilon_bound,
            alpha_grid=alpha_grid,
            history_window=history_window,
            binary_model=binary_model,
            history_representation=history_representation,
        )
        truth = float(
            df[(df["T"] == pseudo_t) & (df["A"] == 1)]["Y"].iloc[0]
        )
        for key in candidate_keys:
            validation_losses[key].append(
                (candidate_set.estimates[key] - truth) ** 2
            )

    mean_validation_losses = {
        key: float(np.mean(losses))
        for key, losses in validation_losses.items()
    }
    chosen_key = min(
        candidate_keys,
        key=lambda key: (
            mean_validation_losses[key],
            float(key.split(":")[1]),
            key,
        ),
    )
    selected_path, alpha_text = chosen_key.split(":")
    chosen_alpha = float(alpha_text)

    candidate_set = _single_time_candidates(
        df,
        target_t=T_lagged,
        history_cutoff=T0,
        binary=binary,
        ridge_alpha=ridge_alpha,
        n_folds=n_folds,
        epsilon_bound=epsilon_bound,
        alpha_grid=alpha_grid,
        history_window=history_window,
        binary_model=binary_model,
        history_representation=history_representation,
    )
    w_star = candidate_set.weights[chosen_key]
    estimate = candidate_set.estimates[chosen_key]
    epsilon = candidate_set.residual_epsilon
    root_exists = candidate_set.residual_root_exists
    sc_estimate = float(candidate_set.w0 @ candidate_set.donor_y)
    ascm_estimate = float(
        candidate_set.mu_target
        + candidate_set.w0
        @ (candidate_set.donor_y - candidate_set.mu_donor_full)
    )
    return TSCResult(
        estimate=estimate,
        plugin_estimate=candidate_set.mu_target,
        ascm_estimate=ascm_estimate,
        sc_estimate=sc_estimate,
        w0=candidate_set.w0,
        w_star=w_star,
        raw_epsilon=epsilon,
        selected_path=selected_path,
        shrinkage_alpha=float(chosen_alpha),
        moment_before=float(candidate_set.w0 @ candidate_set.residual),
        moment_after=float(w_star @ candidate_set.residual),
        effective_sample_size_before=float(
            1.0 / np.sum(candidate_set.w0**2)
        ),
        effective_sample_size_after=float(1.0 / np.sum(w_star**2)),
        validation_losses=mean_validation_losses,
        root_exists=root_exists,
        nuisance_oof_rmse=float(
            np.sqrt(np.mean(candidate_set.residual**2))
        ),
    )
