"""Target-time nuisance fitting (Ridge, or Logistic for binary outcomes)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

def _history_features(
    df: pd.DataFrame,
    units: list[str],
    history_cutoff: int,
    history_window: int | None = None,
    history_representation: str = "full",
) -> np.ndarray:
    if history_representation not in {"full", "summaries"}:
        raise ValueError("Unknown history representation")
    history_times = sorted(df.loc[df["T"] < history_cutoff, "T"].unique())
    if history_window is not None:
        if history_window < 1:
            raise ValueError("history_window must be positive or None")
        history_times = history_times[-history_window:]
    history = (
        df[df["T"] < history_cutoff]
        .pivot(index="abb", columns="T", values="Y")
        .reindex(index=units, columns=history_times)
        .to_numpy(dtype=float)
    )
    if history.shape[1] == 0:
        raise ValueError("At least one history period is required")
    time = np.arange(history.shape[1], dtype=float)
    centered_time = time - time.mean()
    denominator = max(float(centered_time @ centered_time), 1.0)
    slopes = (history @ centered_time / denominator).reshape(-1, 1)
    summaries = np.column_stack(
        [
            history.mean(axis=1),
            history.std(axis=1),
            history[:, -1],
            history[:, -min(3, history.shape[1]) :].mean(axis=1),
        ]
    )
    if history_representation == "summaries":
        return np.column_stack([summaries, slopes])
    return np.column_stack([history, summaries, slopes])


def _target_time_nuisance(
    df: pd.DataFrame,
    target_t: int,
    history_cutoff: int,
    binary: bool,
    ridge_alpha: float,
    n_folds: int,
    cross_fit_donors: bool = True,
    return_full_donor: bool = False,
    history_window: int | None = None,
    binary_model: str = "logistic",
    history_representation: str = "full",
) -> tuple:
    """Cross-fit target-time donor predictions and fit the treated prediction.

    Donors are split across units.  For each fold, the outcome regression is
    trained on the other donors' target-time outcomes, then predicts the held-
    out donors. Consequently each TSC residual ``Y_j - m_hat_j`` never uses
    donor j's own target-time outcome to fit ``m_hat_j``. A second model fitted
    on every donor predicts the treated unit and produces the in-sample donor
    predictions used only by ASCM.
    """
    target_rows = df[df["T"] == target_t].sort_values("abb")
    donors = target_rows[target_rows["A"] == 0]
    treated = target_rows[target_rows["A"] == 1]
    if len(treated) != 1:
        raise ValueError(f"Expected one treated unit at T={target_t}")
    donor_units = donors["abb"].astype(str).tolist()
    treated_unit = str(treated["abb"].iloc[0])
    all_units = donor_units + [treated_unit]
    x_cols = sorted(
        [column for column in df if column.startswith("x")],
        key=lambda column: int(column[1:]),
    )
    current_x = (
        target_rows.set_index("abb")
        .reindex(all_units)[x_cols]
        .to_numpy(dtype=float)
    )
    if binary_model not in {"logistic", "ridge"}:
        raise ValueError("binary_model must be 'logistic' or 'ridge'")
    history_x = _history_features(df, all_units, history_cutoff, history_window, history_representation)
    design = np.column_stack([current_x, history_x])
    donor_design = design[:-1]
    treated_design = design[-1:]

    mean = donor_design.mean(axis=0)
    std = donor_design.std(axis=0)
    std[std < 1e-8] = 1.0
    donor_design = (donor_design - mean) / std
    treated_design = (treated_design - mean) / std
    donor_y = donors["Y"].to_numpy(dtype=float)

    n_splits = min(n_folds, len(donors))
    if cross_fit_donors and n_splits < 2:
        raise ValueError("TSC cross-fitting requires at least two donor units")
    folds = (
        KFold(n_splits=n_splits, shuffle=True, random_state=2026 + target_t)
        if cross_fit_donors
        else None
    )
    donor_oof = np.zeros(len(donors), dtype=float)
    if binary and binary_model == "logistic":
        # Binary nuisance predictions must be probabilities.  Logistic
        # regression improves on fitting an unbounded Ridge outcome and then
        # clipping it after the fact.
        if cross_fit_donors:
            assert folds is not None
            for train_index, test_index in folds.split(donor_design):
                train_y = donor_y[train_index].astype(int)
                if len(np.unique(train_y)) < 2:
                    donor_oof[test_index] = float(train_y.mean())
                    continue
                model = LogisticRegression(
                    C=1.0 / max(ridge_alpha, 1e-8),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                )
                model.fit(donor_design[train_index], train_y)
                donor_oof[test_index] = model.predict_proba(
                    donor_design[test_index]
                )[:, 1]
        full_y = donor_y.astype(int)
        if len(np.unique(full_y)) < 2:
            donor_full = np.full(len(donors), float(full_y.mean()))
            target_prediction = float(full_y.mean())
        else:
            full_model = LogisticRegression(
                C=1.0 / max(ridge_alpha, 1e-8),
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
            )
            full_model.fit(donor_design, full_y)
            full_predictions = full_model.predict_proba(
                np.vstack([donor_design, treated_design])
            )[:, 1]
            donor_full = full_predictions[:-1]
            target_prediction = float(full_predictions[-1])
        donor_oof = np.clip(donor_oof, 1e-3, 1.0 - 1e-3)
        donor_full = np.clip(donor_full, 1e-3, 1.0 - 1e-3)
        target_prediction = float(np.clip(target_prediction, 1e-3, 1.0 - 1e-3))
    else:
        if cross_fit_donors:
            assert folds is not None
            for train_index, test_index in folds.split(donor_design):
                model = Ridge(alpha=ridge_alpha)
                model.fit(donor_design[train_index], donor_y[train_index])
                donor_oof[test_index] = model.predict(donor_design[test_index])

        full_model = Ridge(alpha=ridge_alpha)
        full_model.fit(donor_design, donor_y)
        donor_full = full_model.predict(donor_design)
        target_prediction = float(full_model.predict(treated_design)[0])
    if not cross_fit_donors:
        donor_oof = donor_full.copy()
    if return_full_donor:
        return donor_units, donor_oof, donor_full, target_prediction, donor_y
    return donor_units, donor_oof, target_prediction, donor_y
