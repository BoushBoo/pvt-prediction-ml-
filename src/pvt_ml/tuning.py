"""Explicit inner CV, with serial fits for deterministic neural execution."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from .evaluation import metrics
from .results import fit_audit


def select_model(estimator, grid, X, y, splits, positions):
    candidates, audits = [], []
    best_score, best_params = float("inf"), None
    for candidate, params in enumerate(ParameterGrid(grid)):
        scores = []
        for fold, (tr, va) in enumerate(splits, 1):
            fitted = clone(estimator).set_params(**params).fit(X.iloc[tr], y.iloc[tr])
            score = metrics(y.iloc[va], fitted.predict(X.iloc[va]))["rmse"]
            scores.append(score)
            audits.append(
                {
                    "candidate": candidate,
                    "inner_fold": fold,
                    "validation_positions": np.asarray(positions)[va].tolist(),
                    **fit_audit(fitted, np.asarray(positions)[tr]),
                }
            )
        mean = float(np.mean(scores))
        candidates.append(
            {
                "candidate": candidate,
                "params": params,
                "mean_validation_rmse": mean,
                "std_validation_rmse": float(np.std(scores)),
                **{f"split{i}_rmse": score for i, score in enumerate(scores, 1)},
            }
        )
        if mean < best_score:
            best_score, best_params = mean, params
    fitted = clone(estimator).set_params(**best_params).fit(X, y)
    audits.append({"stage": "outer_train_refit", **fit_audit(fitted, positions)})
    return fitted, best_params, best_score, pd.DataFrame(candidates), audits
