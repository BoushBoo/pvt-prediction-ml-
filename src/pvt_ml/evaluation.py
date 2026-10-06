"""Metrics computed exclusively from held-out predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRICS = ["mse", "rmse", "mae", "r2", "pearson_r"]


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not np.isfinite(pred).all():
        raise ValueError("Model produced nonfinite predictions")
    mse = float(mean_squared_error(y, pred))
    corr = (
        float(np.corrcoef(y, pred)[0, 1])
        if np.ptp(y) > 0 and np.ptp(pred) > 0
        else None
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred, force_finite=False)) if np.ptp(y) > 0 else None,
        "pearson_r": corr,
    }


def summarize(predictions):
    folds = []
    pooled = []
    for (name, fold), p in predictions.groupby(["model", "fold"], sort=False):
        folds.append({"model": name, "fold": fold, **metrics(p.actual, p.predicted)})
    for name, p in predictions.groupby("model", sort=False):
        pooled.append({"model": name, **metrics(p.actual, p.predicted)})
    fold_metrics = pd.DataFrame(folds)
    summary = fold_metrics.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return fold_metrics, summary, pd.DataFrame(pooled)
