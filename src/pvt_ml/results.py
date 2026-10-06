"""Run manifests, environment versions and auditable fit/split provenance."""

import json
import platform
import subprocess
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
from sklearn.ensemble import StackingRegressor

from .neural import NeuralRegressor


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n"
    )


def environment():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]
        },
        "git_revision": revision,
        "git_dirty": dirty,
    }


def fit_audit(estimator, positions):
    positions = np.asarray(positions)
    audit = {"fit_positions": positions.tolist()}
    if isinstance(estimator, NeuralRegressor):
        audit.update(
            stopping_train_positions=positions[
                estimator.stopping_train_positions_
            ].tolist(),
            stopping_validation_positions=positions[
                estimator.stopping_validation_positions_
            ].tolist(),
            selected_epochs=estimator.selected_epochs_,
            validation_losses=estimator.validation_losses_,
            stopping_scaler_mean=estimator.stopping_scaler_.mean_.tolist(),
            refit_scaler_mean=estimator.scaler_.mean_.tolist(),
        )
    if isinstance(estimator, StackingRegressor):
        audit["stack_splits"] = [
            {
                "train_positions": positions[tr].tolist(),
                "validation_positions": positions[va].tolist(),
            }
            for tr, va in estimator.cv.split(positions)
        ]
    return audit


def save_evaluation(out, predictions):
    from .evaluation import summarize

    folds, summary, pooled = summarize(predictions)
    predictions.to_csv(out / "predictions.csv", index=False)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    summary.to_csv(out / "summary.csv")
    pooled.to_csv(out / "pooled_metrics.csv", index=False)
    return summary
