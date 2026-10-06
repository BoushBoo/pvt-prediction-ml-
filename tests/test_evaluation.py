import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

import pvt_ml.experiment as exp


class RecordingRegressor(RegressorMixin, BaseEstimator):
    fits = []

    def fit(self, X, y):
        self.fits.append(set(X.index))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_outer_isolation_and_saved_metrics(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=exp.FEATURES)
    df["Pb"] = 1000 + 50 * df.Rs
    RecordingRegressor.fits = []
    monkeypatch.setattr(
        exp, "model_spec", lambda name, seed: (RecordingRegressor(), {})
    )
    exp.run_experiment(df, "Pb", tmp_path, ["recording"], 3, 2)
    splits = json.loads((tmp_path / "splits.json").read_text())
    # For each fold: two inner fits followed by refit on the full outer training set.
    for i, split in enumerate(splits):
        fits = RecordingRegressor.fits[i * 3 : (i + 1) * 3]
        train, test = set(split["train_positions"]), set(split["test_positions"])
        assert all(s <= train and not s & test for s in fits)
        assert fits[-1] == train
    preds = pd.read_csv(tmp_path / "predictions.csv")
    assert sorted(preds.row_position) == list(range(60))
    results = pd.read_csv(tmp_path / "fold_metrics.csv")
    for fold, p in preds.groupby("fold"):
        recorded = results[results.fold == fold].iloc[0]
        for metric, val in exp.metrics(p.actual, p.predicted).items():
            assert (
                pd.isna(recorded[metric])
                if val is None
                else np.isclose(recorded[metric], val)
            )


def test_repeatability(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=exp.FEATURES)
    df["Bob"] = 1.3 + 0.05 * df.Rs
    a = exp.run_experiment(df, "Bob", tmp_path / "a", ["linear", "ridge"], 3, 2)
    b = exp.run_experiment(df, "Bob", tmp_path / "b", ["linear", "ridge"], 3, 2)
    pd.testing.assert_frame_equal(a, b)


def test_excel_style_columns(tmp_path):
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(30, 4)), columns=["TF", "RS", "GG", "API "])
    df["PB"] = 500 + df.RS
    exp.run_experiment(df, "Pb", tmp_path, ["linear"], 3, 2)
    assert pd.read_csv(tmp_path / "predictions.csv").shape[0] == 30
