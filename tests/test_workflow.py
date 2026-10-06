import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pvt_ml.data import FEATURES, load_data, validate_data
from pvt_ml.evaluation import summarize
from pvt_ml.experiment import run_experiment
from pvt_ml.models import PHASE1, PHASE2, model_spec
from pvt_ml.neural import NeuralRegressor


def data(n=48):
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=FEATURES)
    df["Pb"] = 10 + 2 * df.Rs + df.Tf**2
    df["Bob"] = 1.3 + 0.05 * df.Rs
    return df


@pytest.mark.parametrize("suffix", [".csv", ".xls", ".xlsx"])
def test_formats_and_no_row_removal(tmp_path, suffix):
    df = data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.columns = [f" {c.upper()} " for c in df.columns]
    path = tmp_path / ("data" + suffix)
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".xlsx":
        df.to_excel(path, index=False)
    else:
        # Legacy XLS writer is test-only; real loader uses xlrd.
        xlwt = pytest.importorskip("xlwt")
        wb = xlwt.Workbook()
        ws = wb.add_sheet("data")
        for j, col in enumerate(df.columns):
            ws.write(0, j, col)
        for i, row in enumerate(df.to_numpy(), 1):
            for j, val in enumerate(row):
                ws.write(i, j, float(val))
        wb.save(str(path))
    loaded, target = validate_data(load_data(path), " pb ")
    assert target == "Pb" and len(loaded) == len(df)
    np.testing.assert_allclose(loaded, df.iloc[:, :5])


def test_bad_schema_and_values_are_rejected():
    df = data()
    with pytest.raises(ValueError, match="duplicate"):
        validate_data(df.assign(TF=df.Tf), "Pb")
    df.loc[0, "Rs"] = np.nan
    with pytest.raises(ValueError, match="no automatic cleaning"):
        validate_data(df, "Pb")


def test_original_stacking_architecture():
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    estimator, grid = model_spec("stacking", 7)
    assert [name for name, _ in estimator.estimators] == [
        "lr",
        "ridge",
        "lasso",
        "dt",
        "knn",
        "svr",
        "xgb",
    ]
    assert estimator.final_estimator.__class__.__name__ == "CatBoostRegressor"
    assert estimator.passthrough and estimator.cv.n_splits == 5
    assert grid == {}
    for name in ["ridge", "lasso", "knn", "svr"]:
        assert dict(estimator.estimators)[name].steps[0][0] == "standardscaler"


def test_neural_architectures_and_scaler_isolation():
    pytest.importorskip("tensorflow")
    X, y = data()[FEATURES], data().Bob
    a = NeuralRegressor(layers=(8, 4), epochs=3, batch_size=8, random_state=7).fit(X, y)
    b = clone(a).fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=0, atol=0)
    np.testing.assert_allclose(
        a.stopping_scaler_.mean_, X.iloc[a.stopping_train_positions_].mean(), rtol=1e-6
    )
    np.testing.assert_allclose(a.scaler_.mean_, X.mean(), rtol=1e-6)
    assert not set(a.stopping_train_positions_) & set(a.stopping_validation_positions_)
    nn, grid = model_spec("nn", 7)
    assert [item["layers"][0] for item in grid] == [(64, 32), (128, 64), (128, 64, 32)]
    dnn, _ = model_spec("dnn", 7)
    assert dnn.layers == (256, 128, 64)


def test_all_families_smoke_and_artifact_consistency(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    models = PHASE1 + PHASE2
    # Short explicit grids exercise the real architectures, not the expensive production search.
    spaces = {name: {} for name in models}
    spaces.update(
        random_forest={"n_estimators": [5]},
        xgboost={"n_estimators": [5]},
        catboost={"iterations": [5]},
        nn={"epochs": [2]},
        dnn={"epochs": [2]},
        stacking={"final_estimator__iterations": [5], "xgb__n_estimators": [5]},
    )
    run_experiment(data(), "Bob", tmp_path, models, 3, 2, search_spaces=spaces)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["status"] == "complete" and "tensorflow_cpu" in {
        k.replace("-", "_") for k in meta["packages"]
    }
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert len(pred) == 48 * len(models)
    for name, p in pred.groupby("model"):
        assert sorted(p.row_position) == list(range(48))
        assert (tmp_path / f"{name}_oof.png").exists()
    folds, summary, pooled = summarize(pred)
    pd.testing.assert_frame_equal(
        folds, pd.read_csv(tmp_path / "fold_metrics.csv"), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        summary, pd.read_csv(tmp_path / "summary.csv", index_col=0), check_dtype=False
    )
    pd.testing.assert_frame_equal(
        pooled, pd.read_csv(tmp_path / "pooled_metrics.csv"), check_dtype=False
    )
    for split in json.loads((tmp_path / "splits.json").read_text()):
        train, test = set(split["train_positions"]), set(split["test_positions"])
        for inner in split["inner_splits"]:
            assert set(inner["train_positions"]) <= train
            assert set(inner["validation_positions"]) <= train
        for name in models:
            audits = json.loads(
                (tmp_path / f"{name}_fold{split['fold']}_fit_audit.json").read_text()
            )
            for audit in audits:
                fit = set(audit["fit_positions"])
                assert fit <= train and not fit & test
                if "validation_positions" in audit:
                    assert not fit & set(audit["validation_positions"])
                if "stopping_validation_positions" in audit:
                    early = set(audit["stopping_validation_positions"])
                    assert early <= fit and not early & set(
                        audit["stopping_train_positions"]
                    )
                for stack in audit.get("stack_splits", []):
                    assert set(stack["train_positions"]) <= fit
                    assert set(stack["validation_positions"]) <= fit
                    assert not set(stack["train_positions"]) & set(
                        stack["validation_positions"]
                    )


def test_overwrite_protection(tmp_path):
    (tmp_path / "previous").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        run_experiment(data(), "Pb", tmp_path, ["linear"], 3, 2)
    assert (tmp_path / "previous").read_text() == "keep"


def test_outer_labels_cannot_change_that_folds_selected_model(tmp_path):
    """Changing only fold 1's held-out labels cannot alter its tuning or predictions."""
    from sklearn.model_selection import KFold

    df = data(60)
    _, test = next(KFold(3, shuffle=True, random_state=42).split(df))
    changed = df.copy()
    changed.loc[test, "Pb"] += 100000
    spaces = {"ridge": {"ridge__alpha": [0.01, 10, 1000]}}
    for frame, directory in [(df, tmp_path / "a"), (changed, tmp_path / "b")]:
        run_experiment(frame, "Pb", directory, ["ridge"], 3, 2, search_spaces=spaces)
    params_a = json.loads((tmp_path / "a/selected_parameters.json").read_text())[0]
    params_b = json.loads((tmp_path / "b/selected_parameters.json").read_text())[0]
    assert params_a == params_b
    a = pd.read_csv(tmp_path / "a/predictions.csv").query("fold == 1")
    b = pd.read_csv(tmp_path / "b/predictions.csv").query("fold == 1")
    np.testing.assert_array_equal(a.predicted, b.predicted)


def test_preprocessing_fits_only_inner_training_rows(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    observed = []
    original = StandardScaler.fit

    def recording_fit(self, X, y=None, **kwargs):
        observed.append(set(X.index))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    run_experiment(
        data(60),
        "Pb",
        tmp_path,
        ["ridge"],
        3,
        2,
        search_spaces={"ridge": {"ridge__alpha": [1]}},
    )
    splits = json.loads((tmp_path / "splits.json").read_text())
    for fold, split in enumerate(splits):
        expected = [set(s["train_positions"]) for s in split["inner_splits"]]
        expected.append(set(split["train_positions"]))
        assert observed[fold * 3 : (fold + 1) * 3] == expected
