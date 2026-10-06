"""One shared nested-CV workflow for every target, dataset and model family."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, ParameterGrid

from .data import FEATURES, validate_data
from .evaluation import metrics
from .models import model_spec
from .plotting import plot_predictions
from .results import environment, save_evaluation, save_json
from .tuning import select_model


def run_experiment(
    df,
    target,
    output_dir,
    models=("linear", "extra_trees", "xgboost"),
    outer_folds=5,
    inner_folds=5,
    seed=42,
    n_jobs=1,
    search_spaces=None,
    source=None,
):
    if n_jobs != 1:
        raise ValueError("Use n_jobs=1: fits run serially to control neural RNG state")
    if not models or len(set(models)) != len(models):
        raise ValueError("Provide a nonempty list of unique models")
    data, target = validate_data(df, target)
    X, y = data[FEATURES], data[target]
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("Both CV levels require at least two folds")
    outer = list(KFold(outer_folds, shuffle=True, random_state=seed).split(X))
    if min(len(te) for _, te in outer) < 2:
        raise ValueError("Each outer test fold needs at least two observations for R²")
    nested = [
        list(KFold(inner_folds, shuffle=True, random_state=seed + fold).split(tr))
        for fold, (tr, _) in enumerate(outer, 1)
    ]
    # Resolve optional dependencies and validate grids before creating outputs.
    specs = {name: model_spec(name, seed) for name in models}
    search_spaces = search_spaces or {}
    if set(search_spaces) - set(models):
        raise ValueError("Search-space overrides must refer to requested models")
    for name, (estimator, default) in specs.items():
        grid = search_spaces.get(name, default)
        candidates = list(ParameterGrid(grid))
        if not candidates:
            raise ValueError(f"Empty candidate grid for {name}")
        for params in candidates:
            clone(estimator).set_params(**params)
        min_train = min(len(tr) for splits in nested for tr, _ in splits)
        if (
            name == "knn"
            and max(
                p.get("kneighborsregressor__n_neighbors", 5)
                for p in ParameterGrid(grid)
            )
            > min_train
        ):
            raise ValueError(
                "Too few inner training samples for the KNN grid; specify a smaller grid explicitly"
            )
        if isinstance(estimator, StackingRegressor):
            for params in ParameterGrid(grid):
                candidate = clone(estimator).set_params(**params)
                k = candidate.cv.get_n_splits()
                if min_train - int(np.ceil(min_train / k)) < 5:
                    raise ValueError(
                        "Too few training samples for the original stacking architecture"
                    )
        specs[name] = (estimator, grid)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(
            "Output directory must be empty; use a new directory to preserve prior runs"
        )
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(data, index=True).values.tobytes()
    ).hexdigest()
    manifest = {
        "target": target,
        "features": FEATURES,
        "models": list(models),
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "rows": len(data),
        "data_sha256": fingerprint,
        "source": source,
        "evaluation": "nested shuffled KFold; sample-mixture generalization",
        "selection": "minimum mean inner-fold RMSE; first candidate wins ties",
        "final_model_selection": "not performed; each family evaluated independently",
        "search_spaces": {name: grid for name, (_, grid) in specs.items()},
        **environment(),
        "status": "running",
    }
    save_json(out / "metadata.json", manifest)
    splits = []
    for fold, ((tr, te), inner) in enumerate(zip(outer, nested), 1):
        splits.append(
            {
                "fold": fold,
                "train_positions": tr.tolist(),
                "test_positions": te.tolist(),
                "inner_splits": [
                    {
                        "train_positions": tr[a].tolist(),
                        "validation_positions": tr[b].tolist(),
                    }
                    for a, b in inner
                ],
            }
        )
    save_json(out / "splits.json", splits)
    pd.DataFrame(
        {"row_position": np.arange(len(df)), "source_index": [str(i) for i in df.index]}
    ).to_csv(out / "row_identifiers.csv", index=False)
    predictions, params = [], []
    try:
        for name, (estimator, grid) in specs.items():
            for fold, ((tr, te), inner) in enumerate(zip(outer, nested), 1):
                selected, best, score, search, audits = select_model(
                    estimator, grid, X.iloc[tr], y.iloc[tr], inner, tr
                )
                pred = selected.predict(X.iloc[te])
                metrics(y.iloc[te], pred)  # Validate predictions before recording.
                params.append(
                    {
                        "model": name,
                        "fold": fold,
                        "best_params": best,
                        "inner_selection_rmse": score,
                        "estimator": repr(selected),
                        "resolved_parameters": {
                            k: repr(v)
                            for k, v in selected.get_params(deep=True).items()
                        },
                    }
                )
                search.to_csv(out / f"{name}_fold{fold}_search.csv", index=False)
                save_json(out / f"{name}_fold{fold}_fit_audit.json", audits)
                predictions.extend(
                    {
                        "model": name,
                        "fold": fold,
                        "row_position": int(pos),
                        "actual": float(actual),
                        "predicted": float(p),
                    }
                    for pos, actual, p in zip(te, y.iloc[te], pred)
                )
                # Checkpoints remain inspectable if a later model fails.
                pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
                save_json(out / "selected_parameters.json", params)
        preds = pd.DataFrame(predictions)
        summary = save_evaluation(out, preds)
        plot_predictions(preds, out, target)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(out / "metadata.json", manifest)
        raise
    manifest["status"] = "complete"
    save_json(out / "metadata.json", manifest)
    return summary
