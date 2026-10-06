"""Original model families and explicit, training-only search spaces."""

from importlib import import_module

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .neural import NeuralRegressor

PHASE1 = ("linear", "ridge", "lasso", "tree", "random_forest", "knn", "svr")
PHASE2 = ("xgboost", "catboost", "nn", "stacking", "extra_trees", "dnn")
ALL_MODELS = ("dummy",) + PHASE1 + PHASE2


def model_spec(name, seed):
    def scaled(m):
        return make_pipeline(StandardScaler(), m)

    specs = {
        "dummy": (DummyRegressor(), {}),
        "linear": (LinearRegression(), {}),
        "ridge": (scaled(Ridge()), {"ridge__alpha": np.logspace(-3, 1, 10).tolist()}),
        "lasso": (
            scaled(Lasso(max_iter=10000)),
            {"lasso__alpha": np.logspace(-3, 1, 10).tolist()},
        ),
        "tree": (
            DecisionTreeRegressor(random_state=seed),
            {"max_depth": list(range(1, 21))},
        ),
        "knn": (
            scaled(KNeighborsRegressor()),
            {"kneighborsregressor__n_neighbors": list(range(1, 21))},
        ),
        "svr": (
            scaled(SVR()),
            {
                "svr__C": np.logspace(-3, 2, 10).tolist(),
                "svr__epsilon": np.linspace(0.01, 1, 5).tolist(),
            },
        ),
        "extra_trees": (
            ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=1),
            {},
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
            {"max_depth": [None, 10], "min_samples_leaf": [1, 3]},
        ),
    }
    if name in specs:
        return specs[name]
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(random_state=seed, n_jobs=1), {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [4, 6, 8],
        }
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(random_seed=seed, verbose=False, thread_count=1), {
            "iterations": [500, 1000],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6],
            "l2_leaf_reg": [3, 5, 7],
        }
    if name in ("nn", "dnn"):
        import_module(
            "tensorflow"
        )  # Fail before a run starts if this extra is missing.

        if name == "nn":
            grid = [
                {"layers": [layers], "epochs": [epochs], "batch_size": [batch]}
                for layers, epochs, batch in [
                    ((64, 32), 100, 16),
                    ((128, 64), 100, 16),
                    ((128, 64, 32), 150, 32),
                ]
            ]
            return NeuralRegressor(random_state=seed), grid
        return NeuralRegressor(layers=(256, 128, 64), random_state=seed), {
            "epochs": [100, 150],
            "batch_size": [16, 32],
        }
    if name == "stacking":
        from catboost import CatBoostRegressor
        from xgboost import XGBRegressor

        bases = [
            ("lr", LinearRegression()),
            ("ridge", scaled(Ridge())),
            ("lasso", scaled(Lasso(max_iter=10000))),
            ("dt", DecisionTreeRegressor(max_depth=10, random_state=seed)),
            ("knn", scaled(KNeighborsRegressor(n_neighbors=5))),
            ("svr", scaled(SVR())),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
        # Each base pipeline fits its scaler independently during stack OOF fits.
        stack = StackingRegressor(
            estimators=bases,
            final_estimator=CatBoostRegressor(
                verbose=False, random_seed=seed, thread_count=1
            ),
            passthrough=True,
            cv=KFold(5, shuffle=True, random_state=seed),
            n_jobs=1,
        )
        return stack, {}  # Original architecture has fixed configurations.
    raise ValueError(f"Unknown model: {name}. Choose from {ALL_MODELS}")
