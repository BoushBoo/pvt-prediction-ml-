"""Cloneable Keras regression with training-only epoch selection and refitting."""

import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def stopping_split(n, seed, fraction):
    return train_test_split(np.arange(n), test_size=fraction, random_state=seed)


class NeuralRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        layers=(64, 32),
        epochs=100,
        batch_size=16,
        patience=10,
        validation_fraction=0.2,
        random_state=42,
    ):
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _build(self, width):
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        tf.config.experimental.enable_op_determinism()
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=(width,))]
            + [tf.keras.layers.Dense(units, activation="relu") for units in self.layers]
            + [tf.keras.layers.Dense(1, activation="linear")]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        return model

    def _epoch(self, model, X, y, rng):
        # Explicit batches avoid tf.data thread pools and make ordering reproducible.
        order = rng.permutation(len(X))
        for start in range(0, len(X), self.batch_size):
            batch = order[start : start + self.batch_size]
            model.train_on_batch(X[batch], y[batch])

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience and batch_size must be positive")
        tr, val = stopping_split(len(X), self.random_state, self.validation_fraction)
        self.stopping_train_positions_, self.stopping_validation_positions_ = tr, val
        # Validation data cannot affect even the scaler used for epoch selection.
        self.stopping_scaler_ = StandardScaler().fit(X[tr])
        Xtr = self.stopping_scaler_.transform(X[tr])
        Xval = self.stopping_scaler_.transform(X[val])
        trial = self._build(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        best_loss, wait = float("inf"), 0
        self.validation_losses_ = []
        for epoch in range(1, self.epochs + 1):
            self._epoch(trial, Xtr, y[tr], rng)
            pred = np.asarray(trial(Xval, training=False))
            loss = float(np.mean((y[val] - pred) ** 2))
            if not np.isfinite(loss):
                raise ValueError("Neural training produced nonfinite validation loss")
            self.validation_losses_.append(loss)
            if loss < best_loss:
                best_loss, wait, self.selected_epochs_ = loss, 0, epoch
            else:
                wait += 1
                if wait >= self.patience:
                    break
        # Fit on all data supplied to this fit, for the training-selected epoch count.
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = self._build(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        scaled = self.scaler_.transform(X)
        for _ in range(self.selected_epochs_):
            self._epoch(self.model_, scaled, y, rng)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return np.asarray(
            self.model_(
                self.scaler_.transform(np.asarray(X, dtype=np.float32)), training=False
            )
        ).ravel()
