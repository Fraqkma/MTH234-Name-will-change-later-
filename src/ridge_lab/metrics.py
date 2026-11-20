"""Metrics shared by the scientific core and later experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Return mean squared error for two equal-length, finite vectors."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or actual.size == 0 or predicted.shape != actual.shape:
        raise ValueError("y_true and y_pred must be non-empty vectors with matching shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    return float(np.mean(np.square(actual - predicted)))


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Return root mean squared error in the target's units."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Return mean absolute error in the target's units."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mse(actual, predicted)  # Reuse shape and finiteness checks.
    return float(np.mean(np.abs(actual - predicted)))


def r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Return validation-window R-squared, possibly negative."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residual_mse = mse(actual, predicted)
    variance = float(np.mean(np.square(actual - actual.mean())))
    if variance == 0:
        raise ValueError("R-squared is undefined for a constant target")
    return 1.0 - residual_mse / variance
