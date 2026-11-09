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
