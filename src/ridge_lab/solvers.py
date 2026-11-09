"""Numerically stable least-squares and Ridge solvers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _training_data(design: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[FloatArray, FloatArray]:
    matrix = np.asarray(design, dtype=float)
    targets = np.asarray(y, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("design must be a non-empty two-dimensional matrix")
    if targets.ndim != 1 or targets.size != matrix.shape[0]:
        raise ValueError("y must be one-dimensional with one target per design row")
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise ValueError("design and y must contain only finite values")
    return matrix, targets


def fit_ols(design: NDArray[np.float64], y: NDArray[np.float64]) -> FloatArray:
    """Fit ordinary least squares with NumPy's SVD-based least-squares solver."""
    matrix, targets = _training_data(design, y)
    return np.linalg.lstsq(matrix, targets, rcond=None)[0]


def ridge(design: NDArray[np.float64], y: NDArray[np.float64], lam: float) -> FloatArray:
    """Fit Ridge by augmented least squares, leaving the first-column intercept free."""
    matrix, targets = _training_data(design, y)
    if isinstance(lam, bool) or not isinstance(lam, (int, float, np.integer, np.floating)):
        raise TypeError("lambda must be a number")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and non-negative")
    if lam == 0:
        return fit_ols(matrix, targets)

    penalty = np.eye(matrix.shape[1])
    penalty[0, 0] = 0.0
    augmented_design = np.vstack((matrix, np.sqrt(lam) * penalty))
    augmented_targets = np.concatenate((targets, np.zeros(matrix.shape[1])))
    return np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)[0]
