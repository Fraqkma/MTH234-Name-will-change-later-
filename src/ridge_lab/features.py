"""Polynomial design matrices and train-fitted feature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def make_A(x: NDArray[np.float64], degree: int) -> FloatArray:
    """Return [1, x, ..., x**degree] for a finite one-dimensional input."""
    if isinstance(degree, bool) or not isinstance(degree, int):
        raise TypeError("degree must be an integer")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("x must be a non-empty, finite one-dimensional array")
    return np.vander(values, N=degree + 1, increasing=True)


def make_bike_design(hours: NDArray[np.int64], working_days: NDArray[np.int64]) -> FloatArray:
    """One intercept, hour-0 reference coding, working-day main effect, and interaction."""
    hour_values = np.asarray(hours)
    working_values = np.asarray(working_days)
    if (hour_values.ndim != 1 or working_values.shape != hour_values.shape
            or hour_values.size == 0):
        raise ValueError("hours and working_days must be non-empty matching vectors")
    if (not np.issubdtype(hour_values.dtype, np.integer)
            or not np.issubdtype(working_values.dtype, np.integer)
            or np.any((hour_values < 0) | (hour_values > 23))
            or np.any((working_values != 0) & (working_values != 1))):
        raise ValueError("hours must be 0..23 and working_days must be 0 or 1")
    hour_effects = (hour_values[:, None] == np.arange(1, 24)).astype(float)
    work = working_values.astype(float)[:, None]
    return np.column_stack((np.ones(hour_values.size), hour_effects, work,
                            hour_effects * work))


@dataclass(frozen=True)
class PolynomialScaler:
    """Standardize non-intercept columns using statistics from training data."""

    mean: FloatArray
    scale: FloatArray

    def transform(self, design: NDArray[np.float64]) -> FloatArray:
        matrix = np.asarray(design, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.size + 1:
            raise ValueError("design matrix does not match the fitted polynomial scaler")
        if not np.isfinite(matrix).all():
            raise ValueError("design matrix must contain only finite values")
        transformed = matrix.copy()
        transformed[:, 1:] = (transformed[:, 1:] - self.mean) / self.scale
        return transformed


def fit_polynomial_scaler(train_design: NDArray[np.float64]) -> PolynomialScaler:
    """Fit means and standard deviations on training features only."""
    matrix = np.asarray(train_design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("train_design must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("train_design must contain only finite values")
    if not np.allclose(matrix[:, 0], 1.0):
        raise ValueError("the first design-matrix column must be the intercept")
    mean = matrix[:, 1:].mean(axis=0)
    scale = matrix[:, 1:].std(axis=0)
    scale[scale == 0.0] = 1.0
    mean.setflags(write=False)
    scale.setflags(write=False)
    return PolynomialScaler(mean=mean, scale=scale)
