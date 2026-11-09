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
