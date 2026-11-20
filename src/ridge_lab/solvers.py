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


def lasso(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    diagnostics: dict | None = None,
) -> FloatArray:
    """Minimize SSE/(2n) + alpha*L1(non-intercept) by coordinate descent."""
    matrix, targets = _training_data(design, y)
    if not np.allclose(matrix[:, 0], 1.0):
        raise ValueError("the first design column must be a single intercept")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError("alpha must be a number")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if alpha == 0:
        return fit_ols(matrix, targets)
    if max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")

    features = matrix[:, 1:]
    means = features.mean(axis=0)
    centered = features - means
    centered_y = targets - targets.mean()
    gram = centered.T @ centered / matrix.shape[0]
    covariance = centered.T @ centered_y / matrix.shape[0]
    coefficients = np.zeros(features.shape[1], dtype=float)
    gradient = covariance.copy()
    for iteration in range(1, max_iter + 1):
        largest_change = 0.0
        for column in range(features.shape[1]):
            norm = gram[column, column]
            if norm == 0:
                continue
            rho = gradient[column] + norm * coefficients[column]
            updated = np.sign(rho) * max(abs(rho) - alpha, 0.0) / norm
            largest_change = max(largest_change, abs(updated - coefficients[column]))
            delta = updated - coefficients[column]
            coefficients[column] = updated
            gradient -= gram[:, column] * delta
        if iteration % 500 == 0:
            active = np.flatnonzero(coefficients != 0)
            if active.size:
                trial = np.zeros_like(coefficients)
                trial[active] = np.linalg.lstsq(
                    gram[np.ix_(active, active)],
                    covariance[active] - alpha * np.sign(coefficients[active]),
                    rcond=None,
                )[0]
                trial_gradient = covariance - gram @ trial
                trial_kkt = np.max(np.where(
                    trial != 0,
                    np.abs(trial_gradient - alpha * np.sign(trial)),
                    np.maximum(np.abs(trial_gradient) - alpha, 0),
                ))
                same_sign = np.array_equal(np.sign(trial[active]), np.sign(coefficients[active]))
                if same_sign:
                    coefficients = trial
                    gradient = trial_gradient
                    if trial_kkt <= tol:
                        largest_change = 0.0
                else:
                    crossing = active[trial[active] * coefficients[active] <= 0]
                    steps = coefficients[crossing] / (coefficients[crossing] - trial[crossing])
                    step = float(np.min(steps))
                    coefficients += step * (trial - coefficients)
                    coefficients[np.abs(coefficients) < 1e-12] = 0.0
                    gradient = covariance - gram @ coefficients
        if largest_change < tol:
            intercept = targets.mean() - means @ coefficients
            if diagnostics is not None:
                diagnostics.update(iterations=iteration, max_change=largest_change,
                                   kkt_residual=float(np.max(np.where(
                                       coefficients != 0,
                                       np.abs(gradient - alpha * np.sign(coefficients)),
                                       np.maximum(np.abs(gradient) - alpha, 0),
                                   ))), converged=True)
            return np.concatenate(([intercept], coefficients))
    if diagnostics is not None:
        diagnostics.update(iterations=max_iter, max_change=largest_change,
                           kkt_residual=float(np.max(np.where(
                               coefficients != 0,
                               np.abs(gradient - alpha * np.sign(coefficients)),
                               np.maximum(np.abs(gradient) - alpha, 0),
                           ))), converged=False)
    raise RuntimeError("Lasso coordinate descent did not converge")
