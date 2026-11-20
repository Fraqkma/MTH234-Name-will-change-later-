"""Development-only paired OLS, Ridge, and Lasso comparison for bike demand."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from ridge_lab.bike import HOUR_CSV_SHA256, BikeData, BikeSplit
from ridge_lab.features import make_bike_design
from ridge_lab.metrics import mae, r2, rmse
from ridge_lab.solvers import fit_ols, lasso, ridge

FIRST_HOLDOUT_DATE = date(2012, 8, 8)
DEVELOPMENT_ROW_COUNT = 13_915
REVIEWED_BIKE_PENALTIES = (("OLS", 0.0), ("Ridge", 0.0), ("Lasso", 0.001))


@dataclass(frozen=True)
class BikeWindowMetrics:
    """Validation metrics for one future time window."""

    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class BikeDevelopmentResult:
    """One method/penalty assessed on five shared development windows."""

    method: str
    penalty: float
    windows: tuple[BikeWindowMetrics, ...]
    mean_rmse: float
    mean_mae: float
    mean_r2: float
    rmse_std: float


@dataclass(frozen=True)
class BikeFittedModel:
    """One development-only final fit, with no predictions or scores."""

    method: str
    penalty: float
    coefficients: NDArray[np.float64]


@dataclass(frozen=True)
class BikeRefit:
    """Auditable full-development fit membership and selected models."""

    source_sha256: str
    training_row_ids: tuple[str, ...]
    training_start: date
    training_end: date
    models: tuple[BikeFittedModel, ...]


def evaluate_bike_development(
    data: BikeData,
    split: BikeSplit,
    *,
    ridge_penalties: tuple[float, ...] = (0.0, 1.0, 10.0, 100.0),
    lasso_penalties: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
) -> tuple[BikeDevelopmentResult, ...]:
    """Evaluate only development folds; sealed holdout indices are never read."""
    if not ridge_penalties or not lasso_penalties:
        raise ValueError("Ridge and Lasso candidate grids must be non-empty")
    development = set(split.development)
    if any(set(fold.train + fold.validation) - development for fold in split.folds):
        raise ValueError("a development fold contains a holdout row")

    candidates = (("OLS", 0.0), *(('Ridge', p) for p in ridge_penalties),
                  *(('Lasso', p) for p in lasso_penalties))
    results = []
    for method, penalty in candidates:
        windows = []
        for fold in split.folds:
            train = np.array(fold.train, dtype=int)
            validation = np.array(fold.validation, dtype=int)
            train_design = make_bike_design(data.hours[train], data.working_days[train])
            validation_design = make_bike_design(
                data.hours[validation], data.working_days[validation]
            )
            train_y = data.counts[train]
            if method == "OLS":
                coefficients = fit_ols(train_design, train_y)
            elif method == "Ridge":
                coefficients = ridge(train_design, train_y, penalty)
            else:
                coefficients = lasso(train_design, train_y, penalty)
            actual = data.counts[validation]
            predicted = validation_design @ coefficients
            windows.append(BikeWindowMetrics(
                rmse=rmse(actual, predicted),
                mae=mae(actual, predicted),
                r2=r2(actual, predicted),
            ))
        results.append(BikeDevelopmentResult(
            method=method,
            penalty=penalty,
            windows=tuple(windows),
            mean_rmse=float(np.mean([window.rmse for window in windows])),
            mean_mae=float(np.mean([window.mae for window in windows])),
            mean_r2=float(np.mean([window.r2 for window in windows])),
            rmse_std=float(np.std([window.rmse for window in windows])),
        ))
    return tuple(results)


def select_bike_settings(
    results: tuple[BikeDevelopmentResult, ...],
) -> dict[str, BikeDevelopmentResult]:
    """Choose each method's lowest development mean RMSE, then smaller penalty."""
    return {
        method: min(
            (result for result in results if result.method == method),
            key=lambda result: (result.mean_rmse, result.penalty),
        )
        for method in ("OLS", "Ridge", "Lasso")
    }


def refit_bike_development(
    data: BikeData,
    split: BikeSplit,
    selected: Mapping[str, BikeDevelopmentResult],
) -> BikeRefit:
    """Fit reviewed settings once on every development row, without scoring."""
    if data.sha256 != HOUR_CSV_SHA256 or split.holdout_start != FIRST_HOLDOUT_DATE:
        raise ValueError("bike source or holdout cutoff differs from the reviewed protocol")
    row_count = len(data.dates)
    if not (len(data.hours) == len(data.working_days) == len(data.counts)
            == len(data.row_ids) == row_count):
        raise ValueError("bike data columns have inconsistent lengths")
    expected = tuple(index for index, day in enumerate(data.dates) if day < FIRST_HOLDOUT_DATE)
    if len(expected) != DEVELOPMENT_ROW_COUNT or split.development != expected:
        raise ValueError("refit requires exactly the 13,915 approved development rows")
    expected_holdout = tuple(index for index, day in enumerate(data.dates)
                             if day >= FIRST_HOLDOUT_DATE)
    if split.holdout != expected_holdout:
        raise ValueError("holdout membership differs from the approved cutoff")
    if set(selected) != {method for method, _ in REVIEWED_BIKE_PENALTIES}:
        raise ValueError("refit requires OLS, Ridge, and Lasso selections")
    for method, reviewed_penalty in REVIEWED_BIKE_PENALTIES:
        choice = selected[method]
        if choice.method != method or choice.penalty != reviewed_penalty:
            raise ValueError(f"{method} setting differs from the reviewed development selection")

    indices = np.asarray(expected, dtype=int)
    design = make_bike_design(data.hours[indices], data.working_days[indices])
    targets = data.counts[indices]
    models = []
    for method, _ in REVIEWED_BIKE_PENALTIES:
        choice = selected[method]
        if method == "OLS":
            fitted = fit_ols(design, targets)
        elif method == "Ridge":
            fitted = ridge(design, targets, choice.penalty)
        else:
            fitted = lasso(design, targets, choice.penalty)
        coefficients = np.asarray(fitted, dtype=float).copy()
        if coefficients.shape != (design.shape[1],) or not np.isfinite(coefficients).all():
            raise ValueError(f"{method} returned invalid coefficients")
        coefficients.setflags(write=False)
        models.append(BikeFittedModel(method, choice.penalty, coefficients))
    return BikeRefit(
        source_sha256=data.sha256,
        training_row_ids=tuple(data.row_ids[index] for index in expected),
        training_start=data.dates[expected[0]],
        training_end=data.dates[expected[-1]],
        models=tuple(models),
    )


def bike_development_payload() -> dict:
    """Normal site entry point: selection and refit cannot access reserved targets."""
    from dataclasses import asdict

    from ridge_lab.bike import load_bike_hourly, make_bike_split

    data = load_bike_hourly()
    split = make_bike_split(data)
    selected = select_bike_settings(evaluate_bike_development(data, split))
    fitted = refit_bike_development(data, split, selected)
    return {
        "scope": "development-only", "source_sha256": data.sha256,
        "training_rows": len(fitted.training_row_ids),
        "training_end": fitted.training_end.isoformat(),
        "results": [asdict(result) for result in selected.values()],
    }
