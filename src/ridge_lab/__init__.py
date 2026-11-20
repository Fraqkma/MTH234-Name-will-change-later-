"""Framework-independent math and data functions for The Least Squares Showdown."""

from ridge_lab.bike import BikeData, BikeFold, BikeSplit, load_bike_hourly, make_bike_split
from ridge_lab.bike_experiment import (
    BikeFittedModel,
    BikeRefit,
    evaluate_bike_development,
    refit_bike_development,
    select_bike_settings,
)
from ridge_lab.datasets import DatasetBundle, DatasetSplit, make_dataset
from ridge_lab.features import PolynomialScaler, fit_polynomial_scaler, make_A, make_bike_design
from ridge_lab.metrics import mae, mse, r2, rmse
from ridge_lab.solvers import fit_ols, lasso, ridge

__all__ = [
    "BikeData",
    "BikeFittedModel",
    "BikeFold",
    "BikeRefit",
    "BikeSplit",
    "DatasetBundle",
    "DatasetSplit",
    "PolynomialScaler",
    "evaluate_bike_development",
    "fit_ols",
    "fit_polynomial_scaler",
    "lasso",
    "load_bike_hourly",
    "mae",
    "make_A",
    "make_bike_design",
    "make_bike_split",
    "make_dataset",
    "mse",
    "r2",
    "refit_bike_development",
    "ridge",
    "rmse",
    "select_bike_settings",
]
