"""Framework-independent math and data functions for the Ridge Regression Lab."""

from ridge_lab.datasets import DatasetBundle, DatasetSplit, make_dataset
from ridge_lab.features import PolynomialScaler, fit_polynomial_scaler, make_A
from ridge_lab.metrics import mse
from ridge_lab.solvers import fit_ols, ridge

__all__ = [
    "DatasetBundle",
    "DatasetSplit",
    "PolynomialScaler",
    "fit_ols",
    "fit_polynomial_scaler",
    "make_A",
    "make_dataset",
    "mse",
    "ridge",
]
