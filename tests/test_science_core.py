import numpy as np
import pytest

from ridge_lab import fit_ols, fit_polynomial_scaler, make_A, mse, ridge


def test_polynomial_design_and_scaler_use_train_statistics_only() -> None:
    train = make_A(np.array([0.0, 1.0]), degree=2)
    scaler = fit_polynomial_scaler(train)
    validation = scaler.transform(make_A(np.array([2.0]), degree=2))

    assert train.shape == (2, 3)
    assert np.array_equal(train[:, 0], np.ones(2))
    assert np.allclose(train[:, 1:], [[0.0, 0.0], [1.0, 1.0]])
    assert np.array_equal(scaler.mean, [0.5, 0.5])
    assert np.allclose(validation[:, 1:], [[3.0, 7.0]])


def test_ols_ridge_zero_and_unpenalized_intercept() -> None:
    x = np.array([-1.0, 0.0, 1.0])
    design = make_A(x, degree=1)
    y = 3.0 + 2.0 * x

    ols = fit_ols(design, y)
    ridge_zero = ridge(design, y, 0.0)
    ridge_strong = ridge(design, y, 100.0)

    assert np.allclose(ols, [3.0, 2.0])
    assert np.allclose(ridge_zero, ols)
    assert ridge_strong[0] == pytest.approx(3.0)
    assert abs(ridge_strong[1]) < abs(ols[1])
    assert mse(y, design @ ols) == pytest.approx(0.0)


@pytest.mark.parametrize("degree", [-1, 1.5])
def test_design_matrix_rejects_invalid_degree(degree: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_A(np.array([0.0, 1.0]), degree)


def test_ridge_rejects_negative_lambda() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ridge(np.ones((2, 1)), np.ones(2), -1.0)
