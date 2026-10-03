import numpy as np

from linear_models.linear_regression import LinearRegression


def test_loss_is_half_mean_squared_error():
    model = LinearRegression()
    model.coef_ = np.array([0.0])
    model.intercept_ = 0.0
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 3.0])

    result = model._loss(X, y)

    assert np.isclose(result, 2.5)


def test_loss_uses_weights_and_intercept():
    model = LinearRegression()
    model.coef_ = np.array([1.0, -1.0])
    model.intercept_ = 0.5
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])

    result = model._loss(X, y)

    assert np.isclose(result, 2.125)
