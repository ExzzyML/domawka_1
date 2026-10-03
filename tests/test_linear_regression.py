import numpy as np
import pytest

from linear_models.base import NotFittedError
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


def test_gradient_returns_pair_with_expected_values():
    model = LinearRegression()
    model.coef_ = np.array([0.0])
    model.intercept_ = 0.0
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 3.0])

    grad_w, grad_b = model._gradient(X, y)

    assert grad_w.shape == (1,)
    assert np.allclose(grad_w, [-3.5])
    assert np.isclose(grad_b, -2.0)


def _numerical_gradient(model, X, y, eps=1e-6):
    """Градиент _loss по (coef_, intercept_) через центральную разность."""
    grad_w = np.zeros_like(model.coef_)
    for j in range(model.coef_.shape[0]):
        original = model.coef_[j]
        model.coef_[j] = original + eps
        loss_plus = model._loss(X, y)
        model.coef_[j] = original - eps
        loss_minus = model._loss(X, y)
        model.coef_[j] = original
        grad_w[j] = (loss_plus - loss_minus) / (2 * eps)

    original_b = model.intercept_
    model.intercept_ = original_b + eps
    loss_plus = model._loss(X, y)
    model.intercept_ = original_b - eps
    loss_minus = model._loss(X, y)
    model.intercept_ = original_b
    grad_b = (loss_plus - loss_minus) / (2 * eps)

    return grad_w, grad_b


def test_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    y = rng.normal(size=5)
    model = LinearRegression()
    model.coef_ = np.array([0.5, -1.0, 2.0])
    model.intercept_ = 0.3

    grad_w, grad_b = model._gradient(X, y)
    num_grad_w, num_grad_b = _numerical_gradient(model, X, y)

    assert np.allclose(grad_w, num_grad_w, atol=1e-6)
    assert np.isclose(grad_b, num_grad_b, atol=1e-6)


def test_predict_returns_linear_output():
    model = LinearRegression()
    model.coef_ = np.array([2.0, -1.0])
    model.intercept_ = 0.5
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = model.predict(X)

    assert result.shape == (2,)
    assert np.allclose(result, [0.5, 2.5])


def test_predict_requires_fitted_model():
    model = LinearRegression()
    with pytest.raises(NotFittedError):
        model.predict([[1.0, 2.0]])


def test_fit_returns_self_and_creates_weights():
    model = LinearRegression(max_iter=1)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])

    result = model.fit(X, y)

    assert result is model
    assert model.coef_.shape == (2,)
    assert np.isscalar(model.intercept_)


def test_fit_recovers_line_without_penalty():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] - 1.0
    model = LinearRegression(learning_rate=0.1, max_iter=5000, tol=0.0)

    model.fit(X, y)

    assert np.allclose(model.coef_, [2.0], atol=1e-4)
    assert np.isclose(model.intercept_, -1.0, atol=1e-4)
    assert model.n_iter_ == 5000
    assert len(model.loss_history_) == 5000
    assert np.isclose(model.loss_history_[0], 1.245)
    assert model.loss_history_[-1] < 1e-8


def test_fit_validates_params():
    model = LinearRegression(solver="banana")
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 3.0])

    with pytest.raises(ValueError, match="solver"):
        model.fit(X, y)
