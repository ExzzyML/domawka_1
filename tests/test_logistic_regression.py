import numpy as np

from linear_models.logistic_regression import LogisticRegression, _sigmoid
from tests.helpers import numerical_gradient


def test_sigmoid_known_values():
    z = np.array([-2.0, 0.0, 2.0])

    result = _sigmoid(z)

    assert result.shape == (3,)
    assert np.allclose(result, [0.1192029220, 0.5, 0.8807970780])


def test_sigmoid_extreme_values_no_overflow():
    z = np.array([-1000.0, 1000.0])

    with np.errstate(over="raise", invalid="raise"):
        result = _sigmoid(z)

    assert np.allclose(result, [0.0, 1.0])


def test_loss_is_mean_log_loss():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 0.0])

    result = model._loss(X, y)

    assert np.isclose(result, 1.2200948)


def test_loss_extreme_wrong_predictions_equals_margin():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[-1000.0], [1000.0]])
    y = np.array([1.0, 0.0])

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        result = model._loss(X, y)

    assert np.isclose(result, 1000.0)


def test_loss_extreme_correct_predictions_is_zero():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[-1000.0], [1000.0]])
    y = np.array([0.0, 1.0])

    with np.errstate(divide="raise", over="raise", invalid="raise"):
        result = model._loss(X, y)

    assert np.isclose(result, 0.0)


def test_gradient_returns_pair_with_expected_values():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 0.0])

    grad_w, grad_b = model._gradient(X, y)

    assert grad_w.shape == (1,)
    assert np.isclose(grad_w[0], 0.7463264)
    assert np.isclose(grad_b, 0.3059278)


def test_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    model = LogisticRegression()
    model.coef_ = np.array([0.5, -1.0, 2.0])
    model.intercept_ = 0.3

    grad_w, grad_b = model._gradient(X, y)
    num_grad_w, num_grad_b = numerical_gradient(model, X, y)

    assert np.allclose(grad_w, num_grad_w, atol=1e-6)
    assert np.isclose(grad_b, num_grad_b, atol=1e-6)
