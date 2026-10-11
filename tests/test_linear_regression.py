import numpy as np
import pytest

from linear_models.base import NotFittedError
from linear_models.linear_regression import LinearRegression
from tests.helpers import numerical_gradient


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


def test_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    y = rng.normal(size=5)
    model = LinearRegression()
    model.coef_ = np.array([0.5, -1.0, 2.0])
    model.intercept_ = 0.3

    grad_w, grad_b = model._gradient(X, y)
    num_grad_w, num_grad_b = numerical_gradient(model, X, y)

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


def test_fit_rejects_analytic_solver_with_l1():
    model = LinearRegression(solver="analytic", penalty="l1")
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 3.0])

    with pytest.raises(ValueError, match="analytic"):
        model.fit(X, y)


def test_fit_without_intercept_keeps_intercept_zero():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] + 1.0
    model = LinearRegression(
        fit_intercept=False, learning_rate=0.05, max_iter=2000, tol=0.0
    )

    model.fit(X, y)

    assert model.intercept_ == 0.0
    assert np.allclose(model.coef_, [7 / 3], atol=1e-4)
    assert np.isclose(model.loss_history_[0], 8.05859375)


def test_fit_l2_shrinks_weights_to_ridge_solution():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] - 1.0
    model = LinearRegression(
        penalty="l2", alpha=0.5, learning_rate=0.1, max_iter=5000, tol=0.0
    )

    model.fit(X, y)

    assert np.allclose(model.coef_, [1.6], atol=1e-4)
    assert np.isclose(model.intercept_, -0.2, atol=1e-4)
    assert np.isclose(model.loss_history_[0], 1.495)
    assert np.isclose(model.loss_history_[-1], 0.8)


def test_fit_l1_shrinks_weights_by_constant_amount():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] - 1.0
    model = LinearRegression(
        penalty="l1", alpha=0.5, learning_rate=0.1, max_iter=5000, tol=0.0
    )

    model.fit(X, y)

    assert np.allclose(model.coef_, [1.75], atol=1e-4)
    assert np.isclose(model.intercept_, -0.5, atol=1e-4)
    assert np.isclose(model.loss_history_[0], 1.745)
    assert np.isclose(model.loss_history_[-1], 0.9375)


def test_fit_analytic_matches_normal_equation_without_intercept():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 3.0])
    model = LinearRegression(solver="analytic", fit_intercept=False)

    model.fit(X, y)

    assert np.allclose(model.coef_, [1.4])
    assert model.intercept_ == 0.0


def test_residual_is_prediction_minus_target():
    model = LinearRegression()
    model.coef_ = np.array([1.0, -1.0])
    model.intercept_ = 0.5
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1.0, 2.0])

    result = model._residual(X, y)

    assert result.shape == (2,)
    assert np.allclose(result, [-1.5, -2.5])


def test_fit_analytic_with_intercept_centers_data():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 9.0])
    model = LinearRegression(solver="analytic", fit_intercept=True)

    model.fit(X, y)

    assert np.allclose(model.coef_, [3.5])
    assert np.isclose(model.intercept_, -2.0)


def test_fit_analytic_l2_shrinks_weight_not_intercept():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 9.0])
    model = LinearRegression(solver="analytic", penalty="l2", alpha=0.5)

    model.fit(X, y)

    assert np.allclose(model.coef_, [2.0])
    assert np.isclose(model.intercept_, 1.0)


def test_fit_analytic_duplicate_columns_gives_minimum_norm_solution():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([2.0, 4.0, 6.0])
    model = LinearRegression(solver="analytic")

    model.fit(X, y)

    assert np.allclose(model.coef_, [1.0, 1.0])
    assert np.isclose(model.intercept_, 0.0)
    assert np.allclose(model.predict(X), y)


def test_fit_analytic_sets_n_iter_and_loss_history():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 9.0])
    model = LinearRegression(solver="analytic", penalty="l2", alpha=0.5)

    model.fit(X, y)

    assert model.n_iter_ == 1
    assert np.allclose(model.loss_history_, [2.0])


def test_analytic_and_gradient_solvers_agree():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
    y = np.array([3.0, 3.0, 8.0, 7.0])
    analytic = LinearRegression(solver="analytic")
    gradient = LinearRegression(
        solver="gradient", learning_rate=0.1, max_iter=2000, tol=0.0
    )

    analytic.fit(X, y)
    gradient.fit(X, y)

    assert np.allclose(analytic.coef_, [0.875, 1.375])
    assert np.isclose(analytic.intercept_, -0.375)
    assert np.allclose(gradient.coef_, analytic.coef_, atol=1e-6)
    assert np.isclose(gradient.intercept_, analytic.intercept_, atol=1e-6)


def test_fit_stops_early_when_loss_change_is_below_tol():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * X[:, 0] - 1.0
    model = LinearRegression(learning_rate=0.1, max_iter=5000, tol=1e-6)

    model.fit(X, y)

    changes = np.abs(np.diff(model.loss_history_))
    assert model.n_iter_ < model.max_iter
    assert len(model.loss_history_) == model.n_iter_
    assert changes[-1] < 1e-6
    assert np.all(changes[:-1] >= 1e-6)


def test_fit_rejects_mismatched_x_and_y_and_stays_unfitted():
    model = LinearRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="samples"):
        model.fit(X, y)

    assert not hasattr(model, "coef_")


def test_default_params_and_no_weights_before_fit():
    model = LinearRegression()

    assert vars(model) == {
        "fit_intercept": True,
        "solver": "gradient",
        "penalty": None,
        "alpha": 0.0,
        "learning_rate": 0.01,
        "max_iter": 1000,
        "tol": 1e-6,
        "verbose": False,
        "random_state": None,
        "l1_ratio": 0.5,
    }


def _fitted_model(coef, intercept):
    model = LinearRegression()
    model.coef_ = np.array([coef])
    model.intercept_ = intercept
    return model


def test_score_is_zero_when_predicting_the_mean():
    model = _fitted_model(0.0, 2.0)  # z = [2, 2, 2]
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])

    assert np.isclose(model.score(X, y), 0.0)


def test_score_is_negative_for_worse_than_mean():
    model = _fitted_model(-1.0, 3.0)  # z = [3, 2, 1]
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])

    assert np.isclose(model.score(X, y), -3.0)


def test_score_is_one_for_perfect_fit():
    model = _fitted_model(1.0, 1.0)  # z = [1, 2, 3]
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])

    assert np.isclose(model.score(X, y), 1.0)


def test_score_constant_target_follows_sklearn():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([5.0, 5.0, 5.0])

    assert _fitted_model(0.0, 5.0).score(X, y) == 1.0
    assert _fitted_model(0.0, 4.0).score(X, y) == 0.0


def test_score_requires_fitted_model():
    with pytest.raises(NotFittedError):
        LinearRegression().score([[1.0]], [1.0])


def test_score_rejects_y_with_wrong_length():
    model = _fitted_model(1.0, 0.0)
    with pytest.raises(ValueError, match="same length"):
        model.score([[1.0], [2.0]], [1.0])


def test_score_rejects_2d_y():
    model = _fitted_model(1.0, 0.0)
    with pytest.raises(ValueError, match="1-D"):
        model.score([[1.0], [2.0]], [[1.0], [2.0]])


def test_analytic_solver_rejects_elasticnet():
    model = LinearRegression(solver="analytic", penalty="elasticnet", alpha=0.1)
    with pytest.raises(ValueError, match="analytic"):
        model.fit([[1.0], [2.0]], [1.0, 2.0])


def test_elasticnet_with_l1_ratio_zero_equals_l2():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0]])
    y = np.array([1.0, 2.0, 4.0, 5.0])
    mixed = LinearRegression(penalty="elasticnet", l1_ratio=0.0, alpha=0.1).fit(X, y)
    pure = LinearRegression(penalty="l2", alpha=0.1).fit(X, y)

    assert np.allclose(mixed.coef_, pure.coef_)
    assert np.isclose(mixed.intercept_, pure.intercept_)
    assert mixed.n_iter_ == pure.n_iter_


def test_elasticnet_matches_sklearn_reference():
    # Эталон получен из sklearn.linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5,
    # tol=1e-12) на тех же данных. Субградиент даёт малые, но не точные нули,
    # поэтому допуск 1e-3.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = X @ np.array([2.0, 0.0, -1.0]) + 0.5
    model = LinearRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        alpha=0.1,
        learning_rate=0.05,
        max_iter=1000,
        tol=0.0,
    ).fit(X, y)

    assert np.allclose(model.coef_, [1.87145047, 0.0, -0.83291678], atol=1e-3)
    assert np.isclose(model.intercept_, 0.43046137, atol=1e-3)
