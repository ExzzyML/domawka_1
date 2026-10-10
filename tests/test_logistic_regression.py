import numpy as np
import pytest

from linear_models.base import NotFittedError
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


@pytest.mark.parametrize(
    "bad_y",
    [
        [0.0, 2.0, 1.0],
        [-1.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, float("nan"), 1.0],
    ],
)
def test_validate_binary_y_rejects_non_binary_values(bad_y):
    model = LogisticRegression()
    with pytest.raises(ValueError, match="only 0 and 1"):
        model._validate_binary_y(np.array(bad_y))


@pytest.mark.parametrize("bad_y", [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
def test_validate_binary_y_rejects_single_class(bad_y):
    model = LogisticRegression()
    with pytest.raises(ValueError, match="both classes"):
        model._validate_binary_y(np.array(bad_y))


def test_validate_binary_y_accepts_both_classes():
    model = LogisticRegression()
    assert model._validate_binary_y(np.array([0.0, 1.0, 1.0, 0.0])) is None


def test_analytic_solver_is_not_supported():
    model = LogisticRegression(solver="analytic")
    with pytest.raises(ValueError, match="only solver='gradient'"):
        model._validate_solver()


def test_default_solver_is_valid():
    assert LogisticRegression()._validate_solver() is None


def test_fit_returns_self_and_creates_weights():
    model = LogisticRegression(max_iter=1)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, 1.0, 1.0])

    result = model.fit(X, y)

    assert result is model
    assert model.coef_.shape == (2,)
    assert np.isscalar(model.intercept_)


def test_fit_validates_params():
    model = LogisticRegression(solver="banana")
    X = np.array([[1.0], [2.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="solver"):
        model.fit(X, y)


def test_fit_rejects_analytic_solver():
    model = LogisticRegression(solver="analytic")
    X = np.array([[1.0], [2.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="only solver='gradient'"):
        model.fit(X, y)


def test_fit_rejects_non_binary_y():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="only 0 and 1"):
        model.fit(X, y)


def test_fit_rejects_single_class_y():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="both classes"):
        model.fit(X, y)


@pytest.mark.parametrize(
    "y_values",
    [[False, False, True, True], [0, 0, 1, 1]],
    ids=["bool", "int"],
)
def test_fit_accepts_bool_and_int_labels(y_values):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    reference = LogisticRegression(max_iter=50).fit(X, np.array([0.0, 0.0, 1.0, 1.0]))

    model = LogisticRegression(max_iter=50).fit(X, np.array(y_values))

    assert reference.coef_[0] > 0
    assert np.allclose(model.coef_, reference.coef_)
    assert np.isclose(model.intercept_, reference.intercept_)


def test_threshold_default_is_half():
    assert LogisticRegression().threshold == 0.5


def test_threshold_and_base_params_can_be_set():
    model = LogisticRegression(threshold=0.3, alpha=0.1)

    assert model.threshold == 0.3
    assert model.alpha == 0.1


@pytest.mark.parametrize(
    "bad_value",
    [-0.1, 1.5, "0.5", None, True, float("nan"), float("inf")],
)
def test_invalid_threshold_raises(bad_value):
    model = LogisticRegression(threshold=bad_value)
    with pytest.raises(ValueError, match="threshold"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [0, 0.0, 0.5, 1, 1.0, np.float64(0.3)])
def test_valid_threshold_accepted(good_value):
    LogisticRegression(threshold=good_value)._validate_params()


def test_predict_proba_known_values():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[-2.0], [0.0], [2.0]])

    proba = model.predict_proba(X)

    assert proba.shape == (3, 2)
    assert np.allclose(
        proba,
        [[0.8807970780, 0.1192029220], [0.5, 0.5], [0.1192029220, 0.8807970780]],
    )


def test_predict_proba_rows_sum_to_one_and_stay_in_unit_interval():
    model = LogisticRegression()
    model.coef_ = np.array([1.0, -2.0])
    model.intercept_ = 0.5
    X = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 1.5], [1000.0, 0.0], [-1000.0, 0.0]])

    with np.errstate(over="raise", invalid="raise"):
        proba = model.predict_proba(X)

    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_predict_proba_before_fit_raises():
    model = LogisticRegression()

    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict_proba(np.array([[1.0]]))


def test_predict_proba_rejects_wrong_n_features():
    model = LogisticRegression()
    model.coef_ = np.array([1.0, 2.0])
    model.intercept_ = 0.0

    with pytest.raises(ValueError, match="features"):
        model.predict_proba(np.array([[1.0], [2.0]]))


def test_predict_known_values():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[-2.0], [-0.5], [0.5], [2.0]])

    pred = model.predict(X)

    assert pred.shape == (4,)
    assert np.issubdtype(pred.dtype, np.integer)
    assert np.array_equal(pred, [0, 0, 1, 1])


@pytest.mark.parametrize(
    "threshold, expected",
    [(0.3, [0, 1, 1, 1]), (0.7, [0, 0, 0, 1])],
)
def test_predict_uses_custom_threshold(threshold, expected):
    model = LogisticRegression(threshold=threshold)
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[-2.0], [-0.5], [0.5], [2.0]])

    pred = model.predict(X)

    assert np.array_equal(pred, expected)


def test_predict_probability_equal_to_threshold_gives_class_zero():
    model = LogisticRegression()
    model.coef_ = np.zeros(2)
    model.intercept_ = 0.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pred = model.predict(X)

    assert np.array_equal(pred, [0, 0])
