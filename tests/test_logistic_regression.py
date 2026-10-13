import numpy as np
import pytest

from linear_models.base import ConvergenceWarning, NotFittedError
from linear_models.logistic_regression import LogisticRegression, _sigmoid, _softmax
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
    with pytest.raises(ValueError, match="'gradient', 'proximal' or 'newton'"):
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

    with pytest.raises(ValueError, match="'gradient', 'proximal' or 'newton'"):
        model.fit(X, y)


def test_fit_rejects_non_integer_labels():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="integer class labels"):
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


def test_predict_before_fit_raises():
    model = LogisticRegression()

    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict(np.array([[1.0]]))


def test_predict_rejects_wrong_n_features():
    model = LogisticRegression()
    model.coef_ = np.array([1.0, 2.0])
    model.intercept_ = 0.0

    with pytest.raises(ValueError, match="features"):
        model.predict(np.array([[1.0], [2.0]]))


def test_fit_decreases_loss_on_separable_data():
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.5, max_iter=200)

    model.fit(X, y)

    history = np.array(model.loss_history_)
    assert np.all(np.diff(history) <= 1e-12)
    assert history[-1] < 0.1 * history[0]


def test_fit_separates_linearly_separable_data():
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.5, max_iter=200)

    model.fit(X, y)

    assert model.coef_[0] > 0
    assert np.array_equal(model.predict(X), y)


def test_l2_penalty_shrinks_weights():
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    plain = LogisticRegression(learning_rate=0.5, max_iter=200).fit(X, y)
    ridge = LogisticRegression(
        learning_rate=0.5, max_iter=200, penalty="l2", alpha=0.1
    ).fit(X, y)

    assert np.linalg.norm(ridge.coef_) < 0.5 * np.linalg.norm(plain.coef_)


def test_l2_penalty_does_not_regularize_intercept():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1, 1, 1])
    model = LogisticRegression(
        learning_rate=0.05, max_iter=5000, penalty="l2", alpha=10.0
    ).fit(X, y)

    assert abs(model.coef_[0]) < 0.1
    assert model.intercept_ > 1.0


def test_fit_without_intercept_keeps_intercept_zero():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    with_b = LogisticRegression(learning_rate=0.5, max_iter=200).fit(X, y)
    no_b = LogisticRegression(learning_rate=0.5, max_iter=200, fit_intercept=False).fit(
        X, y
    )

    assert abs(with_b.intercept_) > 0.1
    assert no_b.intercept_ == 0.0


def test_refit_starts_from_zero_weights():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_first = np.array([0, 0, 1, 1])
    y_second = np.array([1, 1, 0, 0])
    fresh = LogisticRegression(learning_rate=0.5, max_iter=50).fit(X, y_second)
    model = LogisticRegression(learning_rate=0.5, max_iter=50)

    model.fit(X, y_first)
    model.fit(X, y_second)

    assert np.allclose(model.coef_, fresh.coef_)
    assert np.isclose(model.intercept_, fresh.intercept_)


def test_l1_penalty_suppresses_weak_feature_more_than_l2():
    X = np.array(
        [
            [-3.0, 1.0],
            [-2.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [2.0, 1.0],
            [3.0, -1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    params = {"learning_rate": 0.5, "max_iter": 500}
    plain = LogisticRegression(**params).fit(X, y)
    ridge = LogisticRegression(penalty="l2", alpha=0.1, **params).fit(X, y)
    lasso = LogisticRegression(penalty="l1", alpha=0.1, **params).fit(X, y)

    ridge_ratio = abs(ridge.coef_[1]) / abs(ridge.coef_[0])
    lasso_ratio = abs(lasso.coef_[1]) / abs(lasso.coef_[0])

    assert abs(lasso.coef_[1]) < 0.1 * abs(plain.coef_[1])
    assert lasso_ratio < 0.5 * ridge_ratio


def test_l1_penalty_does_not_regularize_intercept():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1, 1, 1])
    model = LogisticRegression(
        learning_rate=0.05, max_iter=5000, penalty="l1", alpha=1.0
    ).fit(X, y)

    assert abs(model.coef_[0]) < 0.1
    assert model.intercept_ > 1.0


def test_fit_stops_early_when_loss_change_is_below_tol():
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.5, max_iter=1000, tol=1e-3)

    model.fit(X, y)

    changes = np.abs(np.diff(model.loss_history_))
    assert model.n_iter_ < model.max_iter
    assert len(model.loss_history_) == model.n_iter_
    assert changes[-1] < 1e-3
    assert np.all(changes[:-1] >= 1e-3)


def test_fit_runs_all_iterations_when_tol_is_zero():
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(learning_rate=0.5, max_iter=30, tol=0.0)

    model.fit(X, y)

    assert model.n_iter_ == 30
    assert len(model.loss_history_) == 30


def test_fit_rejects_mismatched_x_and_y_and_stays_unfitted():
    model = LogisticRegression()
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="samples"):
        model.fit(X, y)

    assert not hasattr(model, "coef_")


def test_default_params_and_no_weights_before_fit():
    model = LogisticRegression()

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
        "momentum": 0.0,
        "nesterov": False,
        "lr_schedule": None,
        "lr_decay": 0.9,
        "lr_step_size": 10,
        "store_gradients": False,
        "callbacks": None,
        "threshold": 0.5,
        "batch_size": None,
        "shuffle": True,
    }


@pytest.mark.parametrize(
    "penalty, alpha, expected",
    [
        (None, 0.0, 0.65001),
        ("l2", 2.0, 0.71251),
        ("l1", 2.0, 1.15001),
    ],
)
def test_first_loss_history_value_includes_penalty(penalty, alpha, expected):
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 0.0])
    model = LogisticRegression(
        learning_rate=1.0, max_iter=1, penalty=penalty, alpha=alpha
    )

    model.fit(X, y)

    assert len(model.loss_history_) == 1
    assert np.isclose(model.loss_history_[0], expected, atol=1e-5)


@pytest.mark.parametrize(
    "bad_value",
    [5.0, -1.0, float("nan"), True, None, "abc"],
)
def test_predict_rejects_invalid_threshold(bad_value):
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    model.threshold = bad_value

    with pytest.raises(ValueError, match="threshold"):
        model.predict(np.array([[1.0]]))


def test_predict_proba_does_not_depend_on_threshold():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    model.threshold = "abc"
    X = np.array([[-2.0], [0.0], [2.0]])

    proba = model.predict_proba(X)

    assert np.allclose(
        proba,
        [[0.8807970780, 0.1192029220], [0.5, 0.5], [0.1192029220, 0.8807970780]],
    )


def _fitted_logistic():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = -1.5  # z = x - 1.5, predict = [0, 0, 1, 1]
    return model


def test_score_is_accuracy():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1])

    assert np.isclose(_fitted_logistic().score(X, y), 0.75)


def test_score_requires_fitted_model():
    with pytest.raises(NotFittedError):
        LogisticRegression().score([[1.0]], [1])


def test_score_rejects_y_with_wrong_length():
    with pytest.raises(ValueError, match="same length"):
        _fitted_logistic().score([[1.0], [2.0]], [1])


def test_elasticnet_with_l1_ratio_one_equals_l1():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    mixed = LogisticRegression(penalty="elasticnet", l1_ratio=1.0, alpha=0.05).fit(X, y)
    pure = LogisticRegression(penalty="l1", alpha=0.05).fit(X, y)

    assert np.allclose(mixed.coef_, pure.coef_)
    assert np.isclose(mixed.intercept_, pure.intercept_)
    assert mixed.n_iter_ == pure.n_iter_


def test_gradient_history_of_logistic_regression():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(store_gradients=True, max_iter=20, tol=0.0).fit(X, y)

    assert model.gradient_history_.shape == (20, 3)


def _logistic_weighted_data():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [1.5, 1.5]])
    y = np.array([0, 0, 1, 1, 1])
    return X, y


def test_weighted_logistic_loss_known_value():
    model = LogisticRegression()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    X = np.array([[0.0], [2.0]])
    y = np.array([0.0, 1.0])

    # per-object losses: ln 2 and ln(1 + e^2) - 2
    assert np.isclose(model._loss(X, y, np.array([1.0, 3.0])), 0.2684828)


def test_weighted_logistic_gradient_matches_numerical_gradient():
    X, y = _logistic_weighted_data()
    weights = np.array([1.0, 2.0, 0.5, 3.0, 1.0])
    model = LogisticRegression()
    model.coef_ = np.array([0.3, -0.2])
    model.intercept_ = 0.1

    grad_w, grad_b = model._gradient(X, y, weights)
    num_w, num_b = numerical_gradient(model, X, y, sample_weight=weights)

    assert np.allclose(grad_w, num_w, atol=1e-6)
    assert np.isclose(grad_b, num_b, atol=1e-6)


def test_logistic_integer_weights_equal_repeated_rows():
    X, y = _logistic_weighted_data()
    counts = np.array([2, 1, 3, 1, 2])
    params = {
        "penalty": "l2",
        "alpha": 0.05,
        "learning_rate": 0.1,
        "max_iter": 200,
        "tol": 0.0,
    }
    weighted = LogisticRegression(**params).fit(X, y, sample_weight=counts)
    repeated = LogisticRegression(**params).fit(
        np.repeat(X, counts, axis=0), np.repeat(y, counts)
    )

    assert np.allclose(weighted.coef_, repeated.coef_)
    assert np.isclose(weighted.intercept_, repeated.intercept_)
    assert np.allclose(weighted.loss_history_, repeated.loss_history_)


def test_logistic_zero_weight_equals_removing_the_row():
    X, y = _logistic_weighted_data()
    weights = np.array([1.0, 2.0, 0.0, 1.0, 1.0])
    keep = weights > 0
    params = {"learning_rate": 0.1, "max_iter": 200, "tol": 0.0}
    zeroed = LogisticRegression(**params).fit(X, y, sample_weight=weights)
    removed = LogisticRegression(**params).fit(
        X[keep], y[keep], sample_weight=weights[keep]
    )

    assert np.allclose(zeroed.coef_, removed.coef_)
    assert np.isclose(zeroed.intercept_, removed.intercept_)


def test_logistic_all_ones_weights_equal_no_weights():
    X, y = _logistic_weighted_data()
    params = {"penalty": "l1", "alpha": 0.02, "max_iter": 100, "tol": 0.0}
    plain = LogisticRegression(**params).fit(X, y)
    ones = LogisticRegression(**params).fit(X, y, sample_weight=np.ones(len(y)))

    assert np.allclose(plain.coef_, ones.coef_)
    assert np.isclose(plain.intercept_, ones.intercept_)


def test_logistic_needs_both_classes_with_positive_weight():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="both classes"):
        LogisticRegression().fit(X, y, sample_weight=[1.0, 1.0, 0.0])


def test_logistic_fit_rejects_bad_sample_weight():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="sample_weight"):
        model.fit([[0.0], [1.0]], [0, 1], sample_weight=[1.0])

    assert not hasattr(model, "coef_")


def test_weighted_accuracy():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1])

    # predict = [0, 0, 1, 1]; correct = [yes, no, yes, yes]; weights 1 + 1 + 1 of 6
    result = _fitted_logistic().score(X, y, sample_weight=[1.0, 3.0, 1.0, 1.0])

    assert np.isclose(result, 0.5)


def test_logistic_mini_batch_reduces_loss():
    X, y = _logistic_weighted_data()
    model = LogisticRegression(
        batch_size=2, learning_rate=0.1, max_iter=50, tol=0.0, random_state=0
    ).fit(X, y)

    assert model.n_iter_ == 50
    assert model.loss_history_[-1] < model.loss_history_[0]


# --- solver="newton" и solver="proximal" ---

_X_TINY = np.array([[0.0], [1.0], [2.0], [3.0]])
_Y_TINY = np.array([0.0, 1.0, 0.0, 1.0])


def _noisy_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    z = X @ np.array([1.5, -1.0]) + 0.3
    y = (rng.random(60) < 1 / (1 + np.exp(-z))).astype(float)
    return X, y, rng


def _model_at_zero():
    model = LogisticRegression()
    model.coef_ = np.zeros(1)
    model.intercept_ = 0.0
    return model


def test_gradient_and_hessian_at_zero():
    # p = 0.5 у всех: r = p - y = (0.5, -0.5, 0.5, -0.5), p(1 - p) = 0.25
    # g = (mean(x·r), mean(r)) = (-0.25, 0)
    # H = 0.25 · [[mean(x²), mean(x)], [mean(x), 1]] = [[0.875, 0.375], [0.375, 0.25]]
    grad, hessian = _model_at_zero()._gradient_and_hessian(_X_TINY, _Y_TINY)

    assert np.allclose(grad, [-0.25, 0.0])
    assert np.allclose(hessian, [[0.875, 0.375], [0.375, 0.25]])


def test_gradient_and_hessian_match_numerical_derivatives():
    X, y, rng = _noisy_problem()
    model = LogisticRegression(penalty="l2", alpha=0.3)
    model.coef_ = np.array([0.4, -0.7])
    model.intercept_ = 0.2
    weights = rng.integers(0, 4, 60).astype(float)
    grad, hessian = model._gradient_and_hessian(X, y, weights)

    def grad_at(theta):
        m = LogisticRegression(penalty="l2", alpha=0.3)
        m.coef_, m.intercept_ = theta[:2].copy(), theta[2]
        return m._gradient_and_hessian(X, y, weights)[0]

    def objective_at(theta):
        m = LogisticRegression(penalty="l2", alpha=0.3)
        m.coef_, m.intercept_ = theta[:2].copy(), theta[2]
        return m._objective(X, y, weights)

    theta = np.array([0.4, -0.7, 0.2])
    eps = 1e-6
    unit = np.eye(3)
    num_grad = [
        (objective_at(theta + eps * e) - objective_at(theta - eps * e)) / (2 * eps)
        for e in unit
    ]
    num_hess = np.array(
        [
            (grad_at(theta + eps * e) - grad_at(theta - eps * e)) / (2 * eps)
            for e in unit
        ]
    )
    assert np.allclose(grad, num_grad, atol=1e-7)
    assert np.allclose(hessian, num_hess, atol=1e-7)


def test_first_newton_step_known_values():
    # H^-1 = [[3.2, -4.8], [-4.8, 11.2]], d = H^-1 g = (-0.8, 1.2),
    # полный шаг принят: θ = 0 - d = (0.8, -1.2)
    model = LogisticRegression(solver="newton", max_iter=1, tol=0.0)

    model.fit(_X_TINY, _Y_TINY)

    assert np.allclose(model.coef_, [0.8])
    assert np.isclose(model.intercept_, -1.2)
    assert np.isclose(model.loss_history_[0], 0.5881488598689919)


def test_line_search_halves_overlong_step():
    # Направление 8·d: t = 1 даёт J = 1.62, t = 1/2 даёт J = 0.896 (обе больше
    # J(0) = log 2 = 0.693), t = 1/4 даёт θ = (1.6, -2.4) и J = 0.629: принято
    model = _model_at_zero()
    grad, hessian = model._gradient_and_hessian(_X_TINY, _Y_TINY)
    direction = 8 * np.linalg.solve(hessian, grad)

    model._newton_line_search(_X_TINY, _Y_TINY, None, grad, direction)

    assert np.allclose(model.coef_, [1.6])
    assert np.isclose(model.intercept_, -2.4)
    assert np.isclose(model._objective(_X_TINY, _Y_TINY), 0.6289684090508637)


def test_line_search_keeps_params_for_ascent_direction():
    model = _model_at_zero()
    grad, hessian = model._gradient_and_hessian(_X_TINY, _Y_TINY)

    model._newton_line_search(
        _X_TINY, _Y_TINY, None, grad, -np.linalg.solve(hessian, grad)
    )

    assert np.array_equal(model.coef_, [0.0])
    assert model.intercept_ == 0.0


@pytest.mark.parametrize(
    "penalty, alpha, coef, intercept",
    [
        # sklearn LogisticRegression(C=np.inf, tol=1e-14)
        (None, 0.0, [2.253126134013, -1.490916504179], 0.160542672492),
        # sklearn LogisticRegression(C=1 / (60 · 0.1), tol=1e-14)
        ("l2", 0.1, [0.900090072581, -0.549622549638], 0.017722020898),
    ],
)
def test_newton_matches_sklearn_in_few_iterations(penalty, alpha, coef, intercept):
    X, y, _ = _noisy_problem()
    model = LogisticRegression(solver="newton", penalty=penalty, alpha=alpha, tol=1e-12)

    model.fit(X, y)

    assert np.allclose(model.coef_, coef, atol=1e-8)
    assert np.isclose(model.intercept_, intercept, atol=1e-8)
    assert model.n_iter_ <= 8


def test_newton_converges_quadratically():
    # Расстояние до минимума по цели примерно возводится в квадрат на каждом шаге
    X, y, _ = _noisy_problem()
    model = LogisticRegression(solver="newton", tol=1e-15).fit(X, y)

    gaps = np.array(model.loss_history_[:4]) - model.loss_history_[-1]
    assert np.all(np.diff(model.loss_history_) <= 0)
    assert gaps[1] < 0.2 * gaps[0]
    assert gaps[2] < 0.05 * gaps[1]
    assert gaps[3] < 1e-3 * gaps[2]


def test_newton_needs_far_fewer_iterations_than_gradient():
    X, y, _ = _noisy_problem()
    newton = LogisticRegression(solver="newton", tol=1e-12).fit(X, y)
    gradient = LogisticRegression(learning_rate=0.5, tol=1e-12, max_iter=100_000).fit(
        X, y
    )

    assert newton.n_iter_ * 50 < gradient.n_iter_
    assert np.allclose(newton.coef_, gradient.coef_, atol=1e-3)


def test_newton_weighted_matches_sklearn():
    # Эталон: sklearn LogisticRegression(C=1 / (Σw · 0.1)).fit(..., sample_weight=w)
    X, y, rng = _noisy_problem()
    weights = rng.integers(0, 4, 60).astype(float)
    model = LogisticRegression(solver="newton", penalty="l2", alpha=0.1, tol=1e-12)

    model.fit(X, y, sample_weight=weights)

    assert np.allclose(model.coef_, [1.072794573940, -0.559093801561], atol=1e-8)
    assert np.isclose(model.intercept_, -0.256077946674, atol=1e-8)


def test_newton_without_intercept_and_gradient_history():
    model = LogisticRegression(
        solver="newton", fit_intercept=False, store_gradients=True, tol=1e-12
    )

    model.fit(_X_TINY, _Y_TINY)

    assert model.intercept_ == 0.0
    assert model.gradient_history_.shape == (model.n_iter_, 2)
    # первая строка: градиент в нуле, (mean(x·r), 0) с нулевым столбцом интерцепта
    assert np.allclose(model.gradient_history_[0], [-0.25, 0.0])


@pytest.mark.parametrize("penalty", ["l1", "elasticnet"])
def test_newton_rejects_non_smooth_penalty(penalty):
    model = LogisticRegression(solver="newton", penalty=penalty, alpha=0.1)
    with pytest.raises(ValueError, match="twice differentiable"):
        model.fit(_X_TINY, _Y_TINY)


def test_newton_rejects_batch_size():
    model = LogisticRegression(solver="newton", batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        model.fit(_X_TINY, _Y_TINY)


def test_newton_warning_mentions_newton():
    model = LogisticRegression(solver="newton", max_iter=1, tol=1e-12)
    with pytest.warns(ConvergenceWarning, match="Newton's method did not converge"):
        model.fit(_X_TINY, _Y_TINY)


def test_proximal_l1_zeroes_noise_feature_like_sklearn_saga():
    # Эталон: sklearn LogisticRegression(l1_ratio=1, solver="saga",
    # C=1 / (60 · 0.05), tol=1e-14); третий признак шумовой
    X, y, rng = _noisy_problem()
    X = np.column_stack([X, rng.normal(size=60)])
    params = {"penalty": "l1", "alpha": 0.05, "learning_rate": 0.5, "tol": 0.0}
    proximal = LogisticRegression(solver="proximal", max_iter=20_000, **params)
    gradient = LogisticRegression(solver="gradient", max_iter=20_000, **params)

    proximal.fit(X, y)
    gradient.fit(X, y)

    assert np.allclose(proximal.coef_, [1.24118206666, -0.681651723044, 0.0], atol=1e-9)
    assert np.isclose(proximal.intercept_, 0.034093540657, atol=1e-9)
    assert proximal.coef_[2] == 0.0
    assert abs(gradient.coef_[2]) > 0.01


# --- несколько классов (softmax) ---

_X_THREE = np.array([[0.0], [1.0], [2.0]])
_Y_THREE = np.array([0, 1, 2])


def _three_class_problem():
    """60 точек вокруг трёх центров, веса 0..3 и два шумовых признака."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [2.5, 0.0], [0.0, 2.5]])
    y = np.repeat([0, 1, 2], 20)
    X = centers[y] + rng.normal(size=(60, 2))
    order = rng.permutation(60)
    X, y = X[order], y[order]
    weights = rng.integers(0, 4, 60).astype(float)
    noise = rng.normal(size=(60, 2))
    return X, y, weights, noise


def _multiclass_model_at_zero(n_features=1, n_classes=3, **params):
    model = LogisticRegression(**params)
    model.coef_ = np.zeros((n_classes, n_features))
    model.intercept_ = np.zeros(n_classes)
    return model


def test_softmax_known_values_and_stability():
    result = _softmax(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))

    assert np.allclose(result[0], [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(result[1], [0.09003057317, 0.24472847106, 0.66524095577])
    # без вычитания максимума exp(1000) дал бы inf
    stable = _softmax(np.array([[1000.0, 1001.0]]))
    assert np.allclose(stable, [[1 / (1 + np.e), np.e / (1 + np.e)]])


def test_fit_with_three_classes_creates_matrix_weights():
    X, y, _, _ = _three_class_problem()

    model = LogisticRegression(max_iter=5).fit(X, y)

    assert model.coef_.shape == (3, 2)
    assert model.intercept_.shape == (3,)
    assert np.array_equal(model.classes_, [0, 1, 2])


def test_binary_fit_keeps_vector_weights_and_sets_classes():
    X, y, _, _ = _three_class_problem()

    model = LogisticRegression(max_iter=5).fit(X, (y > 0).astype(int))

    assert model.coef_.shape == (2,)
    assert np.isscalar(model.intercept_)
    assert np.array_equal(model.classes_, [0, 1])


def test_fit_rejects_single_class_that_is_not_binary():
    with pytest.raises(ValueError, match="at least two different classes"):
        LogisticRegression().fit([[1.0], [2.0], [3.0]], [2, 2, 2])


def test_fit_rejects_class_with_zero_total_weight():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2, 2])
    with pytest.raises(ValueError, match="class 2 has zero total sample_weight"):
        LogisticRegression().fit(X, y, sample_weight=[1.0, 1.0, 0.0, 0.0])


def test_multiclass_loss_and_gradient_at_zero():
    # Все P = 1/3: потери log 3. Остатки P − Y по объектам:
    # (-2/3, 1/3, 1/3), (1/3, -2/3, 1/3), (1/3, 1/3, -2/3).
    # grad_W[k] = mean(x · остаток_k) = (1/3, 0, -1/3), столбцы остатков дают
    # нулевой градиент по интерцептам.
    model = _multiclass_model_at_zero()

    grad_w, grad_b = model._gradient(_X_THREE, _Y_THREE)

    assert np.isclose(model._loss(_X_THREE, _Y_THREE), np.log(3))
    assert grad_w.shape == (3, 1)
    assert np.allclose(grad_w[:, 0], [1 / 3, 0.0, -1 / 3])
    assert np.allclose(grad_b, [0.0, 0.0, 0.0])


def test_one_step_on_three_classes_known_values():
    # lr = 0.5: W = -0.5 · (1/3, 0, -1/3) = (-1/6, 0, 1/6), интерцепты 0.
    # Потери на каждом объекте: log Σ e^z − z_y при z = W · x
    model = LogisticRegression(learning_rate=0.5, max_iter=1, tol=0.0)

    model.fit(_X_THREE, _Y_THREE)

    assert np.allclose(model.coef_[:, 0], [-1 / 6, 0.0, 1 / 6])
    assert np.allclose(model.intercept_, 0.0)
    third, sixth = np.exp(1 / 3), np.exp(1 / 6)
    expected = np.mean(
        [
            np.log(3),
            np.log(1 / sixth + 1 + sixth),
            np.log(1 / third + 1 + third) - 1 / 3,
        ]
    )
    assert np.isclose(model.loss_history_[0], expected)
    assert np.isclose(expected, 1.0028136494)


def _numerical_multiclass_gradient(model, X, y, weights, eps=1e-6):
    grad_w = np.zeros_like(model.coef_)
    for index in np.ndindex(*model.coef_.shape):
        original = model.coef_[index]
        model.coef_[index] = original + eps
        plus = model._loss(X, y, weights)
        model.coef_[index] = original - eps
        minus = model._loss(X, y, weights)
        model.coef_[index] = original
        grad_w[index] = (plus - minus) / (2 * eps)
    grad_b = np.zeros_like(model.intercept_)
    for k in range(len(grad_b)):
        original = model.intercept_[k]
        model.intercept_[k] = original + eps
        plus = model._loss(X, y, weights)
        model.intercept_[k] = original - eps
        minus = model._loss(X, y, weights)
        model.intercept_[k] = original
        grad_b[k] = (plus - minus) / (2 * eps)
    return grad_w, grad_b


def test_multiclass_gradient_matches_numerical_gradient_with_weights():
    X, y, weights, _ = _three_class_problem()
    model = _multiclass_model_at_zero(n_features=2)
    model.coef_ = np.array([[0.4, -0.7], [-0.2, 0.3], [0.1, 0.5]])
    model.intercept_ = np.array([0.2, -0.1, 0.05])

    grad_w, grad_b = model._gradient(X, y, weights)
    num_w, num_b = _numerical_multiclass_gradient(model, X, y, weights)

    assert np.allclose(grad_w, num_w, atol=1e-7)
    assert np.allclose(grad_b, num_b, atol=1e-7)
    assert np.isclose(np.sum(grad_b), 0.0, atol=1e-12)  # сумма по классам нулевая


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_hessian_and_gradient_match_numerical_derivatives(fit_intercept):
    X, y, weights, _ = _three_class_problem()
    params = {"penalty": "l2", "alpha": 0.3, "fit_intercept": fit_intercept}
    model = _multiclass_model_at_zero(n_features=2, **params)
    model.coef_ = np.array([[0.4, -0.7], [-0.2, 0.3], [0.1, 0.5]])
    if fit_intercept:
        model.intercept_ = np.array([0.2, -0.1, 0.05])
    theta = model._params_vector()
    grad, hessian = model._gradient_and_hessian(X, y, weights)

    def at(vector):
        other = _multiclass_model_at_zero(n_features=2, **params)
        other._set_params_vector(vector)
        return other

    eps = 1e-6
    unit = np.eye(len(theta))
    num_grad = [
        (
            at(theta + eps * e)._objective(X, y, weights)
            - at(theta - eps * e)._objective(X, y, weights)
        )
        / (2 * eps)
        for e in unit
    ]
    num_hessian = np.array(
        [
            (
                at(theta + eps * e)._gradient_and_hessian(X, y, weights)[0]
                - at(theta - eps * e)._gradient_and_hessian(X, y, weights)[0]
            )
            / (2 * eps)
            for e in unit
        ]
    )
    assert grad.shape == (len(theta),)
    assert hessian.shape == (len(theta), len(theta))
    assert np.allclose(grad, num_grad, atol=1e-7)
    assert np.allclose(hessian, num_hessian, atol=1e-6)


def test_params_vector_roundtrip_for_multiclass():
    model = _multiclass_model_at_zero(n_features=2)
    model.coef_ = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    model.intercept_ = np.array([7.0, 8.0, 9.0])

    theta = model._params_vector()

    assert np.array_equal(theta, [1, 2, 7, 3, 4, 8, 5, 6, 9])
    model._set_params_vector(theta + 1)
    assert np.array_equal(model.coef_, [[2, 3], [4, 5], [6, 7]])
    assert np.array_equal(model.intercept_, [8, 9, 10])


@pytest.mark.parametrize(
    "penalty, alpha, coef, intercept, atol",
    [
        # sklearn LogisticRegression(C=np.inf, tol=1e-14); данные почти разделимы,
        # поэтому веса большие и допуск шире
        (
            None,
            0.0,
            [
                [-2.784167076, -1.7756218541],
                [7.3404791604, -2.2315882534],
                [-4.5563120845, 4.0072101075],
            ],
            [6.1977324239, -5.2204910777, -0.9772413461],
            1e-5,
        ),
        # sklearn LogisticRegression(C=1 / (60 · 0.1), tol=1e-14)
        (
            "l2",
            0.1,
            [
                [-0.50133922155, -0.538166464692],
                [0.942143790198, -0.329743211944],
                [-0.440804568648, 0.867909676637],
            ],
            [1.168594680961, -0.569914658054, -0.598680022907],
            1e-6,
        ),
    ],
)
def test_multiclass_newton_matches_sklearn(penalty, alpha, coef, intercept, atol):
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(solver="newton", penalty=penalty, alpha=alpha, tol=1e-13)

    model.fit(X, y)

    assert np.allclose(model.coef_, coef, atol=atol)
    assert np.allclose(model.intercept_, intercept, atol=atol)
    assert model.n_iter_ <= 15
    assert np.all(np.diff(model.loss_history_) <= 1e-12)


def test_multiclass_gradient_descent_matches_sklearn_with_l2():
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(
        penalty="l2", alpha=0.1, learning_rate=0.5, tol=1e-13, max_iter=100_000
    )

    model.fit(X, y)

    assert np.allclose(
        model.coef_,
        [
            [-0.50133922155, -0.538166464692],
            [0.942143790198, -0.329743211944],
            [-0.440804568648, 0.867909676637],
        ],
        atol=1e-5,
    )
    assert np.allclose(
        model.intercept_, [1.168594680961, -0.569914658054, -0.598680022907], atol=1e-5
    )


def test_multiclass_newton_weighted_matches_sklearn():
    # Эталон: sklearn LogisticRegression(C=1 / (Σw · 0.1)).fit(..., sample_weight=w)
    X, y, weights, _ = _three_class_problem()
    model = LogisticRegression(solver="newton", penalty="l2", alpha=0.1, tol=1e-13)

    model.fit(X, y, sample_weight=weights)

    assert np.isclose(weights.sum(), 108.0)
    assert np.allclose(
        model.coef_,
        [
            [-0.596312436564, -0.530730677902],
            [0.956573498286, -0.310556363565],
            [-0.360261061722, 0.841287041466],
        ],
        atol=1e-6,
    )
    assert np.allclose(
        model.intercept_, [1.159057535827, -0.555141547563, -0.603915988264], atol=1e-6
    )


def test_multiclass_newton_without_intercept_matches_sklearn():
    # Эталон: sklearn LogisticRegression(C=1 / (60 · 0.1), fit_intercept=False)
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(
        solver="newton", penalty="l2", alpha=0.1, fit_intercept=False, tol=1e-13
    )

    model.fit(X, y)

    assert np.array_equal(model.intercept_, [0.0, 0.0, 0.0])
    assert np.allclose(
        model.coef_,
        [
            [-0.43166317976, -0.219687658222],
            [0.769906053859, -0.389934272062],
            [-0.338242874098, 0.609621930284],
        ],
        atol=1e-6,
    )


def test_multiclass_integer_weights_equal_repeated_rows():
    X, y, weights, _ = _three_class_problem()
    counts = weights.astype(int)
    params = {"solver": "newton", "penalty": "l2", "alpha": 0.1, "tol": 1e-13}
    weighted = LogisticRegression(**params).fit(X, y, sample_weight=weights)
    repeated = LogisticRegression(**params).fit(
        np.repeat(X, counts, axis=0), np.repeat(y, counts)
    )

    assert np.allclose(weighted.coef_, repeated.coef_, atol=1e-8)
    assert np.allclose(weighted.intercept_, repeated.intercept_, atol=1e-8)


def test_two_class_softmax_equals_binary_model_with_doubled_step():
    # Метки {1, 2} включают softmax с K = 2. Из нуля строки весов остаются
    # противоположными (W0 = -W1), разность w = W1 - W0 равна двоичным весам,
    # а шаг спуска по w вдвое больше шага по W1: lr двоичной модели = 2 · lr.
    X, y, _, _ = _three_class_problem()
    binary_y = (y > 0).astype(int)
    binary = LogisticRegression(learning_rate=0.2, max_iter=40, tol=0.0)
    softmax = LogisticRegression(learning_rate=0.1, max_iter=40, tol=0.0)

    binary.fit(X, binary_y)
    softmax.fit(X, binary_y + 1)

    assert np.array_equal(softmax.classes_, [1, 2])
    assert softmax.coef_.shape == (2, 2)
    assert np.allclose(softmax.coef_[1] - softmax.coef_[0], binary.coef_, atol=1e-12)
    assert np.isclose(softmax.intercept_[1] - softmax.intercept_[0], binary.intercept_)
    assert np.allclose(softmax.loss_history_, binary.loss_history_, atol=1e-12)


def test_two_class_softmax_newton_gives_binary_probabilities():
    X, y, _, _ = _three_class_problem()
    binary_y = (y > 0).astype(int)
    binary = LogisticRegression(solver="newton", tol=1e-13).fit(X, binary_y)
    softmax = LogisticRegression(solver="newton", tol=1e-13).fit(X, binary_y + 1)

    assert np.allclose(binary.predict_proba(X), softmax.predict_proba(X), atol=1e-9)


def test_predict_proba_and_predict_for_three_classes():
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(penalty="l2", alpha=0.1, learning_rate=0.5, max_iter=500)
    model.fit(X, y)

    proba = model.predict_proba(X)
    predicted = model.predict(X)

    assert proba.shape == (60, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.array_equal(predicted, np.argmax(proba, axis=1))
    assert predicted.dtype.kind == "i"
    assert model.score(X, y) > 0.85


def test_predict_returns_original_labels_in_sorted_order():
    X, y, _, _ = _three_class_problem()
    labels = np.array([3, 5, 7])[y]
    model = LogisticRegression(penalty="l2", alpha=0.1, learning_rate=0.5, max_iter=500)

    model.fit(X, labels)

    assert np.array_equal(model.classes_, [3, 5, 7])
    assert set(np.unique(model.predict(X))) <= {3, 5, 7}
    assert model.score(X, labels) > 0.85


def test_threshold_is_ignored_for_three_classes():
    X, y, _, _ = _three_class_problem()
    low = LogisticRegression(max_iter=50, threshold=0.1).fit(X, y)
    high = LogisticRegression(max_iter=50, threshold=0.9).fit(X, y)

    assert np.array_equal(low.predict(X), high.predict(X))


def test_multiclass_predict_rejects_wrong_n_features():
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(max_iter=5).fit(X, y)
    with pytest.raises(ValueError, match="3 features"):
        model.predict_proba(np.zeros((2, 3)))


def test_refit_switches_between_binary_and_multiclass():
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(max_iter=5)

    model.fit(X, y)
    assert model.coef_.shape == (3, 2)
    model.fit(X, (y > 0).astype(int))
    assert model.coef_.shape == (2,)
    assert model.predict_proba(X).shape == (60, 2)


def test_multiclass_without_intercept_keeps_zero_vector():
    X, y, _, _ = _three_class_problem()

    model = LogisticRegression(fit_intercept=False, max_iter=20, tol=0.0).fit(X, y)

    assert np.array_equal(model.intercept_, [0.0, 0.0, 0.0])
    assert np.all(model.coef_ != 0.0)


def test_multiclass_proximal_l1_matches_sklearn_saga_with_zeros():
    # Эталон: sklearn LogisticRegression(l1_ratio=1, solver="saga",
    # C=1 / (60 · 0.05), tol=1e-14) на признаках [X, шум]; те же 9 нулей
    X, y, _, noise = _three_class_problem()
    X = np.column_stack([X, noise])
    params = {"penalty": "l1", "alpha": 0.05, "learning_rate": 0.5, "tol": 0.0}
    proximal = LogisticRegression(solver="proximal", max_iter=30_000, **params)
    gradient = LogisticRegression(solver="gradient", max_iter=30_000, **params)

    proximal.fit(X, y)
    gradient.fit(X, y)

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.926320150437, 0.0, 0.0, 0.0],
            [0.0, 1.791409455719, 0.0, -0.062728617559],
        ]
    )
    assert np.allclose(proximal.coef_, expected, atol=1e-9)
    assert np.allclose(
        proximal.intercept_,
        [1.474802320365, -0.687887276274, -0.786915044091],
        atol=1e-9,
    )
    assert np.sum(proximal.coef_ == 0.0) == 9
    assert np.sum(gradient.coef_ == 0.0) == 0


def test_multiclass_gradient_history_layout_and_shape():
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(store_gradients=True, max_iter=3, tol=0.0).fit(X, y)

    history = model.gradient_history_
    grad_w, grad_b = _multiclass_model_at_zero(n_features=2)._gradient(X, y)

    assert history.shape == (3, 9)  # K · (d + 1) = 3 · 3
    # первая строка: градиент в нуле, по классам (w_k1, w_k2, b_k)
    assert np.allclose(history[0].reshape(3, 3)[:, :2], grad_w)
    assert np.allclose(history[0].reshape(3, 3)[:, 2], grad_b)


@pytest.mark.parametrize("solver", ["gradient", "newton"])
def test_multiclass_gradient_history_without_intercept_has_zero_columns(solver):
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(
        solver=solver, fit_intercept=False, store_gradients=True, max_iter=3, tol=0.0
    ).fit(X, y)

    assert model.gradient_history_.shape == (3, 9)
    assert np.all(model.gradient_history_.reshape(3, 3, 3)[:, :, 2] == 0.0)


def test_multiclass_mini_batch_and_nesterov_reduce_loss():
    X, y, _, _ = _three_class_problem()
    sgd = LogisticRegression(
        batch_size=8, learning_rate=0.1, max_iter=100, tol=0.0, random_state=0
    ).fit(X, y)
    nesterov = LogisticRegression(
        momentum=0.9, nesterov=True, learning_rate=0.1, max_iter=100, tol=0.0
    ).fit(X, y)

    assert sgd.loss_history_[-1] < 0.5 * sgd.loss_history_[0]
    assert nesterov.loss_history_[-1] < 0.5 * nesterov.loss_history_[0]


def test_multiclass_newton_rejects_non_smooth_penalty():
    X, y, _, _ = _three_class_problem()
    model = LogisticRegression(solver="newton", penalty="l1", alpha=0.1)
    with pytest.raises(ValueError, match="twice differentiable"):
        model.fit(X, y)
