import warnings

import numpy as np
import pytest

from linear_models.base import BaseLinearModel, ConvergenceWarning, NotFittedError
from linear_models.linear_regression import LinearRegression
from linear_models.logistic_regression import LogisticRegression


class DummyModel(BaseLinearModel):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return self._linear_output(X)

    def _loss(self, X, y, sample_weight=None):
        return 0.0

    def _gradient(self, X, y, sample_weight=None):
        return 0.0, 0.0


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseLinearModel()


def test_dummy_model_stores_params():
    model = DummyModel(alpha=0.5, learning_rate=0.1, penalty="l2")
    assert model.alpha == 0.5
    assert model.learning_rate == 0.1
    assert model.penalty == "l2"
    assert model.max_iter == 1000


def test_linear_output():
    model = DummyModel()
    model.coef_ = np.array([1.0, 1.0])
    model.intercept_ = 0.5
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = model._linear_output(X)

    assert result.shape == (2,)
    assert np.allclose(result, [3.5, 7.5])


def test_invalid_solver_raises():
    model = DummyModel(solver="banana")
    with pytest.raises(ValueError, match="solver"):
        model._validate_params()


def test_invalid_penalty_raises():
    model = DummyModel(penalty="banana")
    with pytest.raises(ValueError, match="penalty"):
        model._validate_params()


@pytest.mark.parametrize("bad_value", [0, -0.1, "0.1", None, True, float("nan")])
def test_invalid_learning_rate_raises(bad_value):
    model = DummyModel(learning_rate=bad_value)
    with pytest.raises(ValueError, match="learning_rate"):
        model._validate_params()


def test_default_params_are_valid():
    DummyModel()._validate_params()


@pytest.mark.parametrize("bad_value", [0, -5, 10.5, 1000.0, "100", None, True])
def test_invalid_max_iter_raises(bad_value):
    model = DummyModel(max_iter=bad_value)
    with pytest.raises(ValueError, match="max_iter"):
        model._validate_params()


def test_numpy_integer_max_iter_is_valid():
    DummyModel(max_iter=np.int64(5))._validate_params()


@pytest.mark.parametrize("bad_value", [-0.1, "0.1", None, True, float("nan")])
def test_invalid_tol_raises(bad_value):
    model = DummyModel(tol=bad_value)
    with pytest.raises(ValueError, match="tol"):
        model._validate_params()


def test_numpy_tol_is_valid():
    DummyModel(tol=np.float64(1e-6))._validate_params()


@pytest.mark.parametrize("good_value", [0, 0.0, 1e-6])
def test_valid_tol_accepted(good_value):
    DummyModel(tol=good_value)._validate_params()


@pytest.mark.parametrize("bad_value", [-0.1, "0.1", None, True, float("nan")])
def test_invalid_alpha_raises(bad_value):
    model = DummyModel(alpha=bad_value)
    with pytest.raises(ValueError, match="alpha"):
        model._validate_params()


def test_numpy_alpha_is_valid():
    DummyModel(alpha=np.float64(1e-6))._validate_params()


@pytest.mark.parametrize("good_value", [0, 0.0, 1e-6])
def test_valid_alpha_accepted(good_value):
    DummyModel(alpha=good_value)._validate_params()


def test_X_must_be_2d():
    model = DummyModel()
    X = np.array([1.0, 2.0, 3.0])  # ndim == 1, а нужно 2
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="X must be 2"):
        model._validate_X_y(X, y)


def test_y_must_be_1d():
    model = DummyModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2), корректный
    y = np.array([[1.0], [2.0], [3.0]])  # (3, 1), ndim == 2, а нужно 1
    with pytest.raises(ValueError, match="y must be 1"):
        model._validate_X_y(X, y)


def test_X_and_y_must_have_same_length():
    model = DummyModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 объекта, shape (3, 2)
    y = np.array([1.0, 2.0])  # 2 значения, shape (2,)
    with pytest.raises(ValueError, match="samples"):
        model._validate_X_y(X, y)


def test_valid_X_y_are_returned_as_arrays():
    model = DummyModel()
    X_out, y_out = model._validate_X_y([[1, 2], [3, 4]], [5, 6])
    assert X_out.shape == (2, 2)
    assert y_out.shape == (2,)
    assert X_out.dtype == float


@pytest.mark.parametrize("bad_X", [[1.0, 2.0, 3.0], [[[1.0]]]])
def test_validate_X_rejects_wrong_ndim(bad_X):
    model = DummyModel()
    with pytest.raises(ValueError, match="X must be 2"):
        model._validate_X(bad_X)


def test_validate_X_returns_float_array():
    model = DummyModel()
    X_out = model._validate_X([[1, 2], [3, 4]])
    assert X_out.shape == (2, 2)
    assert X_out.dtype == float


def test_check_is_fitted_raises_before_fit():
    model = DummyModel()
    with pytest.raises(NotFittedError, match="not fitted"):
        model._check_is_fitted()


def test_check_is_fitted_passes_after_coef_set():
    model = DummyModel()
    model.coef_ = np.array([1.0])
    model.intercept_ = 0.0
    model._check_is_fitted()


def test_predict_input_rejects_wrong_n_features():
    model = DummyModel()
    model.coef_ = np.array([1.0, 1.0])  # обучена на 2 признаках
    model.intercept_ = 0.0
    X = np.array([[1.0, 2.0, 3.0]])  # пришло 3 признака
    with pytest.raises(ValueError, match="features"):
        model._validate_predict_input(X)


def test_predict_input_requires_fitted_model():
    model = DummyModel()
    with pytest.raises(NotFittedError):
        model._validate_predict_input([[1.0, 2.0]])


def test_predict_input_accepts_correct_input():
    model = DummyModel()
    model.coef_ = np.array([1.0, 1.0])
    model.intercept_ = 0.0
    X_out = model._validate_predict_input([[1, 2], [3, 4]])
    assert X_out.shape == (2, 2)


def _model_with_weights(penalty):
    model = DummyModel(penalty=penalty, alpha=0.1)
    model.coef_ = np.array([3.0, -4.0])
    model.intercept_ = 100.0
    return model


@pytest.mark.parametrize(
    "penalty, expected",
    [(None, 0.0), ("l2", 1.25), ("l1", 0.7)],
)
def test_penalty_loss(penalty, expected):
    model = _model_with_weights(penalty)
    assert np.isclose(model._penalty_loss(), expected)


@pytest.mark.parametrize(
    "penalty, expected",
    [(None, [0.0, 0.0]), ("l2", [0.3, -0.4]), ("l1", [0.1, -0.1])],
)
def test_penalty_gradient(penalty, expected):
    model = _model_with_weights(penalty)

    result = model._penalty_gradient()

    assert result.shape == (2,)
    assert np.allclose(result, expected)


class FixedLossModel(DummyModel):
    def _loss(self, X, y, sample_weight=None):
        return 2.5


@pytest.mark.parametrize(
    "penalty, expected",
    [(None, 2.5), ("l2", 3.75), ("l1", 3.2)],
)
def test_objective_is_data_loss_plus_penalty(penalty, expected):
    model = FixedLossModel(penalty=penalty, alpha=0.1)
    model.coef_ = np.array([3.0, -4.0])
    model.intercept_ = 100.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    result = model._objective(X, y)

    assert np.isclose(result, expected)


class QuadraticModel(DummyModel):
    """L(w, b) = 0.5 * sum(w**2) + 0.5 * b**2, градиент (w, b)."""

    def _loss(self, X, y, sample_weight=None):
        return 0.5 * np.sum(self.coef_**2) + 0.5 * self.intercept_**2

    def _gradient(self, X, y, sample_weight=None):
        return self.coef_.copy(), self.intercept_


def test_fit_gradient_follows_descent_steps():
    model = QuadraticModel(learning_rate=0.5, max_iter=3, tol=0.0)
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    model._fit_gradient(X, y)

    assert np.allclose(model.coef_, [0.5, -0.25])
    assert np.isclose(model.intercept_, 0.25)
    assert model.n_iter_ == 3
    assert len(model.loss_history_) == 3
    assert np.allclose(model.loss_history_, [3.0, 0.75, 0.1875])


def test_fit_gradient_includes_penalty_gradient():
    model = QuadraticModel(
        penalty="l2", alpha=0.5, learning_rate=0.5, max_iter=1, tol=0.0
    )
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    model._fit_gradient(X, y)

    assert np.allclose(model.coef_, [1.0, -0.5])
    assert np.isclose(model.intercept_, 1.0)
    assert model.n_iter_ == 1
    assert len(model.loss_history_) == 1
    assert np.allclose(model.loss_history_, [1.4375])


def test_fit_gradient_stops_early_when_objective_change_is_below_tol():
    model = QuadraticModel(learning_rate=0.5, max_iter=10, tol=1.0)
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    model._fit_gradient(X, y)

    assert model.n_iter_ == 3
    assert model.n_iter_ < model.max_iter
    assert len(model.loss_history_) == 3
    assert np.allclose(model.loss_history_, [3.0, 0.75, 0.1875])
    assert np.allclose(model.coef_, [0.5, -0.25])
    assert np.isclose(model.intercept_, 0.25)


def test_fit_gradient_runs_all_iterations_when_tol_is_zero():
    model = QuadraticModel(learning_rate=0.5, max_iter=10, tol=0.0)
    model.coef_ = np.array([0.0, 0.0])
    model.intercept_ = 0.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    model._fit_gradient(X, y)

    assert model.n_iter_ == 10
    assert len(model.loss_history_) == 10


def test_fit_gradient_keeps_intercept_when_fit_intercept_is_false():
    model = QuadraticModel(fit_intercept=False, learning_rate=0.5, max_iter=1, tol=0.0)
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])

    model._fit_gradient(X, y)

    assert np.allclose(model.coef_, [2.0, -1.0])
    assert model.intercept_ == 2.0
    assert model.n_iter_ == 1
    assert np.allclose(model.loss_history_, [4.5])


def _elasticnet_model(l1_ratio, penalty="elasticnet"):
    model = DummyModel(penalty=penalty, alpha=0.1, l1_ratio=l1_ratio)
    model.coef_ = np.array([3.0, -4.0])
    model.intercept_ = 100.0
    return model


def test_penalty_loss_elasticnet():
    # alpha * (0.25 * 7 + 0.75 * 0.5 * 25) = 0.1 * 11.125
    assert np.isclose(_elasticnet_model(0.25)._penalty_loss(), 1.1125)


def test_penalty_gradient_elasticnet():
    # 0.1 * (0.25 * sign(w) + 0.75 * w) = 0.1 * [2.5, -3.25]
    result = _elasticnet_model(0.25)._penalty_gradient()

    assert result.shape == (2,)
    assert np.allclose(result, [0.25, -0.325])


@pytest.mark.parametrize("l1_ratio, penalty", [(0.0, "l2"), (1.0, "l1")])
def test_elasticnet_edges_match_pure_penalties(l1_ratio, penalty):
    mixed = _elasticnet_model(l1_ratio)
    pure = _elasticnet_model(0.5, penalty=penalty)

    assert np.isclose(mixed._penalty_loss(), pure._penalty_loss())
    assert np.allclose(mixed._penalty_gradient(), pure._penalty_gradient())


def test_default_l1_ratio_is_half():
    assert DummyModel().l1_ratio == 0.5


@pytest.mark.parametrize(
    "bad_value", [-0.1, 1.1, "0.5", None, True, float("nan"), float("inf")]
)
def test_invalid_l1_ratio_raises(bad_value):
    model = DummyModel(penalty="elasticnet", l1_ratio=bad_value)
    with pytest.raises(ValueError, match="l1_ratio"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [0, 0.0, 0.3, 1, 1.0, np.float64(0.7)])
def test_valid_l1_ratio_accepted(good_value):
    DummyModel(penalty="elasticnet", l1_ratio=good_value)._validate_params()


def _quadratic_after_three_steps(**params):
    model = QuadraticModel(learning_rate=0.5, max_iter=3, tol=0.0, **params)
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])
    model._fit_gradient(X, y)
    return model


def test_default_momentum_and_schedule():
    model = DummyModel()

    assert model.momentum == 0.0
    assert model.lr_schedule is None
    assert model.lr_decay == 0.9


def test_fit_gradient_with_momentum_overshoots_minimum():
    # v = 0.5 * v + g, w = w - 0.5 * v; для w0 = 4: 4 -> 2 -> 0 -> -1
    model = _quadratic_after_three_steps(momentum=0.5)

    assert np.allclose(model.coef_, [-1.0, 0.5])
    assert np.isclose(model.intercept_, -0.5)
    assert np.allclose(model.loss_history_, [3.0, 0.0, 0.75])


def test_fit_gradient_with_exponential_schedule():
    # шаги 0.5, 0.25, 0.125; для w0 = 4: 4 -> 2 -> 1.5 -> 1.3125
    model = _quadratic_after_three_steps(lr_schedule="exponential", lr_decay=0.5)

    assert np.allclose(model.coef_, [1.3125, -0.65625])
    assert np.isclose(model.intercept_, 0.65625)
    assert np.allclose(model.loss_history_, [3.0, 1.6875, 1.2919921875])


@pytest.mark.parametrize(
    "schedule, decay, expected",
    [
        (None, 0.5, [0.5, 0.5, 0.5]),
        ("exponential", 0.5, [0.5, 0.25, 0.125]),
        ("inverse_time", 1.0, [0.5, 0.25, 0.5 / 3]),
    ],
)
def test_learning_rate_at(schedule, decay, expected):
    model = DummyModel(learning_rate=0.5, lr_schedule=schedule, lr_decay=decay)

    result = [model._learning_rate_at(t) for t in range(3)]

    assert np.allclose(result, expected)


@pytest.mark.parametrize("bad_value", [-0.1, 1.0, 1.5, "0.5", None, True, float("nan")])
def test_invalid_momentum_raises(bad_value):
    model = DummyModel(momentum=bad_value)
    with pytest.raises(ValueError, match="momentum"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [0, 0.0, 0.5, 0.99, np.float64(0.9)])
def test_valid_momentum_accepted(good_value):
    DummyModel(momentum=good_value)._validate_params()


@pytest.mark.parametrize("bad_value", ["banana", 1, True, "Exponential"])
def test_invalid_lr_schedule_raises(bad_value):
    model = DummyModel(lr_schedule=bad_value)
    with pytest.raises(ValueError, match="lr_schedule"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [None, "exponential", "inverse_time"])
def test_valid_lr_schedule_accepted(good_value):
    DummyModel(lr_schedule=good_value)._validate_params()


@pytest.mark.parametrize("bad_value", [0, -0.5, "0.5", None, True, float("nan")])
def test_invalid_lr_decay_raises(bad_value):
    model = DummyModel(lr_decay=bad_value)
    with pytest.raises(ValueError, match="lr_decay"):
        model._validate_params()


def test_exponential_decay_above_one_raises():
    model = DummyModel(lr_schedule="exponential", lr_decay=1.5)
    with pytest.raises(ValueError, match="lr_decay"):
        model._validate_params()


def test_inverse_time_decay_above_one_is_valid():
    DummyModel(lr_schedule="inverse_time", lr_decay=1.5)._validate_params()


def _quadratic_model(**params):
    params = {"learning_rate": 0.5, "max_iter": 3, "tol": 0.0, **params}
    model = QuadraticModel(**params)
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    return model


_XY = (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.0, 1.0]))


def test_default_store_gradients_and_callbacks():
    model = DummyModel()

    assert model.store_gradients is False
    assert model.callbacks is None


def test_gradient_history_is_off_by_default():
    model = _quadratic_model()
    model._fit_gradient(*_XY)

    assert not hasattr(model, "gradient_history_")


def test_gradient_history_stores_gradient_before_each_step():
    model = _quadratic_model(store_gradients=True)
    model._fit_gradient(*_XY)

    expected = [[4.0, -2.0, 2.0], [2.0, -1.0, 1.0], [1.0, -0.5, 0.5]]
    assert model.gradient_history_.shape == (3, 3)
    assert np.allclose(model.gradient_history_, expected)


def test_gradient_history_has_zero_intercept_column_without_intercept():
    model = _quadratic_model(store_gradients=True, fit_intercept=False)
    model._fit_gradient(*_XY)

    expected = [[4.0, -2.0, 0.0], [2.0, -1.0, 0.0], [1.0, -0.5, 0.0]]
    assert np.allclose(model.gradient_history_, expected)


def test_gradient_history_includes_penalty_gradient():
    model = _quadratic_model(store_gradients=True, penalty="l2", alpha=0.5, max_iter=1)
    model._fit_gradient(*_XY)

    assert np.allclose(model.gradient_history_, [[6.0, -3.0, 2.0]])


def test_gradient_history_is_removed_when_refit_without_storing():
    model = _quadratic_model(store_gradients=True)
    model._fit_gradient(*_XY)
    model.store_gradients = False
    model._fit_gradient(*_XY)

    assert not hasattr(model, "gradient_history_")


def test_callback_receives_model_iteration_and_loss():
    calls = []

    def record(model, iteration, loss):
        calls.append((model, iteration, loss))

    model = _quadratic_model(callbacks=[record])
    model._fit_gradient(*_XY)

    assert [c[1] for c in calls] == [1, 2, 3]
    assert np.allclose([c[2] for c in calls], [3.0, 0.75, 0.1875])
    assert all(c[0] is model for c in calls)


def test_callback_returning_true_stops_training():
    model = _quadratic_model(callbacks=[lambda m, i, loss: i == 2], max_iter=10)
    model._fit_gradient(*_XY)

    assert model.n_iter_ == 2
    assert len(model.loss_history_) == 2


def test_callback_returning_other_values_does_not_stop():
    model = _quadratic_model(callbacks=[lambda m, i, loss: 1])
    model._fit_gradient(*_XY)

    assert model.n_iter_ == 3


def test_all_callbacks_are_called_even_if_first_requests_stop():
    calls = []

    def stopper(model, iteration, loss):
        return True

    def recorder(model, iteration, loss):
        calls.append(iteration)

    model = _quadratic_model(callbacks=(stopper, recorder), max_iter=10)
    model._fit_gradient(*_XY)

    assert model.n_iter_ == 1
    assert calls == [1]


@pytest.mark.parametrize("bad_value", [print, "f", 5, [1], [print, "x"], {"a": print}])
def test_invalid_callbacks_raise(bad_value):
    model = DummyModel(callbacks=bad_value)
    with pytest.raises(ValueError, match="callbacks"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [None, [], [print], (print, len)])
def test_valid_callbacks_accepted(good_value):
    DummyModel(callbacks=good_value)._validate_params()


@pytest.mark.parametrize("bad_value", [1, 0, "yes", None])
def test_invalid_store_gradients_raises(bad_value):
    model = DummyModel(store_gradients=bad_value)
    with pytest.raises(ValueError, match="store_gradients"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [True, False])
def test_valid_store_gradients_accepted(good_value):
    DummyModel(store_gradients=good_value)._validate_params()


def test_verbose_is_silent_by_default(capsys):
    _quadratic_model()._fit_gradient(*_XY)

    assert capsys.readouterr().out == ""


def test_verbose_prints_first_iteration_and_summary(capsys):
    _quadratic_model(verbose=True)._fit_gradient(*_XY)

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "iter 1: J=3",
        "stopped at iteration 3: J=0.1875, reason: max_iter",
    ]


def test_verbose_prints_every_hundredth_iteration(capsys):
    model = _quadratic_model(verbose=True, learning_rate=0.01, max_iter=250)
    model._fit_gradient(*_XY)

    iter_lines = [
        line.split(":")[0]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("iter ")
    ]
    assert iter_lines == ["iter 1", "iter 100", "iter 200"]


def test_verbose_reports_tol_and_callback_reasons(capsys):
    _quadratic_model(verbose=True, tol=1e-3, max_iter=100)._fit_gradient(*_XY)
    tol_out = capsys.readouterr().out
    model = _quadratic_model(verbose=True, callbacks=[lambda m, i, loss: i == 2])
    model._fit_gradient(*_XY)
    callback_out = capsys.readouterr().out

    assert "reason: tol" in tol_out
    assert "stopped at iteration 2" in callback_out
    assert "reason: callback" in callback_out


class RecordingModel(QuadraticModel):
    """Запоминает, какие sample_weight пришли в _loss и _gradient."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    def _loss(self, X, y, sample_weight=None):
        self.seen.append(sample_weight)
        return super()._loss(X, y, sample_weight)

    def _gradient(self, X, y, sample_weight=None):
        self.seen.append(sample_weight)
        return super()._gradient(X, y, sample_weight)


def test_validate_sample_weight_none_stays_none():
    assert DummyModel()._validate_sample_weight(None, 3) is None


def test_validate_sample_weight_returns_float_array():
    result = DummyModel()._validate_sample_weight([1, 2, 0], 3)

    assert result.dtype == float
    assert np.array_equal(result, [1.0, 2.0, 0.0])


@pytest.mark.parametrize(
    "bad_value",
    [
        [[1.0, 2.0, 3.0]],
        [1.0, 2.0],
        [1.0, -1.0, 2.0],
        [1.0, float("nan"), 2.0],
        [1.0, float("inf"), 2.0],
        [0.0, 0.0, 0.0],
        ["a", "b", "c"],
    ],
)
def test_validate_sample_weight_rejects_bad_values(bad_value):
    with pytest.raises(ValueError, match="sample_weight"):
        DummyModel()._validate_sample_weight(bad_value, 3)


def test_total_weight_and_apply_weight():
    weights = np.array([3.0, 1.0])
    values = np.array([2.0, 5.0])

    assert DummyModel._total_weight(None, 2) == 2
    assert DummyModel._total_weight(weights, 2) == 4.0
    assert DummyModel._apply_weight(values, None) is values
    assert np.array_equal(DummyModel._apply_weight(values, weights), [6.0, 5.0])


def test_fit_gradient_passes_sample_weight_to_loss_and_gradient():
    weights = np.array([3.0, 1.0])
    model = RecordingModel(learning_rate=0.5, max_iter=2, tol=0.0)
    model.coef_ = np.array([4.0, -2.0])
    model.intercept_ = 2.0
    model._fit_gradient(*_XY, sample_weight=weights)

    assert len(model.seen) == 4
    assert all(seen is weights for seen in model.seen)


def test_default_batch_size_shuffle_and_random_state():
    model = DummyModel()

    assert model.batch_size is None
    assert model.shuffle is True
    assert model.random_state is None


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "2", True, float("nan")])
def test_invalid_batch_size_raises(bad_value):
    model = DummyModel(batch_size=bad_value)
    with pytest.raises(ValueError, match="batch_size"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [None, 1, 5, np.int64(3)])
def test_valid_batch_size_accepted(good_value):
    DummyModel(batch_size=good_value)._validate_params()


@pytest.mark.parametrize("bad_value", [1, 0, "yes", None])
def test_invalid_shuffle_raises(bad_value):
    model = DummyModel(shuffle=bad_value)
    with pytest.raises(ValueError, match="shuffle"):
        model._validate_params()


@pytest.mark.parametrize("bad_value", [-1, 1.5, "a", True, float("nan")])
def test_invalid_random_state_raises(bad_value):
    model = DummyModel(random_state=bad_value)
    with pytest.raises(ValueError, match="random_state"):
        model._validate_params()


@pytest.mark.parametrize("good_value", [None, 0, 42, np.int64(7)])
def test_valid_random_state_accepted(good_value):
    DummyModel(random_state=good_value)._validate_params()


def _batches(n_samples, batch_size, **params):
    model = DummyModel(**params)
    rng = np.random.default_rng(params.get("random_state"))
    return list(model._epoch_batches(n_samples, batch_size, rng))


def test_epoch_batches_without_shuffle_keep_order_and_last_is_smaller():
    result = _batches(5, 2, shuffle=False)

    assert [list(b) for b in result] == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize("batch_size", [5, 8])
def test_epoch_batches_full_batch_is_a_single_none(batch_size):
    assert _batches(5, batch_size) == [None]


def test_epoch_batches_with_shuffle_are_a_reproducible_permutation():
    first = _batches(7, 3, shuffle=True, random_state=0)
    second = _batches(7, 3, shuffle=True, random_state=0)

    assert [len(b) for b in first] == [3, 3, 1]
    assert sorted(np.concatenate(first)) == list(range(7))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


# --- NaN, inf и нечисловые значения во входных данных ---

_BAD_NUMBERS = [float("nan"), float("inf"), -float("inf")]


@pytest.mark.parametrize("bad", _BAD_NUMBERS)
def test_validate_X_rejects_nan_and_inf(bad):
    X = [[1.0, 2.0], [3.0, bad]]
    with pytest.raises(ValueError, match="X must not contain NaN or infinity"):
        DummyModel()._validate_X(X)


@pytest.mark.parametrize("bad", _BAD_NUMBERS)
def test_validate_X_y_rejects_nan_and_inf_in_y(bad):
    X = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError, match="y must not contain NaN or infinity"):
        DummyModel()._validate_X_y(X, [1.0, bad])


def test_validate_X_rejects_strings():
    with pytest.raises(ValueError, match="X must contain only numbers"):
        DummyModel()._validate_X([["a", "b"], ["c", "d"]])


def test_validate_X_y_rejects_strings_in_y():
    with pytest.raises(ValueError, match="y must contain only numbers"):
        DummyModel()._validate_X_y([[1.0], [2.0]], ["a", "b"])


def test_numeric_strings_are_still_accepted():
    X_out, y_out = DummyModel()._validate_X_y([["1", "2"]], ["3"])
    assert X_out.dtype == float and y_out.dtype == float


@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(solver="analytic"),
        LinearRegression(solver="gradient", max_iter=5),
        LogisticRegression(max_iter=5),
    ],
)
def test_fit_rejects_nan_in_X(model):
    X = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 0.0], [3.0, 1.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.fit(X, y)


@pytest.mark.parametrize(
    "model",
    [LinearRegression(solver="analytic"), LogisticRegression(max_iter=5)],
)
def test_predict_and_score_reject_nan(model):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model.fit(X, y)
    X_bad = X.copy()
    X_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="X must not contain NaN or infinity"):
        model.predict(X_bad)
    with pytest.raises(ValueError, match="y must not contain NaN or infinity"):
        model.score(X, [0.0, np.nan, 0.0, 1.0])


# --- ConvergenceWarning: не сошлись за max_iter или разошлись ---

_X_LINE = np.arange(6, dtype=float).reshape(-1, 1)
_Y_LINE = 2 * _X_LINE[:, 0] + 1


def _fit_without_warnings(model, X=_X_LINE, y=_Y_LINE):
    """Обучить, превращая любое предупреждение в ошибку."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return model.fit(X, y)


def test_convergence_warning_is_a_user_warning():
    assert issubclass(ConvergenceWarning, UserWarning)


def test_warns_when_max_iter_is_reached():
    model = LinearRegression(learning_rate=0.01, max_iter=3, tol=1e-6)
    with pytest.warns(ConvergenceWarning, match="did not converge in max_iter=3"):
        model.fit(_X_LINE, _Y_LINE)
    assert model.n_iter_ == 3


def test_no_warning_when_converged_by_tol():
    model = _fit_without_warnings(
        LinearRegression(learning_rate=0.05, max_iter=10_000, tol=1e-10)
    )
    assert model.n_iter_ < 10_000


def test_no_warning_when_tol_is_zero():
    model = _fit_without_warnings(LinearRegression(max_iter=5, tol=0.0))
    assert model.n_iter_ == 5


def test_no_warning_when_callback_stops_training():
    model = _fit_without_warnings(
        LinearRegression(max_iter=100, callbacks=[lambda m, i, loss: i == 2])
    )
    assert model.n_iter_ == 2


def test_no_warning_for_analytic_solver():
    _fit_without_warnings(LinearRegression(solver="analytic"))


def test_divergence_stops_training_with_one_clear_warning():
    model = LinearRegression(learning_rate=10.0, max_iter=1000)
    X = np.array([[0.0, 100.0], [1.0, -50.0], [2.0, 80.0], [3.0, -90.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    messages = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert len(messages) == 1 and "diverged" in str(messages[0].message)
    assert all(issubclass(w.category, ConvergenceWarning) for w in caught)
    assert model.n_iter_ < 1000
    assert not np.isfinite(model.loss_history_[-1])


def test_verbose_reports_diverged_reason(capsys):
    model = LinearRegression(learning_rate=10.0, max_iter=1000, verbose=True)
    with pytest.warns(ConvergenceWarning):
        model.fit(np.array([[0.0], [100.0], [-80.0]]), np.array([0.0, 1.0, 2.0]))
    assert "reason: diverged" in capsys.readouterr().out