import numpy as np
import pytest

from linear_models.base import BaseLinearModel, NotFittedError


class DummyModel(BaseLinearModel):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return self._linear_output(X)

    def _loss(self, X, y):
        return 0.0

    def _gradient(self, X, y):
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
    def _loss(self, X, y):
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

    def _loss(self, X, y):
        return 0.5 * np.sum(self.coef_**2) + 0.5 * self.intercept_**2

    def _gradient(self, X, y):
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
