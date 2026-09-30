import numpy as np
import pytest

from linear_models.base import BaseLinearModel


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
    X = np.array([1.0, 2.0, 3.0])       # ndim == 1, а нужно 2
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="X must be 2"):
        model._validate_X_y(X, y)


def test_y_must_be_1d():
    model = DummyModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # (3, 2), корректный
    y = np.array([[1.0], [2.0], [3.0]])                   # (3, 1), ndim == 2, а нужно 1
    with pytest.raises(ValueError, match="y must be 1"):
        model._validate_X_y(X, y)


def test_X_and_y_must_have_same_length():
    model = DummyModel()
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])   # 3 объекта, shape (3, 2)
    y = np.array([1.0, 2.0])                              # 2 значения, shape (2,)
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