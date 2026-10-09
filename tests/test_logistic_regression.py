import numpy as np

from linear_models.logistic_regression import _sigmoid


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
