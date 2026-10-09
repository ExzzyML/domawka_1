import numpy as np

from linear_models.base import BaseLinearModel


def _sigmoid(z):
    """Устойчивая сигмоида 1 / (1 + exp(-z))."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class LogisticRegression(BaseLinearModel):
    def fit(self, X, y):
        # TODO: будет в следующих шагах
        raise NotImplementedError

    def predict(self, X):
        # TODO: будет в следующих шагах
        raise NotImplementedError

    def _validate_binary_y(self, y):
        if not np.all((y == 0) | (y == 1)):
            raise ValueError(f"y must contain only 0 and 1, got values {np.unique(y)}")
        if np.unique(y).size < 2:
            raise ValueError("y must contain both classes 0 and 1")

    def _loss(self, X, y):
        z = self._linear_output(X)
        return np.mean(np.logaddexp(0, z) - y * z)

    def _gradient(self, X, y):
        residual = _sigmoid(self._linear_output(X)) - y
        n = X.shape[0]
        grad_w = (X.T @ residual) / n
        grad_b = np.sum(residual) / n
        return grad_w, grad_b
