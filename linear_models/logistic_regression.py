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

    def _loss(self, X, y):
        z = self._linear_output(X)
        return np.mean(np.logaddexp(0, z) - y * z)

    def _gradient(self, X, y):
        # TODO: будет в следующих шагах
        raise NotImplementedError
