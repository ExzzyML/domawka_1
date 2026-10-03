import numpy as np

from linear_models.base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    def fit(self, X, y):
        raise NotImplementedError  # появится в 4e

    def predict(self, X):
        raise NotImplementedError  # появится в 4d

    def _loss(self, X, y):
        residual = self._linear_output(X) - y
        return np.sum(residual**2) / (2 * X.shape[0])

    def _gradient(self, X, y):
        residual = self._linear_output(X) - y
        n = X.shape[0]
        grad_w = (X.T @ residual) / n
        grad_b = np.sum(residual) / n
        return grad_w, grad_b
