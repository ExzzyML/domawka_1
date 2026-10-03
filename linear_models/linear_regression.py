import numpy as np

from linear_models.base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    def fit(self, X, y):
        X, y = self._validate_X_y(X, y)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self._fit_gradient(X, y)
        return self

    def predict(self, X):
        X = self._validate_predict_input(X)
        return self._linear_output(X)

    def _loss(self, X, y):
        residual = self._linear_output(X) - y
        return np.sum(residual**2) / (2 * X.shape[0])

    def _gradient(self, X, y):
        residual = self._linear_output(X) - y
        n = X.shape[0]
        grad_w = (X.T @ residual) / n
        grad_b = np.sum(residual) / n
        return grad_w, grad_b
