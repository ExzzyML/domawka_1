import numpy as np

from linear_models.base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    def fit(self, X, y):
        self._validate_params()
        self._validate_solver_penalty()
        X, y = self._validate_X_y(X, y)
        if self.solver == "analytic":
            self._fit_analytic(X, y)
            return self
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self._fit_gradient(X, y)
        return self

    def _fit_analytic(self, X, y):
        if self.fit_intercept:
            x_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xc = X - x_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y
        if self.penalty == "l2":
            n = X.shape[0]
            d = Xc.shape[1]
            penalty_rows = np.sqrt(n * self.alpha) * np.eye(d)
            Xc = np.vstack([Xc, penalty_rows])
            yc = np.concatenate([yc, np.zeros(d)])
        coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        self.coef_ = coef
        if self.fit_intercept:
            self.intercept_ = y_mean - x_mean @ coef
        else:
            self.intercept_ = 0.0
        self.n_iter_ = 1
        self.loss_history_ = [self._objective(X, y)]

    def _validate_solver_penalty(self):
        if self.solver == "analytic" and self.penalty not in (None, "l2"):
            raise ValueError(
                f"solver='analytic' supports only penalty None or 'l2', "
                f"got {self.penalty!r}"
            )

    def predict(self, X):
        X = self._validate_predict_input(X)
        return self._linear_output(X)

    def _residual(self, X, y):
        residual = self._linear_output(X) - y
        return residual

    def _loss(self, X, y):
        residual = self._residual(X, y)
        return np.sum(residual**2) / (2 * X.shape[0])

    def _gradient(self, X, y):
        residual = self._residual(X, y)
        n = X.shape[0]
        grad_w = (X.T @ residual) / n
        grad_b = np.sum(residual) / n
        return grad_w, grad_b
