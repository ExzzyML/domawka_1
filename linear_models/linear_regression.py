import numpy as np

from linear_models.base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    """Линейная регрессия: z = X @ coef_ + intercept_, потери 1/(2n) * сумма (z - y)^2.

    solver="analytic" даёт точное решение (penalty None или "l2"),
    solver="gradient" работает с любым penalty.
    """

    def fit(self, X, y, sample_weight=None):
        """Обучить на X формы (n, d) и y формы (n,); вернуть self.

        sample_weight: веса объектов (n,), неотрицательные, сумма > 0; вес k
        равносилен k копиям объекта. None: все веса равны 1.
        """
        self._validate_params()
        self._validate_solver_penalty()
        X, y = self._validate_X_y(X, y)
        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        if self.solver == "analytic":
            self._fit_analytic(X, y, sample_weight)
            return self
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self._fit_gradient(X, y, sample_weight)
        return self

    def _fit_analytic(self, X, y, sample_weight=None):
        n = X.shape[0]
        weights = np.ones(n) if sample_weight is None else sample_weight
        total = np.sum(weights)
        root = np.sqrt(weights)
        if self.fit_intercept:
            x_mean = np.average(X, axis=0, weights=weights)
            y_mean = np.average(y, weights=weights)
            Xc = X - x_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y
        Xc = Xc * root[:, None]
        yc = yc * root
        if self.penalty == "l2":
            d = Xc.shape[1]
            penalty_rows = np.sqrt(total * self.alpha) * np.eye(d)
            Xc = np.vstack([Xc, penalty_rows])
            yc = np.concatenate([yc, np.zeros(d)])
        coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        self.coef_ = coef
        if self.fit_intercept:
            self.intercept_ = y_mean - x_mean @ coef
        else:
            self.intercept_ = 0.0
        self.n_iter_ = 1
        self.loss_history_ = [self._objective(X, y, sample_weight)]

    def _validate_solver_penalty(self):
        if self.solver == "analytic" and self.penalty not in (None, "l2"):
            raise ValueError(
                f"solver='analytic' supports only penalty None or 'l2', "
                f"got {self.penalty!r}"
            )
        if self.solver == "analytic" and self.store_gradients:
            raise ValueError(
                "store_gradients=True is not supported by solver='analytic': "
                "it has no iterations"
            )

    def predict(self, X):
        """Вернуть предсказания z = X @ coef_ + intercept_ формы (n,)."""

        X = self._validate_predict_input(X)
        return self._linear_output(X)

    def score(self, X, y, sample_weight=None):
        """R^2 = 1 - SS_res / SS_tot (с весами: взвешенные суммы и среднее)."""
        X, y = self._validate_score_input(X, y)
        weights = self._validate_sample_weight(sample_weight, len(y))
        y = y.astype(float)
        residual = y - self.predict(X)
        y_mean = np.mean(y) if weights is None else np.average(y, weights=weights)
        centered = y - y_mean
        ss_res = np.sum(self._apply_weight(residual, weights) * residual)
        ss_tot = np.sum(self._apply_weight(centered, weights) * centered)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def _residual(self, X, y):
        residual = self._linear_output(X) - y
        return residual

    def _loss(self, X, y, sample_weight=None):
        residual = self._residual(X, y)
        total = self._total_weight(sample_weight, X.shape[0])
        weighted = self._apply_weight(residual, sample_weight)
        return np.sum(weighted * residual) / (2 * total)

    def _gradient(self, X, y, sample_weight=None):
        residual = self._residual(X, y)
        total = self._total_weight(sample_weight, X.shape[0])
        weighted = self._apply_weight(residual, sample_weight)
        grad_w = (X.T @ weighted) / total
        grad_b = np.sum(weighted) / total
        return grad_w, grad_b
