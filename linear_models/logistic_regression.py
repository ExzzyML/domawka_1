import numpy as np

from linear_models.base import BaseLinearModel, _is_number


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
    """Логистическая регрессия для двух классов (0 и 1), solver только "gradient".

    Потери: среднее значение log(1 + e^z) - y * z. Класс 1 присваивается,
    если P(1) строго больше threshold (по умолчанию 0.5).
    """

    def __init__(self, threshold=0.5, **kwargs):
        """threshold: порог для predict; остальное см. BaseLinearModel."""

        super().__init__(**kwargs)
        self.threshold = threshold

    def _validate_threshold(self):
        if not _is_number(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError(
                f"threshold must be a number between 0 and 1, got {self.threshold!r}"
            )

    def _validate_params(self):
        super()._validate_params()
        self._validate_threshold()

    def fit(self, X, y):
        """Обучить на X формы (n, d) и y из {0, 1} (нужны оба класса); вернуть self."""

        self._validate_params()
        self._validate_solver()
        X, y = self._validate_X_y(X, y)
        self._validate_binary_y(y)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self._fit_gradient(X, y)
        return self

    def predict_proba(self, X):
        """Вернуть вероятности формы (n, 2): столбцы P(класс 0) и P(класс 1)."""

        X = self._validate_predict_input(X)
        p1 = _sigmoid(self._linear_output(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """Вернуть классы 0 и 1 (тип int): 1, если P(1) > threshold."""

        self._validate_threshold()
        p1 = self.predict_proba(X)[:, 1]
        return (p1 > self.threshold).astype(int)

    def _validate_binary_y(self, y):
        if not np.all((y == 0) | (y == 1)):
            raise ValueError(f"y must contain only 0 and 1, got values {np.unique(y)}")
        if np.unique(y).size < 2:
            raise ValueError("y must contain both classes 0 and 1")

    def _validate_solver(self):
        if self.solver != "gradient":
            raise ValueError(
                f"LogisticRegression supports only solver='gradient', "
                f"got {self.solver!r}"
            )

    def _loss(self, X, y):
        z = self._linear_output(X)
        return np.mean(np.logaddexp(0, z) - y * z)

    def _gradient(self, X, y):
        residual = _sigmoid(self._linear_output(X)) - y
        n = X.shape[0]
        grad_w = (X.T @ residual) / n
        grad_b = np.sum(residual) / n
        return grad_w, grad_b

    def score(self, X, y):
        """Accuracy: доля верно предсказанных классов."""
        X, y = self._validate_score_input(X, y)
        return float(np.mean(self.predict(X) == y))
