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

    def fit(self, X, y, sample_weight=None):
        """Обучить на X формы (n, d) и y из {0, 1}; вернуть self.

        Нужны оба класса среди объектов с положительным весом. sample_weight:
        веса объектов (n,), неотрицательные, сумма > 0; None: все веса равны 1.
        """
        self._validate_params()
        self._validate_solver()
        X, y = self._validate_X_y(X, y)
        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        self._validate_binary_y(y, sample_weight)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self._fit_gradient(X, y, sample_weight)
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

    def _validate_binary_y(self, y, sample_weight=None):
        if not np.all((y == 0) | (y == 1)):
            raise ValueError(f"y must contain only 0 and 1, got values {np.unique(y)}")
        used = y if sample_weight is None else y[sample_weight > 0]
        if np.unique(used).size < 2:
            raise ValueError(
                "y must contain both classes 0 and 1 "
                "(among samples with positive sample_weight)"
            )

    def _validate_solver(self):
        if self.solver != "gradient":
            raise ValueError(
                f"LogisticRegression supports only solver='gradient', "
                f"got {self.solver!r}"
            )

    def score(self, X, y, sample_weight=None):
        """Accuracy: доля верно предсказанных классов (с весами: взвешенная доля)."""
        X, y = self._validate_score_input(X, y)
        weights = self._validate_sample_weight(sample_weight, len(y))
        correct = self.predict(X) == y
        if weights is None:
            return float(np.mean(correct))
        return float(np.average(correct, weights=weights))

    def _loss(self, X, y, sample_weight=None):
        z = self._linear_output(X)
        per_object = np.logaddexp(0, z) - y * z
        total = self._total_weight(sample_weight, X.shape[0])
        return np.sum(self._apply_weight(per_object, sample_weight)) / total

    def _gradient(self, X, y, sample_weight=None):
        residual = _sigmoid(self._linear_output(X)) - y
        total = self._total_weight(sample_weight, X.shape[0])
        weighted = self._apply_weight(residual, sample_weight)
        grad_w = (X.T @ weighted) / total
        grad_b = np.sum(weighted) / total
        return grad_w, grad_b
