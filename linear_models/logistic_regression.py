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
    """Логистическая регрессия для двух классов (0 и 1).

    Потери: среднее значение log(1 + e^z) - y * z. Класс 1 присваивается,
    если P(1) строго больше threshold (по умолчанию 0.5).

    solver: "gradient" (любой penalty), "proximal" (проксимальный спуск, точные
    нули при L1 и elasticnet) или "newton" (метод Ньютона с дроблением шага,
    penalty только None или "l2"; learning_rate, momentum и расписание шага
    не используются).
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
        if self.solver == "newton":
            self._fit_newton(X, y, sample_weight)
        else:
            self._fit_gradient(X, y, sample_weight)
        return self

    def _fit_newton(self, X, y, sample_weight=None):
        """Метод Ньютона; предупреждения numpy о переполнении отключены."""
        with np.errstate(over="ignore", invalid="ignore"):
            return self._newton_method(X, y, sample_weight)

    def _newton_method(self, X, y, sample_weight=None):
        def step(i, gradients):
            grad, hessian = self._gradient_and_hessian(X, y, sample_weight)
            if self.store_gradients:
                gradients.append(grad if self.fit_intercept else np.append(grad, 0.0))
            direction = np.linalg.lstsq(hessian, grad, rcond=None)[0]
            self._newton_line_search(X, y, sample_weight, grad, direction)

        return self._run_iterations(X, y, sample_weight, step)

    def _params_vector(self):
        """θ = (w, b) при fit_intercept, иначе θ = w."""
        if self.fit_intercept:
            return np.append(self.coef_, self.intercept_)
        return self.coef_.copy()

    def _set_params_vector(self, theta):
        d = self.coef_.shape[0]
        self.coef_ = theta[:d].copy()
        if self.fit_intercept:
            self.intercept_ = float(theta[d])

    def _gradient_and_hessian(self, X, y, sample_weight=None):
        """Градиент g и гессиан H цели J по θ = (w, b) (или по w без интерцепта).

        g = Xa^T (s · (p - y)) / Σs + alpha·(w, 0),
        H = Xa^T diag(s · p(1 - p)) Xa / Σs + alpha·diag(1, ..., 1, 0),
        где Xa = [X, 1] (столбец единиц только при fit_intercept), s веса
        объектов, p = σ(z). Штраф только L2 (или его нет), интерцепт не штрафуется.
        """
        n, d = X.shape
        p = _sigmoid(self._linear_output(X))
        total = self._total_weight(sample_weight, n)
        Xa = np.column_stack([X, np.ones(n)]) if self.fit_intercept else X
        residual = self._apply_weight(p - y, sample_weight)
        curvature = self._apply_weight(p * (1 - p), sample_weight)
        grad = Xa.T @ residual / total
        hessian = (Xa.T * curvature) @ Xa / total
        if self.penalty == "l2":
            grad[:d] += self.alpha * self.coef_
            hessian[np.arange(d), np.arange(d)] += self.alpha
        return grad, hessian

    def _newton_line_search(self, X, y, sample_weight, grad, direction):
        """Шаг θ ← θ − t·direction с дроблением t = 1, 1/2, 1/4, ...

        Берётся первое t, при котором выполнено условие Армихо
        J(θ − t·d) ≤ J(θ) − 1e-4 · t · (g · d). Вдали от решения полный шаг
        Ньютона может увеличить цель (особенно у почти разделимых данных);
        дробление гарантирует, что цель не растёт. Если подходящего t нет
        (уже в минимуме с точностью float), параметры не меняются.
        """
        theta = self._params_vector()
        current = self._objective(X, y, sample_weight)
        slope = float(grad @ direction)
        t = 1.0
        while t > 1e-10:
            self._set_params_vector(theta - t * direction)
            if self._objective(X, y, sample_weight) <= current - 1e-4 * t * slope:
                return
            t /= 2
        self._set_params_vector(theta)

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
        if self.solver not in ("gradient", "proximal", "newton"):
            raise ValueError(
                "LogisticRegression supports solver 'gradient', 'proximal' or "
                f"'newton', got {self.solver!r}"
            )
        if self.solver == "newton" and self.penalty not in (None, "l2"):
            raise ValueError(
                "solver='newton' supports only penalty None or 'l2' "
                f"(the objective must be twice differentiable), got {self.penalty!r}"
            )
        if self.solver == "newton" and self.batch_size is not None:
            raise ValueError(
                "batch_size is not supported by solver='newton': "
                "each step uses all samples"
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
