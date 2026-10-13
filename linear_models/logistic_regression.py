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


def _softmax(z):
    """Устойчивая softmax по строкам: exp(z_ik) / sum_j exp(z_ij)."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)


class LogisticRegression(BaseLinearModel):
    """Логистическая регрессия: два класса (0 и 1) или несколько (softmax).

    Если y состоит из 0 и 1, обучается двоичная модель: coef_ формы (d,),
    потери mean(log(1 + e^z) - y * z), класс 1 присваивается, если P(1)
    строго больше threshold (по умолчанию 0.5). Иначе y должен содержать
    целые метки классов, и обучается мультиномиальная (softmax) модель:
    coef_ формы (K, d), intercept_ формы (K,), потери
    mean(log Σ_j e^{z_ij} - z_{i,y_i}), класс выбирается по наибольшей
    вероятности (threshold не используется). Метки в classes_ отсортированы.

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
        """Обучить на X формы (n, d) и y из {0, 1} или целых меток; вернуть self.

        Нужны хотя бы два класса, и у каждого класса суммарный вес объектов
        положителен. sample_weight: веса объектов (n,), неотрицательные,
        сумма > 0; None: все веса равны 1.
        """
        self._validate_params()
        self._validate_solver()
        X, y = self._validate_X_y(X, y)
        sample_weight = self._validate_sample_weight(sample_weight, X.shape[0])
        classes = self._validate_labels(y, sample_weight)
        n_features = X.shape[1]
        if classes.size == 2 and classes[0] == 0 and classes[1] == 1:
            self.coef_ = np.zeros(n_features)
            self.intercept_ = 0.0
        else:
            y = np.searchsorted(classes, y)
            self.coef_ = np.zeros((classes.size, n_features))
            self.intercept_ = np.zeros(classes.size)
        self.classes_ = classes
        if self.solver == "newton":
            self._fit_newton(X, y, sample_weight)
        else:
            self._fit_gradient(X, y, sample_weight)
        return self

    def _is_multiclass(self):
        return self.coef_.ndim == 2

    def _linear_output(self, X):
        """z = X w + b (n,) у двух классов, z = X Wᵀ + b (n, K) у нескольких."""
        if self._is_multiclass():
            return X @ self.coef_.T + self.intercept_
        return super()._linear_output(X)

    def _fit_newton(self, X, y, sample_weight=None):
        """Метод Ньютона; предупреждения numpy о переполнении отключены."""
        with np.errstate(over="ignore", invalid="ignore"):
            return self._newton_method(X, y, sample_weight)

    def _newton_method(self, X, y, sample_weight=None):
        def step(i, gradients):
            grad, hessian = self._gradient_and_hessian(X, y, sample_weight)
            if self.store_gradients:
                gradients.append(self._newton_gradient_row(grad))
            direction = np.linalg.lstsq(hessian, grad, rcond=None)[0]
            self._newton_line_search(X, y, sample_weight, grad, direction)

        return self._run_iterations(X, y, sample_weight, step)

    def _newton_gradient_row(self, grad):
        """Плоский градиент Ньютона как строка gradient_history_."""
        if not self._is_multiclass():
            return grad if self.fit_intercept else np.append(grad, 0.0)
        by_class = grad.reshape(self.coef_.shape[0], -1)
        if not self.fit_intercept:
            by_class = np.column_stack([by_class, np.zeros(by_class.shape[0])])
        return by_class.ravel()

    def _params_vector(self):
        """θ = (w, b) при fit_intercept, иначе θ = w.

        У нескольких классов строки (w_k, b_k) идут подряд: форма (K, d + 1)
        разворачивается в вектор длины K·(d + 1) (без интерцепта K·d).
        """
        if self._is_multiclass():
            columns = [self.coef_]
            if self.fit_intercept:
                columns.append(self.intercept_[:, None])
            return np.hstack(columns).ravel()
        if self.fit_intercept:
            return np.append(self.coef_, self.intercept_)
        return self.coef_.copy()

    def _set_params_vector(self, theta):
        if self._is_multiclass():
            n_classes, d = self.coef_.shape
            by_class = theta.reshape(n_classes, -1)
            self.coef_ = by_class[:, :d].copy()
            if self.fit_intercept:
                self.intercept_ = by_class[:, d].copy()
            return
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
        if self._is_multiclass():
            return self._multinomial_gradient_and_hessian(X, y, sample_weight)
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

    def _multinomial_gradient_and_hessian(self, X, y, sample_weight=None):
        """Градиент и гессиан цели softmax по θ = строки (w_k, b_k) подряд.

        Здесь y номера классов 0..K-1, Xa = [X, 1] (столбец единиц только при
        fit_intercept), P = softmax(Z). Градиент (K, da) = (P − Y)ᵀ Xa / Σs,
        где s веса объектов, Y one-hot. Блок гессиана (k, l) равен
        Xaᵀ diag(s·(δ_kl P_k − P_k P_l)) Xa / Σs. Штраф L2 добавляет
        alpha·w_k к градиенту и alpha к диагонали гессиана на местах весов
        (интерцепты не штрафуются).
        """
        n, d = X.shape
        n_classes = self.coef_.shape[0]
        P = _softmax(self._linear_output(X))
        total = self._total_weight(sample_weight, n)
        Xa = np.column_stack([X, np.ones(n)]) if self.fit_intercept else X
        da = Xa.shape[1]
        residual = P.copy()
        residual[np.arange(n), y] -= 1.0
        weights = np.ones(n) if sample_weight is None else sample_weight
        grad = (residual * weights[:, None]).T @ Xa / total
        curvature = P[:, :, None] * np.eye(n_classes) - P[:, :, None] * P[:, None, :]
        curvature = curvature * weights[:, None, None]
        hessian = np.einsum("nkl,ni,nj->kilj", curvature, Xa, Xa, optimize=True)
        hessian = hessian.reshape(n_classes * da, n_classes * da) / total
        if self.penalty == "l2":
            grad[:, :d] += self.alpha * self.coef_
            weight_positions = (
                np.arange(n_classes)[:, None] * da + np.arange(d)
            ).ravel()
            hessian[weight_positions, weight_positions] += self.alpha
        return grad.ravel(), hessian

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
        """Вернуть вероятности формы (n, K); столбцы идут в порядке classes_.

        Два класса: столбцы P(класс 0) и P(класс 1). Несколько классов:
        softmax от X Wᵀ + b, строки суммируются в 1.
        """

        X = self._validate_predict_input(X)
        if self._is_multiclass():
            return _softmax(self._linear_output(X))
        p1 = _sigmoid(self._linear_output(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        """Вернуть метки классов (тип int).

        Два класса: 1, если P(1) > threshold. Несколько классов: класс с
        наибольшей вероятностью (при равенстве первый в classes_); threshold
        не используется.
        """

        self._validate_threshold()
        proba = self.predict_proba(X)
        if self._is_multiclass():
            return self.classes_[np.argmax(proba, axis=1)]
        return (proba[:, 1] > self.threshold).astype(int)

    def _validate_binary_y(self, y, sample_weight=None):
        if not np.all((y == 0) | (y == 1)):
            raise ValueError(f"y must contain only 0 and 1, got values {np.unique(y)}")
        used = y if sample_weight is None else y[sample_weight > 0]
        if np.unique(used).size < 2:
            raise ValueError(
                "y must contain both classes 0 and 1 "
                "(among samples with positive sample_weight)"
            )

    def _validate_labels(self, y, sample_weight=None):
        """Проверить метки и вернуть отсортированный массив классов.

        y из {0, 1}: двоичный случай (оба класса нужны среди объектов с
        положительным весом). Иначе метки должны быть целыми, классов не
        меньше двух, и у каждого класса суммарный вес положителен.
        """
        if np.all((y == 0) | (y == 1)):
            self._validate_binary_y(y, sample_weight)
            return np.array([0, 1])
        if not np.all(y == np.round(y)):
            raise ValueError(
                "y must contain integer class labels (0 and 1 for two classes), "
                f"got values {np.unique(y)}"
            )
        classes = np.unique(y).astype(int)
        if classes.size < 2:
            raise ValueError("y must contain at least two different classes")
        if sample_weight is not None:
            for label in classes:
                if np.sum(sample_weight[y == label]) <= 0:
                    raise ValueError(
                        f"class {label} has zero total sample_weight: "
                        "every class needs a positive weight"
                    )
        return classes

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
        if self._is_multiclass():
            # y номера классов 0..K-1: logsumexp(z_i) − z_{i, y_i}
            top = np.max(z, axis=1)
            log_sum = top + np.log(np.sum(np.exp(z - top[:, None]), axis=1))
            per_object = log_sum - z[np.arange(len(y)), y]
            total = self._total_weight(sample_weight, X.shape[0])
            return np.sum(self._apply_weight(per_object, sample_weight)) / total
        per_object = np.logaddexp(0, z) - y * z
        total = self._total_weight(sample_weight, X.shape[0])
        return np.sum(self._apply_weight(per_object, sample_weight)) / total

    def _gradient(self, X, y, sample_weight=None):
        if self._is_multiclass():
            residual = _softmax(self._linear_output(X))
            residual[np.arange(len(y)), y] -= 1.0
            total = self._total_weight(sample_weight, X.shape[0])
            if sample_weight is not None:
                residual = residual * sample_weight[:, None]
            return residual.T @ X / total, np.sum(residual, axis=0) / total
        residual = _sigmoid(self._linear_output(X)) - y
        total = self._total_weight(sample_weight, X.shape[0])
        weighted = self._apply_weight(residual, sample_weight)
        grad_w = (X.T @ weighted) / total
        grad_b = np.sum(weighted) / total
        return grad_w, grad_b
