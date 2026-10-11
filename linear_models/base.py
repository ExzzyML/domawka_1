import math
import numbers
from abc import ABC, abstractmethod

import numpy as np


def _is_number(value):
    """True для обычного конечного числа (не bool, не nan, не inf)."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_int(value):
    """True для обычного целого числа (не bool)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


class NotFittedError(ValueError):
    """Модель используется до обучения."""


class BaseLinearModel(ABC):
    """Общая основа линейных моделей: параметры, проверки и градиентный спуск.

    Параметры: fit_intercept (обучать свободный член), solver ("gradient" или
    "analytic"), penalty (None, "l1", "l2" или "elasticnet"), alpha (сила штрафа,
    >= 0), learning_rate (шаг спуска, > 0), max_iter (максимум итераций, целое >= 1),
    tol (порог раннего останова по изменению цели, >= 0; 0.0 отключает останов),
    verbose и random_state (принимаются, но не используются), l1_ratio (доля L1
    в штрафе "elasticnet", число из [0, 1]; при других penalty не используется).

    После fit доступны coef_, intercept_, n_iter_ и loss_history_.
    """

    def __init__(
        self,
        fit_intercept=True,
        solver="gradient",
        penalty=None,
        alpha=0.0,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
        verbose=False,
        random_state=None,
        l1_ratio=0.5,
    ):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.l1_ratio = l1_ratio

    def _linear_output(self, X):
        return X @ self.coef_ + self.intercept_

    def _penalty_loss(self):
        """Штраф alpha * R(w) (число). intercept_ не участвует."""
        if self.penalty is None:
            return 0.0
        if self.penalty == "l2":
            return self.alpha * 0.5 * np.sum(self.coef_**2)
        if self.penalty == "l1":
            return self.alpha * np.sum(np.abs(self.coef_))
        if self.penalty == "elasticnet":
            l1 = np.sum(np.abs(self.coef_))
            l2 = 0.5 * np.sum(self.coef_**2)
            return self.alpha * (self.l1_ratio * l1 + (1 - self.l1_ratio) * l2)

    def _penalty_gradient(self):
        """Градиент штрафа по w (массив формы coef_). intercept_ не участвует."""
        if self.penalty is None:
            return np.zeros_like(self.coef_)
        if self.penalty == "l2":
            return self.alpha * self.coef_
        if self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        if self.penalty == "elasticnet":
            l1_part = self.l1_ratio * np.sign(self.coef_)
            l2_part = (1 - self.l1_ratio) * self.coef_
            return self.alpha * (l1_part + l2_part)

    def _objective(self, X, y):
        """Полная цель J = потери на данных + штраф (число)."""
        return self._loss(X, y) + self._penalty_loss()

    def _fit_gradient(self, X, y):
        """Градиентный спуск: обновляет coef_ и intercept_, ведёт историю."""
        self.loss_history_ = []
        for i in range(self.max_iter):
            grad_w, grad_b = self._gradient(X, y)
            grad_w = grad_w + self._penalty_gradient()
            self.coef_ = self.coef_ - self.learning_rate * grad_w
            if self.fit_intercept:
                self.intercept_ = self.intercept_ - self.learning_rate * grad_b
            self.loss_history_.append(self._objective(X, y))
            self.n_iter_ = i + 1
            if (
                len(self.loss_history_) > 1
                and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol
            ):
                break
        return self

    def _validate_params(self):
        if self.solver not in ("gradient", "analytic"):
            raise ValueError(
                f"solver must be 'gradient' or 'analytic', got {self.solver!r}"
            )
        if self.penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError(
                "penalty must be None, 'l1', 'l2' or 'elasticnet', "
                f"got {self.penalty!r}"
            )
        if not _is_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be a positive number, got {self.learning_rate!r}"
            )
        if not _is_int(self.max_iter) or self.max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {self.max_iter!r}")
        if not _is_number(self.tol) or self.tol < 0:
            raise ValueError(f"tol must be a non-negative number, got {self.tol!r}")
        if not _is_number(self.alpha) or self.alpha < 0:
            raise ValueError(f"alpha must be a non-negative number, got {self.alpha!r}")
        if not _is_number(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                f"l1_ratio must be a number in [0, 1], got {self.l1_ratio!r}"
            )

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional (n_samples, n_features), got ndim={X.ndim}"
            )
        return X

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional (n_samples,), got ndim={y.ndim}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]} samples"
            )
        return X, y

    def _check_is_fitted(self):
        if not (hasattr(self, "coef_") and hasattr(self, "intercept_")):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet; call fit before predict"
            )

    def _validate_predict_input(self, X):
        self._check_is_fitted()
        X = self._validate_X(X)
        n_features = self.coef_.shape[0]
        if X.shape[1] != n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted "
                f"with {n_features} features"
            )
        return X

    def _validate_score_input(self, X, y):
        X = self._validate_predict_input(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got {y.ndim}-D")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y must have the same length, got {X.shape[0]} and {len(y)}"
            )
        return X, y

    @abstractmethod
    def fit(self, X, y):
        """Обучить модель и вернуть self."""

    @abstractmethod
    def predict(self, X):
        """Вернуть предсказания для X."""

    @abstractmethod
    def _loss(self, X, y):
        """Вернуть значение потерь на данных (число), без регуляризации."""

    @abstractmethod
    def _gradient(self, X, y):
        """Вернуть пару (grad_w, grad_b): градиент потерь без регуляризации."""
