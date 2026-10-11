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
    verbose (печать прогресса), random_state (пока не используется), l1_ratio (доля L1
    в штрафе "elasticnet", число из [0, 1]; при других penalty не используется).
    Дополнительно: momentum (коэффициент скорости β, число из [0, 1); 0.0 — обычный
    спуск), lr_schedule (None, "exponential" или "inverse_time": шаг убывает с номером
    итерации) и lr_decay (параметр убывания, число > 0, для "exponential" не больше 1).
    store_gradients (True: сохранить gradient_history_ формы (n_iter_, d + 1), строка
    i градиент цели перед шагом i, последний столбец интерцепт), callbacks (список
    функций cb(model, iteration, loss), вызываются после каждой итерации; True из
    функции останавливает обучение). verbose=True печатает прогресс.

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
        momentum=0.0,
        lr_schedule=None,
        lr_decay=0.9,
        store_gradients=False,
        callbacks=None,
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
        self.momentum = momentum
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.store_gradients = store_gradients
        self.callbacks = callbacks

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

    def _learning_rate_at(self, t):
        """Шаг на итерации t (t = 0, 1, ...) с учётом расписания."""
        if self.lr_schedule is None:
            return self.learning_rate
        if self.lr_schedule == "exponential":
            return self.learning_rate * self.lr_decay**t
        if self.lr_schedule == "inverse_time":
            return self.learning_rate / (1 + self.lr_decay * t)

    def _fit_gradient(self, X, y):
        """Градиентный спуск (с momentum и расписанием шага): обновляет
        coef_ и intercept_, ведёт историю, вызывает callbacks."""
        self.loss_history_ = []
        self.__dict__.pop("gradient_history_", None)
        gradients = []
        velocity_w = np.zeros_like(self.coef_)
        velocity_b = 0.0
        reason = "max_iter"
        for i in range(self.max_iter):
            lr = self._learning_rate_at(i)
            grad_w, grad_b = self._gradient(X, y)
            grad_w = grad_w + self._penalty_gradient()
            if self.store_gradients:
                gradients.append(
                    np.append(grad_w, grad_b if self.fit_intercept else 0.0)
                )
            velocity_w = self.momentum * velocity_w + grad_w
            self.coef_ = self.coef_ - lr * velocity_w
            if self.fit_intercept:
                velocity_b = self.momentum * velocity_b + grad_b
                self.intercept_ = self.intercept_ - lr * velocity_b
            loss = self._objective(X, y)
            self.loss_history_.append(loss)
            self.n_iter_ = i + 1
            if self.verbose and (i == 0 or (i + 1) % 100 == 0):
                print(f"iter {i + 1}: J={loss:.6g}")
            if self._run_callbacks(i + 1, loss):
                reason = "callback"
                break
            if (
                len(self.loss_history_) > 1
                and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol
            ):
                reason = "tol"
                break
        if self.store_gradients:
            self.gradient_history_ = np.array(gradients)
        if self.verbose:
            print(
                f"stopped at iteration {self.n_iter_}: "
                f"J={self.loss_history_[-1]:.6g}, reason: {reason}"
            )
        return self

    def _run_callbacks(self, iteration, loss):
        """Вызвать все callbacks; True, если хотя бы один вернул True."""
        stop = False
        for callback in self.callbacks or ():
            result = callback(self, iteration, loss)
            if isinstance(result, (bool, np.bool_)) and result:
                stop = True
        return stop

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
        if self.lr_schedule not in (None, "exponential", "inverse_time"):
            raise ValueError(
                "lr_schedule must be None, 'exponential' or 'inverse_time', "
                f"got {self.lr_schedule!r}"
            )
        if not _is_number(self.lr_decay) or self.lr_decay <= 0:
            raise ValueError(
                f"lr_decay must be a positive number, got {self.lr_decay!r}"
            )
        if self.lr_schedule == "exponential" and self.lr_decay > 1:
            raise ValueError(
                "lr_decay must be <= 1 for lr_schedule='exponential', "
                f"got {self.lr_decay!r}"
            )
        if not _is_number(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError(
                f"momentum must be a number in [0, 1), got {self.momentum!r}"
            )
        if self.store_gradients is not True and self.store_gradients is not False:
            raise ValueError(
                f"store_gradients must be True or False, got {self.store_gradients!r}"
            )
        if self.callbacks is not None and not (
            isinstance(self.callbacks, (list, tuple))
            and all(callable(cb) for cb in self.callbacks)
        ):
            raise ValueError(
                "callbacks must be None or a list/tuple of callables, "
                f"got {self.callbacks!r}"
            )
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
