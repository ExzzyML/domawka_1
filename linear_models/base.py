import math
import numbers
import warnings
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


class ConvergenceWarning(UserWarning):
    """Градиентный спуск не сошёлся: исчерпан max_iter или цель стала inf/nan."""


class BaseLinearModel(ABC):
    """Общая основа линейных моделей: параметры, проверки и градиентный спуск.

    Параметры: fit_intercept (обучать свободный член), solver ("gradient" или
    "analytic"), penalty (None, "l1", "l2" или "elasticnet"), alpha (сила штрафа,
    >= 0), learning_rate (шаг спуска, > 0), max_iter (максимум итераций, целое >= 1),
    tol (порог раннего останова по изменению цели, >= 0; 0.0 отключает останов),
    verbose (печать прогресса), random_state (None или целое >= 0, для перемешивания), l1_ratio (доля L1
    в штрафе "elasticnet", число из [0, 1]; при других penalty не используется).

    Дополнительно: momentum (коэффициент скорости β, число из [0, 1); 0.0 — обычный
    спуск), nesterov (True: момент Нестерова, требует momentum > 0), lr_schedule
    (None, "exponential", "inverse_time" или "step": шаг убывает с номером итерации),
    lr_decay (параметр убывания, число > 0, для "exponential" и "step" не больше 1) и
    lr_step_size (для "step": шаг умножается на lr_decay каждые lr_step_size итераций).
    store_gradients (True: сохранить gradient_history_ формы (n_iter_, d + 1), строка
    i градиент цели перед шагом i, последний столбец интерцепт), callbacks (список
    функций cb(model, iteration, loss), вызываются после каждой итерации; True из
    функции останавливает обучение). verbose=True печатает прогресс, batch_size (None:
    градиент по всем объектам; целое >= 1: размер батча, 1 это SGD)
    и shuffle (перемешивать объекты в каждой эпохе; random_state задаёт
    перемешивание). При батчах итерация это эпоха: проход по данным, на каждый
    батч один шаг; loss_history_ и n_iter_ считаются по эпохам.

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
        nesterov=False,
        lr_schedule=None,
        lr_decay=0.9,
        lr_step_size=10,
        store_gradients=False,
        callbacks=None,
        batch_size=None,
        shuffle=True,
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
        self.nesterov = nesterov
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.lr_step_size = lr_step_size
        self.store_gradients = store_gradients
        self.callbacks = callbacks
        self.batch_size = batch_size
        self.shuffle = shuffle

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

    def _objective(self, X, y, sample_weight=None):
        """Полная цель J = потери на данных + штраф (число)."""
        return self._loss(X, y, sample_weight) + self._penalty_loss()

    def _learning_rate_at(self, t):
        """Шаг на итерации t (t = 0, 1, ...) с учётом расписания."""
        if self.lr_schedule is None:
            return self.learning_rate
        if self.lr_schedule == "exponential":
            return self.learning_rate * self.lr_decay**t
        if self.lr_schedule == "inverse_time":
            return self.learning_rate / (1 + self.lr_decay * t)
        if self.lr_schedule == "step":
            return self.learning_rate * self.lr_decay ** (t // self.lr_step_size)

    def _momentum_step(self, velocity, grad):
        """Новая скорость и направление шага: w = w - lr * direction.

        Обычный момент (heavy ball): v = β·v + g, direction = v.
        Нестеров (форма PyTorch): v = β·v + g, direction = g + β·v, то есть
        шаг делается так, будто скорость уже применили ещё раз («взгляд вперёд»).
        При β = 0 оба варианта дают direction = g, обычный спуск.
        """
        velocity = self.momentum * velocity + grad
        if self.nesterov:
            return velocity, grad + self.momentum * velocity
        return velocity, velocity

    def _epoch_batches(self, n_samples, batch_size, rng):
        """Батчи одной эпохи: None (все данные) или массивы индексов."""
        if batch_size >= n_samples:
            yield None
            return
        order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
        for start in range(0, n_samples, batch_size):
            yield order[start : start + batch_size]

    def _fit_gradient(self, X, y, sample_weight=None):
        """Градиентный спуск; предупреждения numpy о переполнении отключены.

        При расходимости numpy сам печатал бы «overflow encountered» на каждой
        операции; вместо этого спуск останавливается с одним понятным
        ConvergenceWarning (см. _gradient_descent).
        """
        with np.errstate(over="ignore", invalid="ignore"):
            return self._gradient_descent(X, y, sample_weight)

    def _gradient_descent(self, X, y, sample_weight=None):
        """Градиентный спуск по эпохам (с mini-batch, momentum и расписанием шага).

        Итерация это эпоха: один проход по данным, в котором на каждый батч
        делается один шаг. Обновляет coef_ и intercept_, ведёт историю,
        вызывает callbacks. Если цель стала inf/nan, спуск останавливается
        (reason "diverged"); при исчерпании max_iter с tol > 0 и при
        расходимости выдаётся ConvergenceWarning.
        """
        self.loss_history_ = []
        self.__dict__.pop("gradient_history_", None)
        gradients = []
        velocity_w = np.zeros_like(self.coef_)
        velocity_b = 0.0
        reason = "max_iter"
        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size is None else self.batch_size
        rng = np.random.default_rng(self.random_state)
        for i in range(self.max_iter):
            lr = self._learning_rate_at(i)
            for batch in self._epoch_batches(n_samples, batch_size, rng):
                if batch is None:
                    X_b, y_b, w_b = X, y, sample_weight
                else:
                    X_b, y_b = X[batch], y[batch]
                    w_b = None if sample_weight is None else sample_weight[batch]
                    if w_b is not None and np.sum(w_b) == 0:
                        continue
                grad_w, grad_b = self._gradient(X_b, y_b, w_b)
                grad_w = grad_w + self._penalty_gradient()
                if self.store_gradients:
                    gradients.append(
                        np.append(grad_w, grad_b if self.fit_intercept else 0.0)
                    )
                velocity_w, step_w = self._momentum_step(velocity_w, grad_w)
                self.coef_ = self.coef_ - lr * step_w
                if self.fit_intercept:
                    velocity_b, step_b = self._momentum_step(velocity_b, grad_b)
                    self.intercept_ = self.intercept_ - lr * step_b
            loss = self._objective(X, y, sample_weight)
            self.loss_history_.append(loss)
            self.n_iter_ = i + 1
            if not np.isfinite(loss):
                reason = "diverged"
                break
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
        self._warn_if_not_converged(reason)
        if self.verbose:
            print(
                f"stopped at iteration {self.n_iter_}: "
                f"J={self.loss_history_[-1]:.6g}, reason: {reason}"
            )
        return self

    def _warn_if_not_converged(self, reason):
        """ConvergenceWarning при расходимости или при исчерпании max_iter.

        При tol == 0 останов по tol отключён сознательно, и прогон всех
        max_iter итераций это ожидаемое поведение: предупреждения нет.
        """
        if reason == "diverged":
            warnings.warn(
                f"objective became {self.loss_history_[-1]} at iteration "
                f"{self.n_iter_}: gradient descent diverged, training stopped; "
                "decrease learning_rate or standardize features",
                ConvergenceWarning,
                stacklevel=5,
            )
        elif reason == "max_iter" and self.tol > 0:
            warnings.warn(
                f"gradient descent did not converge in max_iter={self.max_iter} "
                f"iterations (last objective change >= tol={self.tol}); "
                "increase max_iter or learning_rate",
                ConvergenceWarning,
                stacklevel=5,
            )

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
        if self.lr_schedule not in (None, "exponential", "inverse_time", "step"):
            raise ValueError(
                "lr_schedule must be None, 'exponential', 'inverse_time' or 'step', "
                f"got {self.lr_schedule!r}"
            )
        if not _is_number(self.lr_decay) or self.lr_decay <= 0:
            raise ValueError(
                f"lr_decay must be a positive number, got {self.lr_decay!r}"
            )
        if self.lr_schedule in ("exponential", "step") and self.lr_decay > 1:
            raise ValueError(
                f"lr_decay must be <= 1 for lr_schedule={self.lr_schedule!r}, "
                f"got {self.lr_decay!r}"
            )
        if not _is_int(self.lr_step_size) or self.lr_step_size < 1:
            raise ValueError(
                f"lr_step_size must be an integer >= 1, got {self.lr_step_size!r}"
            )
        if not _is_number(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError(
                f"momentum must be a number in [0, 1), got {self.momentum!r}"
            )
        if self.nesterov is not True and self.nesterov is not False:
            raise ValueError(f"nesterov must be True or False, got {self.nesterov!r}")
        if self.nesterov and self.momentum == 0:
            raise ValueError("nesterov=True requires momentum > 0")
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
        if self.batch_size is not None and (
            not _is_int(self.batch_size) or self.batch_size < 1
        ):
            raise ValueError(
                f"batch_size must be None or an integer >= 1, got {self.batch_size!r}"
            )
        if self.shuffle is not True and self.shuffle is not False:
            raise ValueError(f"shuffle must be True or False, got {self.shuffle!r}")
        if self.random_state is not None and (
            not _is_int(self.random_state) or self.random_state < 0
        ):
            raise ValueError(
                f"random_state must be None or an integer >= 0, "
                f"got {self.random_state!r}"
            )
        if not _is_number(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                f"l1_ratio must be a number in [0, 1], got {self.l1_ratio!r}"
            )

    def _validate_sample_weight(self, sample_weight, n_samples):
        """None остаётся None; иначе вернуть веса как float-массив (n,)."""
        if sample_weight is None:
            return None
        try:
            weights = np.asarray(sample_weight, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("sample_weight must contain numbers") from None
        if weights.ndim != 1:
            raise ValueError(
                f"sample_weight must be 1-dimensional, got ndim={weights.ndim}"
            )
        if weights.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight has {weights.shape[0]} values, "
                f"but there are {n_samples} samples"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite and non-negative")
        if np.sum(weights) <= 0:
            raise ValueError("sample_weight must have a positive sum")
        return weights

    @staticmethod
    def _total_weight(sample_weight, n_samples):
        """Знаменатель средних: n без весов, иначе сумма весов."""
        return n_samples if sample_weight is None else np.sum(sample_weight)

    @staticmethod
    def _apply_weight(values, sample_weight):
        """values без весов, иначе sample_weight * values."""
        return values if sample_weight is None else sample_weight * values

    @staticmethod
    def _as_finite_float_array(values, name):
        """Вернуть values как float-массив; числа обязаны быть конечными.

        Строки и прочие нечисловые значения, а также NaN и ±inf дают
        понятный ValueError с именем массива (X, y).
        """
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain only numbers") from None
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must not contain NaN or infinity")
        return array

    def _validate_X(self, X):
        X = self._as_finite_float_array(X, "X")
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional (n_samples, n_features), got ndim={X.ndim}"
            )
        return X

    def _validate_X_y(self, X, y):
        X = self._validate_X(X)
        y = self._as_finite_float_array(y, "y")
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
        self._as_finite_float_array(y, "y")
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got {y.ndim}-D")
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y must have the same length, got {X.shape[0]} and {len(y)}"
            )
        return X, y

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Обучить модель и вернуть self."""

    @abstractmethod
    def predict(self, X):
        """Вернуть предсказания для X."""

    @abstractmethod
    def _loss(self, X, y, sample_weight=None):
        """Вернуть потери на данных (число), без регуляризации.

        С весами это взвешенное среднее: деление на сумму весов, а не на n.
        """

    @abstractmethod
    def _gradient(self, X, y, sample_weight=None):
        """Вернуть пару (grad_w, grad_b): градиент потерь без регуляризации."""
