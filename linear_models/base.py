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


class BaseLinearModel(ABC):

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

    def _linear_output(self, X):
        return X @ self.coef_ + self.intercept_

    def _validate_params(self):
        if self.solver not in ("gradient", "analytic"):
            raise ValueError(
                f"solver must be 'gradient' or 'analytic', got {self.solver!r}"
            )
        if self.penalty not in (None, "l1", "l2"):
            raise ValueError(
                f"penalty must be None, 'l1' or 'l2', got {self.penalty!r}"
            )
        if not _is_number(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be a positive number, got {self.learning_rate!r}"
            )
        if not _is_int(self.max_iter) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be an integer >= 1, got {self.max_iter!r}"
            )
        if not _is_number(self.tol) or self.tol < 0:
            raise ValueError(
                f"tol must be a non-negative number, got {self.tol!r}"
            )
        if not _is_number(self.alpha) or self.alpha < 0:
            raise ValueError(
                f"alpha must be a non-negative number, got {self.alpha!r}"
            )


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