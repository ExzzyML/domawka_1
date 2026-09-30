from abc import ABC, abstractmethod

import numpy as np


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