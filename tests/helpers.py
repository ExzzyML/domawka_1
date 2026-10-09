import numpy as np


def numerical_gradient(model, X, y, eps=1e-6):
    """Градиент _loss по (coef_, intercept_) через центральную разность."""
    grad_w = np.zeros_like(model.coef_)
    for j in range(model.coef_.shape[0]):
        original = model.coef_[j]
        model.coef_[j] = original + eps
        loss_plus = model._loss(X, y)
        model.coef_[j] = original - eps
        loss_minus = model._loss(X, y)
        model.coef_[j] = original
        grad_w[j] = (loss_plus - loss_minus) / (2 * eps)

    original_b = model.intercept_
    model.intercept_ = original_b + eps
    loss_plus = model._loss(X, y)
    model.intercept_ = original_b - eps
    loss_minus = model._loss(X, y)
    model.intercept_ = original_b
    grad_b = (loss_plus - loss_minus) / (2 * eps)

    return grad_w, grad_b
