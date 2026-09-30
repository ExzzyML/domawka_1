import pytest

from linear_models.base import BaseLinearModel


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseLinearModel()