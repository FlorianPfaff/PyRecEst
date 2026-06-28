import pytest
from pyrecest._backend.numpy import random


def test_normal_rejects_boolean_location_parameter():
    with pytest.raises(TypeError):
        random.normal(loc=True)
