import pytest
from pyrecest._backend import numpy as backend


def test_uniform_rejects_ragged_bounds_with_normalized_type_error():
    with pytest.raises(TypeError, match="low must be real numeric"):
        backend.random.uniform([[0.0], [0.5, 0.75]], 1.0)
    with pytest.raises(TypeError, match="high must be real numeric"):
        backend.random.uniform(0.0, [[1.0], [1.5, 2.0]])
