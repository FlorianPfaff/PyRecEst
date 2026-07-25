import pytest
from pyrecest._backend.numpy import random


@pytest.mark.parametrize(
    ("low", "high", "expected_message"),
    [
        ([[0.0], [0.5, 0.75]], 1.0, "low must be real numeric"),
        (0.0, [[1.0], [1.5, 1.75]], "high must be real numeric"),
    ],
)
def test_uniform_rejects_ragged_bounds_with_normalized_type_error(
    low, high, expected_message
):
    with pytest.raises(TypeError, match=expected_message):
        random.uniform(low, high)
