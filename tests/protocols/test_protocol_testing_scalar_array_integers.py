"""Regression tests for exact zero-dimensional integer protocol values."""

import numpy as np
import pytest
from pyrecest.protocols.testing import (
    ProtocolAssertionError,
    assert_shape,
    assert_supports_dim,
    assert_supports_input_dim,
    assert_supports_sampling,
    assert_supports_transition_sampling,
    assert_trailing_dimension,
)


class _ArrayLike:
    def __init__(self, shape):
        self.shape = shape


class _Dimensions:
    dim = np.array(2, dtype=np.int64)
    input_dim = np.array(3, dtype=np.uint64)


class _Distribution:
    received_n = None

    def sample(self, n):
        self.received_n = n
        return _ArrayLike((n, 2))


class _TransitionModel:
    received_n = None

    def sample_next(self, state, n=1):
        self.received_n = n
        return _ArrayLike((n, len(state)))


def test_zero_dim_integer_arrays_are_valid_protocol_dimensions():
    obj = _Dimensions()

    assert assert_supports_dim(obj) == 2
    assert assert_supports_input_dim(obj) == 3
    assert assert_shape(_ArrayLike((np.array(2, dtype=np.int64),)), (2,)) == (2,)
    assert assert_trailing_dimension(
        _ArrayLike((1, np.array(3, dtype=np.uint64))),
        np.array(3, dtype=np.int64),
    ) == (1, 3)


def test_sampling_helpers_forward_normalized_python_integer_counts():
    distribution = _Distribution()
    transition_model = _TransitionModel()

    assert_shape(
        assert_supports_sampling(distribution, np.array(4, dtype=np.int64)),
        (4, 2),
    )
    assert_shape(
        assert_supports_transition_sampling(
            transition_model,
            (0.0, 1.0),
            np.array(5, dtype=np.uint64),
        ),
        (5, 2),
    )

    assert distribution.received_n == 4
    assert type(distribution.received_n) is int
    assert transition_model.received_n == 5
    assert type(transition_model.received_n) is int


@pytest.mark.parametrize(
    "value",
    [
        np.array(1.0),
        np.array(True),
        np.array([1], dtype=np.int64),
        np.ma.array(1, mask=True),
    ],
)
def test_non_exact_or_missing_scalar_arrays_remain_rejected(value):
    with pytest.raises(ProtocolAssertionError, match="must be an integer"):
        assert_supports_sampling(_Distribution(), value)
