"""Regression tests for temporal values used as protocol dimensions."""

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


class _DimensionObject:
    dim = np.timedelta64(2, "ns")
    input_dim = np.datetime64("1970-01-01T00:00:00.000000003", "ns")


class _Distribution:
    def sample(self, n):
        return _ArrayLike((n,))


class _TransitionModel:
    def sample_next(self, state, n=1):
        return _ArrayLike((n, len(state)))


_TEMPORAL_VALUES = (
    np.timedelta64(2, "ns"),
    np.timedelta64(2, "us"),
    np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
)


@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_shape_helpers_reject_temporal_dimensions(temporal_value):
    with pytest.raises(ProtocolAssertionError, match="must be an integer"):
        assert_shape(_ArrayLike((temporal_value,)), (2,))

    with pytest.raises(ProtocolAssertionError, match="must be an integer"):
        assert_shape(_ArrayLike((2,)), (temporal_value,))

    with pytest.raises(ProtocolAssertionError, match="must be an integer"):
        assert_trailing_dimension(_ArrayLike((2,)), temporal_value)


@pytest.mark.parametrize("temporal_value", _TEMPORAL_VALUES)
def test_sampling_helpers_reject_temporal_counts(temporal_value):
    with pytest.raises(ProtocolAssertionError, match="n must be an integer"):
        assert_supports_sampling(_Distribution(), temporal_value)

    with pytest.raises(ProtocolAssertionError, match="n must be an integer"):
        assert_supports_transition_sampling(_TransitionModel(), (0.0,), temporal_value)


def test_dimension_protocol_helpers_reject_temporal_attributes():
    obj = _DimensionObject()

    with pytest.raises(ProtocolAssertionError, match="dim must be an integer"):
        assert_supports_dim(obj)

    with pytest.raises(ProtocolAssertionError, match="input_dim must be an integer"):
        assert_supports_input_dim(obj)


def test_numpy_integer_dimensions_remain_supported():
    assert assert_shape(_ArrayLike((np.int64(2),)), (2,)) == (2,)
    assert assert_trailing_dimension(_ArrayLike((2, np.int64(3))), np.int64(3)) == (
        2,
        3,
    )
